"""Manages MV PF data-generation jobs: lifecycle, background execution,
status tracking. Mirrors gridarena/diffusion/run_manager.py's registry shape."""

import json
import logging
import multiprocessing
import re
import uuid
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from gridarena.pf_datagen.schemas import PFDataGenRequest, PFDataGenRunStatus

logger = logging.getLogger(__name__)

RUNS_DIR = Path(__file__).resolve().parent / "runs"
GROUPS_DIR = RUNS_DIR / "groups"

# run_ids are server-generated (uuid4 prefix, 8 hex chars); anything else is
# rejected before it is used to build a path under RUNS_DIR.
_RUN_ID_RE = re.compile(r"[0-9a-f]{8}")


def _status_from_data(run_id: str, data: dict) -> PFDataGenRunStatus:
    return PFDataGenRunStatus(
        run_id=run_id,
        mv_grid_id=data.get("mv_grid_id", ""),
        status=data.get("status", "unknown"),
        created_at=data.get("created_at", ""),
        completed_at=data.get("completed_at"),
        error=data.get("error"),
        config=data.get("config"),
        total_scenarios=data.get("total_scenarios", 0),
        completed_scenarios=data.get("completed_scenarios", 0),
        failed_scenarios=data.get("failed_scenarios", 0),
        n_topology_variants=data.get("n_topology_variants", 1),
        seed_used=data.get("seed_used"),
        metrics=data.get("metrics"),
        group_id=data.get("group_id"),
    )


class PFDataGenRunRegistry:
    def __init__(self, max_workers: int = 1):
        self._runs: dict[str, PFDataGenRunStatus] = {}
        self._pool: Optional[ProcessPoolExecutor] = None
        self._max_workers = max_workers

    def startup(self):
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        ctx = multiprocessing.get_context("spawn")
        self._pool = ProcessPoolExecutor(max_workers=self._max_workers, mp_context=ctx)
        self._rebuild_from_disk()
        logger.info("PFDataGenRunRegistry started with %d workers (spawn)", self._max_workers)

    def shutdown(self):
        if self._pool:
            # cancel_futures only drops queued (not-yet-started) work; a
            # worker already running is otherwise orphaned on restart, so
            # forcefully terminate any live worker processes too. This
            # relies on ProcessPoolExecutor's private _processes attribute --
            # concurrent.futures has no public API for this.
            for p in list(getattr(self._pool, "_processes", {}).values()):
                if p.is_alive():
                    p.terminate()
            self._pool.shutdown(wait=False, cancel_futures=True)
            logger.info("PFDataGenRunRegistry shut down")

    def _rebuild_from_disk(self):
        for config_file in RUNS_DIR.glob("*/run_config.json"):
            try:
                with open(config_file) as f:
                    data = json.load(f)
                run_id = data["run_id"]
                if data.get("status") == "running":
                    data["status"] = "failed"
                    data["error"] = "Server restarted during job"
                    with open(config_file, "w") as f:
                        json.dump(data, f, indent=2)
                self._runs[run_id] = _status_from_data(run_id, data)
            except Exception as e:
                logger.warning("Failed to load PF datagen run config %s: %s", config_file, e)

    # ── Job creation ──

    def start_run(self, mv_grid_id: str, request: PFDataGenRequest, group_id: Optional[str] = None) -> str:
        from gridarena.pf_datagen.job import run_pf_datagen

        run_id = str(uuid.uuid4())[:8]
        run_dir = RUNS_DIR / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        now = datetime.now(timezone.utc).isoformat()

        run_config = request.model_dump(mode="json")
        run_config.update({
            "run_id": run_id,
            "mv_grid_id": mv_grid_id,
            "status": "queued",
            "created_at": now,
            "config": request.model_dump(mode="json"),
            "group_id": group_id,
        })

        config_path = run_dir / "run_config.json"
        with open(config_path, "w") as f:
            json.dump(run_config, f, indent=2)

        self._runs[run_id] = _status_from_data(run_id, run_config)

        future = self._pool.submit(run_pf_datagen, str(run_dir))

        def _on_done(fut):
            try:
                fut.result()  # surface subprocess exceptions into the log
            except Exception as e:
                # The worker crashed (OOM, segfault, BrokenProcessPool, ...) before
                # it could write its own "failed" status to run_config.json -- mark
                # it failed directly instead of falling through to the disk read
                # below, which would otherwise re-read a stale "running"/"queued"
                # status and silently mask the crash.
                logger.error("PF datagen run %s raised: %s", run_id, e, exc_info=True)
                if run_id in self._runs:
                    self._runs[run_id].status = "failed"
                    self._runs[run_id].error = str(e)
                return
            try:
                with open(config_path) as f:
                    data = json.load(f)
                self._runs[run_id] = _status_from_data(run_id, data)
                logger.info("PF datagen run %s finished: %s", run_id, data.get("status"))
            except Exception as e:
                if run_id in self._runs:
                    self._runs[run_id].status = "failed"
                    self._runs[run_id].error = str(e)
                logger.error("PF datagen run %s callback error: %s", run_id, e)

        future.add_done_callback(_on_done)
        return run_id

    # ── Groups ──

    def start_group(self, grids: list, request: PFDataGenRequest, skipped: list) -> str:
        """One run per grid of `grids` ([{mv_grid_id, n_nodes,
        n_connection_points}]), sharing a group id. Each grid's seed is
        derived from the group seed and its position, so the grids don't
        share their random draws and the group reproduces from one seed."""
        import numpy as np

        from gridarena.pf_datagen.package import grid_seeds

        group_id = str(uuid.uuid4())[:8]
        seed = request.seed if request.seed is not None else int(np.random.SeedSequence().entropy % (2**32))
        runs = []
        for g, grid_seed in zip(grids, grid_seeds(seed, len(grids))):
            run_id = self.start_run(g["mv_grid_id"], request.model_copy(update={"seed": grid_seed}), group_id)
            runs.append({**g, "run_id": run_id, "seed": grid_seed})
        group = {"group_id": group_id, "created_at": datetime.now(timezone.utc).isoformat(), "seed": seed,
                 "runs": runs, "skipped": skipped, "config": request.model_dump(mode="json")}
        GROUPS_DIR.mkdir(parents=True, exist_ok=True)
        with open(GROUPS_DIR / f"{group_id}.json", "w") as f:
            json.dump(group, f, indent=2)
        return group_id

    def get_group(self, group_id: str) -> Optional[dict]:
        if not _RUN_ID_RE.fullmatch(group_id):
            return None
        path = GROUPS_DIR / f"{group_id}.json"
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)

    def list_groups(self) -> list:
        out = []
        for path in GROUPS_DIR.glob("*.json"):
            try:
                with open(path) as f:
                    g = json.load(f)
                out.append({"group_id": g["group_id"], "created_at": g.get("created_at"),
                            "n_runs": len(g.get("runs", [])), "n_skipped": len(g.get("skipped", []))})
            except Exception as e:
                logger.warning("Failed to load PF datagen group %s: %s", path, e)
        return sorted(out, key=lambda g: g["created_at"] or "", reverse=True)

    # ── Queries ──

    def _fresh(self, run_id: str) -> PFDataGenRunStatus:
        """A queued or running run's status as its worker last wrote it to
        run_config.json (the worker is another process, so the in-memory copy
        only changes when the run ends). A finished or crashed run keeps its
        in-memory status; an unreadable file (mid-write) keeps the last one."""
        status = self._runs[run_id]
        if status.status not in ("queued", "running"):
            return status
        try:
            with open(RUNS_DIR / run_id / "run_config.json") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return status
        if self._runs.get(run_id) is status:          # not finished meanwhile by _on_done
            self._runs[run_id] = _status_from_data(run_id, data)
        return self._runs[run_id]

    def get_run(self, run_id: str) -> Optional[PFDataGenRunStatus]:
        if run_id in self._runs:
            return self._fresh(run_id)
        if not _RUN_ID_RE.fullmatch(run_id):
            return None
        config_path = RUNS_DIR / run_id / "run_config.json"
        if config_path.exists():
            with open(config_path) as f:
                data = json.load(f)
            return _status_from_data(run_id, data)
        return None

    def get_report(self, run_id: str) -> Optional[dict]:
        """The run's diversity report (report.json, written when the run
        finishes), or None if there's none yet."""
        if not _RUN_ID_RE.fullmatch(run_id):
            return None
        path = RUNS_DIR / run_id / "report.json"
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)

    def list_runs(self, mv_grid_id: Optional[str] = None) -> list[PFDataGenRunStatus]:
        runs = sorted((self._fresh(rid) for rid in list(self._runs)), key=lambda r: r.created_at, reverse=True)
        if mv_grid_id is not None:
            runs = [r for r in runs if r.mv_grid_id == mv_grid_id]
        return runs


pf_datagen_registry = PFDataGenRunRegistry()
