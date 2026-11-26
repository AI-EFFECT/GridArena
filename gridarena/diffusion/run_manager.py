"""Manages diffusion jobs (train / privacy check / epoch search): lifecycle,
background execution, status tracking."""

import json
import logging
import multiprocessing
import re
import uuid
from concurrent.futures import Future, ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from gridarena.diffusion.schemas import (
    DiffusionEpochSearchRequest,
    DiffusionPrivacyCheckRequest,
    DiffusionRunStatus,
    DiffusionTrainRequest,
)
from gridarena.run_cancellation import request_cancel

logger = logging.getLogger(__name__)

RUNS_DIR = Path(__file__).resolve().parent / "runs"

# run_id is always server-generated as uuid.uuid4().hex[:8]; enforced here as
# defense-in-depth before the id is used to build a filesystem path.
_RUN_ID_RE = re.compile(r"^[0-9a-f]{8}$")


def _status_from_data(run_id: str, data: dict) -> DiffusionRunStatus:
    """Build a DiffusionRunStatus from a run_config.json dict."""
    status = data.get("status", "unknown")
    return DiffusionRunStatus(
        run_id=run_id,
        grid_ids=data.get("grid_ids", []),
        status=status,
        created_at=data.get("created_at", ""),
        completed_at=data.get("completed_at"),
        # epoch search reports its max_epochs; other jobs their num_epochs.
        num_epochs=data.get("max_epochs") or data.get("num_epochs", 0),
        error=data.get("error"),
        metrics=data.get("metrics"),
        dataset_size=data.get("dataset_size", 0),
        job_type=data.get("job_type", "train"),
        architecture=data.get("architecture"),
    )


class DiffusionRunRegistry:
    def __init__(self, max_workers: int = 1):
        self._runs: dict[str, DiffusionRunStatus] = {}
        self._futures: dict[str, Future] = {}
        self._pool: Optional[ProcessPoolExecutor] = None
        # Separate single-slot pool for evaluation / generation, so inference
        # neither imports torch into the API process nor waits behind training.
        self._infer_pool: Optional[ProcessPoolExecutor] = None
        self._max_workers = max_workers

    def startup(self):
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        ctx = multiprocessing.get_context("spawn")
        self._pool = ProcessPoolExecutor(max_workers=self._max_workers, mp_context=ctx)
        self._infer_pool = ProcessPoolExecutor(max_workers=1, mp_context=ctx)
        self._rebuild_from_disk()
        logger.info("DiffusionRunRegistry started with %d workers (spawn)", self._max_workers)

    def run_inference(self, fn, *args) -> Future:
        """Run a lightweight job entry point (see the ``jobs`` module) in the
        inference worker and return its Future."""
        if self._infer_pool is None:
            raise RuntimeError("Registry not started.")
        return self._infer_pool.submit(fn, *args)

    def shutdown(self):
        if self._infer_pool:
            self._infer_pool.shutdown(wait=False, cancel_futures=True)
        if self._pool:
            # cancel_futures only drops queued (not-yet-started) work; a
            # worker already training is otherwise orphaned on restart, so
            # forcefully terminate any live worker processes too. This
            # relies on ProcessPoolExecutor's private _processes attribute --
            # concurrent.futures has no public API for this.
            for p in list(getattr(self._pool, "_processes", {}).values()):
                if p.is_alive():
                    p.terminate()
            self._pool.shutdown(wait=False, cancel_futures=True)
            logger.info("DiffusionRunRegistry shut down")

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
                logger.warning("Failed to load run config %s: %s", config_file, e)

    # ── Job creation ──

    def _create_run(self, request, job_type: str, entry_fn) -> str:
        run_id = str(uuid.uuid4())[:8]
        run_dir = RUNS_DIR / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        now = datetime.now(timezone.utc).isoformat()

        run_config = request.model_dump()
        run_config.update({
            "run_id": run_id,
            "job_type": job_type,
            "status": "queued",
            "created_at": now,
        })

        config_path = run_dir / "run_config.json"
        with open(config_path, "w") as f:
            json.dump(run_config, f, indent=2)

        self._runs[run_id] = _status_from_data(run_id, run_config)

        future = self._pool.submit(entry_fn, str(run_dir))
        self._futures[run_id] = future

        def _on_done(fut):
            self._futures.pop(run_id, None)
            if fut.cancelled():
                # Cancelled while still queued: the worker never ran.
                self._mark_cancelled(run_id)
                return
            try:
                fut.result()  # surface subprocess exceptions into the log
            except Exception as e:
                # The worker crashed (OOM, segfault, BrokenProcessPool, ...) before
                # it could write its own "failed" status to run_config.json -- mark
                # it failed directly instead of falling through to the disk read
                # below, which would otherwise re-read a stale "running"/"queued"
                # status and silently mask the crash.
                logger.error("Diffusion %s job %s raised: %s", job_type, run_id, e, exc_info=True)
                if run_id in self._runs:
                    self._runs[run_id].status = "failed"
                    self._runs[run_id].error = str(e)
                return
            try:
                with open(config_path) as f:
                    data = json.load(f)
                self._runs[run_id] = _status_from_data(run_id, data)
                logger.info("Diffusion %s run %s finished: %s", job_type, run_id, data.get("status"))
            except Exception as e:
                if run_id in self._runs:
                    self._runs[run_id].status = "failed"
                    self._runs[run_id].error = str(e)
                logger.error("Diffusion run %s callback error: %s", run_id, e)

        future.add_done_callback(_on_done)
        return run_id

    # The job entry points live in a lightweight module so the API process never
    # imports torch/diffusers; the heavy imports happen inside the worker.
    def start_training(self, request: DiffusionTrainRequest) -> str:
        from gridarena.diffusion.jobs import run_training_job
        return self._create_run(request, "train", run_training_job)

    def start_privacy_check(self, request: DiffusionPrivacyCheckRequest) -> str:
        from gridarena.diffusion.jobs import run_privacy_check_job
        return self._create_run(request, "privacy_check", run_privacy_check_job)

    def start_epoch_search(self, request: DiffusionEpochSearchRequest) -> str:
        from gridarena.diffusion.jobs import run_epoch_search_job
        return self._create_run(request, "epoch_search", run_epoch_search_job)

    # ── Cancellation ──

    def cancel(self, run_id: str) -> Optional[DiffusionRunStatus]:
        """Ask a queued or running job to stop.

        A queued job is dropped from the pool straight away. A running one is
        signalled through its cancel marker and stops at the next batch, after
        which its status becomes "cancelled".
        """
        run = self.get_run(run_id)
        if run is None or run.status not in ("queued", "running"):
            return run
        request_cancel(self.get_run_dir(run_id))
        future = self._futures.get(run_id)
        if future is not None and future.cancel():
            return self._runs.get(run_id, run)
        if run_id in self._runs:
            self._runs[run_id].cancel_requested = True
        return self._runs.get(run_id, run)

    def _mark_cancelled(self, run_id: str):
        config_path = RUNS_DIR / run_id / "run_config.json"
        try:
            with open(config_path) as f:
                data = json.load(f)
            data.update({"status": "cancelled", "completed_at": datetime.now(timezone.utc).isoformat()})
            with open(config_path, "w") as f:
                json.dump(data, f, indent=2)
            self._runs[run_id] = _status_from_data(run_id, data)
        except Exception as e:
            logger.warning("Could not record cancellation for diffusion run %s: %s", run_id, e)
            if run_id in self._runs:
                self._runs[run_id].status = "cancelled"
        logger.info("Diffusion run %s cancelled before it started", run_id)

    # ── Queries ──

    def get_run(self, run_id: str) -> Optional[DiffusionRunStatus]:
        if run_id in self._runs:
            return self._runs[run_id]
        if not _RUN_ID_RE.match(run_id):
            return None
        config_path = RUNS_DIR / run_id / "run_config.json"
        if config_path.exists():
            with open(config_path) as f:
                data = json.load(f)
            return _status_from_data(run_id, data)
        return None

    def list_runs(self) -> list[DiffusionRunStatus]:
        return sorted(self._runs.values(), key=lambda r: r.created_at, reverse=True)

    def get_run_dir(self, run_id: str) -> Path:
        if not _RUN_ID_RE.match(run_id):
            raise ValueError(f"Invalid run_id: {run_id!r}")
        return RUNS_DIR / run_id


diffusion_registry = DiffusionRunRegistry()
