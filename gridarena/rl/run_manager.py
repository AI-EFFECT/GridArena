"""Manages training runs: lifecycle, background execution, status tracking."""

import json
import logging
import multiprocessing
import re
import uuid
from concurrent.futures import Future, ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from gridarena.rl.schemas import RunStatus, TrainRequest
from gridarena.run_cancellation import request_cancel

logger = logging.getLogger(__name__)

RUNS_DIR = Path(__file__).resolve().parent / "runs"

# run_id is always server-generated as uuid.uuid4().hex[:8]; enforced here as
# defense-in-depth before the id is used to build a filesystem path.
_RUN_ID_RE = re.compile(r"^[0-9a-f]{8}$")


class RunRegistry:
    def __init__(self, max_workers: int = 2):
        self._runs: dict[str, RunStatus] = {}
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
        logger.info("RunRegistry started with %d workers (spawn)", self._max_workers)

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
            logger.info("RunRegistry shut down")

    def _rebuild_from_disk(self):
        for config_file in RUNS_DIR.glob("*/run_config.json"):
            try:
                with open(config_file) as f:
                    data = json.load(f)
                run_id = data["run_id"]
                status = data.get("status", "unknown")
                if status == "running":
                    status = "failed"
                    data["status"] = "failed"
                    data["error"] = "Server restarted during training"
                    with open(config_file, "w") as f:
                        json.dump(data, f, indent=2)

                self._runs[run_id] = RunStatus(
                    run_id=run_id,
                    grid_id=data["grid_id"],
                    agent_type=data["agent_type"],
                    status=status,
                    created_at=data["created_at"],
                    completed_at=data.get("completed_at"),
                    timesteps=data.get("timesteps", 0),
                    error=data.get("error"),
                    metrics=data.get("metrics"),
                )
            except Exception as e:
                logger.warning("Failed to load run config %s: %s", config_file, e)

    def start_training(self, request: TrainRequest, grid_config_dict: dict) -> str:
        from gridarena.rl.jobs import run_training_job

        run_id = str(uuid.uuid4())[:8]
        run_dir = RUNS_DIR / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        now = datetime.now(timezone.utc).isoformat()

        run_config = {
            "run_id": run_id,
            "grid_id": request.grid_id,
            "agent_type": request.agent_type,
            "timesteps": request.timesteps,
            "status": "queued",
            "created_at": now,
            "device": request.device,
            "active_wrappers": request.active_wrappers,
            "wrapper_config": request.wrapper_config.model_dump(),
            "seed": request.seed,
            "grid_config": grid_config_dict,
        }

        config_path = run_dir / "run_config.json"
        with open(config_path, "w") as f:
            json.dump(run_config, f, indent=2)

        self._runs[run_id] = RunStatus(
            run_id=run_id,
            grid_id=request.grid_id,
            agent_type=request.agent_type,
            status="queued",
            created_at=now,
            timesteps=request.timesteps,
        )

        future = self._pool.submit(run_training_job, str(run_dir))
        self._futures[run_id] = future

        def _on_done(fut):
            self._futures.pop(run_id, None)
            if fut.cancelled():
                # Cancelled while still queued: the worker never ran.
                self._mark_cancelled(run_id)
                return
            try:
                result = fut.result()
                with open(config_path) as f:
                    data = json.load(f)
                self._runs[run_id] = RunStatus(
                    run_id=run_id,
                    grid_id=request.grid_id,
                    agent_type=request.agent_type,
                    status=data.get("status", "completed"),
                    created_at=now,
                    completed_at=data.get("completed_at"),
                    timesteps=request.timesteps,
                    error=data.get("error"),
                    metrics=data.get("metrics"),
                )
                logger.info("Run %s finished: %s", run_id, data.get("status"))
            except Exception as e:
                self._runs[run_id].status = "failed"
                self._runs[run_id].error = str(e)
                logger.error("Run %s crashed: %s", run_id, e, exc_info=True)

        future.add_done_callback(_on_done)
        return run_id

    def cancel(self, run_id: str) -> Optional[RunStatus]:
        """Ask a queued or running run to stop.

        A queued run is dropped from the pool straight away. A running one is
        signalled through its cancel marker and stops at the next training
        step check, after which its status becomes "cancelled".
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
        now = datetime.now(timezone.utc).isoformat()
        try:
            with open(config_path) as f:
                data = json.load(f)
            data.update({"status": "cancelled", "completed_at": now})
            with open(config_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning("Could not record cancellation for run %s: %s", run_id, e)
        if run_id in self._runs:
            self._runs[run_id].status = "cancelled"
            self._runs[run_id].completed_at = now
            self._runs[run_id].cancel_requested = False
        logger.info("Run %s cancelled before it started", run_id)

    def get_run(self, run_id: str) -> Optional[RunStatus]:
        if run_id in self._runs:
            return self._runs[run_id]
        if not _RUN_ID_RE.match(run_id):
            return None
        config_path = RUNS_DIR / run_id / "run_config.json"
        if config_path.exists():
            with open(config_path) as f:
                data = json.load(f)
            return RunStatus(
                run_id=run_id,
                grid_id=data["grid_id"],
                agent_type=data["agent_type"],
                status=data.get("status", "unknown"),
                created_at=data["created_at"],
                completed_at=data.get("completed_at"),
                timesteps=data.get("timesteps", 0),
                error=data.get("error"),
                metrics=data.get("metrics"),
            )
        return None

    def list_runs(self) -> list[RunStatus]:
        return sorted(self._runs.values(), key=lambda r: r.created_at, reverse=True)

    def get_run_dir(self, run_id: str) -> Path:
        if not _RUN_ID_RE.match(run_id):
            raise ValueError(f"Invalid run_id: {run_id!r}")
        return RUNS_DIR / run_id


registry = RunRegistry()
