"""Code-submission benchmark runs: queueing, background execution, status tracking.

A benchmark plugs in through a BenchmarkTask: it builds the hidden test inputs
(and keeps the ground truth on the server side), names the function the user's
script must define, and scores the outputs. The user's code only ever sees the
inputs, never the truth -- the truth stays in this process.

Each run lives in RUNS_DIR/<submission_id>/:

    run.json        status record (what the API returns)
    log.txt         the submission's stdout/stderr
    code/           the uploaded files (kept, for reproducibility)

plus, only while the run is being evaluated, the harness work files
(harness.py, manifest.json, inputs/, outputs/, status.json).
"""

import json
import logging
import os
import re
import shutil
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from gridarena.benchmarks.submissions import sandbox
from gridarena.benchmarks.submissions.packaging import ENTRYPOINT, install_submission

logger = logging.getLogger(__name__)

RUNS_DIR = Path(__file__).resolve().parent / "runs"
HARNESS = Path(__file__).resolve().parent / "harness.py"

# submission_id is always server-generated as uuid4().hex[:12]; enforced as
# defense-in-depth before it is used to build a filesystem path.
_ID_RE = re.compile(r"^[0-9a-f]{12}$")

LOG_TAIL_CHARS = 8000

_WORK_FILES = ("harness.py", "manifest.json", "status.json", "inputs", "outputs")


@dataclass
class PreparedInputs:
    """``calls`` maps an input name to the positional args of one call of the
    user's function (JSON-serialisable); ``truth`` is whatever ``score`` needs
    and never leaves the server."""
    calls: Dict[str, list]
    truth: Any


@dataclass(frozen=True)
class BenchmarkTask:
    name: str
    function: str
    build_inputs: Callable[[dict], PreparedInputs]          # settings -> inputs
    score: Callable[[Dict[str, dict], Any], dict]            # (outputs by input name, truth) -> score


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tail(path: Path, chars: int = LOG_TAIL_CHARS) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text if len(text) <= chars else "...\n" + text[-chars:]


class SubmissionRegistry:
    def __init__(self, runs_dir: Path = RUNS_DIR, max_workers: Optional[int] = None):
        self._runs_dir = runs_dir
        self._max_workers = max_workers or int(os.getenv("GRIDARENA_SUBMISSION_WORKERS", "1"))
        self._pool: Optional[ThreadPoolExecutor] = None
        self._lock = threading.Lock()
        self._started = False

    # ── lifecycle ──

    def _ensure_started(self) -> None:
        # Lazy, so routers work without the app lifespan (tests, in-process LLM tools).
        with self._lock:
            if self._started:
                return
            self._runs_dir.mkdir(parents=True, exist_ok=True)
            self._fail_interrupted_runs()
            self._pool = ThreadPoolExecutor(max_workers=self._max_workers, thread_name_prefix="submission")
            self._started = True

    def shutdown(self) -> None:
        sandbox.terminate_all()
        with self._lock:
            if self._pool:
                self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None
            self._started = False

    def _fail_interrupted_runs(self) -> None:
        for run_file in self._runs_dir.glob("*/run.json"):
            try:
                record = json.loads(run_file.read_text(encoding="utf-8"))
                if record.get("status") in ("queued", "running"):
                    record.update(status="failed", completed_at=_now(),
                                  error="The server restarted before this submission finished.")
                    run_file.write_text(json.dumps(record, indent=2), encoding="utf-8")
                    self._cleanup_work(run_file.parent)
            except Exception as e:
                logger.warning("Failed to load submission record %s: %s", run_file, e)

    # ── records ──

    def _run_dir(self, submission_id: str) -> Path:
        if not _ID_RE.match(submission_id):
            raise ValueError(f"Invalid submission id: {submission_id!r}")
        return self._runs_dir / submission_id

    def _read(self, submission_id: str) -> Optional[dict]:
        try:
            path = self._run_dir(submission_id) / "run.json"
        except ValueError:
            return None
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _update(self, submission_id: str, **fields) -> dict:
        with self._lock:
            record = self._read(submission_id) or {}
            record.update(fields)
            path = self._run_dir(submission_id) / "run.json"
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(record, indent=2), encoding="utf-8")
            tmp.replace(path)
            return record

    def get(self, submission_id: str, benchmark: str) -> Optional[dict]:
        """The record plus the tail of its log, or None if unknown / another benchmark's."""
        self._ensure_started()
        record = self._read(submission_id)
        if record is None or record.get("benchmark") != benchmark:
            return None
        return {**record, "log_tail": _tail(self._run_dir(submission_id) / "log.txt")}

    def list(self, benchmark: str, submission_name: Optional[str] = None) -> List[dict]:
        """Records of one benchmark, newest first (no logs)."""
        self._ensure_started()
        records = []
        for run_file in self._runs_dir.glob("*/run.json"):
            try:
                record = json.loads(run_file.read_text(encoding="utf-8"))
            except Exception:
                continue
            if record.get("benchmark") != benchmark:
                continue
            if submission_name and record.get("submission_name") != submission_name:
                continue
            records.append(record)
        return sorted(records, key=lambda r: r.get("created_at", ""), reverse=True)

    # ── submit / execute ──

    def submit(self, task: BenchmarkTask, *, filename: str, data: bytes,
               submission_name: str, settings: dict) -> dict:
        """Validate and store the upload, queue its evaluation and return its record.
        Raises packaging.SubmissionError for a malformed upload."""
        self._ensure_started()
        submission_id = uuid.uuid4().hex[:12]
        run_dir = self._run_dir(submission_id)
        try:
            install_submission(filename, data, run_dir / "code", task.function)
        except Exception:
            shutil.rmtree(run_dir, ignore_errors=True)
            raise

        record = {
            "submission_id": submission_id,
            "benchmark": task.name,
            "submission_name": submission_name,
            "settings": settings,
            "filename": Path(filename).name,
            "status": "queued",
            "created_at": _now(),
            "started_at": None,
            "completed_at": None,
            "score": None,
            "error": None,
        }
        (run_dir / "run.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        self._pool.submit(self._execute, task, submission_id, settings)
        return record

    def _execute(self, task: BenchmarkTask, submission_id: str, settings: dict) -> None:
        run_dir = self._run_dir(submission_id)
        self._update(submission_id, status="running", started_at=_now())
        try:
            score, error = self._evaluate(task, submission_id, run_dir, settings)
        except Exception as e:
            logger.exception("Submission %s crashed during evaluation", submission_id)
            score, error = None, f"Internal error while evaluating the submission: {e}"
        finally:
            self._cleanup_work(run_dir)
        self._update(submission_id, status="failed" if error else "completed",
                     completed_at=_now(), score=score, error=error)

    def _evaluate(self, task: BenchmarkTask, submission_id: str, run_dir: Path, settings: dict):
        try:
            prepared = task.build_inputs(settings)
        except Exception as e:
            logger.warning("Could not build %s test inputs: %s", task.name, e)
            return None, f"Could not load the test data: {e}"

        names = sorted(prepared.calls)
        (run_dir / "inputs").mkdir()
        for name in names:
            with open(run_dir / "inputs" / f"{name}.json", "w", encoding="utf-8") as f:
                json.dump({"args": prepared.calls[name]}, f)
        (run_dir / "manifest.json").write_text(json.dumps(
            {"entrypoint": ENTRYPOINT, "function": task.function, "inputs": names}), encoding="utf-8")
        shutil.copy(HARNESS, run_dir / "harness.py")

        result = sandbox.run_harness(run_dir, run_dir / "log.txt", run_id=submission_id)
        if result.timed_out:
            return None, f"The submission did not finish within {sandbox.timeout_seconds():g} s."

        status_file = run_dir / "status.json"
        status = json.loads(status_file.read_text(encoding="utf-8")) if status_file.exists() else {}
        if not status.get("loaded"):
            detail = (status.get("error") or "").strip().splitlines()
            return None, ("The submission could not be loaded: "
                          + (detail[-1] if detail else f"harness exited with code {result.returncode}")
                          + " (see the log for the full traceback).")

        outputs = {}
        for name in names:
            out_file = run_dir / "outputs" / f"{name}.json"
            try:
                outputs[name] = json.loads(out_file.read_text(encoding="utf-8"))
            except Exception:
                outputs[name] = {"ok": False, "error": "No output was written (the process exited early).",
                                 "seconds": None}
        return task.score(outputs, prepared.truth), None

    @staticmethod
    def _cleanup_work(run_dir: Path) -> None:
        # The inputs are hidden test data: don't leave copies on disk.
        for name in _WORK_FILES:
            path = run_dir / name
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            elif path.exists():
                path.unlink()


submission_registry = SubmissionRegistry()
