"""Cooperative cancellation for background training jobs (RL and diffusion).

Jobs run in ProcessPoolExecutor workers. Terminating a worker process breaks
the whole pool (every later submission fails with BrokenProcessPool), so a
cancel request instead drops a marker file in the run directory. The training
loops poll for it between steps and stop cleanly, recording status "cancelled".
"""

from pathlib import Path

CANCEL_MARKER = "CANCEL"


class TrainingCancelled(Exception):
    """Raised inside a job when its run has been asked to stop."""


def request_cancel(run_dir) -> None:
    (Path(run_dir) / CANCEL_MARKER).touch()


def is_cancel_requested(run_dir) -> bool:
    return (Path(run_dir) / CANCEL_MARKER).exists()


def raise_if_cancelled(run_dir) -> None:
    if is_cancel_requested(run_dir):
        raise TrainingCancelled("Training cancelled by user")


class CancelCheck:
    """Callable that reports whether the run was cancelled, polling the marker
    file at most every ``every`` calls (a filesystem stat per training step is
    wasteful, especially on slow mounts)."""

    def __init__(self, run_dir, every: int = 1):
        self.run_dir = run_dir
        self.every = max(1, every)
        self._calls = 0
        self.cancelled = False

    def __call__(self) -> bool:
        if not self.cancelled:
            self._calls += 1
            if self._calls % self.every == 0:
                self.cancelled = is_cancel_requested(self.run_dir)
        return self.cancelled
