"""Runs harness.py over a prepared submission work directory, isolated from the app.

Two backends, picked with GRIDARENA_SUBMISSION_SANDBOX:

``subprocess`` (default)
    A child Python process (the server's own interpreter, so the user gets the
    same packages: numpy, pandas, scipy, torch, ...) with a stripped environment:
    no database DSN, API keys or other secrets are inherited. It still runs as
    the server's OS user, so it can read the files that user can read and open
    network connections -- only use it for trusted submitters (local or
    in-house deployments).

``docker``
    A throw-away container with no network, a read-only root filesystem,
    memory/CPU/process limits and only the work directory mounted. Use this for
    any deployment that accepts code from untrusted users. The image must have
    Python and the packages submissions may import; by default it is the
    GridArena image itself (build it with ``docker build -t gridarena .``).

Settings (environment variables):
    GRIDARENA_SUBMISSION_SANDBOX   subprocess | docker            (subprocess)
    GRIDARENA_SUBMISSION_TIMEOUT   seconds for the whole run      (900)
    GRIDARENA_SUBMISSION_IMAGE     docker image                   (gridarena)
    GRIDARENA_SUBMISSION_MEMORY    docker --memory                (4g)
    GRIDARENA_SUBMISSION_CPUS      docker --cpus                  (2)
"""

import logging
import os
import signal
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Only these variables reach the user's process; everything else (GRID_DB_DSN,
# DATABASE_URL, LLM_API_KEY, ...) is dropped.
_ENV_ALLOWLIST = (
    "PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "TMPDIR", "HOME", "USERPROFILE",
    "LANG", "LC_ALL", "CUDA_VISIBLE_DEVICES",
)

_active: set = set()
_active_lock = threading.Lock()


@dataclass
class SandboxResult:
    returncode: int
    timed_out: bool


def backend() -> str:
    return os.getenv("GRIDARENA_SUBMISSION_SANDBOX", "subprocess").strip().lower()


def timeout_seconds() -> float:
    return float(os.getenv("GRIDARENA_SUBMISSION_TIMEOUT", "900"))


def _clean_env() -> dict:
    env = {k: os.environ[k] for k in _ENV_ALLOWLIST if k in os.environ}
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _subprocess_command(workdir: Path) -> list:
    return [sys.executable, str(workdir / "harness.py"), str(workdir)]


def _docker_command(workdir: Path, container_name: str) -> list:
    cmd = [
        "docker", "run", "--rm", "--name", container_name,
        "--network", "none",
        "--read-only", "--tmpfs", "/tmp",
        "--memory", os.getenv("GRIDARENA_SUBMISSION_MEMORY", "4g"),
        "--cpus", os.getenv("GRIDARENA_SUBMISSION_CPUS", "2"),
        "--pids-limit", "512",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "-e", "PYTHONUNBUFFERED=1", "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "HOME=/tmp",
        "-v", f"{workdir}:/work",
        "-w", "/work/code",
        "--entrypoint", "python",
    ]
    if hasattr(os, "getuid"):
        # Write outputs as the server's user, so it can read and delete them.
        cmd += ["--user", f"{os.getuid()}:{os.getgid()}"]
    return cmd + [os.getenv("GRIDARENA_SUBMISSION_IMAGE", "gridarena"), "/work/harness.py", "/work"]


def _kill(proc: subprocess.Popen, container_name: str | None) -> None:
    if container_name:
        # Killing the docker CLI leaves the container running; stop it by name.
        subprocess.run(["docker", "kill", container_name], capture_output=True, timeout=30)
    if os.name != "nt":
        try:
            os.killpg(proc.pid, signal.SIGKILL)  # also the user's own child processes
        except ProcessLookupError:
            pass
    else:
        proc.kill()


def run_harness(workdir: Path, log_path: Path, *, run_id: str, timeout: float | None = None) -> SandboxResult:
    """Run harness.py on ``workdir`` with the configured backend, appending its
    stdout/stderr to ``log_path``. Blocks until it exits or times out."""
    timeout = timeout_seconds() if timeout is None else timeout
    container_name = None
    if backend() == "docker":
        container_name = f"gridarena-submission-{run_id}"
        cmd = _docker_command(workdir, container_name)
        env = None  # the docker CLI needs the normal environment (DOCKER_HOST, ...)
    elif backend() == "subprocess":
        cmd = _subprocess_command(workdir)
        env = _clean_env()
    else:
        raise ValueError(f"Unknown GRIDARENA_SUBMISSION_SANDBOX backend: {backend()!r}")

    with open(log_path, "ab") as log:
        proc = subprocess.Popen(
            cmd, cwd=workdir / "code", env=env, stdout=log, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=(os.name != "nt"),
        )
        with _active_lock:
            _active.add((proc, container_name))
        try:
            return SandboxResult(returncode=proc.wait(timeout=timeout), timed_out=False)
        except subprocess.TimeoutExpired:
            logger.warning("Submission %s timed out after %s s; killing it", run_id, timeout)
            _kill(proc, container_name)
            proc.wait()
            return SandboxResult(returncode=proc.returncode, timed_out=True)
        finally:
            with _active_lock:
                _active.discard((proc, container_name))


def terminate_all() -> None:
    """Kill every running submission (called on server shutdown)."""
    with _active_lock:
        running = list(_active)
    for proc, container_name in running:
        if proc.poll() is None:
            _kill(proc, container_name)
