"""Cancellation of background RL and diffusion runs."""

import importlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import numpy as np
import pytest
from fastapi import HTTPException

from gridarena.diffusion.schemas import DiffusionRunStatus, DiffusionTrainingConfig, DiffusionTrainRequest
from gridarena.rl.schemas import RunStatus, TrainRequest
from gridarena.routers import diffusion as diff_router
from gridarena.routers import rl as rl_router
from gridarena.run_cancellation import (
    CancelCheck,
    TrainingCancelled,
    is_cancel_requested,
    raise_if_cancelled,
    request_cancel,
)

# `gridarena.diffusion` / `gridarena.rl` attribute access resolves to the routers
# of the same name, so load the run managers by module path.
diff_rm = importlib.import_module("gridarena.diffusion.run_manager")
rl_rm = importlib.import_module("gridarena.rl.run_manager")


def _rl_status(status, **kw):
    return RunStatus(run_id="abc12345", grid_id="g1", agent_type="single",
                     status=status, created_at="2026-01-01T00:00:00", **kw)


def _diff_status(status, **kw):
    return DiffusionRunStatus(run_id="abc12345", grid_ids=["g1"], status=status,
                              created_at="2026-01-01T00:00:00", **kw)


# ── Marker helpers ──


def test_marker_roundtrip(tmp_path):
    assert not is_cancel_requested(tmp_path)
    raise_if_cancelled(tmp_path)  # no-op while not requested
    request_cancel(tmp_path)
    assert is_cancel_requested(tmp_path)
    with pytest.raises(TrainingCancelled):
        raise_if_cancelled(tmp_path)


def test_cancel_check_polls_every_n_calls(tmp_path):
    check = CancelCheck(tmp_path, every=3)
    request_cancel(tmp_path)
    assert [check(), check(), check()] == [False, False, True]
    assert check() is True  # stays cancelled without re-polling


# ── Router endpoints ──


@pytest.mark.asyncio
@patch("gridarena.routers.rl.registry")
async def test_rl_cancel_not_found(mock_registry):
    mock_registry.get_run.return_value = None
    with pytest.raises(HTTPException) as exc:
        await rl_router.cancel_run("abc12345")
    assert exc.value.status_code == 404


@pytest.mark.asyncio
@patch("gridarena.routers.rl.registry")
async def test_rl_cancel_already_finished(mock_registry):
    mock_registry.get_run.return_value = _rl_status("completed")
    with pytest.raises(HTTPException) as exc:
        await rl_router.cancel_run("abc12345")
    assert exc.value.status_code == 409
    mock_registry.cancel.assert_not_called()


@pytest.mark.asyncio
@patch("gridarena.routers.rl.registry")
async def test_rl_cancel_running(mock_registry):
    mock_registry.get_run.return_value = _rl_status("running")
    mock_registry.cancel.return_value = _rl_status("running", cancel_requested=True)
    resp = await rl_router.cancel_run("abc12345")
    assert resp["status"] == "running"
    assert "next step" in resp["message"]
    mock_registry.cancel.assert_called_once_with("abc12345")


@pytest.mark.asyncio
@patch("gridarena.routers.diffusion.diffusion_registry")
async def test_diffusion_cancel_queued(mock_registry):
    mock_registry.get_run.return_value = _diff_status("queued")
    mock_registry.cancel.return_value = _diff_status("cancelled")
    resp = await diff_router.cancel_diffusion_run("abc12345")
    assert resp == {"run_id": "abc12345", "status": "cancelled", "message": "Run cancelled."}


@pytest.mark.asyncio
@patch("gridarena.routers.diffusion.diffusion_registry")
async def test_diffusion_cancel_already_cancelled(mock_registry):
    mock_registry.get_run.return_value = _diff_status("cancelled")
    with pytest.raises(HTTPException) as exc:
        await diff_router.cancel_diffusion_run("abc12345")
    assert exc.value.status_code == 409


# ── Registries ──


@pytest.fixture
def blocking_job():
    """A job that holds the single pool worker until released."""
    release = threading.Event()

    def job(run_dir):
        release.wait(10)
        return {"status": "completed"}

    yield job, release
    release.set()


def test_rl_registry_cancels_queued_and_flags_running(tmp_path, monkeypatch, blocking_job):
    job, release = blocking_job
    monkeypatch.setattr(rl_rm, "RUNS_DIR", tmp_path)
    rl_jobs = importlib.import_module("gridarena.rl.jobs")  # gridarena.rl is shadowed by routers.rl
    monkeypatch.setattr(rl_jobs, "run_training_job", job)
    reg = rl_rm.RunRegistry()
    reg._pool = ThreadPoolExecutor(max_workers=1)

    req = TrainRequest(grid_id="g1")
    running_id = reg.start_training(req, {})
    queued_id = reg.start_training(req, {})

    queued = reg.cancel(queued_id)
    assert queued.status == "cancelled"
    on_disk = json.loads((tmp_path / queued_id / "run_config.json").read_text())
    assert on_disk["status"] == "cancelled"

    running = reg.cancel(running_id)
    assert running.cancel_requested is True
    assert is_cancel_requested(tmp_path / running_id)

    release.set()
    reg._pool.shutdown(wait=True)


def test_diffusion_registry_cancels_queued(tmp_path, monkeypatch, blocking_job):
    job, release = blocking_job
    monkeypatch.setattr(diff_rm, "RUNS_DIR", tmp_path)
    diff_jobs = importlib.import_module("gridarena.diffusion.jobs")
    monkeypatch.setattr(diff_jobs, "run_training_job", job)
    reg = diff_rm.DiffusionRunRegistry()
    reg._pool = ThreadPoolExecutor(max_workers=1)

    req = DiffusionTrainRequest(grid_ids=["g1"], num_epochs=1)
    reg.start_training(req)
    queued_id = reg.start_training(req)

    assert reg.cancel(queued_id).status == "cancelled"
    assert reg.get_run(queued_id).status == "cancelled"

    release.set()
    reg._pool.shutdown(wait=True)


def test_registry_cancel_ignores_finished_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(rl_rm, "RUNS_DIR", tmp_path)
    reg = rl_rm.RunRegistry()
    reg._runs["abc12345"] = _rl_status("completed")
    assert reg.cancel("abc12345").status == "completed"
    assert not is_cancel_requested(tmp_path / "abc12345")


# ── Worker side ──


def _write_config(run_dir, **data):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_config.json").write_text(json.dumps({"status": "queued", **data}))


def test_rl_worker_exits_when_cancelled_before_start(tmp_path):
    from gridarena.rl.training import run_training

    _write_config(tmp_path, grid_config={}, agent_type="single", timesteps=1000)
    request_cancel(tmp_path)
    assert run_training(str(tmp_path)) == {"status": "cancelled"}
    assert json.loads((tmp_path / "run_config.json").read_text())["status"] == "cancelled"


def test_diffusion_worker_exits_when_cancelled_before_start(tmp_path):
    from gridarena.diffusion.training import run_diffusion_training

    _write_config(tmp_path, grid_ids=["g1"])
    request_cancel(tmp_path)
    assert run_diffusion_training(str(tmp_path)) == {"status": "cancelled"}
    assert json.loads((tmp_path / "run_config.json").read_text())["status"] == "cancelled"


def test_diffusion_training_loop_stops_and_saves_nothing(tmp_path):
    from gridarena.diffusion.model_core import train_diffusion_model

    rng = np.random.default_rng(0)
    data = rng.normal(size=(4, 4, 48)).astype("float32")
    config = DiffusionTrainingConfig(image_size=(4, 48), num_epochs=2, train_batch_size=2,
                                     power_subtract=0.0, power_rescale=1.0)
    with pytest.raises(TrainingCancelled):
        train_diffusion_model(data, config, str(tmp_path), should_stop=lambda: True)
    assert not (tmp_path / "model").exists()
