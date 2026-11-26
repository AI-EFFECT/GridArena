"""Background database preparation (gridarena/startup.py) and the loading page
the middleware serves meanwhile. The database is mocked throughout."""
import time
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import gridarena.database as db
from gridarena import startup
from gridarena.app import app

_DB_CALLS = ["create_grid_database", "create_anon_map_database", "create_phase_guesses_database",
             "create_topology_guesses_database", "create_voltage_timestamp_map_table",
             "create_vc_guesses_database", "create_voltage_control_solutions", "create_base_tables",
             "create_state_estimates_table", "create_mv_database", "create_pf_datagen_database",
             "historical_record_counts_need_fill", "ensure_historical_record_counts"]


def _patched_db(stack, **overrides):
    """Mock every database call the preparation makes; returns the mocks."""
    conn, cursor = MagicMock(), MagicMock()
    cursor.fetchone.return_value = (True,)
    mocks = {"get_db_connection": stack.enter_context(
        patch.object(db, "get_db_connection", return_value=(conn, cursor)))}
    for name in _DB_CALLS:
        mocks[name] = stack.enter_context(patch.object(db, name, **overrides.get(name, {})))
    mocks["historical_record_counts_need_fill"].return_value = False
    return mocks


def test_preparation_runs_every_step_once():
    s = startup.DatabaseStartup()
    with ExitStack() as stack:
        mocks = _patched_db(stack)
        s._begin()
        s._run_once()
    st = s.status()
    assert st["state"] == "ready" and st["progress"] == 1.0 and st["error"] is None
    assert [x["status"] for x in st["steps"]] == ["done"] * 5
    for name in ("create_mv_database", "create_pf_datagen_database", "ensure_historical_record_counts"):
        mocks[name].assert_called_once()
    assert not s.blocks_requests()


def test_failed_step_reports_error_and_retry_recovers():
    s = startup.DatabaseStartup()
    with ExitStack() as stack:
        _patched_db(stack, create_mv_database={"side_effect": RuntimeError("relation lock timeout")})
        s._begin()
        s._run_once()
    st = s.status()
    assert st["state"] == "failed" and "relation lock timeout" in st["error"]
    assert [x["status"] for x in st["steps"]] == ["done", "done", "failed", "pending", "pending"]
    assert s.blocks_requests()

    with ExitStack() as stack:
        _patched_db(stack)
        s.start()
        s._thread.join(timeout=10)
    assert s.state == "ready"


def test_progress_uses_the_estimate_of_a_long_step():
    s = startup.DatabaseStartup()
    s._begin()
    for step in s.steps[:4]:
        step.status = "done"
    long_step = s.steps[4]
    long_step.status, long_step.estimate_s = "running", 100.0
    long_step.started = time.monotonic() - 50.0          # half of the estimate elapsed
    # weights: 4 quick steps of 1 + one of 100 -> (4 + 50) / 104
    assert s.status()["progress"] == pytest.approx(54 / 104, abs=0.01)
    long_step.started = time.monotonic() - 500.0         # overrun: capped at 95 % of the step
    assert s.status()["progress"] == pytest.approx((4 + 95) / 104, abs=0.01)


@pytest.fixture
def preparing():
    """The app as it is while the database is being prepared."""
    saved = (startup.db_startup.state, startup.db_startup.steps, startup.db_startup.started)
    startup.db_startup._begin()
    startup.db_startup.steps[0].status = "done"
    startup.db_startup.steps[1].status = "running"
    with patch("gridarena.settings_store.db_configured", return_value=True):
        yield TestClient(app)
    startup.db_startup.state, startup.db_startup.steps, startup.db_startup.started = saved


def test_pages_show_the_loading_screen_while_preparing(preparing):
    r = preparing.get("/mv_grid/ui", headers={"accept": "text/html"})
    assert r.status_code == 503 and r.headers["retry-after"] == "2"
    assert "Starting GridArena" in r.text and "Grid and benchmark tables" in r.text


def test_api_answers_503_while_preparing(preparing):
    r = preparing.get("/mv_grid/list", headers={"accept": "application/json"})
    assert r.status_code == 503
    body = r.json()
    assert body["startup"]["state"] == "running" and "preparing the database" in body["detail"]


def test_startup_status_and_static_files_stay_reachable(preparing):
    r = preparing.get("/startup/status")
    assert r.status_code == 200 and r.json()["state"] == "running"
    assert preparing.get("/static/css/tokens.css").status_code == 200


def test_requests_pass_through_when_preparation_never_started():
    assert startup.DatabaseStartup().blocks_requests() is False
