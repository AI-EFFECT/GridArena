import time
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import gridarena.benchmarks.voltage_control.get_measurement_data as vc
import gridarena.database as db
import gridarena.routers._submissions as submissions_router
import gridarena.routers.voltage_control_benchmark as vc_router
from gridarena.app import app
from gridarena.benchmarks.submissions import BenchmarkTask, SubmissionRegistry
from gridarena.benchmarks.voltage_control import evaluation as vc_eval

from tests._benchmark_helpers import delete_grids, register_grid, wait_for

GRID_ID = "grid_001"
VOLTAGE_LIMIT = 230

client = TestClient(app)
BENCH_PREFIX = "/voltage_control_benchmark"


# -------------------------------------------------
# Training scenarios (Overvoltages / Undervoltages)
# -------------------------------------------------
@pytest.mark.asyncio
@patch("gridarena.routers.voltage_control_benchmark.db.get_db_connection")
@patch("gridarena.routers.voltage_control_benchmark.vc.get_all_measurements_over")
@patch("gridarena.routers.voltage_control_benchmark.vc.get_all_measurements_under")
async def test_get_training_scenarios_success(
    mock_under, mock_over, mock_get_db
):
    """Should fetch measurements and organize data correctly."""
    conn, cursor = MagicMock(), MagicMock()
    mock_get_db.return_value = (conn, cursor)

    mock_over.return_value = [
        (GRID_ID, "NODE1", "R", "2025-01-01T00:00:00", 10, 5, 240, 0.1),
    ]
    cursor.fetchall.return_value = [
        ("NODE1", "R", 230, 9.8, 4.9, "2025-01-01T00:00:00")
    ]

    result = await vc_router.get_training_voltage_control_data(
        VOLTAGE_LIMIT, Scenario="Overvoltages", difficulty="clean"
    )
    assert "grids" in result
    assert GRID_ID in result["grids"]
    assert "measurements" in result["grids"][GRID_ID]["2025-01-01T00:00:00"]
    conn.commit.assert_called_once()
    conn.close.assert_called_once()


@pytest.mark.asyncio
@patch("gridarena.routers.voltage_control_benchmark.db.get_db_connection")
@patch("gridarena.routers.voltage_control_benchmark.vc.get_all_measurements_over")
async def test_get_training_scenarios_db_error(mock_get_meas, mock_get_db):
    """Should raise HTTP 500 on DB error and call rollback."""
    conn, cursor = MagicMock(), MagicMock()
    mock_get_db.return_value = (conn, cursor)
    mock_get_meas.side_effect = Exception("SQL fail")

    with pytest.raises(HTTPException) as exc:
        await vc_router.get_training_voltage_control_data(
            VOLTAGE_LIMIT, Scenario="Overvoltages", difficulty="clean"
        )
    assert exc.value.status_code == 500
    assert "SQL fail" in exc.value.detail
    conn.rollback.assert_called_once()
    conn.close.assert_called_once()


# -------------------------------------------------
# Snapshot scoring
# -------------------------------------------------
MEAS = [
    {"node_id": "N1", "phase": "R", "power_active": 10.0, "power_reactive": 1.0,
     "voltage_magnitude": 245.0, "voltage_angle": 0.0},
    {"node_id": "N2", "phase": "R", "power_active": 5.0, "power_reactive": 1.0,
     "voltage_magnitude": 232.0, "voltage_angle": 0.0},
]
SOLUTION = [{"node_id": "N1", "phase": "R", "corrected_voltage": 240.0,
             "adjusted_power_active": 8.0, "adjusted_power_reactive": 1.0}]


def test_score_snapshot_matching_reference_scores_on_effort():
    prediction = [{"node_id": "N1", "phase": "R", "corrected_voltage": 240.3, "adjusted_power_active": 8.0}]
    row = vc_eval.score_snapshot(MEAS, SOLUTION, prediction)
    assert row["mode"] == "ref_voltage_then_effort"
    assert row["reference_voltage_mae"] == 0.3
    assert row["timestamp_score"] == round(vc.effort_factor(2.0 / 15.0), 4)
    assert (row["nodes"], row["changed_nodes"], row["invalid"]) == (2, 1, 0)


def test_score_snapshot_no_answer_counts_as_unchanged():
    row = vc_eval.score_snapshot(MEAS, SOLUTION, [])
    assert row["mode"] == "ref_voltage_miss_penalised"
    assert row["reference_voltage_mae"] == 5.0  # measured 245 vs reference 240
    assert row["timestamp_score"] == round(1.0 / (1.0 + 25.0), 4)


def test_score_snapshot_without_reference_uses_outcome():
    unchanged = vc_eval.score_snapshot(MEAS, [], None)
    assert unchanged["mode"] == "outcome_only_fallback"
    assert unchanged["timestamp_score"] == 0.2  # only the power term, on both nodes

    fixed = vc_eval.score_snapshot(MEAS, [], [
        {"node_id": "N1", "phase": "R", "corrected_voltage": 235.0},
        {"node_id": "N2", "phase": "S", "corrected_voltage": 230.0},   # wrong phase: ignored
        {"node_id": "N2", "corrected_voltage": "low"},                 # invalid
    ])
    assert fixed["invalid"] == 1 and fixed["changed_nodes"] == 1
    assert fixed["timestamp_score"] > unchanged["timestamp_score"]


def test_score_predictions_weights_nodes_and_reports_errors():
    truth = {
        "0000": {"grid_id": "gA", "datetime": "2025-07-01T12:00:00", "measurements": MEAS, "solutions": SOLUTION},
        "0001": {"grid_id": "gB", "datetime": "2025-07-01T13:00:00", "measurements": MEAS[:1], "solutions": []},
    }
    outputs = {
        "0000": {"ok": True, "seconds": 0.1, "result": [
            {"node_id": "N1", "phase": "R", "corrected_voltage": 240.0, "adjusted_power_active": 10.0}]},
        "0001": {"ok": False, "error": "Traceback\nRuntimeError: no solver", "seconds": 0.2},
    }
    score = vc_eval.score_predictions(outputs, truth)
    assert score["per_snapshot"][0]["timestamp_score"] == 1.0   # reference matched, no effort
    gb = score["per_snapshot"][1]
    assert gb["error"] == "RuntimeError: no solver" and gb["mode"] == "outcome_only_fallback"
    assert score["score"] == round((1.0 * 2 + gb["timestamp_score"] * 1) / 3, 4)
    assert (score["snapshots"], score["nodes"], score["failed_snapshots"], score["grids"]) == (2, 3, 1, 2)


def test_build_test_inputs_from_snapshots():
    snapshots = [{"grid_id": "gA", "datetime": datetime(2025, 7, 1, 12), "measurements": MEAS, "solutions": SOLUTION}]
    settings = {"voltage_limit": 241.5, "scenario": "Overvoltages", "difficulty": "clean"}
    prepared = vc_eval.build_test_inputs(settings, snapshots=snapshots)
    (snapshot,) = prepared.calls["0000"]
    assert snapshot == {
        "datetime": "2025-07-01T12:00:00", "scenario": "Overvoltages", "voltage_limit": 241.5,
        "nominal_voltage": 230.0, "voltage_band": 0.05, "measurements": MEAS,
    }
    assert prepared.truth["0000"]["solutions"] == SOLUTION
    with pytest.raises(Exception, match="No test snapshots"):
        vc_eval.build_test_inputs(settings, snapshots=[])


# -------------------------------------------------
# Running submissions and the HTTP endpoints (no database)
# -------------------------------------------------
SNAPSHOTS = [
    {"grid_id": "gA", "datetime": datetime(2025, 7, 1, 12), "measurements": MEAS, "solutions": SOLUTION},
    {"grid_id": "gA", "datetime": datetime(2025, 7, 1, 13), "measurements": MEAS, "solutions": []},
]

# Curtails the over-limit nodes and reports the voltage the reference would reach.
CURTAIL_SCRIPT = """
def predict(snapshot):
    actions = []
    for m in snapshot["measurements"]:
        if m["voltage_magnitude"] > snapshot["voltage_limit"]:
            actions.append({"node_id": m["node_id"], "phase": m["phase"], "corrected_voltage": 240.0,
                            "adjusted_power_active": 0.8 * m["power_active"]})
    return actions
"""


def _task():
    return BenchmarkTask(
        name=vc_eval.VOLTAGE_CONTROL_TASK.name, function="predict",
        build_inputs=lambda settings: vc_eval.build_test_inputs(settings, snapshots=SNAPSHOTS),
        score=vc_eval.score_predictions,
    )


@pytest.fixture
def registry(tmp_path):
    reg = SubmissionRegistry(runs_dir=tmp_path / "runs")
    yield reg
    reg.shutdown()


@pytest.fixture
def patched_router(registry, monkeypatch):
    monkeypatch.setattr(submissions_router, "submission_registry", registry)
    monkeypatch.setattr(vc_router, "VOLTAGE_CONTROL_TASK", _task())
    return registry


def test_submission_template_runs_as_a_submission(registry):
    source = vc_router._SUBMISSION_TEMPLATE.read_bytes()
    record = registry.submit(_task(), filename="t.py", data=source, submission_name="template",
                             settings={"voltage_limit": 241.5, "scenario": "Overvoltages", "difficulty": "clean"})
    done = wait_for(registry, record["submission_id"], "voltage_control")
    assert done["status"] == "completed", done
    # "No action" leaves N1 5 V from the reference, and earns only the power term without one.
    assert [r["mode"] for r in done["score"]["per_snapshot"]] == ["ref_voltage_miss_penalised", "outcome_only_fallback"]
    assert done["score"]["failed_snapshots"] == 0


def test_download_submission_template():
    resp = client.get(f"{BENCH_PREFIX}/submission-template")
    assert resp.status_code == 200
    assert "voltage_control_submission.py" in resp.headers["content-disposition"]
    assert "def predict(snapshot)" in resp.text


def test_submit_poll_and_list_over_http(patched_router):
    resp = client.post(
        f"{BENCH_PREFIX}/submissions",
        files={"file": ("model.py", CURTAIL_SCRIPT.encode(), "text/x-python")},
        data={"submission_name": "alice_v1", "voltage_limit": "241.5", "Scenario": "Overvoltages"},
    )
    assert resp.status_code == 202, resp.text
    submission_id = resp.json()["submission_id"]

    deadline = time.time() + 60
    while True:
        record = client.get(f"{BENCH_PREFIX}/submissions/{submission_id}").json()
        if record["status"] in ("completed", "failed") or time.time() > deadline:
            break
        time.sleep(0.1)
    assert record["status"] == "completed", record
    assert record["settings"] == {"voltage_limit": 241.5, "scenario": "Overvoltages", "difficulty": "clean"}
    assert record["score"]["per_snapshot"][0]["mode"] == "ref_voltage_then_effort"

    listed = client.get(f"{BENCH_PREFIX}/submissions", params={"submission_name": "alice_v1"}).json()
    assert [r["submission_id"] for r in listed["submissions"]] == [submission_id]


@pytest.mark.parametrize("data, detail", [
    ({"submission_name": "a"}, "voltage_limit"),
    ({"submission_name": "a", "voltage_limit": "-3"}, "voltage_limit"),
    ({"submission_name": "a", "voltage_limit": "241", "Scenario": "Blackout"}, "Scenario"),
])
def test_submit_rejects_bad_settings(patched_router, data, detail):
    resp = client.post(f"{BENCH_PREFIX}/submissions", files={"file": ("m.py", CURTAIL_SCRIPT.encode())}, data=data)
    assert resp.status_code == 422
    assert detail in resp.text


def test_old_test_data_endpoints_are_gone():
    assert client.get(f"{BENCH_PREFIX}/voltage-control-data", params={"voltage_limit": 230}).status_code == 404
    assert client.get(f"{BENCH_PREFIX}/score", params={"guess_id": "a", "grid_id": "b"}).status_code == 404


# -------------------------------------------------
# Integration (real FastAPI + PostgreSQL)
# -------------------------------------------------
TEST_GRID_IDS = ["vc_grid_train_pg1", "vc_grid_test_pg1"]
SNAPSHOT_TIME = datetime(2025, 7, 1, 12)


@pytest.fixture(scope="module")
def clean_db():
    """Deletes only this module's own test grids, before and after."""
    delete_grids(TEST_GRID_IDS)
    yield
    delete_grids(TEST_GRID_IDS)


def _setup_voltage_grid(grid_id, usage):
    """PT at 230 V, NODE1 in overvoltage (250 V), plus a reference solution for NODE1."""
    def meas(v, p):
        return [{"datetime": SNAPSHOT_TIME.isoformat(), "phase": "R", "power_active": p,
                 "power_reactive": 1.0, "voltage_magnitude": v, "voltage_angle": 0.0}]

    register_grid(client, grid_id, edges=[("PT", "NODE1")],
                  measurements={"PT": meas(230.0, 0.0), "NODE1": meas(250.0, 10.0)}, usage=usage)
    conn, cursor = db.get_db_connection()
    try:
        cursor.execute(
            """INSERT INTO "VoltageControlSolutions"
               (grid_id, datetime, node_id, phase, corrected_voltage, adjusted_power_active, adjusted_power_reactive)
               VALUES (%s, %s, 'NODE1', 'R', 240.0, 8.0, 1.0)""",
            (grid_id, SNAPSHOT_TIME),
        )
        conn.commit()
    finally:
        conn.close()


def test_get_training_voltage_control_data_integration(clean_db):
    grid_id = "vc_grid_train_pg1"
    _setup_voltage_grid(grid_id, {"voltage_control_train"})

    response = client.get(f"{BENCH_PREFIX}/training-scenarios",
                          params={"voltage_limit": 241.5, "Scenario": "Overvoltages"})
    assert response.status_code == 200, response.text
    snapshots = response.json()["grids"][grid_id]
    (snap,) = snapshots.values()
    assert snap["solutions"][0]["corrected_voltage"] == 240.0


def test_script_submission_integration(clean_db, registry, monkeypatch):
    grid_id = "vc_grid_test_pg1"
    _setup_voltage_grid(grid_id, {"voltage_control_test"})
    monkeypatch.setattr(submissions_router, "submission_registry", registry)

    resp = client.post(f"{BENCH_PREFIX}/submissions", files={"file": ("m.py", CURTAIL_SCRIPT.encode())},
                       data={"submission_name": "integration", "voltage_limit": "241.5"})
    assert resp.status_code == 202, resp.text

    done = wait_for(registry, resp.json()["submission_id"], "voltage_control", timeout=120)
    assert done["status"] == "completed", done
    (mine,) = [r for r in done["score"]["per_snapshot"] if r["grid_id"] == grid_id]
    assert (mine["nodes"], mine["changed_nodes"], mine["mode"]) == (2, 1, "ref_voltage_then_effort")
    assert mine["reference_voltage_mae"] == 0.0
