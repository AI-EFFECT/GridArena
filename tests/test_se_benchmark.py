import json
import math
import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from gridarena.app import app
from gridarena.benchmarks.state_estimation import evaluation as se_eval
from gridarena.benchmarks.submissions import BenchmarkTask, SubmissionRegistry
import gridarena.routers._submissions as submissions_router
import gridarena.routers.state_estimation_benchmark as se_router

from tests._benchmark_helpers import delete_grids, register_grid, wait_for

GRID_ID = "grid_001"
BENCH_PREFIX = "/state_estimation_benchmark"

client = TestClient(app)


# ===============================================================
#  GET /training-state-data
# ===============================================================
@pytest.mark.asyncio
@patch("gridarena.routers.state_estimation_benchmark.db.get_db_connection")
@patch("gridarena.routers.state_estimation_benchmark.se.ensure_grid_masks")
@patch("gridarena.routers.state_estimation_benchmark.se.ensure_all_node_masks")
async def test_get_training_state_data_success(mock_node_masks, mock_grid_masks, mock_get_db):
    """Should return structured masked training data."""
    conn = MagicMock()
    cursor = MagicMock()
    mock_get_db.return_value = (conn, cursor)

    row = (
        GRID_ID, "masked_gid_1", "NODE1", "masked_nid_1",
        "2025-01-01T00:00:00", "R", 10.0, 5.0, 230.0, 0.0,
    )
    # fetchall for GridUsage query, fetchmany for batched measurement read
    cursor.fetchall.return_value = [(GRID_ID,)]
    cursor.fetchmany.side_effect = [[row], []]

    result = await se_router.get_training_state_data(noise_difficulty="clean", observability="medium")

    assert "grids" in result
    assert "masked_gid_1" in result["grids"]
    assert "masked_nid_1" in result["grids"]["masked_gid_1"]
    conn.commit.assert_called_once()
    conn.close.assert_called_once()


@pytest.mark.asyncio
@patch("gridarena.routers.state_estimation_benchmark.db.get_db_connection")
@patch("gridarena.routers.state_estimation_benchmark.se.ensure_grid_masks")
@patch("gridarena.routers.state_estimation_benchmark.se.ensure_all_node_masks")
async def test_training_state_data_deterministic(mock_node_masks, mock_grid_masks, mock_get_db):
    """Same difficulty should produce identical noise across calls."""
    row = (GRID_ID, "masked_gid_1", "NODE1", "masked_nid_1",
           "2025-01-01T00:00:00", "R", 10.0, 5.0, 230.0, 0.1)

    def setup():
        conn = MagicMock()
        cursor = MagicMock()
        mock_get_db.return_value = (conn, cursor)
        cursor.fetchall.return_value = [row]
        cursor.fetchmany.side_effect = [[row], []]

    setup()
    r1 = await se_router.get_training_state_data(noise_difficulty="easy", observability="medium")
    setup()
    r2 = await se_router.get_training_state_data(noise_difficulty="easy", observability="medium")

    m1 = r1["grids"]["masked_gid_1"]["masked_nid_1"][0]
    m2 = r2["grids"]["masked_gid_1"]["masked_nid_1"][0]
    assert m1["power_active"] == m2["power_active"]
    assert m1["voltage_magnitude"] == m2["voltage_magnitude"]


@pytest.mark.asyncio
@patch("gridarena.routers.state_estimation_benchmark.db.get_db_connection")
@patch("gridarena.routers.state_estimation_benchmark.se.ensure_grid_masks")
@patch("gridarena.routers.state_estimation_benchmark.se.ensure_all_node_masks")
async def test_get_training_state_data_db_error(mock_node_masks, mock_grid_masks, mock_get_db):
    """Should raise HTTP 500 if DB fails."""
    conn = MagicMock()
    cursor = MagicMock()
    mock_get_db.return_value = (conn, cursor)
    cursor.execute.side_effect = Exception("SQL fail")

    with pytest.raises(HTTPException) as exc:
        await se_router.get_training_state_data(noise_difficulty="clean", observability="medium")
    assert exc.value.status_code == 500
    assert "SQL fail" in exc.value.detail
    conn.close.assert_called_once()


# ===============================================================
#  Test inputs and scoring
# ===============================================================
T0 = datetime(2025, 1, 1)


def _rows(grid_id, series, hours):
    """Hourly measurements: voltage drops with the node index, angle follows the phase."""
    angle = {"R": 0.0, "S": -120.0, "T": 120.0}
    return [
        (grid_id, node, phase, T0 + timedelta(hours=h), 1.0 + h, 0.1, 230.0 - i - 0.1 * h, angle[phase] - 0.1 * i)
        for i, (node, phase) in enumerate(series) for h in range(hours)
    ]


# gA: 4 series over 8 h (3 tasks from the later 4 h); gB: 2 series over 4 h (2 tasks).
MEASUREMENTS = (_rows("gA", [("N1", "R"), ("N2", "R"), ("N3", "S"), ("N4", "T")], 8)
                + _rows("gB", [("M1", "R"), ("M2", "S")], 4))


def _task():
    return BenchmarkTask(
        name=se_eval.STATE_ESTIMATION_TASK.name, function="predict",
        build_inputs=lambda s: se_eval.build_test_inputs(
            s["noise_difficulty"], s["observability"], measurements=MEASUREMENTS),
        score=se_eval.score_predictions,
    )


def test_build_test_inputs_tasks_and_split():
    prepared = se_eval.build_test_inputs("clean", "medium", measurements=MEASUREMENTS)
    grids = [t["grid_id"] for t in prepared.truth.values()]
    assert grids.count("gA") == 3 and grids.count("gB") == 2

    for name, truth in prepared.truth.items():
        (task,) = prepared.calls[name]
        ts = datetime.fromisoformat(task["timestamp"])
        assert ts >= T0 + timedelta(hours=4 if truth["grid_id"] == "gA" else 2)  # later half only
        unknown, known = set(task["unknown"]), set(task["known"])
        assert unknown == set(truth["values"]) and unknown and known and not unknown & known
        assert set(task["history"]) == unknown | known
        for key, records in task["history"].items():
            cutoff = ts - timedelta(hours=6 if key in unknown else 0)
            assert all(datetime.fromisoformat(r["datetime"]) < cutoff for r in records)
        blob = json.dumps(task)
        for hidden in ("gA", "gB", "N1", "M1", '"phase"'):
            assert hidden not in blob


def test_build_test_inputs_same_tasks_fresh_keys():
    a = se_eval.build_test_inputs("hard", "medium", measurements=MEASUREMENTS)
    b = se_eval.build_test_inputs("hard", "medium", measurements=MEASUREMENTS)

    def summary(prepared):
        return sorted(
            (t["grid_id"], prepared.calls[n][0]["timestamp"], sorted(map(tuple, t["values"].values())),
             sorted(m["voltage_magnitude"] for m in prepared.calls[n][0]["known"].values()))
            for n, t in prepared.truth.items()
        )

    assert summary(a) == summary(b)
    keys_a = {k for n in a.calls for k in a.calls[n][0]["history"]}
    keys_b = {k for n in b.calls for k in b.calls[n][0]["history"]}
    assert not keys_a & keys_b


def test_clean_observability_has_nothing_to_estimate():
    prepared = se_eval.build_test_inputs("clean", "clean", measurements=MEASUREMENTS)
    assert all(not task[0]["unknown"] for task in prepared.calls.values())


def test_score_predictions_errors_and_penalties():
    truth = {
        "0000": {"grid_id": "gA", "values": {"k1": [230.0, 0.0], "k2": [228.0, -179.0], "k3": [229.0, 1.0]}},
        "0001": {"grid_id": "gB", "values": {"m1": [231.0, 0.0]}},
    }
    outputs = {
        "0000": {"ok": True, "seconds": 0.5, "result": {
            "k1": {"voltage_magnitude": 231.0, "voltage_angle": 0.0},     # dV = 1
            "k2": {"voltage_magnitude": 228.0, "voltage_angle": 179.0},   # angle wraps: 2 deg
            "k3": {"voltage_magnitude": "high", "voltage_angle": 0.0},    # invalid
        }},
        "0001": {"ok": False, "error": "Traceback\nValueError: boom", "seconds": 0.1},
    }
    score = se_eval.score_predictions(outputs, truth)

    sq = [1.0, 0.0, 229.0 ** 2, 231.0 ** 2]
    sin = [0.0, abs(math.sin(math.radians(2.0))), 1.0, 1.0]
    assert score["score"] == round(sum(sq) / 4 + sum(sin) / 4, 6)
    assert score["rmse_voltage_magnitude"] == round(math.sqrt(0.5), 6)
    assert score["mae_voltage_angle"] == 1.0
    assert (score["estimates"], score["missing"], score["invalid"]) == (4, 1, 1)
    assert (score["tasks"], score["failed_tasks"]) == (2, 1)
    gb = next(g for g in score["per_grid"] if g["grid_id"] == "gB")
    assert gb["error"] == "ValueError: boom"


def test_score_predictions_perfect_is_zero():
    truth = {"0000": {"grid_id": "g", "values": {"k": [230.0, -120.0]}}}
    result = {"k": {"voltage_magnitude": 230.0, "voltage_angle": -120.0}}
    score = se_eval.score_predictions({"0000": {"ok": True, "result": result, "seconds": 0}}, truth)
    assert score["score"] == 0.0


# ===============================================================
#  Running submissions and the HTTP endpoints (no database)
# ===============================================================
@pytest.fixture
def registry(tmp_path):
    reg = SubmissionRegistry(runs_dir=tmp_path / "runs")
    yield reg
    reg.shutdown()


@pytest.fixture
def patched_router(registry, monkeypatch):
    monkeypatch.setattr(submissions_router, "submission_registry", registry)
    monkeypatch.setattr(se_router, "STATE_ESTIMATION_TASK", _task())
    return registry


def test_submission_template_runs_as_a_submission(registry):
    source = se_router._SUBMISSION_TEMPLATE.read_bytes()
    record = registry.submit(_task(), filename="t.py", data=source, submission_name="template",
                             settings={"noise_difficulty": "easy", "observability": "medium"})
    done = wait_for(registry, record["submission_id"], "state_estimation")
    assert done["status"] == "completed", done
    s = done["score"]
    assert s["tasks"] == 5 and s["estimates"] > 0
    assert (s["missing"], s["invalid"], s["failed_tasks"]) == (0, 0, 0)
    # The last-value baseline is off by the hourly drift, well under a volt.
    assert s["rmse_voltage_magnitude"] < 2.0


def test_download_submission_template():
    resp = client.get(f"{BENCH_PREFIX}/submission-template")
    assert resp.status_code == 200
    assert "state_estimation_submission.py" in resp.headers["content-disposition"]
    assert "def predict(task)" in resp.text


def test_submit_poll_and_list_over_http(patched_router):
    script = "def predict(task):\n    return {k: {'voltage_magnitude': 230.0, 'voltage_angle': 0.0} for k in task['unknown']}\n"
    resp = client.post(
        f"{BENCH_PREFIX}/submissions",
        files={"file": ("model.py", script.encode(), "text/x-python")},
        data={"submission_name": "alice_v1", "noise_difficulty": "medium", "observability": "hard"},
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
    assert record["settings"] == {"noise_difficulty": "medium", "observability": "hard"}
    assert record["score"]["missing"] == 0

    listed = client.get(f"{BENCH_PREFIX}/submissions", params={"submission_name": "alice_v1"}).json()
    assert [r["submission_id"] for r in listed["submissions"]] == [submission_id]


@pytest.mark.parametrize("data, detail", [
    ({"submission_name": "a", "observability": "total"}, "observability"),
    ({"submission_name": "a", "noise_difficulty": "loud"}, "noise_difficulty"),
])
def test_submit_rejects_bad_settings(patched_router, data, detail):
    resp = client.post(f"{BENCH_PREFIX}/submissions",
                       files={"file": ("m.py", b"def predict(task):\n    return {}\n")}, data=data)
    assert resp.status_code == 422
    assert detail in resp.text


def test_old_test_data_endpoints_are_gone():
    assert client.get(f"{BENCH_PREFIX}/state-data", params={"user_id": "u"}).status_code == 404
    assert client.get(f"{BENCH_PREFIX}/state-score",
                      params={"user_id": "u", "estimation_id": "e"}).status_code == 404


# ===============================================================
#  Integration (real FastAPI + PostgreSQL)
# ===============================================================
TEST_GRID_IDS = ["se_grid_train_pg1", "se_grid_test_pg1"]


@pytest.fixture(scope="module")
def clean_db():
    """Deletes only this module's own test grids, before and after."""
    delete_grids(TEST_GRID_IDS)
    yield
    delete_grids(TEST_GRID_IDS)


def _setup_state_grid(grid_id, usage):
    register_grid(
        client, grid_id,
        edges=[("PT", "NODE1"), ("NODE1", "NODE2"), ("NODE2", "NODE3")],
        measurements={
            n: [{"datetime": (T0 + timedelta(hours=h)).isoformat(), "phase": "R",
                 "power_active": 1.0 + h, "power_reactive": 0.1,
                 "voltage_magnitude": 230.0 - i - 0.1 * h, "voltage_angle": -0.1 * i}
                for h in range(8)]
            for i, n in enumerate(["NODE1", "NODE2", "NODE3"])
        },
        usage=usage,
    )


def test_get_training_state_data_integration(clean_db):
    _setup_state_grid("se_grid_train_pg1", {"state_estimation_train"})
    response = client.get(f"{BENCH_PREFIX}/training-state-data")
    assert response.status_code == 200, response.text
    sizes = sorted(len(nodes) for nodes in response.json()["grids"].values())
    assert 3 in sizes  # this grid's 3 series, under a masked grid id


def test_script_submission_integration(clean_db, registry, monkeypatch):
    grid_id = "se_grid_test_pg1"
    _setup_state_grid(grid_id, {"state_estimation_test"})
    monkeypatch.setattr(submissions_router, "submission_registry", registry)

    resp = client.post(f"{BENCH_PREFIX}/submissions",
                       files={"file": ("m.py", se_router._SUBMISSION_TEMPLATE.read_bytes())},
                       data={"submission_name": "integration"})
    assert resp.status_code == 202, resp.text

    done = wait_for(registry, resp.json()["submission_id"], "state_estimation", timeout=120)
    assert done["status"] == "completed", done
    mine = next(g for g in done["score"]["per_grid"] if g["grid_id"] == grid_id)
    assert mine["tasks"] == 3 and mine["estimates"] >= 3
    assert (mine["missing"], mine["invalid"]) == (0, 0)
