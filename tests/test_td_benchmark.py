import json
import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from gridarena.app import app
from gridarena.benchmarks.submissions import BenchmarkTask, SubmissionRegistry
from gridarena.benchmarks.topology_discovery import evaluation as td_eval
import gridarena.routers._submissions as submissions_router
import gridarena.routers.topology_discovery_benchmark as topo_router

from tests._benchmark_helpers import delete_grids, register_grid, wait_for

GRID_ID = "grid_test_001"
BENCH_PREFIX = "/topology_discovery_benchmark"

client = TestClient(app)


# ===============================================================
#  GET /training-topology-data
# ===============================================================
@pytest.mark.asyncio
@patch("gridarena.routers.topology_discovery_benchmark.db.get_db_connection")
@patch("gridarena.routers.topology_discovery_benchmark.pi.get_all_measurements")
@patch("gridarena.routers.topology_discovery_benchmark.td.get_all_grids_topology")
async def test_get_training_topology_data_success(mock_get_topo, mock_get_meas, mock_get_db):
    """Should return grid measurement + topology data successfully."""
    conn = MagicMock()
    cursor = MagicMock()
    mock_get_db.return_value = (conn, cursor)

    mock_get_meas.return_value = [
        (GRID_ID, "NODE1", "R", "2025-01-01T00:00:00", 10.0, 5.0, 230.0, 0.1),
        (GRID_ID, "NODE2", "S", "2025-01-01T00:00:00", 12.0, 6.0, 231.0, 0.2),
    ]
    mock_get_topo.return_value = (
        {"grid_test_001": {"NODE1": 0, "NODE2": 1}},
        {"grid_test_001": [(0, 1)]},
        {"grid_test_001": {"(0,1)": {"Length": 100}}},
    )

    result = await topo_router.get_topology_identification_data(difficulty="clean")

    assert "grids" in result
    assert GRID_ID in result["grids"]
    assert "node to corresponding index" in result
    assert "topology (index)" in result
    assert "conn_data" in result

    mock_get_meas.assert_called_once()
    mock_get_topo.assert_called_once()
    conn.commit.assert_called_once()
    conn.close.assert_called_once()


@pytest.mark.asyncio
@patch("gridarena.routers.topology_discovery_benchmark.db.get_db_connection")
@patch("gridarena.routers.topology_discovery_benchmark.pi.get_all_measurements")
@patch("gridarena.routers.topology_discovery_benchmark.td.get_all_grids_topology")
async def test_training_topology_data_deterministic(mock_get_topo, mock_get_meas, mock_get_db):
    """Same difficulty should produce identical noise across calls."""
    def setup_mocks():
        conn = MagicMock()
        cursor = MagicMock()
        mock_get_db.return_value = (conn, cursor)
        mock_get_meas.return_value = [
            (GRID_ID, "NODE1", "R", "2025-01-01T00:00:00", 10.0, 5.0, 230.0, 0.1),
        ]
        mock_get_topo.return_value = (
            {GRID_ID: {"NODE1": 0}},
            {GRID_ID: [(0, 1)]},
            {GRID_ID: [("CBL1", 100.0)]},
        )

    setup_mocks()
    r1 = await topo_router.get_topology_identification_data(difficulty="easy")
    setup_mocks()
    r2 = await topo_router.get_topology_identification_data(difficulty="easy")

    m1 = r1["grids"][GRID_ID]["NODE1"][0]
    m2 = r2["grids"][GRID_ID]["NODE1"][0]
    assert m1["power_active"] == m2["power_active"]
    assert m1["voltage_magnitude"] == m2["voltage_magnitude"]


@pytest.mark.asyncio
@patch("gridarena.routers.topology_discovery_benchmark.db.get_db_connection")
@patch("gridarena.routers.topology_discovery_benchmark.pi.get_all_measurements")
@patch("gridarena.routers.topology_discovery_benchmark.td.get_all_grids_topology")
async def test_get_training_topology_data_db_error(mock_get_topo, mock_get_meas, mock_get_db):
    """Should raise HTTP 500 on DB error."""
    conn = MagicMock()
    cursor = MagicMock()
    mock_get_db.return_value = (conn, cursor)
    mock_get_meas.side_effect = Exception("DB fail")

    with pytest.raises(HTTPException) as exc:
        await topo_router.get_topology_identification_data(difficulty="clean")
    assert exc.value.status_code == 500
    assert "DB fail" in exc.value.detail
    conn.close.assert_called_once()


# ===============================================================
#  Test inputs and scoring
# ===============================================================
T0 = datetime(2025, 1, 1)

# Two grids: gA is PT-N1-N2-N3 (N3 has no meter), gB is PT-M1.
GRIDS = (
    ["gA", "gB"],
    {"gA": ["PT", "N1", "N2", "N3"], "gB": ["PT", "M1"]},
    {"gA": [("PT", "N1"), ("N1", "N2"), ("N2", "N3")], "gB": [("PT", "M1")]},
    [
        ("gA", "N1", "R", T0, 10.0, 5.0, 230.0, 0.0),
        ("gA", "N1", "R", T0 + timedelta(hours=1), 11.0, 5.0, 229.0, 0.1),
        ("gA", "N2", "R", T0, 9.0, 4.0, 228.0, -1.0),
        ("gB", "M1", "S", T0, 7.0, 3.0, 230.5, -120.0),
    ],
)

# Connects every node to the substation.
STAR_SCRIPT = """
def predict(grid):
    return [[grid["root"], n] for n in grid["nodes"] if n != grid["root"]]
"""


def _task(grids=GRIDS):
    return BenchmarkTask(
        name=td_eval.TOPOLOGY_TASK.name, function="predict",
        build_inputs=lambda settings: td_eval.build_test_inputs(settings["difficulty"], grids=grids),
        score=td_eval.score_predictions,
    )


def test_build_test_inputs_hides_ids_and_wiring():
    prepared = td_eval.build_test_inputs("clean", grids=GRIDS)
    assert len(prepared.calls) == 2
    by_grid = {t["grid_id"]: (name, t) for name, t in prepared.truth.items()}

    name, truth = by_grid["gA"]
    (grid,) = prepared.calls[name]
    assert sorted(grid["nodes"]) == truth["nodes"]
    assert len(grid["nodes"]) == 4 and grid["root"] in grid["nodes"]
    assert len(grid["measurements"]) == 2                 # N3 has no meter
    assert len(truth["edges"]) == 3
    blob = json.dumps(grid)
    for hidden in ('"PT"', "N1", "N2", "N3", "gA"):
        assert hidden not in blob
    record = next(iter(grid["measurements"].values()))[0]
    assert set(record) == {"datetime", "phase", "power_active", "power_reactive",
                           "voltage_magnitude", "voltage_angle"}


def test_build_test_inputs_fresh_keys_but_same_noise():
    a = td_eval.build_test_inputs("hard", grids=GRIDS)
    b = td_eval.build_test_inputs("hard", grids=GRIDS)

    def values(prepared):
        return sorted(r["voltage_magnitude"] for name in prepared.calls
                      for recs in prepared.calls[name][0]["measurements"].values() for r in recs)

    keys_a = {n for name in a.calls for n in a.calls[name][0]["nodes"]}
    keys_b = {n for name in b.calls for n in b.calls[name][0]["nodes"]}
    assert not keys_a & keys_b
    assert values(a) == values(b)


def test_score_predictions_precision_recall_f1():
    truth = {
        "0000": {"grid_id": "gA", "nodes": ["a", "b", "c", "d"], "edges": [["a", "b"], ["b", "c"], ["c", "d"]]},
        "0001": {"grid_id": "gB", "nodes": ["x", "y"], "edges": [["x", "y"]]},
    }
    outputs = {
        # 2 correct (one reversed, one duplicated), 1 wrong, 3 invalid
        "0000": {"ok": True, "seconds": 0.2, "result": [
            ["b", "a"], ["a", "b"], ["c", "b"], ["a", "d"], ["a", "a"], ["a", "zz"], "nope"]},
        "0001": {"ok": False, "error": "Traceback\nKeyError: 'root'", "seconds": 0.1},
    }
    score = td_eval.score_predictions(outputs, truth)

    ga, gb = score["per_grid"]
    assert (ga["correct"], ga["extra"], ga["missed"], ga["invalid"]) == (2, 1, 1, 3)
    assert (ga["precision"], ga["recall"]) == (round(2 / 3, 4), round(2 / 3, 4))
    assert gb["error"] == "KeyError: 'root'" and gb["missed"] == 1
    assert (score["correct"], score["extra"], score["missed"]) == (2, 1, 2)
    assert score["precision"] == round(2 / 3, 4)
    assert score["recall"] == 0.5
    assert score["f1"] == round(2 * (2 / 3) * 0.5 / (2 / 3 + 0.5), 4)
    assert score["failed_grids"] == 1


def test_score_predictions_every_pair_is_not_rewarded():
    nodes = [f"n{i}" for i in range(10)]
    truth = {"0000": {"grid_id": "g", "nodes": nodes, "edges": [[nodes[i], nodes[i + 1]] for i in range(9)]}}
    every_pair = [[a, b] for i, a in enumerate(nodes) for b in nodes[i + 1:]]
    score = td_eval.score_predictions({"0000": {"ok": True, "result": every_pair, "seconds": 0}}, truth)
    assert score["recall"] == 1.0
    assert score["f1"] < 0.4


def test_score_predictions_rejects_non_list_result():
    truth = {"0000": {"grid_id": "g", "nodes": ["a", "b"], "edges": [["a", "b"]]}}
    score = td_eval.score_predictions({"0000": {"ok": True, "result": {"a": "b"}, "seconds": 0}}, truth)
    assert score["f1"] == 0.0
    assert "must return a list" in score["per_grid"][0]["error"]


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
    monkeypatch.setattr(topo_router, "TOPOLOGY_TASK", _task())
    return registry


def test_star_script_scores_partially(registry):
    """A star from the substation finds PT-N1 and PT-M1 but not N1-N2 / N2-N3."""
    record = registry.submit(_task(), filename="s.py", data=STAR_SCRIPT.encode(),
                             submission_name="star", settings={"difficulty": "clean"})
    done = wait_for(registry, record["submission_id"], "topology_discovery")
    assert done["status"] == "completed", done
    assert (done["score"]["correct"], done["score"]["extra"], done["score"]["missed"]) == (2, 2, 2)


def test_submission_template_runs_as_a_submission(registry):
    source = topo_router._SUBMISSION_TEMPLATE.read_bytes()
    record = registry.submit(_task(), filename="t.py", data=source,
                             submission_name="template", settings={"difficulty": "medium"})
    done = wait_for(registry, record["submission_id"], "topology_discovery")
    assert done["status"] == "completed", done
    # The baseline builds a spanning tree: n - 1 valid edges per grid.
    assert [g["predicted_edges"] for g in done["score"]["per_grid"]] == [3, 1]
    assert all(g["invalid"] == 0 for g in done["score"]["per_grid"])


def test_download_submission_template():
    resp = client.get(f"{BENCH_PREFIX}/submission-template")
    assert resp.status_code == 200
    assert "topology_discovery_submission.py" in resp.headers["content-disposition"]
    assert "def predict(grid)" in resp.text


def test_submit_poll_and_list_over_http(patched_router):
    resp = client.post(
        f"{BENCH_PREFIX}/submissions",
        files={"file": ("model.py", STAR_SCRIPT.encode(), "text/x-python")},
        data={"submission_name": "alice_v1", "difficulty": "easy"},
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
    assert record["settings"] == {"difficulty": "easy"}
    assert "f1" in record["score"]

    listed = client.get(f"{BENCH_PREFIX}/submissions", params={"submission_name": "alice_v1"}).json()
    assert [r["submission_id"] for r in listed["submissions"]] == [submission_id]
    # Another benchmark doesn't see this submission.
    assert client.get(f"/phase_identification_benchmark/submissions/{submission_id}").status_code == 404


@pytest.mark.parametrize("data, detail", [
    ({"submission_name": "a", "difficulty": "extreme"}, "difficulty"),
    ({"submission_name": "bad name"}, "submission_name"),
])
def test_submit_rejects_bad_settings(patched_router, data, detail):
    resp = client.post(f"{BENCH_PREFIX}/submissions", files={"file": ("m.py", STAR_SCRIPT.encode())}, data=data)
    assert resp.status_code == 422
    assert detail in resp.text


def test_old_test_data_endpoints_are_gone():
    assert client.get(f"{BENCH_PREFIX}/topology-data").status_code == 404
    assert client.get(f"{BENCH_PREFIX}/score", params={"guess_id": "a", "grid_id": "b"}).status_code == 404


# ===============================================================
#  Integration (real FastAPI + PostgreSQL)
# ===============================================================
TEST_GRID_IDS = ["topo_grid_train_pg1", "topo_grid_test_pg1"]


@pytest.fixture(scope="module")
def clean_db():
    """Deletes only this module's own test grids, before and after."""
    delete_grids(TEST_GRID_IDS)
    yield
    delete_grids(TEST_GRID_IDS)


def _setup_topology_grid(grid_id, usage):
    now = datetime(2025, 1, 1).isoformat()
    register_grid(
        client, grid_id,
        edges=[("PT", "NODE1"), ("NODE1", "NODE2"), ("NODE2", "NODE3")],
        measurements={
            n: [{"datetime": now, "phase": "R", "power_active": 10.0, "power_reactive": 5.0,
                 "voltage_magnitude": 230.0 - i, "voltage_angle": -float(i)}]
            for i, n in enumerate(["NODE1", "NODE2", "NODE3"])
        },
        usage=usage,
    )


def test_get_training_topology_data_integration(clean_db):
    grid_id = "topo_grid_train_pg1"
    _setup_topology_grid(grid_id, {"topology_detection_train"})

    response = client.get(f"{BENCH_PREFIX}/training-topology-data")
    assert response.status_code == 200, response.text
    data = response.json()
    assert grid_id in data["grids"]
    assert len(data["topology (index)"][grid_id]) == 3


def test_script_submission_integration(clean_db, registry, monkeypatch):
    grid_id = "topo_grid_test_pg1"
    _setup_topology_grid(grid_id, {"topology_detection_test"})
    monkeypatch.setattr(submissions_router, "submission_registry", registry)

    resp = client.post(f"{BENCH_PREFIX}/submissions", files={"file": ("m.py", STAR_SCRIPT.encode())},
                       data={"submission_name": "integration"})
    assert resp.status_code == 202, resp.text

    done = wait_for(registry, resp.json()["submission_id"], "topology_discovery")
    assert done["status"] == "completed", done
    mine = next(g for g in done["score"]["per_grid"] if g["grid_id"] == grid_id)
    # Star from PT: only PT-NODE1 is a true edge.
    assert (mine["nodes"], mine["true_edges"], mine["predicted_edges"], mine["correct"]) == (4, 3, 3, 1)
