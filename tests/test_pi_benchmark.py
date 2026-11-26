import io
import json
import time
import zipfile
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from gridarena.app import app
from gridarena.benchmarks.phase_identification import evaluation as pi_eval
from gridarena.benchmarks.submissions import BenchmarkTask, PreparedInputs, SubmissionRegistry, packaging
from gridarena.routers import _submissions as submissions_router
from gridarena.routers import phase_identification_benchmark as pi_router
import gridarena.database as db


GRID_ID = "test_grid_001"
BENCH_PREFIX = "/phase_identification_benchmark"

client = TestClient(app)


# ----------------------------------------------------------------------------------------
# DB cleanup for this module only
# ----------------------------------------------------------------------------------------
#
# IMPORTANT: this must never touch grids/data this test file didn't create. An earlier
# version of this fixture ran `TRUNCATE "Node"/"Cable"/"Connection"/... CASCADE`
# unconditionally against whatever database is configured, which destroyed real grid
# topology in the shared dev database when the suite was run (~78k Node rows, ~73k
# Cable/Connection rows across ~5900 grids). grids(grid_id) has ON DELETE CASCADE down
# to Node/Cable/Connection/HistoricalNodes->Measurements, so deleting only this file's
# own grid_id rows from `grids` is enough to fully clean up after these tests while
# leaving every other grid untouched.

# Only the integration tests at the bottom write to the real DB. Keep this list in sync
# with the grid_id values used by _setup_phase_grid(...) calls in this file.
TEST_GRID_IDS_FOR_PHASE = [
    "phase_grid_train_pg1",
    "phase_grid_test_pg1",
]


def _cleanup_phase_test_grids():
    conn, cursor = db.get_db_connection()
    try:
        cursor.execute(
            "DELETE FROM grids WHERE grid_id = ANY(%s)", (TEST_GRID_IDS_FOR_PHASE,)
        )
        conn.commit()
    except Exception:
        # e.g. table doesn't exist yet -> ignore and continue
        conn.rollback()
    finally:
        conn.close()


@pytest.fixture(scope="module")
def clean_db_for_phase_benchmark():
    """
    Deletes only this module's own test grids (by grid_id) before and after the
    integration tests, relying on ON DELETE CASCADE for their dependent rows.
    """
    _cleanup_phase_test_grids()
    yield
    _cleanup_phase_test_grids()


# A few measurements of two grids: (grid_id, node_id, phase, datetime, P, Q, |V|, angle)
MEASUREMENTS = [
    ("gA", "N1", "R", datetime(2025, 1, 1, 0), 10.0, 5.0, 230.0, 0.1),
    ("gA", "N1", "R", datetime(2025, 1, 1, 1), 12.0, 6.0, 231.0, 0.2),
    ("gA", "N2", "S", datetime(2025, 1, 1, 0), 9.0, 4.5, 229.5, -120.0),
    ("gA", "N3", "T", datetime(2025, 1, 1, 0), 8.0, 4.0, 229.0, 120.0),
    ("gB", "M1", "R", datetime(2025, 1, 1, 0), 7.0, 3.0, 230.5, 0.0),
    ("gB", "M1", "S", datetime(2025, 1, 1, 0), 7.5, 3.5, 230.4, -119.0),
]

# Recovers the phase from the voltage angle, so a correct run scores 1.0.
ANGLE_SCRIPT = """
def predict(measurements):
    out = {}
    for key, records in measurements.items():
        a = records[0]["voltage_angle"]
        out[key] = "R" if abs(a) < 60 else ("S" if a < 0 else "T")
    return out
"""


def _wait(registry, submission_id, benchmark, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        record = registry.get(submission_id, benchmark)
        if record["status"] in ("completed", "failed"):
            return record
        time.sleep(0.1)
    raise AssertionError(f"submission {submission_id} did not finish within {timeout} s")


@pytest.fixture
def registry(tmp_path):
    reg = SubmissionRegistry(runs_dir=tmp_path / "runs")
    yield reg
    reg.shutdown()


@pytest.fixture
def phase_task():
    """The real phase task, fed MEASUREMENTS instead of the database."""
    return BenchmarkTask(
        name=pi_eval.PHASE_TASK.name,
        function="predict",
        build_inputs=lambda settings: pi_eval.build_test_inputs(settings["difficulty"], measurements=MEASUREMENTS),
        score=pi_eval.score_predictions,
    )


# ----------------------------------------------------------------------------------------
# get_training_phase_data (unchanged endpoint)
# ----------------------------------------------------------------------------------------
@pytest.mark.asyncio
@patch("gridarena.routers.phase_identification_benchmark.db.get_db_connection")
@patch("gridarena.routers.phase_identification_benchmark.pi.get_all_measurements")
async def test_get_training_phase_data_success(mock_get_measurements, mock_get_db):
    """Should organize measurements by grid and node correctly."""
    conn = MagicMock()
    cursor = MagicMock()
    mock_get_db.return_value = (conn, cursor)

    mock_get_measurements.return_value = [
        (GRID_ID, "NODE1", "R", "2025-01-01T00:00:00", 10.0, 5.0, 230.0, 0.1),
        (GRID_ID, "NODE1", "R", "2025-01-01T01:00:00", 12.0, 6.0, 231.0, 0.2),
        (GRID_ID, "NODE2", "S", "2025-01-01T00:00:00", 9.0, 4.5, 229.5, 0.05),
    ]

    result = await pi_router.get_training_phase_identification_data(difficulty="clean")
    assert "grids" in result
    assert GRID_ID in result["grids"]
    assert "NODE1" in result["grids"][GRID_ID]
    assert len(result["grids"][GRID_ID]["NODE1"]) == 2
    conn.close.assert_called_once()


@pytest.mark.asyncio
@patch("gridarena.routers.phase_identification_benchmark.db.get_db_connection")
@patch("gridarena.routers.phase_identification_benchmark.pi.get_all_measurements")
async def test_get_training_phase_data_db_error(mock_get_measurements, mock_get_db):
    """Should raise HTTP 500 on DB error."""
    conn = MagicMock()
    cursor = MagicMock()
    mock_get_db.return_value = (conn, cursor)
    mock_get_measurements.side_effect = Exception("DB fail")

    with pytest.raises(HTTPException) as exc:
        await pi_router.get_training_phase_identification_data(difficulty="clean")
    assert exc.value.status_code == 500
    conn.close.assert_called_once()


@pytest.mark.asyncio
@patch("gridarena.routers.phase_identification_benchmark.db.get_db_connection")
@patch("gridarena.routers.phase_identification_benchmark.pi.get_all_measurements")
async def test_training_data_deterministic(mock_get_measurements, mock_get_db):
    """Same difficulty should produce identical noise across calls."""
    mock_get_db.return_value = (MagicMock(), MagicMock())
    mock_get_measurements.return_value = [
        (GRID_ID, "NODE1", "R", "2025-01-01T00:00:00", 10.0, 5.0, 230.0, 0.1),
    ]

    r1 = await pi_router.get_training_phase_identification_data(difficulty="easy")
    mock_get_db.return_value = (MagicMock(), MagicMock())
    r2 = await pi_router.get_training_phase_identification_data(difficulty="easy")

    m1 = r1["grids"][GRID_ID]["NODE1"][0]
    m2 = r2["grids"][GRID_ID]["NODE1"][0]
    assert m1["power_active"] == m2["power_active"]
    assert m1["voltage_magnitude"] == m2["voltage_magnitude"]


# ----------------------------------------------------------------------------------------
# Test inputs and scoring
# ----------------------------------------------------------------------------------------
def test_build_test_inputs_hides_ids_and_labels():
    prepared = pi_eval.build_test_inputs("clean", measurements=MEASUREMENTS)

    assert len(prepared.calls) == 2  # one predict() call per grid
    by_grid = {t["grid_id"]: (name, t["labels"]) for name, t in prepared.truth.items()}
    assert set(by_grid) == {"gA", "gB"}

    name, labels = by_grid["gA"]
    (payload,) = prepared.calls[name]
    assert set(payload) == set(labels)  # same random keys, one per (node, phase) series
    assert sorted(labels.values()) == ["R", "S", "T"]
    blob = json.dumps(payload)
    for hidden in ("N1", "N2", "N3", "gA", '"phase"'):
        assert hidden not in blob
    assert {len(records) for records in payload.values()} == {1, 2}
    assert set(next(iter(payload.values()))[0]) == {
        "datetime", "power_active", "power_reactive", "voltage_magnitude", "voltage_angle",
    }


def test_build_test_inputs_fresh_keys_but_same_noise():
    a = pi_eval.build_test_inputs("hard", measurements=MEASUREMENTS)
    b = pi_eval.build_test_inputs("hard", measurements=MEASUREMENTS)

    def values(prepared):
        out = {}
        for name, t in prepared.truth.items():
            for key, records in prepared.calls[name][0].items():
                out[(t["grid_id"], t["labels"][key], records[0]["voltage_magnitude"])] = records
        return out

    keys_a = {k for name in a.calls for k in a.calls[name][0]}
    keys_b = {k for name in b.calls for k in b.calls[name][0]}
    assert not keys_a & keys_b
    assert values(a) == values(b)


def test_score_predictions_counts_missing_invalid_and_errors():
    truth = {
        "0000": {"grid_id": "gA", "labels": {"k1": "R", "k2": "S", "k3": "T", "k4": "R"}},
        "0001": {"grid_id": "gB", "labels": {"m1": "S"}},
    }
    outputs = {
        "0000": {"ok": True, "result": {"k1": "R", "k2": "S", "k3": "x", "zz": "R"}, "seconds": 0.5},
        "0001": {"ok": False, "error": "Traceback...\nValueError: boom", "seconds": 0.1},
    }
    score = pi_eval.score_predictions(outputs, truth)

    assert score["correct"] == 2
    assert score["total"] == 5
    assert score["accuracy"] == 0.4
    assert score["failed_grids"] == 1
    assert score["runtime_seconds"] == 0.6
    ga, gb = score["per_grid"]
    assert (ga["missing"], ga["invalid"], ga["unknown_keys"], ga["accuracy"]) == (1, 1, 1, 0.5)
    assert gb["error"] == "ValueError: boom"
    assert score["confusion"]["T"]["other"] == 1
    assert score["confusion"]["S"]["other"] == 1


def test_score_predictions_rejects_non_dict_result():
    truth = {"0000": {"grid_id": "gA", "labels": {"k1": "R"}}}
    score = pi_eval.score_predictions({"0000": {"ok": True, "result": ["R"], "seconds": 0.0}}, truth)
    assert score["accuracy"] == 0.0
    assert "must return a dict" in score["per_grid"][0]["error"]


# ----------------------------------------------------------------------------------------
# Upload validation
# ----------------------------------------------------------------------------------------
@pytest.mark.parametrize("source", [
    "def predict(m):\n    return {}\n",
    "class predict:\n    pass\n",
    "from mymodel import run as predict\n",
    "import numpy\nclass M:\n    def f(self, m): return {}\npredict = M().f\n",
])
def test_check_entrypoint_accepts_top_level_predict(source):
    packaging.check_entrypoint(source, "predict")


@pytest.mark.parametrize("source, message", [
    ("def predict(m):\n  return {\n", "syntax error"),
    ("def other(m):\n    return {}\n", "must define a top-level function 'predict'"),
    ("def outer():\n    def predict(m): pass\n", "must define a top-level function 'predict'"),
    ("async def predict(m):\n    return {}\n", "not 'async def'"),
])
def test_check_entrypoint_rejects(source, message):
    with pytest.raises(packaging.SubmissionError, match=message):
        packaging.check_entrypoint(source, "predict")


def _zip(files: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return buf.getvalue()


def test_install_zip_with_single_top_level_folder(tmp_path):
    data = _zip({"mymodel/submission.py": "def predict(m):\n    return {}\n", "mymodel/weights/w.txt": "1"})
    packaging.install_submission("model.zip", data, tmp_path / "code", "predict")
    assert (tmp_path / "code" / "submission.py").exists()
    assert (tmp_path / "code" / "weights" / "w.txt").read_text() == "1"


@pytest.mark.parametrize("files, message", [
    ({"main.py": "def predict(m): pass\n"}, "must contain submission.py"),
    ({"submission.py": "def predict(m): pass\n", "../evil.py": "x"}, "Unsafe path"),
])
def test_install_zip_rejects(tmp_path, files, message):
    with pytest.raises(packaging.SubmissionError, match=message):
        packaging.install_submission("model.zip", _zip(files), tmp_path / "code", "predict")


def test_install_rejects_other_file_types(tmp_path):
    with pytest.raises(packaging.SubmissionError, match=r"\.py file or a \.zip"):
        packaging.install_submission("guesses.json", b"{}", tmp_path / "code", "predict")


# ----------------------------------------------------------------------------------------
# Running submissions (real sandboxed subprocess, no database)
# ----------------------------------------------------------------------------------------
def test_submission_scores_a_correct_script(registry, phase_task, tmp_path):
    record = registry.submit(phase_task, filename="s.py", data=ANGLE_SCRIPT.encode(),
                             submission_name="angle_v1", settings={"difficulty": "clean"})
    assert record["status"] == "queued"

    done = _wait(registry, record["submission_id"], phase_task.name)
    assert done["status"] == "completed", done
    assert done["score"]["accuracy"] == 1.0
    assert done["score"]["total"] == 5
    assert {g["grid_id"] for g in done["score"]["per_grid"]} == {"gA", "gB"}

    # Hidden test inputs are removed once scored; the submitted code is kept.
    run_dir = tmp_path / "runs" / record["submission_id"]
    assert not (run_dir / "inputs").exists()
    assert (run_dir / "code" / "submission.py").exists()


def test_submission_with_numpy_results_and_print_output(registry, phase_task):
    script = (
        "import numpy as np\n"
        "print('loading model')\n"
        "def predict(measurements):\n"
        "    return {k: np.str_('R') for k in measurements}\n"
    )
    record = registry.submit(phase_task, filename="s.py", data=script.encode(),
                             submission_name="np_v1", settings={"difficulty": "easy"})
    done = _wait(registry, record["submission_id"], phase_task.name)
    assert done["status"] == "completed", done
    assert done["score"]["correct"] == 2  # the two "R" series
    assert "loading model" in done["log_tail"]


def test_submission_that_fails_to_import(registry, phase_task):
    script = "import not_a_real_module_xyz\ndef predict(m):\n    return {}\n"
    record = registry.submit(phase_task, filename="s.py", data=script.encode(),
                             submission_name="bad", settings={"difficulty": "clean"})
    done = _wait(registry, record["submission_id"], phase_task.name)
    assert done["status"] == "failed"
    assert "could not be loaded" in done["error"]
    assert "not_a_real_module_xyz" in done["error"]
    assert "Traceback" in done["log_tail"]


def test_submission_where_predict_raises_on_one_grid(registry, phase_task):
    script = (
        "def predict(measurements):\n"
        "    if len(measurements) == 2:\n"
        "        raise ValueError('cannot handle small grids')\n"
        "    return {k: 'R' for k in measurements}\n"
    )
    record = registry.submit(phase_task, filename="s.py", data=script.encode(),
                             submission_name="partial", settings={"difficulty": "clean"})
    done = _wait(registry, record["submission_id"], phase_task.name)
    assert done["status"] == "completed"
    assert done["score"]["failed_grids"] == 1
    gb = next(g for g in done["score"]["per_grid"] if g["grid_id"] == "gB")
    assert gb["error"] == "ValueError: cannot handle small grids"


def test_submission_timeout(registry, phase_task, monkeypatch):
    monkeypatch.setenv("GRIDARENA_SUBMISSION_TIMEOUT", "2")
    script = "import time\ndef predict(m):\n    time.sleep(60)\n"
    record = registry.submit(phase_task, filename="s.py", data=script.encode(),
                             submission_name="slow", settings={"difficulty": "clean"})
    done = _wait(registry, record["submission_id"], phase_task.name, timeout=30)
    assert done["status"] == "failed"
    assert "did not finish within 2 s" in done["error"]


def test_submission_does_not_inherit_secrets(registry, monkeypatch):
    monkeypatch.setenv("GRID_DB_DSN", "postgresql://secret@db/gridarena")
    task = BenchmarkTask(
        name="env_probe", function="predict",
        build_inputs=lambda settings: PreparedInputs(calls={"0000": [None]}, truth=None),
        score=lambda outputs, truth: {"result": outputs["0000"]["result"]},
    )
    script = "import os\ndef predict(_):\n    return os.environ.get('GRID_DB_DSN')\n"
    record = registry.submit(task, filename="s.py", data=script.encode(),
                             submission_name="probe", settings={"difficulty": "clean"})
    done = _wait(registry, record["submission_id"], task.name)
    assert done["status"] == "completed", done
    assert done["score"] == {"result": None}


def test_test_data_failure_is_reported(registry):
    def no_test_grids(difficulty):
        raise Exception("No valid grids found for the specified task and phase.")

    task = BenchmarkTask(name="phase_identification", function="predict",
                         build_inputs=no_test_grids, score=pi_eval.score_predictions)
    record = registry.submit(task, filename="s.py", data=ANGLE_SCRIPT.encode(),
                             submission_name="x", settings={"difficulty": "clean"})
    done = _wait(registry, record["submission_id"], task.name)
    assert done["status"] == "failed"
    assert "No valid grids found" in done["error"]


def test_interrupted_runs_are_marked_failed_on_restart(tmp_path, phase_task):
    run_dir = tmp_path / "runs" / "0123456789ab"
    (run_dir / "inputs").mkdir(parents=True)
    (run_dir / "run.json").write_text(json.dumps({
        "submission_id": "0123456789ab", "benchmark": phase_task.name, "status": "running",
        "created_at": "2025-01-01T00:00:00+00:00",
    }))
    reg = SubmissionRegistry(runs_dir=tmp_path / "runs")
    try:
        record = reg.get("0123456789ab", phase_task.name)
        assert record["status"] == "failed"
        assert "restarted" in record["error"]
        assert not (run_dir / "inputs").exists()
    finally:
        reg.shutdown()


# ----------------------------------------------------------------------------------------
# HTTP endpoints (registry and test data patched, no database)
# ----------------------------------------------------------------------------------------
@pytest.fixture
def patched_router(registry, phase_task, monkeypatch):
    monkeypatch.setattr(submissions_router, "submission_registry", registry)
    monkeypatch.setattr(pi_router, "PHASE_TASK", phase_task)
    return registry


def test_download_submission_template():
    resp = client.get(f"{BENCH_PREFIX}/submission-template")
    assert resp.status_code == 200
    assert "phase_identification_submission.py" in resp.headers["content-disposition"]
    packaging.check_entrypoint(resp.text, "predict")


def test_submission_template_runs_as_a_submission(patched_router, phase_task):
    """The blank template must work out of the box (random baseline)."""
    source = pi_router._SUBMISSION_TEMPLATE.read_bytes()
    record = patched_router.submit(phase_task, filename="t.py", data=source,
                                   submission_name="template", settings={"difficulty": "clean"})
    done = _wait(patched_router, record["submission_id"], phase_task.name)
    assert done["status"] == "completed", done
    assert done["score"]["total"] == 5
    assert all(g["invalid"] == 0 and g["missing"] == 0 for g in done["score"]["per_grid"])


def test_submit_poll_and_list_over_http(patched_router):
    resp = client.post(
        f"{BENCH_PREFIX}/submissions",
        files={"file": ("my_model.py", ANGLE_SCRIPT.encode(), "text/x-python")},
        data={"submission_name": "alice_v1", "difficulty": "medium"},
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
    assert record["settings"] == {"difficulty": "medium"}
    assert record["filename"] == "my_model.py"

    listed = client.get(f"{BENCH_PREFIX}/submissions", params={"submission_name": "alice_v1"}).json()
    assert [r["submission_id"] for r in listed["submissions"]] == [submission_id]
    assert client.get(f"{BENCH_PREFIX}/submissions", params={"submission_name": "bob"}).json() == {"submissions": []}


@pytest.mark.parametrize("filename, content, data, detail", [
    ("m.py", b"def nope(m): pass\n", {"submission_name": "a"}, "must define a top-level function"),
    ("m.py", b"def predict(m): pass\n", {"submission_name": "has space"}, "submission_name"),
    ("m.json", b"{}", {"submission_name": "a"}, ".zip"),
])
def test_submit_rejects_bad_uploads(patched_router, filename, content, data, detail):
    resp = client.post(f"{BENCH_PREFIX}/submissions", files={"file": (filename, content)}, data=data)
    assert resp.status_code == 422
    assert detail in resp.text


def test_get_unknown_submission_is_404(patched_router):
    assert client.get(f"{BENCH_PREFIX}/submissions/ffffffffffff").status_code == 404
    assert client.get(f"{BENCH_PREFIX}/submissions/../../etc").status_code == 404


# ----------------------------------------------------------------------------------------
# Integration tests (real FastAPI + PostgreSQL via app)
# ----------------------------------------------------------------------------------------
def _setup_phase_grid(grid_id: str, usage: dict):
    """Create a minimal grid + measurements via public API endpoints."""
    grid_payload = {
        "grid_id": grid_id,
        "nodes": [
            {"node_id": "PT", "coord_lat": 38.65, "coord_lon": -7.98, "coord_error": 1},
            {"node_id": "NODE1", "coord_lat": 38.651, "coord_lon": -7.981, "coord_error": 1},
            {"node_id": "NODE2", "coord_lat": 38.652, "coord_lon": -7.982, "coord_error": 1},
        ],
        "connections": [
            {
                "connection_id": "LineUID000001",
                "from_node_id": "PT",
                "to_node_id": "NODE1",
                "length": 1000,
                "cable_id": "TypeLine00001",
            },
            {
                "connection_id": "LineUID000002",
                "from_node_id": "NODE1",
                "to_node_id": "NODE2",
                "length": 1000,
                "cable_id": "TypeLine00001",
            },
        ],
        "cables": [
            {
                "cable_id": "TypeLine00001",
                "r_imp_real": 0.01, "r_imp_imag": 0.001,
                "s_imp_real": 0.01, "s_imp_imag": 0.001,
                "t_imp_real": 0.01, "t_imp_imag": 0.001,
                "r_nom_curr": 10000, "s_nom_curr": 10000, "t_nom_curr": 10000,
            }
        ],
    }
    flags = {
        f"{task}_{split}": "NO"
        for task in ("phase_detection", "topology_detection", "voltage_control", "state_estimation")
        for split in ("test", "train")
    }
    flags.update(usage)

    now = datetime.utcnow().isoformat()
    meas_payload = {
        "grid_id": grid_id,
        "historical": [
            {
                "node_id": "NODE1",
                "measurements": [{"datetime": now, "phase": "R", "power_active": 10.0, "power_reactive": 5.0,
                                  "voltage_magnitude": 230.0, "voltage_angle": 0.0}],
            },
            {
                "node_id": "NODE2",
                "measurements": [{"datetime": now, "phase": "S", "power_active": 9.0, "power_reactive": 4.0,
                                  "voltage_magnitude": 229.5, "voltage_angle": -120.0}],
            },
        ],
    }

    resp_grid = client.post(
        "/grid/", params=flags,
        files={"file": ("grid.json", io.BytesIO(json.dumps(grid_payload).encode()), "application/json")},
    )
    assert resp_grid.status_code == 200, resp_grid.text
    resp_meas = client.post(
        "/measurements/",
        files={"file": ("meas.json", io.BytesIO(json.dumps(meas_payload).encode()), "application/json")},
    )
    assert resp_meas.status_code == 200, resp_meas.text


def test_get_training_phase_identification_data_integration(clean_db_for_phase_benchmark):
    """Should return structured measurement data grouped by grid and node."""
    grid_id = "phase_grid_train_pg1"
    _setup_phase_grid(grid_id, {"phase_detection_train": "YES"})

    response = client.get(f"{BENCH_PREFIX}/training-phase-data")
    assert response.status_code == 200, response.text
    data = response.json()

    assert grid_id in data["grids"]
    assert "NODE1" in data["grids"][grid_id]
    assert data["grids"][grid_id]["NODE1"][0]["phase"] == "R"


def test_script_submission_integration(clean_db_for_phase_benchmark, registry, monkeypatch):
    """A script is run on the database's test grids and scored against their true phases."""
    grid_id = "phase_grid_test_pg1"
    _setup_phase_grid(grid_id, {"phase_detection_test": "YES"})
    monkeypatch.setattr(submissions_router, "submission_registry", registry)

    resp = client.post(
        f"{BENCH_PREFIX}/submissions",
        files={"file": ("model.py", ANGLE_SCRIPT.encode())},
        data={"submission_name": "integration"},
    )
    assert resp.status_code == 202, resp.text

    done = _wait(registry, resp.json()["submission_id"], "phase_identification")
    assert done["status"] == "completed", done
    mine = next(g for g in done["score"]["per_grid"] if g["grid_id"] == grid_id)
    assert (mine["series"], mine["correct"]) == (2, 2)

    # The test grid's data isn't downloadable any more.
    assert client.get(f"{BENCH_PREFIX}/phase-data").status_code == 404
