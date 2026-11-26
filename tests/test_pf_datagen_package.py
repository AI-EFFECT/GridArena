"""The standalone dataset package (gridarena/pf_datagen/package.py): inputs
saved and loaded back, the file sink, and a package unzipped and run with
plain Python -- no database -- giving GridArena's rows."""

import csv
import importlib
import json
import os
import subprocess
import sys
import zipfile
from math import sqrt
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import gridarena.database as db
from gridarena.pf_datagen import columns
from gridarena.pf_datagen.grid_loader import MVDER, MVBranch, MVGridTopology, MVShunt
from gridarena.pf_datagen.inputs import (
    load_inputs, prepare_inputs, save_inputs, topology_from_dict, topology_to_dict,
)
from gridarena.pf_datagen.job import execute, request_and_seed, run_pf_datagen, start_run
from gridarena.pf_datagen.package import build_package, code_modules, folder_name, grid_seeds
from gridarena.pf_datagen.schemas import PFDataGenRequest
from gridarena.pf_datagen.sinks import FileSink, _cell

grid_loader = importlib.import_module("gridarena.pf_datagen.grid_loader")
scenario_builder = importlib.import_module("gridarena.pf_datagen.scenario_builder")
der_profiles = importlib.import_module("gridarena.pf_datagen.der_profiles")

NODES = ["SUB", "N1", "N2", "N3", "N4", "T1_LV"]
REQUEST = {
    "historical_database_id": "h1", "scenario_count": 8, "seed": 3, "reassignment_period_timesteps": 4,
    "topology_perturbation": {"type": "n_minus_k", "k": 1},
    "admittance_perturbation": {"enabled": True, "sigma": 0.05},
    "load_level": {"mode": "loadability", "min": 0.2, "max": 1.0, "bins": 4},
    "der": {"source": "add", "groups": [
        {"database_id": "gen_pv", "count": 2, "capacity_mw": 0.2},
        {"database_id": "gen_pv", "count": 1, "capacity_mw": 0.5, "placement": "feeder_end", "control": "q_v"}]},
    "measurements": {"enabled": True},
    "target_mix": {"min_violation_share": 0.25, "pilot_share": 0.25},
    "chunk_commit_size": 5,
}


def _topology():
    lines = [MVBranch(f"C{i}", a, b, "cab", 1500.0, 0.3, 0.12, 250.0, cap_uf_per_km=0.3)
             for i, (a, b) in enumerate([("SUB", "N1"), ("N1", "N2"), ("N2", "N3"), ("N1", "N4")], 1)]
    t1 = MVBranch("T1", "N3", "T1_LV", None, 0.0, 0.0, 0.0, 24.2, kind="transformer",
                  z_ohm=complex(0.01, 0.04) * 15e3 ** 2 / 630e3, rated_kva=630.0, tap_pos=1, tap_step_pct=2.5,
                  tap_min=-2, tap_max=2)
    return MVGridTopology(
        "mv/1", 15.0, ["SUB"], {n: i for i, n in enumerate(NODES)}, lines + [t1],
        node_vn_kv={"T1_LV": sqrt(3) * 0.23}, load_transfer={"N3": [("T1_LV", 1.0)]},
        shunts=[MVShunt("CB1", "N2", 300.0, 2, 1)],
        ders=[MVDER("PV1", "N4", "pv", 500.0, 500.0, "constant", 0.6, "q_v", lat=48.8, lon=2.2)],
        node_coords={n: (48.8 + 0.01 * i, 2.2) for i, n in enumerate(NODES[:5])},
    )


def _scenarios(n=8):
    rng = np.random.default_rng(0)
    out = []
    for t in range(n):
        s = np.zeros(len(NODES), dtype=complex)
        s[1:4] = (rng.uniform(2e5, 9e5, 3) * (1 + 0.3j))
        out.append({"timestamp": f"2024-06-01 {t:02d}:00:00", "S": s})
    return out


def _generation_profile(*args, **kwargs):
    """The generation database's values at the 8 scenarios (pf_datagen/der_profiles.py)."""
    agg = np.array([0.0, 0.0, 0.2, 0.5, 0.8, 0.9, 0.6, 0.1]) * 40e6
    return der_profiles.GenerationProfile(
        database_id="gen_pv", data_type="pv", series_ids=["agg"], values_w=agg[None, :], ratings_w=np.array([36e6]),
        reference_year=2024, reference_max_w=40e6, gen_times=1.7e9 + 3600.0 * np.arange(8), step_s=900.0,
        shift_s=0.0, n_fallback=0)


def _fake_build_scenarios(*args, **kwargs):
    rng = args[6]
    rng.random(5)                   # the builder draws the series assignment from the main generator
    return _scenarios()


def _db_patches(out=None, weights=None):
    out = out if out is not None else {t: [] for t in columns.ROWS}
    weights = weights if weights is not None else {}

    def batch(run_id, bus, branch, ybus, runtime, conn, cursor, summary_rows=(), der_rows=(), measurement_rows=()):
        for t, rows in (("bus", bus), ("branch", branch), ("ybus", ybus), ("runtime", runtime),
                        ("summary", summary_rows), ("der", der_rows), ("measurements", measurement_rows)):
            out[t] += rows

    return [
        patch.object(db, "get_db_connection", return_value=(MagicMock(), MagicMock())),
        patch.object(db, "insert_pf_datagen_batch", side_effect=batch),
        patch.object(db, "insert_pf_datagen_variants", side_effect=lambda r, rows, c, cu: out["variants"].extend(rows)),
        patch.object(db, "insert_pf_datagen_measurement_layout",
                     side_effect=lambda r, rows, c, cu: out["layout"].extend(rows)),
        patch.object(db, "set_pf_datagen_design_weights", side_effect=lambda r, w, c, cu: weights.update(w)),
        patch.object(db, "get_pf_datagen_column_types", return_value={"runtime": {"failure_reason": "string"}}),
        patch.object(grid_loader, "get_mv_grid_topology", return_value=_topology()),
        patch.object(scenario_builder, "build_scenarios", side_effect=_fake_build_scenarios),
        patch.object(der_profiles, "generation_step_s", return_value=900.0),
        patch.object(der_profiles, "build_generation_profile", side_effect=_generation_profile),
    ]


class _Patched:
    def __init__(self, patches):
        self.patches = patches

    def __enter__(self):
        for p in self.patches:
            p.start()

    def __exit__(self, *exc):
        for p in reversed(self.patches):
            p.stop()


def _run_dir(tmp_path, name):
    d = tmp_path / name
    d.mkdir()
    (d / "run_config.json").write_text(json.dumps({**REQUEST, "run_id": "r1", "mv_grid_id": "mv/1"}))
    return d


def _gridarena_run(tmp_path):
    out, weights = {t: [] for t in columns.ROWS}, {}
    run_dir = _run_dir(tmp_path, "gridarena")
    with _Patched(_db_patches(out, weights)):
        result = run_pf_datagen(str(run_dir))
    assert result["status"] == "completed", result
    return out, weights, json.loads((run_dir / "report.json").read_text())


def _expected_csv(rows, table, run_id, weights):
    """GridArena's rows as the file sink writes them: export order, run_id first,
    solve_time_ms dropped (it is a measured time), design weights filled in."""
    pos = columns.index(columns.ROWS[table])
    header = ["run_id"] + list(columns.EXPORT[table])
    out = []
    for r in rows:
        row = [run_id] + [str(_cell(r[pos[c]])) for c in columns.EXPORT[table]]
        if table == "runtime":
            w = weights.get(int(row[header.index("load_scenario_idx")]))
            row[header.index("design_weight")] = "" if w is None else str(w)
        out.append(row)
    return _drop_time(header, out)


def _drop_time(header, rows):
    if "solve_time_ms" not in header:
        return header, rows
    i = header.index("solve_time_ms")
    return header[:i] + header[i + 1:], [r[:i] + r[i + 1:] for r in rows]


def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    return _drop_time(rows[0], rows[1:])


def test_topology_survives_json():
    t = _topology()
    assert topology_from_dict(json.loads(json.dumps(topology_to_dict(t)))) == t


def test_saved_inputs_give_gridarena_rows(tmp_path):
    expected, weights, report = _gridarena_run(tmp_path)
    assert weights and expected["measurements"] and expected["der"]     # the features the check needs

    run_dir = _run_dir(tmp_path, "offline")
    run_data = start_run(str(run_dir))
    request, seed = request_and_seed(run_data)
    with _Patched(_db_patches()):
        inputs = prepare_inputs("mv/1", request, seed, db)
    save_inputs(inputs, request, run_dir / "inputs")
    loaded = load_inputs(run_dir / "inputs")
    result = execute(str(run_dir), run_data, request, loaded, FileSink(run_dir / "results", "r1"))
    assert result["status"] == "completed"

    for table, rows in expected.items():
        assert _read_csv(run_dir / "results" / f"{table}.csv") == _expected_csv(rows, table, "r1", weights), table
    assert json.loads((run_dir / "report.json").read_text()) == report


def test_code_modules_leave_out_the_web_app_and_database():
    modules = code_modules()
    assert "gridarena.pf_datagen.job" in modules and "gridarena.powerflow.powerflow_algorithm" in modules
    assert not [m for m in modules if m.startswith(("gridarena.database", "gridarena.routers", "gridarena.app",
                                                    "gridarena.rl", "gridarena.diffusion"))]


def test_group_seeds_match_group_runs():
    seeds = grid_seeds(11, 3)
    assert seeds == [int(np.random.SeedSequence([11, k]).generate_state(1)[0]) for k in range(3)]
    assert folder_name("mv/1 a") == "mv_1_a"


def _build(tmp_path, grids):
    zip_path = tmp_path / "pkg.zip"
    with _Patched(_db_patches()):
        build_package(zip_path, grids, db, "abcd1234")
    target = tmp_path / "unzipped"
    target.mkdir()
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(target)
    return target / "gridarena_dataset_abcd1234"


def _run_py(root, *args):
    # A clean environment: the package must import its own copy, nothing else.
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    return subprocess.run([sys.executable, "run.py", *args], cwd=root, env=env, capture_output=True, text=True,
                          timeout=300)


def test_package_runs_standalone_and_gives_gridarena_rows(tmp_path):
    expected, weights, report = _gridarena_run(tmp_path)
    request = PFDataGenRequest(**REQUEST)
    root = _build(tmp_path, [("mv/1", request)])

    for name in ("how_to_run.md", "run.py", "requirements.txt", "package.json", "column_types.json",
                 "gridarena/pf_datagen/job.py", "runs/mv_1/run_config.json", "runs/mv_1/inputs/scenarios.npz"):
        assert (root / name).exists(), name
    assert not (root / "gridarena" / "database").exists()
    assert (root / "gridarena" / "__init__.py").read_text() == ""
    assert "python run.py" in (root / "how_to_run.md").read_text(encoding="utf-8")

    proc = _run_py(root)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "8 load scenarios" in proc.stdout
    results = root / "results" / "mv_1"
    run_id = json.loads((results / "run_config.json").read_text())["run_id"]
    for table, rows in expected.items():
        header, got = _read_csv(results / f"{table}.csv")
        # The package has its own run id (in run_id and output_group_id): compare with it as "R".
        exp_header, exp = _expected_csv(rows, table, "R", weights)
        assert header == exp_header
        i_ogid = header.index("output_group_id") if "output_group_id" in header else None
        assert [[v.replace(run_id, "R") for v in r] for r in got] == \
               [[v.replace("r1", "R", 1) if i == i_ogid else v for i, v in enumerate(r)] for r in exp], table
    assert json.loads((results / "report.json").read_text()) == report
    status = json.loads((results / "run_config.json").read_text())
    assert status["status"] == "completed" and status["metrics"]["converged"] == len(expected["runtime"])


def test_package_refuses_an_edited_scenario_field(tmp_path):
    root = _build(tmp_path, [("mv/1", PFDataGenRequest(**REQUEST))])
    cfg_path = root / "runs" / "mv_1" / "run_config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["scenario_count"] = 50
    cfg_path.write_text(json.dumps(cfg))
    proc = _run_py(root)
    assert proc.returncode != 0
    assert "scenario_count" in proc.stderr


def test_package_parquet_and_failed_grid_exit_code(tmp_path):
    pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    ok = PFDataGenRequest(**{**REQUEST, "seed": 5})
    # Asking for more reconfigurations than a tree without switches has fails at solve time.
    bad = PFDataGenRequest(**{**REQUEST, "seed": 6, "reconfiguration": {"enabled": True, "n_configs": 3}})
    root = _build(tmp_path, [("mv/1", ok), ("mv/2", bad)])
    proc = _run_py(root, "--format", "parquet")
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "FAILED" in proc.stdout
    runtime = pq.read_table(root / "results" / "mv_1" / "runtime.parquet")
    assert runtime.num_rows > 0
    assert str(runtime.schema.field("failure_reason").type) == "string"     # typed although all empty
    assert json.loads((root / "results" / "mv_2" / "run_config.json").read_text())["status"] == "failed"
