from datetime import datetime, timedelta

import numpy as np
import pytest
from fastapi.testclient import TestClient

from gridarena.app import app
from gridarena.powerflow import lv_runs
from gridarena.powerflow import three_phase as tp
from gridarena.powerflow.powerflow_algorithm import full_pf

from tests._benchmark_helpers import delete_grids, register_grid

client = TestClient(app)

# PT - 1 - 2 - 3, with a branch 1 - 4; different impedance per phase.
CONNECTIONS = [(0, 1), (1, 2), (2, 3), (1, 4)]
Z = np.array([[0.03 + 0.008j, 0.04 + 0.010j, 0.05 + 0.012j]] * 4)
V_ROOT = np.array([[230 * np.exp(1j * np.radians(a)) for a in tp.NOMINAL_ANGLES_DEG]])


def _unbalanced_load():
    s = np.zeros((1, 5, 3), dtype=complex)
    s[0, 1, 0] = 1000          # R
    s[0, 2, 0] = 3000 + 300j   # R
    s[0, 3, 1] = 2500          # S
    s[0, 4, 2] = -2000         # T: PV generation
    return s


def _residual(grid, s, result):
    """Largest violation of U = V_root - D_p J_p - D_n sum(J) at the solution."""
    u = result.v_phase[:, grid.nodes, :]
    j = np.where(s[:, grid.nodes, :] == 0, 0, np.conj(s[:, grid.nodes, :] / u))
    drop = np.stack([j[:, :, p] @ grid.d_phase[p].T for p in range(3)], axis=-1)
    v_n = j.sum(axis=-1) @ grid.d_neutral.T
    return np.abs(V_ROOT[:, None, :] - drop - v_n[:, :, None] - u).max()


# ----------------------------------------------------------------------------------------
# Three-phase solver
# ----------------------------------------------------------------------------------------
def test_ideal_neutral_matches_single_phase_solver_per_phase():
    s = _unbalanced_load()
    result = tp.solve(tp.build_radial_grid(5, CONNECTIONS, Z, neutral_ratio=0.0), s, V_ROOT)
    assert result.converged.all()
    for p in range(3):
        expected = full_pf(s[0, :, p], np.array(CONNECTIONS), 1 / Z[:, p], V_ROOT[0, p])
        np.testing.assert_allclose(result.v_phase[0, :, p], expected, atol=1e-6)
    np.testing.assert_allclose(result.v_neutral, 0, atol=1e-12)


def test_unbalanced_load_shifts_the_neutral_and_the_other_phases():
    s = _unbalanced_load()
    grid = tp.build_radial_grid(5, CONNECTIONS, Z, neutral_ratio=1.0)
    ideal = tp.solve(tp.build_radial_grid(5, CONNECTIONS, Z, neutral_ratio=0.0), s, V_ROOT)
    result = tp.solve(grid, s, V_ROOT)

    assert result.converged.all()
    assert _residual(grid, s, result) < 1e-5
    assert abs(result.v_neutral[0, 0]) == 0                       # grounded at the substation
    assert abs(result.v_neutral[0, 3]) > 1.0                      # floats away from it
    # Node 3 only draws on S, yet its R and T voltages move once the neutral is modelled.
    assert abs(abs(result.v_phase[0, 3, 0]) - abs(ideal.v_phase[0, 3, 0])) > 0.5
    assert abs(abs(result.v_phase[0, 3, 2]) - abs(ideal.v_phase[0, 3, 2])) > 0.5


def test_balanced_load_has_no_neutral_current():
    s = np.zeros((1, 5, 3), dtype=complex)
    s[0, 2, :] = 2000 + 200j
    z = np.repeat(Z[:, :1], 3, axis=1)
    result = tp.solve(tp.build_radial_grid(5, CONNECTIONS, z, neutral_ratio=1.0), s, V_ROOT)
    np.testing.assert_allclose(result.v_neutral, 0, atol=1e-9)
    mags = np.abs(result.v_phase[0])
    np.testing.assert_allclose(mags, mags[:, :1].repeat(3, axis=1), atol=1e-9)


def test_balanced_inputs_spread_power_and_symmetrise_the_reference():
    s = _unbalanced_load()
    v_root = np.array([[231 * np.exp(0.01j), 229 * np.exp(-2.1j), 230 * np.exp(2.1j)]])
    s_bal, v_bal = tp.balanced_inputs(s, v_root)
    np.testing.assert_allclose(s_bal[0, 2], [(3000 + 300j) / 3] * 3)
    np.testing.assert_allclose(np.abs(v_bal[0]), [230, 230, 230])
    np.testing.assert_allclose(np.degrees(np.angle(v_bal[0] / v_bal[0, 0])), [0, -120, 120], atol=1e-9)


def test_connection_order_and_direction_do_not_matter():
    s = _unbalanced_load()
    reference = tp.solve(tp.build_radial_grid(5, CONNECTIONS, Z, 1.0), s, V_ROOT)
    shuffled = [(4, 1), (3, 2), (1, 0), (2, 1)]
    result = tp.solve(tp.build_radial_grid(5, shuffled, Z[[3, 2, 0, 1]], 1.0), s, V_ROOT)
    np.testing.assert_allclose(result.v_phase, reference.v_phase, atol=1e-9)


def test_parallel_cables_are_combined():
    s = _unbalanced_load()
    single = tp.solve(tp.build_radial_grid(5, CONNECTIONS, Z / 2, 1.0), s, V_ROOT)
    doubled = tp.solve(tp.build_radial_grid(5, CONNECTIONS + CONNECTIONS, np.vstack([Z, Z]), 1.0), s, V_ROOT)
    np.testing.assert_allclose(doubled.v_phase, single.v_phase, atol=1e-9)


def test_loops_are_rejected_and_unconnected_nodes_left_out():
    with pytest.raises(tp.NotRadialError, match="loop"):
        tp.build_radial_grid(5, CONNECTIONS + [(3, 4)], np.vstack([Z, Z[:1]]))

    grid = tp.build_radial_grid(6, CONNECTIONS, Z)          # node 5 has no connection
    assert grid.unreachable == [5]
    s = np.zeros((1, 6, 3), dtype=complex)
    s[0, 5, 0] = 1000
    result = tp.solve(grid, s, V_ROOT)
    assert np.isnan(result.v_phase[0, 5]).all()
    assert not np.isnan(result.v_phase[0, :5]).any()


def test_many_timestamps_at_once_match_one_at_a_time():
    rng = np.random.default_rng(0)
    s = (rng.uniform(-3000, 5000, (6, 5, 3)) + 1j * rng.uniform(0, 500, (6, 5, 3)))
    v_root = V_ROOT.repeat(6, axis=0) * rng.uniform(0.97, 1.03, (6, 1))
    grid = tp.build_radial_grid(5, CONNECTIONS, Z, 1.0)
    batch = tp.solve(grid, s, v_root)
    for t in range(6):
        one = tp.solve(grid, s[t:t + 1], v_root[t:t + 1])
        np.testing.assert_allclose(batch.v_phase[t], one.v_phase[0], atol=1e-6)


def test_reference_voltages_fill_unmeasured_phases():
    ref = lv_runs._root_voltages({"R": (231.0, 1.0), "S": (229.0, -119.0)})
    assert np.isclose(abs(ref[2]), 230.0)
    assert np.isclose(np.degrees(np.angle(ref[2])), 121.0)
    assert lv_runs._root_voltages({}) is None


def test_error_summary_skips_reference_and_unconverged():
    def row(node, comp, meas, converged=True):
        return {"node_id": node, "phase": "R", "converged": converged,
                "computed": {"voltage_magnitude": comp, "voltage_angle": 1.0},
                "measured": None if meas is None else {"voltage_magnitude": meas, "voltage_angle": 0.0}}

    summary = lv_runs.error_summary([
        row("PT", 230.0, 230.0), row("N1", 229.0, 228.0), row("N1", 227.0, 228.0),
        row("N1", 200.0, 228.0, converged=False), row("N2", 226.0, None),
    ])
    assert summary == [{
        "node_id": "N1", "phase": "R", "points": 2, "mae_voltage_magnitude": 1.0,
        "max_abs_voltage_magnitude": 1.0, "bias_voltage_magnitude": 0.0, "mae_voltage_angle": 1.0,
    }]


# ----------------------------------------------------------------------------------------
# Integration (real FastAPI + PostgreSQL)
# ----------------------------------------------------------------------------------------
GRID_ID = "pf_grid_3ph_pg1"
LOOP_GRID_ID = "pf_grid_loop_pg1"
NO_PT_GRID_ID = "pf_grid_nopt_pg1"
T0 = datetime(2025, 7, 1, 12)


@pytest.fixture(scope="module")
def pf_grid():
    """PT - N1 - N2 (500 m each), PT measured on all phases, N1 on R, N2 on S, 3 timestamps."""
    ids = [GRID_ID, LOOP_GRID_ID, NO_PT_GRID_ID]
    delete_grids(ids)

    def rec(h, phase, p, v, ang):
        return {"datetime": (T0 + timedelta(hours=h)).isoformat(), "phase": phase, "power_active": p,
                "power_reactive": 0.1 * p, "voltage_magnitude": v, "voltage_angle": ang}

    pt = [rec(h, ph, 0.0, 231.0, ang) for h in range(3) for ph, ang in zip("RST", tp.NOMINAL_ANGLES_DEG)]
    register_grid(client, GRID_ID, edges=[("PT", "N1"), ("N1", "N2")], usage=set(), measurements={
        "PT": pt,
        "N1": [rec(h, "R", 4000.0 + 500 * h, 228.5, -0.4) for h in range(3)],
        "N2": [rec(h, "S", 6000.0, 225.0, -121.0) for h in range(3)],
    })
    register_grid(client, LOOP_GRID_ID, edges=[("PT", "N1"), ("N1", "N2"), ("N2", "PT")], usage=set(),
                  measurements={"PT": pt})
    register_grid(client, NO_PT_GRID_ID, edges=[("A", "B")], usage=set(),
                  measurements={"A": [rec(0, "R", 0.0, 230.0, 0.0)]})
    yield GRID_ID
    delete_grids(ids)


def test_unbalanced_run_and_results(pf_grid):
    resp = client.post(f"/powerflow/{pf_grid}/run", params={"mode": "unbalanced"})
    assert resp.status_code == 200, resp.text
    summary = resp.json()
    assert (summary["timestamps"], summary["not_converged"], summary["skipped_timestamps"]) == (3, 0, 0)
    assert summary["neutral_impedance_ratio"] == 1.0

    data = client.get(f"/powerflow/data/{pf_grid}/results").json()
    assert data["run"]["mode"] == "unbalanced" and data["reference_node"] == "PT"
    by_key = {(r["node_id"], r["phase"], r["timestamp"]): r for r in data["results"]}
    assert len(by_key) == 3 * 3 * 4                      # 3 nodes x (R, S, T, N) x 3 timestamps

    stamp = str(T0)
    n2_s = by_key[("N2", "S", stamp)]
    assert n2_s["measured"] == {"voltage_magnitude": 225.0, "voltage_angle": -121.0}
    assert n2_s["computed"]["voltage_magnitude"] < 231.0 and n2_s["converged"]
    assert by_key[("N2", "R", stamp)]["measured"] is None    # no meter on R at N2
    assert by_key[("N2", "N", stamp)]["measured"] is None
    assert by_key[("N2", "N", stamp)]["computed"]["voltage_magnitude"] > 0.1
    pt_r = by_key[("PT", "R", stamp)]
    assert abs(pt_r["computed"]["voltage_magnitude"] - 231.0) < 1e-9

    errors = {(e["node_id"], e["phase"]): e for e in data["errors"]}
    assert set(errors) == {("N1", "R"), ("N2", "S")}
    assert errors[("N2", "S")]["points"] == 3

    only_s = client.get(f"/powerflow/data/{pf_grid}/results", params={"phase": "S"}).json()
    assert {r["phase"] for r in only_s["results"]} == {"S"}


def test_balanced_run_replaces_results(pf_grid):
    resp = client.post(f"/powerflow/{pf_grid}/run", params={"mode": "balanced", "start_time": str(T0)})
    assert resp.status_code == 200, resp.text
    assert resp.json()["neutral_impedance_ratio"] is None

    data = client.get(f"/powerflow/data/{pf_grid}/results").json()
    assert data["run"]["mode"] == "balanced"
    assert {r["phase"] for r in data["results"]} == {"R", "S", "T"}      # no neutral rows
    assert len({r["timestamp"] for r in data["results"]}) == 3
    n2 = [r["computed"]["voltage_magnitude"] for r in data["results"]
          if r["node_id"] == "N2" and r["timestamp"] == str(T0)]
    np.testing.assert_allclose(n2, n2[0], atol=1e-9)                    # same on every phase


@pytest.mark.parametrize("grid_id, params, status, detail", [
    (LOOP_GRID_ID, {}, 422, "loop"),
    (NO_PT_GRID_ID, {}, 422, "PT"),
    (GRID_ID, {"start_time": "2030-01-01T00:00:00"}, 404, "No measurements"),
    (GRID_ID, {"mode": "three-phase"}, 422, "mode"),
    (GRID_ID, {"neutral_impedance_ratio": -1}, 422, "neutral_impedance_ratio"),
])
def test_run_errors(pf_grid, grid_id, params, status, detail):
    resp = client.post(f"/powerflow/{grid_id}/run", params=params)
    assert resp.status_code == status, resp.text
    assert detail in resp.text


def test_results_404_without_a_run(pf_grid):
    assert client.get(f"/powerflow/data/{LOOP_GRID_ID}/results").status_code == 404


def test_ui_pages(pf_grid):
    client.post(f"/powerflow/{pf_grid}/run", params={"mode": "unbalanced"})
    page = client.get("/powerflow/ui")
    assert page.status_code == 200 and 'id="mode"' in page.text

    results = client.get(f"/powerflow/ui/{pf_grid}", params={"phase": "S"})
    assert results.status_code == 200
    assert "Computed vs Measured" in results.text and "Unbalanced (three-phase, four-wire)" in results.text
    neutral = client.get(f"/powerflow/ui/{pf_grid}", params={"phase": "N"})
    assert neutral.status_code == 200 and "neutral-to-ground" in neutral.text
