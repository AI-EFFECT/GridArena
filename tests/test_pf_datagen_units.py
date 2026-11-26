"""Unit/convention tests for the MV PF data-generation pipeline.

MV injections are three-phase totals in W and the MV nominal voltage is
line-to-line (docs/api/index.md#units). These pin the per-phase-equivalent
solve in gridarena/pf_datagen/three_phase.py against hand-computed results.
"""

import importlib
import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import gridarena.database as db
from gridarena.database.create_pf_datagen_database import ADDED_COLUMNS
from gridarena.database.insert_pf_datagen_database import (
    _RESULT_TABLES, get_pf_datagen_all_rows, insert_pf_datagen_batch, insert_pf_datagen_measurement_layout,
    insert_pf_datagen_variants,
)
from gridarena.pf_datagen import columns
from gridarena.pf_datagen.grid_loader import MVDER, MVBranch, MVGridTopology, MVShunt
from gridarena.pf_datagen.job import run_pf_datagen
from gridarena.pf_datagen.per_unit import PerUnitBases
from gridarena.pf_datagen.scenario_builder import fetch_series_values
from gridarena.pf_datagen.schemas import PFDataGenRequest
from gridarena.pf_datagen.three_phase import SQRT3, branch_flow, solve_balanced_pf
from gridarena.powerflow.powerflow_algorithm import full_pf

# Module objects for patch.object: as dotted strings, "gridarena.pf_datagen"
# resolves to the router of the same name re-exported by gridarena/__init__.py.
grid_loader = importlib.import_module("gridarena.pf_datagen.grid_loader")
scenario_builder = importlib.import_module("gridarena.pf_datagen.scenario_builder")
three_phase = importlib.import_module("gridarena.pf_datagen.three_phase")
solver = importlib.import_module("gridarena.pf_datagen.solver")

V_LL = 15_000.0     # 15 kV MV feeder
R_LINE = 2.0        # ohm per phase, purely resistive
P_3PH = 3_000_000.0  # 3 MW three-phase load at the far end


def _two_bus(p_three_phase, r=R_LINE):
    connections = [[0, 1]]
    admittances = np.array([[1 / complex(r, 0.0)]])
    s = np.array([0.0, p_three_phase], dtype=complex)
    return s, connections, admittances


def _analytic_phase_voltage(p_phase, v0, r):
    # Unity-pf load through a resistor: V1^2 - V0*V1 + P*R = 0, upper root.
    return (v0 + np.sqrt(v0 ** 2 - 4 * p_phase * r)) / 2


def test_two_bus_voltage_and_current_match_the_analytic_solution():
    s, connections, admittances = _two_bus(P_3PH)
    v_phase, converged, _ = solve_balanced_pf(s, connections, admittances, V_LL)
    assert converged

    v0_phase = V_LL / SQRT3
    v1_phase = _analytic_phase_voltage(P_3PH / 3, v0_phase, R_LINE)
    assert abs(v_phase[0]) == pytest.approx(v0_phase, rel=1e-12)
    assert abs(v_phase[1]) == pytest.approx(v1_phase, rel=1e-9)
    # Reported line-to-line magnitude, as stored in MVPFScenarioBus.vm.
    assert abs(v_phase[1]) * SQRT3 == pytest.approx(v1_phase * SQRT3, rel=1e-9)

    i_line, s_flow = branch_flow(v_phase[0], v_phase[1], admittances[0][0])
    assert abs(i_line) == pytest.approx((P_3PH / 3) / v1_phase, rel=1e-9)
    # Three-phase sending-end power = load + three-phase I^2 R losses.
    losses = 3 * abs(i_line) ** 2 * R_LINE
    assert s_flow.real == pytest.approx(P_3PH + losses, rel=1e-9)
    assert s_flow.imag == pytest.approx(0.0, abs=1e-6)


def test_line_current_is_not_sqrt3_too_large():
    """Feeding three-phase totals straight against the line-to-line reference
    (the pre-fix behaviour) gives the same per-unit voltages but a branch
    current sqrt(3) times the real line current."""
    s, connections, admittances = _two_bus(P_3PH)
    v_phase, _, _ = solve_balanced_pf(s, connections, admittances, V_LL)
    v_naive = full_pf(s, connections, admittances, V_LL)

    assert abs(v_phase[1]) * SQRT3 == pytest.approx(abs(v_naive[1]), rel=1e-9)

    i_line, _ = branch_flow(v_phase[0], v_phase[1], admittances[0][0])
    i_naive = (v_naive[0] - v_naive[1]) * admittances[0][0]
    assert abs(i_naive) == pytest.approx(SQRT3 * abs(i_line), rel=1e-9)


def test_watts_not_kilowatts():
    """3 MW on 2 ohm at 15 kV drops ~2.7 %; read as 3000 (i.e. kW stored as W)
    the drop would be ~1000x smaller."""
    s, connections, admittances = _two_bus(P_3PH)
    v_phase, _, _ = solve_balanced_pf(s, connections, admittances, V_LL)
    drop = 1 - abs(v_phase[1]) * SQRT3 / V_LL
    assert 0.02 < drop < 0.03


def test_fetch_series_values_aggregates_phases_in_sql():
    """The three-phase total is computed by the query: one row per
    (series, datetime), phase-less row preferred over the per-phase sum."""
    cursor = MagicMock()
    cursor.fetchall.return_value = [("s1", "t0", 3000.0, None), ("s2", "t0", -500.0, 100.0)]

    values = fetch_series_values("db1", ["s1", "s2"], ["t0"], cursor)

    sql = " ".join(cursor.execute.call_args[0][0].split())
    assert "GROUP BY series_id, datetime" in sql
    assert "COALESCE(SUM(power_active) FILTER (WHERE phase IS NULL), SUM(power_active))" in sql
    # Missing Q stays missing (filled from a drawn power factor later, not read as 0).
    assert values == {("s1", "t0"): (3000.0, None), ("s2", "t0"): (-500.0, 100.0)}


def test_fetch_series_values_empty_inputs_skip_the_query():
    cursor = MagicMock()
    assert fetch_series_values("db1", [], ["t0"], cursor) == {}
    assert fetch_series_values("db1", ["s1"], [], cursor) == {}
    cursor.execute.assert_not_called()


# ── Per-unit output ─────────────────────────────────────────────────────────


def test_per_unit_bases():
    bases = PerUnitBases.for_grid(nominal_voltage_kv=15.0, s_base_mva=1.0)
    assert bases.v_base_v == 15_000.0
    assert bases.s_base_va == 1_000_000.0
    assert bases.z_base_ohm == pytest.approx(225.0)
    assert bases.y_base_s == pytest.approx(1 / 225.0)
    assert bases.as_dict()["z_base_ohm"] == pytest.approx(225.0)


@pytest.mark.parametrize("s_base_mva", [1.0, 10.0, 100.0])
def test_per_unit_columns_match_a_solve_done_in_per_unit(s_base_mva):
    """Converting the SI solve with the bases must equal solving the network
    directly in per-unit (S/S_base, Z/Z_base, V_ref = 1), whatever S_base is."""
    bases = PerUnitBases.for_grid(V_LL / 1e3, s_base_mva)
    s, connections, admittances = _two_bus(P_3PH)
    v_phase, _, _ = solve_balanced_pf(s, connections, admittances, V_LL)
    vm_pu = abs(v_phase) * SQRT3 / bases.v_base_v

    v_pu_solve = full_pf(s / bases.s_base_va, connections, admittances / bases.y_base_s, 1.0 + 0j)
    np.testing.assert_allclose(vm_pu, np.abs(v_pu_solve), rtol=1e-9)

    _, s_flow = branch_flow(v_phase[0], v_phase[1], admittances[0][0])
    i_pu = (v_pu_solve[0] - v_pu_solve[1]) * admittances[0][0] / bases.y_base_s
    assert s_flow.real / bases.s_base_va == pytest.approx((v_pu_solve[0] * np.conj(i_pu)).real, rel=1e-9)


def _placeholder_check(cursor):
    for call in cursor.executemany.call_args_list:
        sql, rows = call[0]
        cols = sql.split("(", 1)[1].split(")", 1)[0]
        n_cols = len([c for c in cols.split(",") if c.strip()])
        assert n_cols == sql.count("%s") == len(rows[0]), sql


def test_insert_columns_placeholders_and_row_lengths_agree():
    """Row tuples built by pf_datagen/records.py and job.py: bus 33, branch 53,
    Y-bus 11, runtime 26, summary 35, DER 24, variant 10, measurement 14,
    measurement layout 8 values (run_id is prepended by the insert)."""
    lengths = {"BUS": 33, "BRANCH": 53, "YBUS": 11, "RUNTIME": 26, "SUMMARY": 35, "DER": 24, "VARIANT": 10,
               "MEASUREMENT": 14, "MEASUREMENT_LAYOUT": 8}
    assert {k: len(getattr(columns, k)) for k in lengths} == lengths
    cursor = MagicMock()
    insert_pf_datagen_batch(
        "run1",
        bus_rows=[tuple(range(33))],
        branch_rows=[tuple(range(53))],
        ybus_rows=[tuple(range(11))],
        runtime_rows=[tuple(range(26))],
        conn=MagicMock(), cursor=cursor,
        summary_rows=[tuple(range(35))],
        der_rows=[tuple(range(24))],
        measurement_rows=[tuple(range(14))],
    )
    insert_pf_datagen_variants("run1", [tuple(range(10))], MagicMock(), cursor)
    insert_pf_datagen_measurement_layout("run1", [tuple(range(8))], MagicMock(), cursor)
    assert cursor.executemany.call_count == 9
    _placeholder_check(cursor)


def test_added_columns_are_written_and_returned():
    cursor = MagicMock()
    insert_pf_datagen_batch(
        "run1", [tuple(range(33))], [tuple(range(53))], [tuple(range(11))], [tuple(range(26))],
        MagicMock(), cursor, summary_rows=[tuple(range(35))], der_rows=[tuple(range(24))],
    )
    insert_pf_datagen_variants("run1", [tuple(range(10))], MagicMock(), cursor)
    sql_by_table = {c[0][0].split('"')[1]: c[0][0] for c in cursor.executemany.call_args_list}
    result_columns = {
        "MVPFScenarioBus": _RESULT_TABLES["bus"][1],
        "MVPFScenarioBranch": _RESULT_TABLES["branch"][1],
        "MVPFScenarioYbus": _RESULT_TABLES["ybus"][1],
        "MVPFScenarioRuntime": _RESULT_TABLES["runtime"][1],
        "MVPFScenarioSummary": _RESULT_TABLES["summary"][1],
        "MVPFScenarioDER": _RESULT_TABLES["der"][1],
        "MVPFTopologyVariant": _RESULT_TABLES["variants"][1],
    }
    assert set(_RESULT_TABLES["summary"][1]) == set(columns.SUMMARY)
    assert set(_RESULT_TABLES["der"][1]) == set(columns.DER)
    for table, cols in ADDED_COLUMNS.items():
        for col, _type in cols:
            assert col in sql_by_table[table], (table, col)
            assert col in result_columns[table], (table, col)


def test_csv_export_joins_list_values():
    cursor = MagicMock()
    cursor.fetchall.return_value = [(0, 0, [], 0, [], "base", 0, [], [], 0),
                                    (3, 1, ["C3"], 2, ["N4", "N5"], "both", 1, ["S1"], ["TIE1", "TIE2"], 0)]
    rows, columns = get_pf_datagen_all_rows("run1", "variants", cursor)
    assert columns == ["topology_variant_idx", "n_disabled", "disabled_connections", "n_islanded", "islanded_nodes",
                       "kind", "config_idx", "open_switches", "closed_ties", "n_disabled_transformers"]
    assert rows == [(0, 0, "", 0, "", "base", 0, "", "", 0), (3, 1, "C3", 2, "N4;N5", "both", 1, "S1", "TIE1;TIE2", 0)]
    assert "ORDER BY topology_variant_idx" in cursor.execute.call_args[0][0]


# ── Fixed-size records (proposal 3.2), end to end through the job ───────────

#          C1      C2      C3      C4
#   SUB ─── N2 ─── N3 ─── N4 ─── N5      15 kV, 1 km of 0.2 + j0.1 ohm/km per branch
FEEDER_NODES = ["SUB", "N2", "N3", "N4", "N5"]
LOAD = complex(200_000.0, 50_000.0)  # three-phase, at every non-substation node

# Row layouts (minus run_id), see pf_datagen/records.py:
#   bus:     ogid, ts, variant, node, pd, qd, vm, va, type, pd_pu, qd_pu, vm_pu, scenario, load_idx, energised,
#            vm_min, vm_max, v_violation, vm_lin, vm_lin_pu, vn_kv
#   branch:  ogid, ts, variant, conn, from, to, pf, qf, r, x, in_service, rating, violation, pf_pu, qf_pu,
#            r_pu, x_pu, scenario, load_idx, pt, qt, p_loss, q_loss, i_mag, loading_pct,
#            pt_pu, qt_pu, p_loss_pu, q_loss_pu, b_s_total, b_pu, pf_lin, qf_lin, pf_lin_pu, qf_lin_pu,
#            emergency_rating_a, loading_emergency_pct, thermal_violation_emergency, kind, tap, tap_pos,
#            yff_r, yff_i, yft_r, yft_i, ytf_r, ytf_i, ytt_r, ytt_i
#   runtime: ogid, ts, variant, solve_ms, converged, scenario, load_idx, n_iter, failure_reason, v_slack_pu
BUS_ENERGISED, BUS_VM, BUS_VA, BUS_VM_PU = 14, 6, 7, 11
BUS_VM_MIN, BUS_VM_MAX, BUS_V_VIOLATION = 15, 16, 17
BR_PF, BR_QF, BR_R, BR_X, BR_IN_SERVICE, BR_THERMAL = 6, 7, 8, 9, 10, 12
BR_PT, BR_QT, BR_P_LOSS, BR_Q_LOSS, BR_I_MAG, BR_LOADING = 19, 20, 21, 22, 23, 24
BR_P_LOSS_PU = 27
BUS_VM_LIN_PU = 19
BR_B_TOTAL, BR_B_PU, BR_PF_LIN, BR_QF_LIN = 29, 30, 31, 32
BR_EMERG, BR_LOADING_EMERG, BR_THERMAL_EMERG = 35, 36, 37
BUS_VN_KV = 20
BR_KIND, BR_TAP, BR_TAP_POS, BR_YFF_R, BR_YFT_R, BR_YTT_R = 38, 39, 40, 41, 43, 47


def _feeder():
    branches = [
        MVBranch(f"C{i}", FEEDER_NODES[i - 1], FEEDER_NODES[i], "cab", 1000.0, 0.2, 0.1, 300.0)
        for i in range(1, 5)
    ]
    return MVGridTopology("mv1", 15.0, ["SUB"], {n: i for i, n in enumerate(FEEDER_NODES)}, branches)


def _scenarios(n):
    s = np.array([0j] + [LOAD] * 4)
    return [{"timestamp": f"2024-01-01 0{t}:00:00", "S": s} for t in range(n)]


def _run_job(tmp_path, topology_perturbation, n_timestamps=2, topology=None, **request_overrides):
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    (run_dir / "run_config.json").write_text(json.dumps({
        "run_id": "r1", "mv_grid_id": "mv1", "historical_database_id": "h1",
        "scenario_count": n_timestamps, "topology_perturbation": topology_perturbation, "seed": 7,
        **request_overrides,
    }))
    out = {"bus": [], "branch": [], "ybus": [], "runtime": [], "summary": [], "der": [], "variants": [],
           "measurements": [], "layout": []}

    def fake_batch(run_id, bus, branch, ybus, runtime, conn, cursor, summary_rows=(), der_rows=(),
                   measurement_rows=()):
        out["bus"] += bus
        out["branch"] += branch
        out["ybus"] += ybus
        out["runtime"] += runtime
        out["summary"] += summary_rows
        out["der"] += der_rows
        out["measurements"] += measurement_rows

    def fake_variants(run_id, rows, conn, cursor):
        out["variants"] += rows

    with patch.object(db, "get_db_connection", return_value=(MagicMock(), MagicMock())), \
            patch.object(db, "create_mv_database"), patch.object(db, "create_pf_datagen_database"), \
            patch.object(db, "insert_pf_datagen_batch", side_effect=fake_batch), \
            patch.object(db, "insert_pf_datagen_variants", side_effect=fake_variants),             patch.object(db, "insert_pf_datagen_measurement_layout",
                         side_effect=lambda r, rows, c, cur: out["layout"].extend(rows)), \
            patch.object(grid_loader, "get_mv_grid_topology", return_value=topology or _feeder()), \
            patch.object(scenario_builder, "build_scenarios", return_value=_scenarios(n_timestamps)):
        result = run_pf_datagen(str(run_dir))
    return result, out


def _sample(rows, variant_idx, load_idx=0):
    return [r for r in rows if r[2] == variant_idx and r[1] == f"2024-01-01 0{load_idx}:00:00"]


def _variant_disabling(out, connection_id):
    return next(v[0] for v in out["variants"] if v[2] == [connection_id])


def test_n_minus_1_writes_every_bus_and_branch_for_every_sample(tmp_path):
    result, out = _run_job(tmp_path, {"type": "n_minus_k", "k": 1})
    assert result["status"] == "completed"

    # 5 variants (base + one per branch) x 2 timestamps, all valid -- including
    # C1 out, where only the substation stays energised.
    assert len(out["runtime"]) == 10
    assert all(r[4] for r in out["runtime"])
    assert result["metrics"]["converged"] == 10 and result["metrics"]["non_converged"] == 0
    assert len(out["bus"]) == 10 * 5
    assert len(out["branch"]) == 10 * 4

    # Integer ids: scenario = load_scenario_idx * n_variants + topology_variant_idx, unique.
    assert sorted(r[5] for r in out["runtime"]) == list(range(10))
    for r in out["runtime"]:
        assert r[5] == r[6] * 5 + r[2]

    # Rows built by the job fit the insert statements.
    cursor = MagicMock()
    insert_pf_datagen_batch("r1", out["bus"][:1], out["branch"][:1], out["ybus"][:1], out["runtime"][:1],
                            MagicMock(), cursor, summary_rows=out["summary"][:1])
    insert_pf_datagen_variants("r1", out["variants"][:1], MagicMock(), cursor)
    _placeholder_check(cursor)


def test_variants_table_describes_each_topology(tmp_path):
    _, out = _run_job(tmp_path, {"type": "n_minus_k", "k": 1})
    assert [v[0] for v in out["variants"]] == [0, 1, 2, 3, 4]
    by_disabled = {tuple(v[2]): v[1:] for v in out["variants"]}
    assert by_disabled[()] == (0, [], 0, [], "base", 0, [], [], 0)     # base topology
    assert by_disabled[("C3",)] == (1, ["C3"], 2, ["N4", "N5"], "outage", 0, [], [], 0)
    assert by_disabled[("C1",)][:4] == (1, ["C1"], 4, ["N2", "N3", "N4", "N5"])


def test_outage_is_flagged_not_dropped(tmp_path):
    _, out = _run_job(tmp_path, {"type": "n_minus_k", "k": 1})
    c3 = _variant_disabling(out, "C3")

    buses = {r[3]: r for r in _sample(out["bus"], c3)}
    for nid in ("N4", "N5"):
        r = buses[nid]
        assert r[BUS_ENERGISED] is False
        assert r[BUS_VM] == r[BUS_VA] == r[BUS_VM_PU] == 0.0
        assert (r[4], r[5]) == (LOAD.real, LOAD.imag)                # demand kept = load lost
    for nid in ("SUB", "N2", "N3"):
        assert buses[nid][BUS_ENERGISED] is True
    assert buses["SUB"][BUS_VM_PU] == pytest.approx(1.0)
    assert 0.9 < buses["N3"][BUS_VM_PU] < 1.0

    branches = {r[3]: r for r in _sample(out["branch"], c3)}
    assert branches["C3"][BR_IN_SERVICE] is False and branches["C3"][BR_PF] == 0.0   # opened
    assert branches["C4"][BR_IN_SERVICE] is True and branches["C4"][BR_PF] == 0.0    # in service, no supply
    assert branches["C1"][BR_PF] > branches["C2"][BR_PF] > 0                          # feeds N2+N3 vs N3
    assert branches["C3"][BR_R] == pytest.approx(0.2) and branches["C3"][BR_X] == pytest.approx(0.1)


def test_substation_only_variant_is_a_valid_sample(tmp_path):
    _, out = _run_job(tmp_path, {"type": "n_minus_k", "k": 1})
    c1 = _variant_disabling(out, "C1")
    buses = {r[3]: r for r in _sample(out["bus"], c1)}
    assert buses["SUB"][BUS_ENERGISED] is True and buses["SUB"][BUS_VM_PU] == pytest.approx(1.0)
    assert all(buses[n][BUS_ENERGISED] is False for n in ("N2", "N3", "N4", "N5"))
    assert all(r[BR_PF] == 0.0 for r in _sample(out["branch"], c1))
    assert _sample(out["ybus"], c1) == []


def test_random_variants_never_include_the_base_topology(tmp_path):
    _, out = _run_job(tmp_path, {"type": "random", "k": 2, "n_variants": 6}, n_timestamps=1)
    assert len(out["variants"]) == 6
    assert all(v[1] >= 1 for v in out["variants"])
    assert len(out["bus"]) == 6 * 5 and len(out["branch"]) == 6 * 4


# ── Limits, both-end flows, losses, loading (proposal 3.3) ──────────────────


def test_branch_flows_balance_to_the_loss():
    s, connections, admittances = _two_bus(P_3PH)
    v_phase, _, _ = solve_balanced_pf(s, connections, admittances, V_LL)
    bases = PerUnitBases.for_grid(V_LL / 1e3, 1.0)
    v_pu = v_phase * SQRT3 / bases.v_base_v
    y_pu = admittances[0][0] * bases.z_base_ohm
    i_from, i_to, s_from, s_to = solver.branch_flows_pu(v_pu[0], v_pu[1], y_pu)
    assert i_to == pytest.approx(-i_from)                         # no shunt or ratio: same current at both ends
    assert s_to.real * bases.s_base_va == pytest.approx(-P_3PH, rel=1e-9)   # delivers the load at the to end
    loss = (s_from + s_to) * bases.s_base_va
    i_line = abs(i_from) * bases.s_base_va / (SQRT3 * bases.v_base_v)
    assert loss.real == pytest.approx(3 * i_line ** 2 * R_LINE, rel=1e-9)
    assert loss.imag == pytest.approx(0.0, abs=1e-6)


def test_voltage_limits_default_and_validation():
    from pydantic import ValidationError

    req = PFDataGenRequest(historical_database_id="h1")
    assert (req.voltage_limits.vm_min, req.voltage_limits.vm_max) == (0.90, 1.10)
    for bad in ({"vm_min": 1.0, "vm_max": 1.0}, {"vm_min": 1.05, "vm_max": 1.10}, {"vm_min": 0.9, "vm_max": 0.95}):
        with pytest.raises(ValidationError):
            PFDataGenRequest(historical_database_id="h1", voltage_limits=bad)


def test_voltage_violations_follow_the_configured_band(tmp_path):
    # Base-case drop on this feeder is ~0.2 % at the far end, so a 0.999 floor
    # flags the far nodes but not the substation or N2.
    _, out = _run_job(tmp_path, {"type": "n_minus_k", "k": 1}, voltage_limits={"vm_min": 0.999, "vm_max": 1.1})
    base = {r[3]: r for r in _sample(out["bus"], 0)}
    for r in base.values():
        assert (r[BUS_VM_MIN], r[BUS_VM_MAX]) == (0.999, 1.1)
        assert r[BUS_V_VIOLATION] is bool(r[BUS_VM_PU] < 0.999)
    flagged = {n for n, r in base.items() if r[BUS_V_VIOLATION]}
    assert "N5" in flagged and "SUB" not in flagged and "N2" not in flagged

    c3 = _variant_disabling(out, "C3")
    islanded = {r[3]: r for r in _sample(out["bus"], c3)}
    assert islanded["N5"][BUS_ENERGISED] is False and islanded["N5"][BUS_V_VIOLATION] is False


def test_default_band_flags_nothing_on_a_healthy_feeder(tmp_path):
    _, out = _run_job(tmp_path, {"type": "none"})
    assert out["bus"] and not any(r[BUS_V_VIOLATION] for r in out["bus"])
    assert all((r[BUS_VM_MIN], r[BUS_VM_MAX]) == (0.9, 1.1) for r in out["bus"])


def test_branch_flows_losses_and_loading(tmp_path):
    _, out = _run_job(tmp_path, {"type": "n_minus_k", "k": 1})
    base = {r[3]: r for r in _sample(out["branch"], 0)}
    for r in base.values():
        assert r[BR_PT] < 0                                           # power delivered at the to end
        assert r[BR_PF] + r[BR_PT] == pytest.approx(r[BR_P_LOSS], rel=1e-9)
        assert r[BR_QF] + r[BR_QT] == pytest.approx(r[BR_Q_LOSS], rel=1e-9)
        assert r[BR_P_LOSS] == pytest.approx(3 * r[BR_I_MAG] ** 2 * r[BR_R], rel=1e-9)
        assert r[BR_Q_LOSS] == pytest.approx(3 * r[BR_I_MAG] ** 2 * r[BR_X], rel=1e-9)
        assert r[BR_LOADING] == pytest.approx(100 * r[BR_I_MAG] / 300.0, rel=1e-12)
        assert r[BR_P_LOSS_PU] == pytest.approx(r[BR_P_LOSS] / 1e6, rel=1e-12)
        assert r[BR_THERMAL] is False
    assert base["C1"][BR_I_MAG] > base["C2"][BR_I_MAG] > base["C3"][BR_I_MAG] > base["C4"][BR_I_MAG] > 0

    # Power balance: what the head branch takes in = all load + all losses.
    buses = _sample(out["bus"], 0)
    total_load = sum(r[4] for r in buses)
    total_loss = sum(r[BR_P_LOSS] for r in base.values())
    assert base["C1"][BR_PF] == pytest.approx(total_load + total_loss, rel=1e-9)

    c3 = _variant_disabling(out, "C3")
    opened = {r[3]: r for r in _sample(out["branch"], c3)}
    for cid in ("C3", "C4"):
        r = opened[cid]
        assert r[BR_I_MAG] == r[BR_LOADING] == r[BR_PT] == r[BR_P_LOSS] == 0.0


# ── Small fixes (proposal 3.4) ──────────────────────────────────────────────

RT_CONVERGED, RT_N_ITER, RT_REASON, RT_V_SLACK = 4, 7, 8, 9


def test_failure_reason_categories():
    from gridarena.pf_datagen.records import failure_reason

    ok = np.array([1.0 + 0j, 0.99 + 0j])
    assert failure_reason(True, ok) is None
    assert failure_reason(False, ok) == "max_iterations"
    assert failure_reason(False, np.array([1.0 + 0j, np.nan])) == "diverged"
    assert failure_reason(True, np.array([1.0 + 0j, np.inf])) == "diverged"   # never trust non-finite output
    assert failure_reason(False, None, ZeroDivisionError("bad")) == "solver_error: ZeroDivisionError: bad"
    assert len(failure_reason(False, None, ValueError("x" * 1000))) == 300


def test_runtime_rows_record_iterations_and_no_reason_when_converged(tmp_path):
    _, out = _run_job(tmp_path, {"type": "n_minus_k", "k": 1})
    c1 = _variant_disabling(out, "C1")
    for r in out["runtime"]:
        assert r[RT_CONVERGED] is True and r[RT_REASON] is None
        assert r[RT_N_ITER] == 0 if r[2] == c1 else r[RT_N_ITER] > 0   # substation-only: nothing solved


def test_solver_error_is_recorded_and_the_sample_skipped(tmp_path):
    real_solve = solver.solve_radial
    calls = {"n": 0}

    def flaky_solve(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise FloatingPointError("singular")
        return real_solve(*args, **kwargs)

    with patch.object(solver, "solve_radial", side_effect=flaky_solve):
        result, out = _run_job(tmp_path, {"type": "none"})

    assert result["metrics"]["non_converged"] == 1 and result["metrics"]["converged"] == 1
    failed = [r for r in out["runtime"] if not r[RT_CONVERGED]]
    assert len(failed) == 1
    assert failed[0][RT_REASON] == "solver_error: FloatingPointError: singular"
    assert failed[0][RT_N_ITER] is None
    # Only the converged sample has bus rows.
    assert {r[0] for r in out["bus"]} == {r[0] for r in out["runtime"] if r[RT_CONVERGED]}


def test_random_variants_are_distinct():
    from gridarena.pf_datagen.perturbation import build_variants
    from gridarena.pf_datagen.schemas import TopologyPerturbationConfig

    topo = _feeder()
    node_ids = list(topo.node_id_to_index)
    # 4 connections, up to 2 outages: 4 + 6 = 10 distinct sets -- ask for all of them.
    cfg = TopologyPerturbationConfig(type="random", k=2, n_variants=10)
    variants = build_variants(cfg, topo.branches, node_ids, "SUB", np.random.default_rng(0))
    sets = [frozenset(v.disabled_connection_ids) for v in variants]
    assert len(set(sets)) == 10
    assert [v.variant_idx for v in variants] == list(range(10))


def test_random_variants_infeasible_request_raises_early():
    from gridarena.pf_datagen.perturbation import build_variants
    from gridarena.pf_datagen.schemas import TopologyPerturbationConfig

    topo = _feeder()
    cfg = TopologyPerturbationConfig(type="random", k=1, n_variants=5)   # only 4 single outages exist
    with pytest.raises(Exception, match="only allow 4"):
        build_variants(cfg, topo.branches, list(topo.node_id_to_index), "SUB", np.random.default_rng(0))


@pytest.mark.parametrize("bad", ["../../etc", "ABCDEF12", "abcdef1", "abcdef123", "abcdef12\n", "abcdef1/"])
def test_registry_rejects_malformed_run_ids(bad, tmp_path, monkeypatch):
    run_manager = importlib.import_module("gridarena.pf_datagen.run_manager")
    monkeypatch.setattr(run_manager, "RUNS_DIR", tmp_path)
    (tmp_path / "abcdef12").mkdir()
    (tmp_path / "abcdef12" / "run_config.json").write_text(json.dumps(
        {"run_id": "abcdef12", "mv_grid_id": "mv1", "status": "completed", "created_at": ""}
    ))
    registry = run_manager.PFDataGenRunRegistry()
    assert registry.get_run("abcdef12").run_id == "abcdef12"
    assert registry.get_run(bad) is None


# ── MV solver, pi-model cables, emergency rating (proposal 1a) ──────────────


def _random_tree(n_nodes, seed):
    """Random radial network: edges in BFS order, parent -> child (what full_pf
    needs), with per-unit impedances and loads on a 15 kV / 1 MVA base."""
    rng = np.random.default_rng(seed)
    parents = [None] + [int(rng.integers(0, i)) for i in range(1, n_nodes)]
    edges = [(f"L{i}", f"n{parents[i]}", f"n{i}") for i in range(1, n_nodes)]
    z_ohm = {cid: complex(rng.uniform(0.05, 0.6), rng.uniform(0.03, 0.4)) for cid, _, _ in edges}
    s_va = np.array([0j] + [complex(rng.uniform(-50e3, 150e3), rng.uniform(-10e3, 50e3)) for _ in range(n_nodes - 1)])
    return [f"n{i}" for i in range(n_nodes)], edges, z_ohm, s_va


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_mv_solver_matches_the_full_pf_reference(seed):
    node_ids, edges, z_ohm, s_va = _random_tree(30, seed)
    bases = PerUnitBases.for_grid(V_LL / 1e3, 1.0)

    idx = {n: i for i, n in enumerate(node_ids)}
    v_ref, converged, _ = solve_balanced_pf(
        s_va, [[idx[a], idx[b]] for _, a, b in edges], np.array([[1 / z_ohm[c]] for c, _, _ in edges]), V_LL,
    )
    assert converged
    vm_ref = np.abs(v_ref) * SQRT3 / bases.v_base_v

    net = solver.build_radial_network(node_ids, "n0", edges, {c: z / bases.z_base_ohm for c, z in z_ohm.items()})
    s_pu = np.array([s_va[idx[n]] for n in net.node_ids]) / bases.s_base_va
    v_pu, converged, n_iter = solver.solve_radial(net, s_pu)
    assert converged and 0 < n_iter < 50
    np.testing.assert_allclose(np.abs(v_pu), [vm_ref[idx[n]] for n in net.node_ids], rtol=1e-9)

    # Same answer whatever the order and direction the connections are given in.
    rng = np.random.default_rng(seed + 100)
    scrambled = [(c, b, a) if rng.random() < 0.5 else (c, a, b) for c, a, b in edges]
    rng.shuffle(scrambled)
    net2 = solver.build_radial_network(list(reversed(node_ids)), "n0", scrambled,
                                       {c: z / bases.z_base_ohm for c, z in z_ohm.items()})
    v2, _, _ = solver.solve_radial(net2, np.array([s_va[idx[n]] for n in net2.node_ids]) / bases.s_base_va)
    by_node = dict(zip(net2.node_ids, v2))
    np.testing.assert_allclose([by_node[n] for n in net.node_ids], v_pu, rtol=1e-12)


def test_meshed_network_is_rejected():
    with pytest.raises(solver.MeshedNetworkError, match="closes a loop"):
        solver.build_radial_network(
            ["a", "b", "c"], "a", [("ab", "a", "b"), ("bc", "b", "c"), ("ca", "c", "a")],
            {"ab": 0.01j, "bc": 0.01j, "ca": 0.01j},
        )


def test_single_node_network_is_solved_without_iterating():
    net = solver.build_radial_network(["sub"], "sub", [], {})
    v, converged, n_iter = solver.solve_radial(net, np.array([0.3 + 0.1j]))
    assert converged and n_iter == 0 and v[0] == 1.0


def test_pi_model_no_load_voltage_rise_matches_the_analytic_solution():
    """Unloaded cable: the receiving end sits above the source because of the
    charging current (Ferranti effect): V1 = V0 / (1 + Z * jB/2)."""
    z, half_b = complex(0.02, 0.05), 0.02j
    net = solver.build_radial_network(["s", "r"], "s", [("c", "s", "r")], {"c": z}, {"c": half_b})
    v, converged, _ = solver.solve_radial(net, np.zeros(2, dtype=complex))
    assert converged
    assert v[1] == pytest.approx(1.0 / (1 + z * half_b), rel=1e-12)
    assert abs(v[1]) > 1.0


def test_lindistflow_is_close_at_light_load_and_exact_without_impedance():
    node_ids, edges, z_ohm, s_va = _random_tree(20, 3)
    bases = PerUnitBases.for_grid(V_LL / 1e3, 1.0)
    idx = {n: i for i, n in enumerate(node_ids)}
    net = solver.build_radial_network(node_ids, "n0", edges, {c: z / bases.z_base_ohm for c, z in z_ohm.items()})
    s_pu = np.array([s_va[idx[n]] for n in net.node_ids]) / bases.s_base_va * 0.1     # light load
    v, _, _ = solver.solve_radial(net, s_pu)
    vm_lin, s_branch = solver.lindistflow(net, s_pu)
    np.testing.assert_allclose(vm_lin, np.abs(v), atol=1e-4)
    # Lossless flow into each branch = the load below it.
    assert s_branch[net.branch_child["L1"]] == pytest.approx(
        sum(s_pu[i] for i in range(len(s_pu)) if _is_below(net, i, net.branch_child["L1"]))
    )


def _is_below(net, i, ancestor):
    while i != -1:
        if i == ancestor:
            return True
        i = net.parent[i]
    return False


def _charged_feeder(cap=0.3, emerg=(360.0, None, 360.0, 360.0), reverse=False):
    topo = _feeder()
    for b, e in zip(topo.branches, emerg):
        b.cap_uf_per_km, b.emerg_curr_a = cap, e
        if reverse:
            b.from_node_id, b.to_node_id = b.to_node_id, b.from_node_id
    if reverse:
        topo.branches.reverse()
    return topo


def test_cable_charging_in_the_output(tmp_path):
    from math import pi

    _, plain = _run_job(tmp_path / "a", {"type": "none"})
    _, charged = _run_job(tmp_path / "b", {"type": "none"}, topology=_charged_feeder())

    expected_b = 2 * pi * 50 * 0.3e-6 * 1.0                         # 1 km of 0.3 uF/km
    for r in _sample(charged["branch"], 0):
        assert r[BR_B_TOTAL] == pytest.approx(expected_b, rel=1e-12)
        assert r[BR_B_PU] == pytest.approx(expected_b * 225.0, rel=1e-12)
        assert r[BR_Q_LOSS] < 0                                       # charging outweighs I^2 X at this load
        assert r[BR_PF] + r[BR_PT] == pytest.approx(r[BR_P_LOSS], rel=1e-9)
    for r in _sample(plain["branch"], 0):
        assert r[BR_B_TOTAL] == 0.0

    vm_plain = {r[3]: r[BUS_VM_PU] for r in _sample(plain["bus"], 0)}
    vm_charged = {r[3]: r[BUS_VM_PU] for r in _sample(charged["bus"], 0)}
    assert all(vm_charged[n] > vm_plain[n] for n in ("N2", "N3", "N4", "N5"))

    # Y-bus diagonal carries the shunt halves.
    diag = {r[3]: complex(r[5], r[6]) for r in _sample(charged["ybus"], 0) if r[3] == r[4]}
    y_series = 1 / complex(0.2, 0.1)
    assert diag["N5"] == pytest.approx(y_series + 0.5j * expected_b, rel=1e-12)
    assert diag["N3"] == pytest.approx(2 * (y_series + 0.5j * expected_b), rel=1e-12)


def test_emergency_rating_columns(tmp_path):
    _, out = _run_job(tmp_path, {"type": "none"}, topology=_charged_feeder())
    branches = {r[3]: r for r in _sample(out["branch"], 0)}
    for cid in ("C1", "C3", "C4"):
        r = branches[cid]
        assert r[BR_EMERG] == 360.0
        assert r[BR_LOADING_EMERG] == pytest.approx(100 * r[BR_I_MAG] / 360.0, rel=1e-12)
        assert r[BR_THERMAL_EMERG] is False
    assert branches["C2"][BR_EMERG] is None                           # no emergency rating -> no check
    assert branches["C2"][BR_LOADING_EMERG] is None and branches["C2"][BR_THERMAL_EMERG] is None


def test_output_does_not_depend_on_how_connections_were_stored(tmp_path):
    _, forward = _run_job(tmp_path / "a", {"type": "n_minus_k", "k": 1}, topology=_charged_feeder())
    _, backward = _run_job(tmp_path / "b", {"type": "n_minus_k", "k": 1}, topology=_charged_feeder(reverse=True))

    # Variant numbering follows the branch order, so key samples by what they disable.
    def keyed(out, table, col):
        disabled = {v[0]: frozenset(v[2]) for v in out["variants"]}
        return {(disabled[r[2]], r[1], r[3]): r[col] for r in out[table]}

    vf, vb = keyed(forward, "bus", BUS_VM_PU), keyed(backward, "bus", BUS_VM_PU)
    assert vf.keys() == vb.keys()
    for k in vf:
        assert vb[k] == pytest.approx(vf[k], rel=1e-12)
    # Flows are reported in each connection's own from/to direction.
    pf_f, pf_b = keyed(forward, "branch", BR_PF), keyed(backward, "branch", BR_PT)
    for k in pf_f:
        assert pf_b[k] == pytest.approx(pf_f[k], rel=1e-9, abs=1e-6)


def test_lindistflow_columns_track_the_exact_solution(tmp_path):
    _, out = _run_job(tmp_path, {"type": "none"}, topology=_charged_feeder())
    for r in _sample(out["bus"], 0):
        assert r[BUS_VM_LIN_PU] == pytest.approx(r[BUS_VM_PU], abs=5e-4)
    for r in _sample(out["branch"], 0):
        assert r[BR_PF_LIN] == pytest.approx(r[BR_PF], rel=0.01)        # lossless vs exact: within 1 %


def test_charging_susceptance_and_cable_schema():
    from pydantic import ValidationError

    from gridarena.pf_datagen.grid_loader import branch_charging_susceptance
    from gridarena.schemas.mv_grid_schema import MVCable

    b = MVBranch("C1", "a", "b", "cab", 2000.0, 0.2, 0.1, 300.0)
    assert branch_charging_susceptance(b) == 0.0
    b.cap_uf_per_km = 0.25
    assert branch_charging_susceptance(b) == pytest.approx(2 * np.pi * 50 * 0.25e-6 * 2.0)

    cable = MVCable(cable_id="X", imp_real=0.1, imp_imag=0.1, nom_curr=300)
    assert cable.cap_uf_per_km is None and cable.emerg_curr is None
    MVCable(cable_id="X", imp_real=0.1, imp_imag=0.1, nom_curr=300, cap_uf_per_km=0.3, emerg_curr=360)
    with pytest.raises(ValidationError):
        MVCable(cable_id="X", imp_real=0.1, imp_imag=0.1, nom_curr=300, cap_uf_per_km=-0.1)


# ── Busbar set-point, transformers, regulators (proposal 1b) ────────────────


def test_transformer_branch_defaults_and_conversion():
    from math import sqrt

    from gridarena.pf_datagen.grid_loader import build_transformer_branch, transformer_impedance_pu

    small, large = transformer_impedance_pu(630, None, None), transformer_impedance_pu(1000, None, None)
    assert abs(small) == pytest.approx(0.04) and abs(large) == pytest.approx(0.06)
    assert small.imag / small.real == pytest.approx(5.0)
    assert transformer_impedance_pu(400, 0.01, 0.045) == complex(0.01, 0.045)

    b, lv_bus, lv_kv = build_transformer_branch("T1", "N5", 630.0, 15.0, 230.0, 0.01, 0.04, "active",
                                                1, 2.5, -2, 2)
    assert (b.kind, b.from_node_id, b.to_node_id, lv_bus) == ("transformer", "N5", "T1_LV", "T1_LV")
    assert lv_kv == pytest.approx(sqrt(3) * 0.230)
    assert b.z_ohm == pytest.approx(complex(0.01, 0.04) * 15e3 ** 2 / 630e3)     # p.u. on own rating -> ohm (MV)
    assert b.nom_curr_a == pytest.approx(630e3 / (sqrt(3) * 15e3))
    assert b.ratio == pytest.approx(1.025) and b.in_service

    off, _, _ = build_transformer_branch("T2", "N5", 400.0, 15.0, None, None, None, "inactive",
                                         None, None, None, None)
    assert not off.in_service and off.tap_pos == 0 and off.ratio == 1.0


def test_load_transfer_splits_by_rating():
    from gridarena.pf_datagen.grid_loader import build_transformer_branch, load_transfer_shares

    ts = [build_transformer_branch(t, "CP", kva, 15.0, 230.0, None, None, "active", 0, 2.5, -2, 2)[0]
          for t, kva in (("TA", 400.0), ("TB", 600.0))]
    ts.append(build_transformer_branch("TC", "CP2", 250.0, 15.0, 230.0, None, None, "active", 0, 2.5, -2, 2)[0])
    shares = load_transfer_shares(ts)
    assert shares["CP"] == [("TA_LV", pytest.approx(0.4)), ("TB_LV", pytest.approx(0.6))]
    assert shares["CP2"] == [("TC_LV", 1.0)]


def test_ratio_branch_in_the_solver():
    # Unloaded: the to side sits at n times the from side.
    net = solver.build_radial_network(["s", "t"], "s", [("T", "s", "t")], {"T": 0.01 + 0.05j},
                                      ratio={"T": 1.05}, directional={"T"})
    v, converged, _ = solver.solve_radial(net, np.zeros(2, dtype=complex), 1.0)
    assert converged and v[1] == pytest.approx(1.05)
    vm_lin, _ = solver.lindistflow(net, np.zeros(2, dtype=complex))
    assert vm_lin[1] == pytest.approx(1.05)

    # Loaded: the flows at the to end deliver exactly the load.
    s = np.array([0, 0.4 + 0.1j])
    v, converged, _ = solver.solve_radial(net, s, 1.0)
    _, _, s_from, s_to = solver.branch_flows_pu(v[0], v[1], 1 / (0.01 + 0.05j), 0j, 1.05)
    assert converged and s_to == pytest.approx(-s[1], rel=1e-9)
    assert (s_from + s_to).real > 0                                    # losses in the series impedance


def test_ratio_branch_must_point_away_from_the_substation():
    with pytest.raises(ValueError, match="points towards the substation"):
        solver.build_radial_network(["s", "t"], "s", [("T", "t", "s")], {"T": 0.05j},
                                    ratio={"T": 1.025}, directional={"T"})


def test_regulator_control_reaches_the_band_or_its_tap_limit():
    net = solver.build_radial_network(
        ["s", "a", "b"], "s", [("L1", "s", "a"), ("L2", "a", "b")], {"L1": 0.02 + 0.02j, "L2": 0.02 + 0.02j},
        directional={"L2"},
    )
    s = np.array([0, 0.5 + 0.2j, 0.5 + 0.2j])
    reg = solver.RegulatorControl("L2", v_set_pu=1.0, deadband_pu=0.01, tap_step_pct=0.625,
                                  tap_min=-16, tap_max=16, tap_pos=0)
    v0, _, _ = solver.solve_radial(net, s)
    assert abs(v0[2]) < 0.97
    v, converged, n_iter, taps, settled = solver.solve_with_regulators(net, s, [reg])
    assert converged and settled and n_iter > 0
    assert abs(abs(v[2]) - 1.0) <= 0.005 and 0 < taps["L2"] <= 16

    capped = solver.RegulatorControl("L2", 1.0, 0.01, 0.625, -16, 2, 0)
    v, converged, _, taps, settled = solver.solve_with_regulators(net, s, [capped])
    assert converged and settled and taps["L2"] == 2 and abs(v[2]) < 0.995     # at its limit, short of the band


def test_failure_reason_for_unsettled_control():
    from gridarena.pf_datagen.records import failure_reason

    assert failure_reason(True, np.array([1.0 + 0j]), settled=False) == "control_not_settled"


def test_slack_voltage_setting_validation():
    from pydantic import ValidationError

    assert PFDataGenRequest(historical_database_id="h1").slack_voltage_pu == 1.0
    PFDataGenRequest(historical_database_id="h1", slack_voltage_pu=1.03)
    PFDataGenRequest(historical_database_id="h1", slack_voltage_pu={"min": 0.97, "max": 1.05})
    for bad in (1.3, 0.5, {"min": 1.05, "max": 1.0}, {"min": 0.7, "max": 1.0}):
        with pytest.raises(ValidationError):
            PFDataGenRequest(historical_database_id="h1", slack_voltage_pu=bad)


def test_fixed_slack_set_point(tmp_path):
    _, out = _run_job(tmp_path, {"type": "n_minus_k", "k": 1}, slack_voltage_pu=1.03)
    assert all(r[RT_V_SLACK] == 1.03 for r in out["runtime"])
    assert all(r[BUS_VM_PU] == pytest.approx(1.03) for r in out["bus"] if r[3] == "SUB")


def test_slack_set_point_range_is_drawn_per_sample_and_reproducibly(tmp_path):
    rng_cfg = {"min": 0.97, "max": 1.05}
    _, a = _run_job(tmp_path / "a", {"type": "n_minus_k", "k": 1}, slack_voltage_pu=rng_cfg)
    _, b = _run_job(tmp_path / "b", {"type": "n_minus_k", "k": 1}, slack_voltage_pu=rng_cfg)
    _, fixed = _run_job(tmp_path / "c", {"type": "n_minus_k", "k": 1})
    va = [r[RT_V_SLACK] for r in a["runtime"]]
    assert all(0.97 <= v <= 1.05 for v in va) and len(set(va)) == len(va)
    assert va == [r[RT_V_SLACK] for r in b["runtime"]]                    # same seed, same draws
    assert a["variants"] == fixed["variants"]                            # own RNG stream: topology unchanged
    slack_of = {r[0]: r[RT_V_SLACK] for r in a["runtime"]}
    for r in a["bus"]:
        if r[3] == "SUB":
            assert r[BUS_VM_PU] == pytest.approx(slack_of[r[0]])


def _feeder_with_transformer(status="active", tap_pos=1):
    from gridarena.pf_datagen.grid_loader import build_transformer_branch, load_transfer_shares

    topo = _feeder()
    t, lv_bus, lv_kv = build_transformer_branch("T1", "N5", 630.0, 15.0, 230.0, 0.01, 0.04, status,
                                                tap_pos, 2.5, -2, 2)
    topo.branches.append(t)
    topo.node_id_to_index[lv_bus] = len(topo.node_id_to_index)
    topo.node_vn_kv = {lv_bus: lv_kv}
    topo.load_transfer = load_transfer_shares([t])
    return topo


def _scenarios_with_lv(n):
    s = np.array([0j] + [LOAD] * 4 + [0j])      # LV bus starts empty; N5's load moves there
    return [{"timestamp": f"2024-01-01 0{t}:00:00", "S": s.copy()} for t in range(n)]


def _run_job_lv(tmp_path, topology, topology_perturbation=None, **overrides):
    """Like _run_job, but with injection arrays sized for the extra LV bus."""
    with patch.object(scenario_builder, "build_scenarios", return_value=_scenarios_with_lv(2)):
        return _run_job_patched(tmp_path, topology_perturbation or {"type": "none"}, topology, overrides)


def _run_job_patched(tmp_path, topology_perturbation, topology, overrides):
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    (run_dir / "run_config.json").write_text(json.dumps({
        "run_id": "r1", "mv_grid_id": "mv1", "historical_database_id": "h1", "scenario_count": 2,
        "topology_perturbation": topology_perturbation, "seed": 7, **overrides,
    }))
    out = {"bus": [], "branch": [], "ybus": [], "runtime": [], "summary": [], "der": [], "variants": [],
           "measurements": [], "layout": []}

    def fake_batch(run_id, bus, branch, ybus, runtime, conn, cursor, summary_rows=(), der_rows=(),
                   measurement_rows=()):
        out["bus"] += bus
        out["branch"] += branch
        out["ybus"] += ybus
        out["runtime"] += runtime
        out["summary"] += summary_rows
        out["der"] += der_rows
        out["measurements"] += measurement_rows

    with patch.object(db, "get_db_connection", return_value=(MagicMock(), MagicMock())), \
            patch.object(db, "create_mv_database"), patch.object(db, "create_pf_datagen_database"), \
            patch.object(db, "insert_pf_datagen_batch", side_effect=fake_batch), \
            patch.object(db, "insert_pf_datagen_variants", side_effect=lambda r, rows, c, cur: out["variants"].extend(rows)),             patch.object(db, "insert_pf_datagen_measurement_layout",
                         side_effect=lambda r, rows, c, cur: out["layout"].extend(rows)), \
            patch.object(grid_loader, "get_mv_grid_topology", return_value=topology):
        result = run_pf_datagen(str(run_dir))
    return result, out


def test_transformer_branch_and_lv_bus_in_the_output(tmp_path):
    from math import sqrt

    result, out = _run_job_lv(tmp_path, _feeder_with_transformer())
    assert result["status"] == "completed", result
    buses = {r[3]: r for r in _sample(out["bus"], 0)}
    assert buses["T1_LV"][BUS_VN_KV] == pytest.approx(sqrt(3) * 0.230)
    assert buses["N2"][BUS_VN_KV] == 15.0
    assert (buses["T1_LV"][4], buses["N5"][4]) == (LOAD.real, 0.0)            # load moved behind the transformer
    lv = buses["T1_LV"]
    assert lv[6] == pytest.approx(lv[BUS_VM_PU] * sqrt(3) * 230.0)             # SI on the LV base (~400 V)
    assert lv[BUS_VM_PU] > buses["N5"][BUS_VM_PU] * 0.99                      # +2.5 % tap offsets the drop

    t1 = next(r for r in _sample(out["branch"], 0) if r[3] == "T1")
    assert (t1[BR_KIND], t1[BR_TAP], t1[BR_TAP_POS]) == ("transformer", pytest.approx(1.025), 1)
    assert t1[BR_PT] == pytest.approx(-LOAD.real, rel=1e-9)                   # delivers the LV load
    s_from_va = abs(complex(t1[BR_PF], t1[BR_QF]))
    assert t1[BR_LOADING] == pytest.approx(100 * s_from_va / 630e3, rel=0.02)  # ~ S / S_rated
    y = 1 / (complex(0.01, 0.04) * 15e3 ** 2 / 630e3) * 225.0                  # system p.u.
    assert complex(t1[BR_YFT_R], t1[BR_YFT_R + 1]) == pytest.approx(-y / 1.025)
    assert complex(t1[BR_YTT_R], t1[BR_YTT_R + 1]) == pytest.approx(y / 1.025 ** 2)
    assert all(r[BR_KIND] == "line" and r[BR_TAP] == 1.0 and r[BR_TAP_POS] is None
               for r in _sample(out["branch"], 0) if r[3] != "T1")


def test_transformers_are_not_outage_candidates_and_inactive_ones_island_their_lv_bus(tmp_path):
    _, out = _run_job_lv(tmp_path / "a", _feeder_with_transformer(), {"type": "n_minus_k", "k": 1})
    assert len(out["variants"]) == 5                                         # base + 4 lines, no T1 outage
    assert all("T1" not in v[2] for v in out["variants"])

    _, off = _run_job_lv(tmp_path / "b", _feeder_with_transformer(status="inactive"))
    assert off["variants"][0][1:5] == (0, [], 1, ["T1_LV"])                  # islanded in the base variant
    lv = next(r for r in _sample(off["bus"], 0) if r[3] == "T1_LV")
    t1 = next(r for r in _sample(off["branch"], 0) if r[3] == "T1")
    assert lv[BUS_ENERGISED] is False and t1[BR_IN_SERVICE] is False and t1[BR_PF] == 0.0


def test_regulated_line_end_to_end(tmp_path):
    topo = _feeder()
    c3 = next(b for b in topo.branches if b.connection_id == "C3")
    c3.kind, c3.regulator_id = "regulated_line", "R1"
    c3.reg_v_set_pu, c3.reg_deadband_pu = 1.03, 0.01
    c3.tap_step_pct, c3.tap_min, c3.tap_max, c3.tap_pos = 0.625, -16, 16, 0
    result, out = _run_job(tmp_path, {"type": "none"}, topology=topo)
    assert result["status"] == "completed" and all(r[RT_REASON] is None for r in out["runtime"])
    buses = {r[3]: r for r in _sample(out["bus"], 0)}
    assert abs(buses["N4"][BUS_VM_PU] - 1.03) <= 0.005
    reg = next(r for r in _sample(out["branch"], 0) if r[3] == "C3")
    assert reg[BR_KIND] == "regulated_line" and reg[BR_TAP_POS] > 0
    assert reg[BR_TAP] == pytest.approx(1 + reg[BR_TAP_POS] * 0.00625)


def test_mv_schema_taps_and_regulators():
    from pydantic import ValidationError

    from gridarena.routers.mv_grid import _validate_mv_grid
    from gridarena.schemas.mv_grid_schema import MVGrid, Regulator, Transformer

    base = dict(transformer_id="T", connection_point_id="CP", rated_power_kva=400, primary_voltage_kv=20)
    Transformer(**base)                                                  # all tap fields default
    Transformer(**base, tap_pos=2)
    for bad in ({"tap_pos": 3}, {"tap_pos": -1, "tap_min": 0}, {"tap_min": 2, "tap_max": -2}):
        with pytest.raises(ValidationError):
            Transformer(**base, **bad)

    r = Regulator(regulator_id="R1", connection_id="C1")
    assert (r.v_set_pu, r.deadband_pu, r.tap_step_pct, r.tap_min, r.tap_max, r.tap_pos) == (1.0, 0.02, 0.625, -16, 16, 0)
    with pytest.raises(ValidationError):
        Regulator(regulator_id="R1", connection_id="C1", tap_pos=17)

    grid = MVGrid(
        mv_grid_id="g", nominal_voltage_kv=20, nodes=[{"node_id": "A"}, {"node_id": "B"}],
        cables=[{"cable_id": "K", "imp_real": 0.1, "imp_imag": 0.1, "nom_curr": 200}],
        connections=[{"connection_id": "C1", "from_node_id": "A", "to_node_id": "B", "cable_id": "K", "length": 100}],
        regulators=[{"regulator_id": "R1", "connection_id": "C1"}, {"regulator_id": "R2", "connection_id": "C1"},
                    {"regulator_id": "R3", "connection_id": "NOPE"}],
    )
    errors = _validate_mv_grid(grid)
    assert any("R2" in e and "already has a regulator" in e for e in errors)
    assert any("R3" in e and "not found" in e for e in errors)


# ── Shunts, ZIP loads, per-sample summary (proposal 1c-i) ───────────────────

BUS = columns.index(columns.BUS)
SUM = columns.index(columns.SUMMARY)


def test_zip_load_model_validation_and_default():
    from pydantic import ValidationError

    req = PFDataGenRequest(historical_database_id="h1")
    assert req.load_model.p.as_tuple() == (0.0, 0.0, 1.0) and req.load_model.q.as_tuple() == (0.0, 0.0, 1.0)
    PFDataGenRequest(historical_database_id="h1", load_model={"p": {"z": 0.4, "i": 0.3, "p": 0.3}})
    with pytest.raises(ValidationError):
        PFDataGenRequest(historical_database_id="h1", load_model={"p": {"z": 0.5, "i": 0.5, "p": 0.5}})
    assert solver.LoadModel().is_constant_power


def test_constant_impedance_load_is_a_voltage_divider():
    """A pure-Z load drawing S0 at 1 p.u. is the admittance conj(S0):
    V1 = V0 * y_line / (y_line + conj(S0))."""
    z_line, s0 = complex(0.02, 0.04), complex(0.6, 0.2)
    net = solver.build_radial_network(["s", "r"], "s", [("c", "s", "r")], {"c": z_line})
    model = solver.LoadModel(p=(1.0, 0.0, 0.0), q=(1.0, 0.0, 0.0))
    v, converged, _ = solver.solve_radial(net, np.array([0, s0]), load_model=model)
    y_line = 1 / z_line
    assert converged and v[1] == pytest.approx(y_line / (y_line + np.conj(s0)), rel=1e-9)
    drawn = model.demand(np.array([0, s0]), np.abs(v))
    assert drawn[1] == pytest.approx(s0 * abs(v[1]) ** 2, rel=1e-12)

    # Constant current: drawn power scales with |V|.
    v_i, _, _ = solver.solve_radial(net, np.array([0, s0]), load_model=solver.LoadModel((0, 1, 0), (0, 1, 0)))
    v_p, _, _ = solver.solve_radial(net, np.array([0, s0]))
    assert abs(v_p[1]) < abs(v_i[1]) < abs(v[1])                       # less voltage-sensitive load, more drop


def test_capacitor_bank_in_the_solver_matches_the_analytic_solution():
    """Unloaded line with a capacitor (y = jB) at the end:
    V1 = V0 * y_line / (y_line + jB) -- a voltage rise."""
    z_line, b = complex(0.02, 0.05), 0.3
    net = solver.build_radial_network(["s", "r"], "s", [("c", "s", "r")], {"c": z_line},
                                      node_y_shunt_pu={"r": 1j * b})
    v, converged, _ = solver.solve_radial(net, np.zeros(2, dtype=complex))
    y_line = 1 / z_line
    assert converged and v[1] == pytest.approx(y_line / (y_line + 1j * b), rel=1e-12)
    assert abs(v[1]) > 1.0
    with pytest.raises(ValueError, match="unknown node"):
        solver.build_radial_network(["s", "r"], "s", [("c", "s", "r")], {"c": z_line}, node_y_shunt_pu={"x": 1j})


def test_shunt_admittance_sign_and_status():
    from gridarena.pf_datagen.grid_loader import shunt_admittance_pu

    shunts = [MVShunt("CB", "N5", 300.0, 2, 1), MVShunt("R", "N3", -200.0, 1, 1),
              MVShunt("OFF", "N4", 300.0, 1, 1, in_service=False), MVShunt("ZERO", "N2", 300.0, 2, 0)]
    y = shunt_admittance_pu(shunts, 1e6)
    assert y == {"N5": pytest.approx(0.3j), "N3": pytest.approx(-0.2j)}   # capacitor +jB, reactor -jB


def _feeder_with_shunt(q_step_kvar=300.0, step=2):
    topo = _feeder()
    topo.shunts = [MVShunt("CB1", "N5", q_step_kvar, 2, step)]
    return topo


def test_capacitor_bank_in_the_output(tmp_path):
    _, plain = _run_job(tmp_path / "a", {"type": "none"})
    _, cap = _run_job(tmp_path / "b", {"type": "none"}, topology=_feeder_with_shunt())

    buses = {r[3]: r for r in _sample(cap["bus"], 0)}
    assert (buses["N5"][BUS["bs"]], buses["N5"][BUS["bs_pu"]], buses["N5"][BUS["gs"]]) == (
        pytest.approx(600e3), pytest.approx(0.6), 0.0)
    assert buses["N4"][BUS["bs"]] == 0.0
    vm_plain = {r[3]: r[BUS_VM_PU] for r in _sample(plain["bus"], 0)}
    assert all(buses[n][BUS_VM_PU] > vm_plain[n] for n in ("N2", "N3", "N4", "N5"))

    # The feeder now exports less reactive power from the substation: C1 carries Q - Q_cap * |V5|^2.
    c1 = next(r for r in _sample(cap["branch"], 0) if r[3] == "C1")
    c1_plain = next(r for r in _sample(plain["branch"], 0) if r[3] == "C1")
    assert c1[BR_QF] < c1_plain[BR_QF] - 500e3

    # Y-bus diagonal carries the bank.
    diag = {r[3]: complex(r[7], r[8]) for r in _sample(cap["ybus"], 0) if r[3] == r[4]}
    assert diag["N5"] == pytest.approx(1 / complex(0.2, 0.1) * 225.0 + 0.6j, rel=1e-12)


def test_reference_demand_and_zip_loads_in_the_output(tmp_path):
    _, pq = _run_job(tmp_path / "a", {"type": "none"})
    for r in _sample(pq["bus"], 0):
        assert (r[BUS["pd"]], r[BUS["qd"]]) == (r[BUS["pd_ref"]], r[BUS["qd_ref"]])

    _, z = _run_job(tmp_path / "b", {"type": "none"},
                    load_model={"p": {"z": 1, "i": 0, "p": 0}, "q": {"z": 1, "i": 0, "p": 0}})
    for r in _sample(z["bus"], 0):
        if r[3] == "SUB":
            continue
        assert r[BUS["pd_ref"]] == LOAD.real
        assert r[BUS["pd"]] == pytest.approx(LOAD.real * r[BUS_VM_PU] ** 2, rel=1e-12)
        assert r[BUS["pd"]] < LOAD.real                                 # below 1 p.u. a Z load draws less
        assert r[BUS["pd_pu"]] == pytest.approx(r[BUS["pd"]] / 1e6)


def test_summary_row_balances_the_feeder(tmp_path):
    _, out = _run_job(tmp_path, {"type": "n_minus_k", "k": 1}, topology=_feeder_with_shunt())
    assert len(out["summary"]) == len(out["runtime"]) == 10
    by_group = {r[0]: r for r in out["summary"]}
    for ogid, srow in by_group.items():
        buses = [r for r in out["bus"] if r[0] == ogid]
        branches = [r for r in out["branch"] if r[0] == ogid]
        on = [r for r in buses if r[BUS_ENERGISED]]
        assert srow[SUM["n_bus_energised"]] + srow[SUM["n_bus_islanded"]] == 5
        assert srow[SUM["vm_pu_min"]] == min(r[BUS_VM_PU] for r in on)
        assert srow[SUM["n_branch_in_service"]] == sum(r[BR_IN_SERVICE] for r in branches)
        assert srow[SUM["max_loading_pct"]] == max(r[BR_LOADING] for r in branches)
        # What the substation supplies leaves through C1 (the substation has no load).
        c1 = next(r for r in branches if r[3] == "C1")
        assert srow[SUM["p_slack"]] == pytest.approx(c1[BR_PF], rel=1e-9, abs=1e-6)
        assert srow[SUM["q_slack"]] == pytest.approx(c1[BR_QF], rel=1e-9, abs=1e-6)
        assert srow[SUM["p_loss"]] == pytest.approx(sum(r[BR_P_LOSS] for r in branches), rel=1e-12)

    base = by_group[next(v for v in out["runtime"] if v[2] == 0)[0]]
    assert base[SUM["n_bus_islanded"]] == 0 and base[SUM["p_load"]] == pytest.approx(4 * LOAD.real)
    assert base[SUM["q_shunt"]] > 500e3                                 # 600 kvar at a voltage near 1 p.u.
    assert base[SUM["p_loss_pu"]] == pytest.approx(base[SUM["p_loss"]] / 1e6)


def test_mv_schema_shunts():
    from pydantic import ValidationError

    from gridarena.routers.mv_grid import _validate_mv_grid
    from gridarena.schemas.mv_grid_schema import MVGrid, Shunt

    sh = Shunt(shunt_id="CB", node_id="A", q_step_kvar=300)
    assert (sh.n_steps, sh.step, sh.status) == (1, 1, "active")
    Shunt(shunt_id="R", node_id="A", q_step_kvar=-150, n_steps=3, step=0)
    for bad in ({"q_step_kvar": 0}, {"q_step_kvar": 300, "n_steps": 2, "step": 3}, {"q_step_kvar": 300, "step": -1}):
        with pytest.raises(ValidationError):
            Shunt(shunt_id="X", node_id="A", **bad)

    grid = MVGrid(
        mv_grid_id="g", nominal_voltage_kv=20, nodes=[{"node_id": "A"}],
        shunts=[{"shunt_id": "CB", "node_id": "A", "q_step_kvar": 300},
                {"shunt_id": "CB", "node_id": "A", "q_step_kvar": 300},
                {"shunt_id": "CB2", "node_id": "NOPE", "q_step_kvar": 300}],
    )
    errors = _validate_mv_grid(grid)
    assert any("Duplicate shunt_id 'CB'" in e for e in errors)
    assert any("CB2" in e and "not found" in e for e in errors)


# ── Generators (DER, proposal 1c-ii) ─────────────────────────────────────────

DER_COL = columns.index(columns.DER)
BRANCH_COL = columns.index(columns.BRANCH)
RT_ROUNDS = columns.RUNTIME.index("control_rounds")


def test_clear_sky_share_follows_the_sun():
    from datetime import datetime

    from gridarena.pf_datagen.der import clear_sky_share

    porto = (41.15, -8.61)
    june = [clear_sky_share(datetime(2024, 6, 21, h, m), "Europe/Lisbon", *porto)
            for h, m in ((3, 0), (9, 0), (12, 0), (13, 36), (15, 0), (18, 0))]
    assert june[0] == 0.0                                                   # night
    assert june[3] == max(june) and june[3] > 0.85                          # solar noon ~13:36 local (UTC+1)
    assert june[2] > june[1] > 0 and june[4] > june[5] > 0
    december = clear_sky_share(datetime(2024, 12, 21, 12, 36), "Europe/Lisbon", *porto)
    assert 0.3 < december < june[3]
    # The same clock time read as UTC instead of Lisbon summer time is an hour later in the sun's day.
    assert clear_sky_share(datetime(2024, 6, 21, 12, 36), "UTC", *porto) == pytest.approx(june[3], abs=1e-3)


def test_daily_clearness_is_per_day_and_reproducible():
    from datetime import datetime

    from gridarena.pf_datagen.der import daily_clearness

    stamps = [datetime(2024, 1, d, h) for d in (2, 1, 3) for h in (6, 12)]
    a = daily_clearness(stamps, 0.3, 1.0, np.random.default_rng(1))
    assert sorted(a) == [datetime(2024, 1, d).date() for d in (1, 2, 3)]
    assert all(0.3 <= v <= 1.0 for v in a.values())
    assert a == daily_clearness(list(reversed(stamps)), 0.3, 1.0, np.random.default_rng(1))
    assert set(daily_clearness(stamps, 0.8, 0.8, np.random.default_rng(1)).values()) == {0.8}


def _der(der_id="G1", node="N5", mode="fixed_pf", s_kva=600.0, value=1.0, **kw):
    return MVDER(der_id, node, kw.pop("der_type", "chp"), s_kva, kw.pop("p_max_kw", s_kva),
                 kw.pop("profile", "constant"), value, mode, **kw)


def test_der_control_limits_and_sign():
    from gridarena.pf_datagen.der import der_control

    under = der_control(_der(cos_phi=0.9, s_kva=1000.0), 600e3, 1e6)
    assert under.p_available == pytest.approx(0.6) and under.q_max == pytest.approx(0.8)     # rating circle
    assert under.tan_phi == pytest.approx(-np.tan(np.arccos(0.9)))                           # absorbs Q
    assert der_control(_der(cos_phi=0.9, excitation="over"), 0, 1e6).tan_phi > 0
    qv = der_control(_der(mode="q_v", s_kva=1000.0), 600e3, 1e6)
    assert qv.q_max == pytest.approx(0.33)                                                   # q_limit_pu x S
    full = der_control(_der(mode="voltage", s_kva=1000.0), 1000e3, 1e6)
    assert full.q_max == 0.0                                                                 # no headroom at full P


def _two_bus_net(z=complex(0.05, 0.08)):
    return solver.build_radial_network(["s", "r"], "s", [("c", "s", "r")], {"c": z})


def test_fixed_pf_der_cancels_a_matching_load():
    net = _two_bus_net()
    s = np.array([0, 0.5 + 0.2j])
    d = solver.DERControl("G", "r", "fixed_pf", p_available=0.5, q_max=1.0, tan_phi=0.4)
    res = solver.solve_with_controls(net, s, ders=[d])
    assert res.converged and res.settled and res.rounds == 1
    assert res.v[1] == pytest.approx(1.0, abs=1e-12)
    assert (res.der_p["G"], res.der_q["G"]) == (pytest.approx(0.5), pytest.approx(0.2))


def test_voltage_control_holds_the_set_point_or_its_limit():
    net = _two_bus_net()
    s = np.array([0, 0.8 + 0.3j])
    d = solver.DERControl("G", "r", "voltage", p_available=0.2, q_max=1.0, v_set_pu=1.0)
    res = solver.solve_with_controls(net, s, ders=[d])
    assert res.settled and abs(res.v[1]) == pytest.approx(1.0, abs=1e-6)
    assert 0 < res.der_q["G"] < 1.0 and not res.der_at_q_limit["G"]

    weak = solver.DERControl("G", "r", "voltage", p_available=0.2, q_max=0.1, v_set_pu=1.0)
    res = solver.solve_with_controls(net, s, ders=[weak])
    assert res.settled and res.der_q["G"] == pytest.approx(0.1) and res.der_at_q_limit["G"]
    assert abs(res.v[1]) < 0.999                                            # PQ at its limit, short of the set-point


def test_two_voltage_controllers_on_one_feeder():
    net = solver.build_radial_network(["s", "a", "b"], "s", [("L1", "s", "a"), ("L2", "a", "b")],
                                      {"L1": 0.03 + 0.06j, "L2": 0.03 + 0.06j})
    s = np.array([0, 0.4 + 0.2j, 0.4 + 0.2j])
    ders = [solver.DERControl("A", "a", "voltage", 0.1, 1.0, v_set_pu=1.0),
            solver.DERControl("B", "b", "voltage", 0.1, 1.0, v_set_pu=1.01)]
    res = solver.solve_with_controls(net, s, ders=ders)
    assert res.settled
    assert abs(res.v[1]) == pytest.approx(1.0, abs=1e-6) and abs(res.v[2]) == pytest.approx(1.01, abs=1e-6)


def test_q_v_and_p_v_settle_on_their_curves():
    net = _two_bus_net(complex(0.08, 0.1))
    s = np.zeros(2, dtype=complex)
    qv = solver.DERControl("G", "r", "q_v", p_available=0.6, q_max=0.3)
    res = solver.solve_with_controls(net, s, ders=[qv])
    vm = abs(res.v[1])
    assert res.settled and vm > 1.01                                        # export raises the voltage ...
    assert res.der_q["G"] == pytest.approx(qv.q_of_v(vm)[0], abs=1e-5) and res.der_q["G"] < 0   # ... so it absorbs

    pv = solver.DERControl("G", "r", "p_v", p_available=1.2, q_max=0.0, pv_curve=(1.06, 1.09))
    res = solver.solve_with_controls(net, s, ders=[pv])
    vm = abs(res.v[1])
    assert res.settled and 1.06 < vm < 1.09
    assert 0 < res.der_p["G"] < 1.2 and res.der_p["G"] == pytest.approx(pv.p_of_v(vm)[0], abs=1e-5)


def _feeder_with_ders(*ders):
    topo = _feeder()
    topo.ders = list(ders)
    return topo


def test_der_in_the_output(tmp_path):
    topo = _feeder_with_ders(
        _der("CHP", "N3", s_kva=500.0, value=0.8, cos_phi=0.95, excitation="over"),
        _der("WT", "N5", mode="voltage", s_kva=1000.0, value=0.3, v_set_pu=1.0, der_type="wind"),
        _der("OFF", "N4", in_service=False),
    )
    result, out = _run_job(tmp_path, {"type": "n_minus_k", "k": 1}, topology=topo)
    assert result["status"] == "completed", result
    assert all(r[RT_REASON] is None and r[RT_ROUNDS] >= 1 for r in out["runtime"])
    assert len(out["der"]) == 3 * len(out["runtime"])

    ogid = next(r[0] for r in out["runtime"] if r[2] == 0)               # base topology, first timestamp
    ders = {r[3]: r for r in out["der"] if r[0] == ogid}
    buses = {r[3]: r for r in out["bus"] if r[0] == ogid}
    chp = ders["CHP"]
    assert chp[DER_COL["p"]] == pytest.approx(400e3) and chp[DER_COL["p_curtailed"]] == 0.0
    assert chp[DER_COL["q"]] == pytest.approx(400e3 * np.tan(np.arccos(0.95)))
    assert buses["N3"][BUS["pg"]] == pytest.approx(400e3) and buses["N3"][BUS["qg_pu"]] > 0
    wt = ders["WT"]
    assert buses["N5"][BUS_VM_PU] == pytest.approx(1.0, abs=1e-6) and buses["N5"][BUS["bus_type"]] == "PV"
    assert wt[DER_COL["in_service"]] and not wt[DER_COL["at_q_limit"]] and wt[DER_COL["q"]] > 0
    off = ders["OFF"]
    assert not off[DER_COL["in_service"]] and off[DER_COL["p"]] == 0.0 and off[DER_COL["p_available"]] == 600e3
    assert buses["N4"][BUS["pg"]] == 0.0

    # The substation supplies load + losses - shunt and DER generation.
    for srow in out["summary"]:
        c1 = next(r for r in out["branch"] if r[0] == srow[0] and r[3] == "C1")
        assert srow[SUM["p_slack"]] == pytest.approx(c1[BR_PF], rel=1e-9, abs=1e-3)   # W, sums of ~MW terms
        assert srow[SUM["q_slack"]] == pytest.approx(c1[BR_QF], rel=1e-9, abs=1e-3)
    base = next(r for r in out["summary"] if r[0] == ogid)
    assert base[SUM["p_gen"]] == pytest.approx(400e3 + 300e3)

    # Islanded DER (C4 out cuts N5 off): out of service, no injection.
    c4_out = _variant_disabling(out, "C4")
    wt_islanded = next(r for r in out["der"] if r[2] == c4_out and r[3] == "WT")
    assert not wt_islanded[DER_COL["in_service"]] and wt_islanded[DER_COL["q"]] == 0.0


def test_p_v_curtailment_in_the_output(tmp_path):
    topo = _feeder_with_ders(_der("PV", "N5", mode="p_v", s_kva=6000.0, value=1.0, der_type="pv",
                                  pv_curve=(1.005, 1.04)))
    _, out = _run_job(tmp_path, {"type": "none"}, topology=topo)
    for r in out["der"]:
        assert 0 < r[DER_COL["p_curtailed"]] < 6000e3
        assert r[DER_COL["p"]] + r[DER_COL["p_curtailed"]] == pytest.approx(6000e3)
    for s in out["summary"]:
        assert s[SUM["p_curtailed"]] > 0 and 1.005 < s[SUM["vm_pu_max"]] < 1.04


def test_solar_profile_at_night_and_reproducible_clearness(tmp_path):
    pv = _der("PV", "N5", mode="q_v", s_kva=1000.0, der_type="pv", profile="solar", lat=41.15, lon=-8.61)
    _, night = _run_job(tmp_path / "a", {"type": "none"}, topology=_feeder_with_ders(pv))
    assert all(r[DER_COL["p_available"]] == 0.0 for r in night["der"])      # the job's timestamps are 00:00-01:00

    from gridarena.pf_datagen.der import p_available_by_scenario

    noon = [{"timestamp": f"2024-06-{d:02d} 13:30:00"} for d in (1, 2, 3)]
    a = p_available_by_scenario([pv], noon, "Europe/Lisbon", 0.3, 1.0, np.random.default_rng([7, 2]))
    b = p_available_by_scenario([pv], noon, "Europe/Lisbon", 0.3, 1.0, np.random.default_rng([7, 2]))
    assert a == b and all(0.25e6 < x["PV"] <= 1e6 for x in a) and len({x["PV"] for x in a}) == 3


def test_request_timezone_and_clearness_validation():
    from pydantic import ValidationError

    req = PFDataGenRequest(historical_database_id="h1")
    assert req.timezone == "Europe/Lisbon" and (req.pv_clearness.min, req.pv_clearness.max) == (0.3, 1.0)
    PFDataGenRequest(historical_database_id="h1", timezone="UTC", pv_clearness={"min": 1, "max": 1})
    for bad in ({"timezone": "Mars/Olympus"}, {"timezone": ""}, {"pv_clearness": {"min": 0.9, "max": 0.5}}):
        with pytest.raises(ValidationError):
            PFDataGenRequest(historical_database_id="h1", **bad)


def test_mv_schema_ders():
    from pydantic import ValidationError

    from gridarena.routers.mv_grid import _validate_mv_grid
    from gridarena.schemas.mv_grid_schema import DER, MVGrid

    pv = DER(der_id="PV", node_id="A", type="pv", s_rated_kva=500)
    assert (pv.profile, pv.p_max_kw, pv.control_mode, pv.qv_curve) == ("solar", 500, "fixed_pf", [0.95, 0.99, 1.01, 1.05])
    assert DER(der_id="B", node_id="A", type="battery", s_rated_kva=100, profile_value=-0.5).profile == "constant"
    for bad in ({"qv_curve": [0.99, 0.95, 1.01, 1.05]}, {"pv_v_start": 1.09, "pv_v_end": 1.06},
                {"profile_value": 1.5}, {"type": "nuclear"}, {"cos_phi": 0}):
        with pytest.raises(ValidationError):
            DER(**{"der_id": "X", "node_id": "A", "type": "chp", "s_rated_kva": 100, **bad})

    grid = MVGrid(
        mv_grid_id="g", nominal_voltage_kv=20,
        nodes=[{"node_id": "S", "node_type": "substation"}, {"node_id": "A"},
               {"node_id": "B", "coord_lat": 41.1, "coord_lon": -8.6}],
        ders=[{"der_id": "D1", "node_id": "S", "type": "chp", "s_rated_kva": 100},
              {"der_id": "D2", "node_id": "A", "type": "pv", "s_rated_kva": 100},
              {"der_id": "D3", "node_id": "B", "type": "pv", "s_rated_kva": 100},
              {"der_id": "D3", "node_id": "NOPE", "type": "chp", "s_rated_kva": 100}],
    )
    errors = _validate_mv_grid(grid)
    assert any("D1" in e and "substation" in e for e in errors)
    assert any("D2" in e and "coordinates" in e for e in errors)
    assert any("Duplicate der_id 'D3'" in e for e in errors) and any("NOPE" in e for e in errors)
    assert not any("D3" in e and "coordinates" in e for e in errors)


def test_mixed_der_controls_settle_on_random_feeders():
    """Regression guard for the joint Newton step and its kink damping:
    random feeders with several DER of every control mode (and sometimes a
    regulator) settle whenever the load alone is solvable, and each settled
    unit satisfies its control law."""
    rng = np.random.default_rng(0)
    for trial in range(60):
        n = int(rng.integers(8, 30))
        nodes = [f"n{i}" for i in range(n)]
        edges = [(f"L{i}", nodes[int(rng.integers(max(0, i - 4), i))], nodes[i]) for i in range(1, n)]
        z = {cid: complex(rng.uniform(0.002, 0.03), rng.uniform(0.002, 0.03)) for cid, _, _ in edges}
        regs = []
        if trial % 3 == 0:
            regs = [solver.RegulatorControl(edges[0][0], 1.0, 0.02, 0.625, -16, 16, 0)]
        net = solver.build_radial_network(nodes, "n0", edges, z, directional={r.connection_id for r in regs})
        s = np.zeros(n, dtype=complex)
        s[1:] = rng.uniform(0, 0.1, n - 1) + 1j * rng.uniform(0, 0.03, n - 1)
        ders = []
        for k, idx in enumerate(rng.choice(np.arange(1, n), size=int(rng.integers(1, 6)), replace=False)):
            mode = ("fixed_pf", "q_v", "p_v", "voltage")[k % 4]
            p_av = float(rng.uniform(0, 1.0))
            s_rated = max(1.1 * p_av, 0.2)
            q_max = min(0.33 * s_rated, np.sqrt(s_rated ** 2 - p_av ** 2))
            ders.append(solver.DERControl(f"D{k}", nodes[idx], mode, p_av, q_max,
                                          tan_phi=-0.33 if mode in ("fixed_pf", "p_v") else 0.0,
                                          v_set_pu=float(rng.uniform(0.99, 1.03))))
        if not solver.solve_radial(net, s)[1]:
            continue
        res = solver.solve_with_controls(net, s, regulators=regs, ders=ders)
        assert res.converged and res.settled, trial
        idx = {nid: i for i, nid in enumerate(net.node_ids)}
        for d in ders:
            vm = abs(res.v[idx[d.node_id]])
            if d.mode == "q_v":
                assert res.der_q[d.der_id] == pytest.approx(d.q_of_v(vm)[0], abs=1e-4)
            elif d.mode == "p_v":
                assert res.der_p[d.der_id] == pytest.approx(d.p_of_v(vm)[0], abs=1e-4)
            elif d.mode == "voltage" and not res.der_at_q_limit[d.der_id]:
                assert vm == pytest.approx(d.v_set_pu, abs=1e-5)


# ── Switches, tie lines, several substations (proposal 1d-i) ────────────────


def test_sweep_solves_a_radial_forest_fed_by_two_substations():
    z = {"A1": 0.02 + 0.03j, "A2": 0.01 + 0.02j, "B1": 0.03 + 0.01j}
    edges = [("A1", "sa", "a1"), ("A2", "a1", "a2"), ("B1", "sb", "b1")]
    s = {"sa": 0, "a1": 0.3 + 0.1j, "a2": 0.2 + 0.05j, "sb": 0, "b1": 0.4 + 0.2j}
    forest = solver.build_radial_network(list(s), ["sa", "sb"], edges, z)
    assert forest.node_ids[:2] == ["sa", "sb"] and list(forest.levels[0]) == [0, 1]
    v, converged, _ = solver.solve_radial(forest, np.array([s[n] for n in forest.node_ids]), 1.02)
    assert converged
    got = dict(zip(forest.node_ids, v))
    for root, part, part_edges in (("sa", ["sa", "a1", "a2"], edges[:2]), ("sb", ["sb", "b1"], edges[2:])):
        single = solver.build_radial_network(part, root, part_edges, z)
        v1, _, _ = solver.solve_radial(single, np.array([s[n] for n in single.node_ids]), 1.02)
        for nid, val in zip(single.node_ids, v1):
            assert got[nid] == pytest.approx(val, rel=1e-12)
    vm_lin, s_branch = solver.lindistflow(forest, np.array([s[n] for n in forest.node_ids]), 1.02)
    assert s_branch[0] == 0 and s_branch[1] == 0 and vm_lin[0] == vm_lin[1] == pytest.approx(1.02)
    r, x = solver._path_matrix(forest, [forest.node_ids.index("a2"), forest.node_ids.index("b1")])
    assert x[0, 1] == 0 and x[0, 0] == pytest.approx(0.05) and r[1, 1] == pytest.approx(0.03)   # separate feeders


def test_sweep_rejects_two_substations_in_one_component_and_loops():
    z = {"A": 0.02j, "B": 0.02j, "C": 0.02j}
    with pytest.raises(solver.MeshedNetworkError, match="fed by substations 'sa' and 'sb'"):
        solver.build_radial_network(["sa", "m", "sb"], ["sa", "sb"], [("A", "sa", "m"), ("B", "m", "sb")], z)
    with pytest.raises(solver.MeshedNetworkError, match="closes a loop"):
        solver.build_radial_network(["s", "a", "b"], ["s"], [("A", "s", "a"), ("B", "a", "b"), ("C", "b", "s")], z)


#   SUB ── C1 ── N2 ── C2 ── N3         SUB2 ── D1 ── M2
#                              └─ T (normally-open tie, N3 - M2) ─┘
TWO_FEEDER_NODES = ["SUB", "SUB2", "N2", "N3", "M2"]


def _two_feeders(tie_open=True):
    def line(cid, a, b, **kw):
        return MVBranch(cid, a, b, "cab", 1000.0, 0.2, 0.1, 300.0, **kw)
    branches = [line("C1", "SUB", "N2"), line("C2", "N2", "N3"), line("D1", "SUB2", "M2"),
                line("T", "N3", "M2", is_switch=True, normally_open=tie_open, in_service=not tie_open)]
    return MVGridTopology("mv1", 15.0, ["SUB", "SUB2"], {n: i for i, n in enumerate(TWO_FEEDER_NODES)}, branches)


def _two_feeder_scenarios(n):
    s = np.array([0j, 0j, LOAD, LOAD, LOAD])
    return [{"timestamp": f"2024-01-01 0{t}:00:00", "S": s} for t in range(n)]


def _run_two_feeders(tmp_path, topology, perturbation=None, **overrides):
    with patch.object(scenario_builder, "build_scenarios", return_value=_two_feeder_scenarios(2)):
        return _run_job_patched(tmp_path, perturbation or {"type": "none"}, topology, overrides)


def test_two_substations_and_an_open_tie_in_the_output(tmp_path):
    result, out = _run_two_feeders(tmp_path, _two_feeders(), slack_voltage_pu=1.02)
    assert result["status"] == "completed", result
    assert all(r[RT_REASON] is None for r in out["runtime"])
    buses = {r[3]: r for r in _sample(out["bus"], 0)}
    assert buses["SUB"][8] == buses["SUB2"][8] == "REF"
    assert buses["SUB"][BUS_VM_PU] == buses["SUB2"][BUS_VM_PU] == pytest.approx(1.02)
    assert all(buses[n][BUS_ENERGISED] for n in TWO_FEEDER_NODES)
    br = {r[3]: r for r in _sample(out["branch"], 0)}
    assert br["T"][BR_IN_SERVICE] is False and br["T"][BRANCH_COL["is_switch"]] is True and br["T"][BR_PF] == 0.0
    assert br["C1"][BRANCH_COL["is_switch"]] is False
    # Each feeder carries its own load: SUB feeds N2 + N3, SUB2 feeds M2.
    assert br["C1"][BR_PF] > 2 * LOAD.real and LOAD.real < br["D1"][BR_PF] < 1.01 * LOAD.real
    summary = next(s for s in out["summary"] if s[2] == 0)
    assert summary[SUM["p_slack"]] == pytest.approx(br["C1"][BR_PF] + br["D1"][BR_PF], rel=1e-9)


def test_open_switches_are_not_outage_candidates(tmp_path):
    topo = _two_feeders()
    assert [b.connection_id for b in topo.outage_candidates] == ["C1", "C2", "D1"]
    _, out = _run_two_feeders(tmp_path, topo, {"type": "n_minus_k", "k": 1})
    assert len(out["variants"]) == 4 and all("T" not in v[2] for v in out["variants"])
    d1_out = _variant_disabling(out, "D1")
    variant = next(v for v in out["variants"] if v[0] == d1_out)
    assert variant[4] == ["M2"]                                             # only the second feeder's node islands


def test_closed_tie_between_two_substations_needs_the_newton_solver(tmp_path):
    _, out = _run_two_feeders(tmp_path, _two_feeders(tie_open=False))
    reasons = {r[RT_REASON] for r in out["runtime"]}
    assert len(reasons) == 1 and reasons.pop().startswith("solver_error: MeshedNetworkError")
    assert out["bus"] == []


def test_substation_connected_component_with_several_roots():
    from gridarena.pf_datagen.grid_loader import substation_connected_component

    topo = _two_feeders()
    active = {"C1", "C2"}                                                  # D1 out, tie open
    assert substation_connected_component(TWO_FEEDER_NODES, topo.branches, active, ["SUB", "SUB2"]) ==         {"SUB", "N2", "N3", "SUB2"}


def test_mv_schema_switches_and_graph_edges():
    from pydantic import ValidationError

    from gridarena.routers.mv_grid import _compute_graph_layout
    from gridarena.schemas.mv_grid_schema import MVConnection

    base = dict(connection_id="T", from_node_id="A", to_node_id="B", cable_id="K", length=100)
    c = MVConnection(**base)
    assert (c.status, c.is_switch, c.normally_open) == ("active", False, False)
    MVConnection(**base, is_switch=True, normally_open=True)
    with pytest.raises(ValidationError):
        MVConnection(**base, normally_open=True)

    graph = _compute_graph_layout({
        "nodes": [{"node_id": n} for n in "ABC"],
        "connections": [{"from_node_id": "A", "to_node_id": "B"},
                        {"from_node_id": "B", "to_node_id": "C", "is_switch": True, "normally_open": True}],
    })
    assert {(e["from"], e["to"]): e["open"] for e in graph["edges"]} == {("A", "B"): False, ("B", "C"): True}


def test_mv_graph_carries_attached_elements():
    from gridarena.routers.mv_grid import _compute_graph_layout

    graph = _compute_graph_layout({
        "nodes": [{"node_id": n} for n in "ABCD"],
        "connections": [
            {"connection_id": "AB", "from_node_id": "A", "to_node_id": "B", "cable_id": "K", "length": 100},
            {"connection_id": "BC", "from_node_id": "B", "to_node_id": "C", "is_switch": True},
            {"connection_id": "CD", "from_node_id": "C", "to_node_id": "D", "status": "inactive"},
            {"connection_id": "AD", "from_node_id": "A", "to_node_id": "D", "is_switch": True,
             "normally_open": True},
        ],
        "connection_points": [{"connection_point_id": "CP", "node_id": "D"}],
        "transformers": [{"transformer_id": "T", "connection_point_id": "CP", "rated_power_kva": 400,
                          "tap_pos": 1, "status": "inactive"}],
        "regulators": [{"regulator_id": "R", "connection_id": "AB", "v_set_pu": 1.0, "deadband_pu": 0.02,
                        "tap_pos": 0, "tap_min": -16, "tap_max": 16, "tap_step_pct": 0.625}],
        "shunts": [{"shunt_id": "S", "node_id": "B", "q_step_kvar": -100, "n_steps": 2, "step": 1}],
        "ders": [{"der_id": "G", "node_id": "C", "type": "pv", "s_rated_kva": 500, "control_mode": "q_v"}],
    })
    edges = {e["id"]: e for e in graph["edges"]}
    assert {k: (e["state"], e["is_switch"], e["open"]) for k, e in edges.items()} == {
        "AB": ("closed", False, False), "BC": ("closed", True, False),
        "CD": ("out_of_service", False, True), "AD": ("normally_open", True, True)}
    assert edges["AB"]["regulator"]["regulator_id"] == "R" and "regulator" not in edges["BC"]
    nodes = {n["id"]: n for n in graph["nodes"]}
    assert nodes["D"]["transformers"] == [{"transformer_id": "T", "rated_power_kva": 400, "tap_pos": 1,
                                          "active": False}]
    assert nodes["B"]["shunts"][0]["q_step_kvar"] == -100 and nodes["B"]["shunts"][0]["active"]
    assert nodes["C"]["ders"] == [{"der_id": "G", "type": "pv", "s_rated_kva": 500, "control_mode": "q_v",
                                   "active": True}]


# ── Newton-Raphson backend (proposal 1d-ii) ─────────────────────────────────

newton = importlib.import_module("gridarena.pf_datagen.newton")
RT_SOLVER = columns.RUNTIME.index("solver")


def _random_radial_case(rng, n):
    nodes = [f"n{i}" for i in range(n)]
    edges = [(f"L{i}", nodes[int(rng.integers(max(0, i - 4), i))], nodes[i]) for i in range(1, n)]
    z = {cid: complex(rng.uniform(0.002, 0.03), rng.uniform(0.002, 0.03)) for cid, _, _ in edges}
    y_sh = {cid: 1j * rng.uniform(0, 0.002) for cid, _, _ in edges}
    ratio_cid = edges[int(rng.integers(0, len(edges)))][0]
    ratio = {ratio_cid: 1.025}
    shunt = {nodes[int(rng.integers(1, n))]: 0.2j}
    s = np.zeros(n, dtype=complex)
    s[1:] = rng.uniform(0, 0.08, n - 1) + 1j * rng.uniform(0, 0.03, n - 1)
    gen = np.zeros(n, dtype=complex)
    gen[int(rng.integers(1, n))] = 0.3 - 0.05j
    return nodes, edges, z, y_sh, ratio, shunt, s, gen


def test_newton_matches_the_sweep_on_radial_feeders():
    rng = np.random.default_rng(5)
    model = solver.LoadModel(p=(0.3, 0.3, 0.4), q=(0.5, 0.0, 0.5))
    for _ in range(25):
        nodes, edges, z, y_sh, ratio, shunt, s, gen = _random_radial_case(rng, int(rng.integers(5, 30)))
        radial = solver.build_radial_network(nodes, "n0", edges, z, y_sh, ratio=ratio, directional=set(ratio),
                                             node_y_shunt_pu=shunt)
        meshed = newton.MeshedNetwork(nodes, "n0", edges, {c: 1 / v for c, v in z.items()}, y_sh, ratio=ratio,
                                      node_y_shunt_pu=shunt)
        assert radial.node_ids == meshed.node_ids
        v_r, ok_r, _ = radial.solve(s, 1.01, load_model=model, s_gen_pu=gen)
        v_n, ok_n, it = meshed.solve(s, 1.01, load_model=model, s_gen_pu=gen)
        assert ok_r and ok_n and it <= 6
        np.testing.assert_allclose(v_n, v_r, rtol=0, atol=1e-9)


def test_newton_two_source_ring_matches_the_equivalent_single_feeder():
    """Two substations at the same voltage, each with line z to node m: the
    same as one substation with z/2 to m."""
    z, s_m = complex(0.04, 0.06), 0.5 + 0.2j
    ring = newton.MeshedNetwork(["a", "b", "m"], ["a", "b"], [("A", "a", "m"), ("B", "b", "m")],
                                {"A": 1 / z, "B": 1 / z})
    v, converged, _ = ring.solve(np.array([0, 0, s_m]), 1.0)
    single = solver.build_radial_network(["a", "m"], "a", [("A", "a", "m")], {"A": z / 2})
    v1, _, _ = single.solve(np.array([0, s_m]), 1.0)
    assert converged and v[2] == pytest.approx(v1[1], abs=1e-10)


def test_newton_sensitivity_is_the_common_path_impedance_on_a_radial_feeder():
    rng = np.random.default_rng(2)
    nodes, edges, z, _, _, _, _, _ = _random_radial_case(rng, 15)
    radial = solver.build_radial_network(nodes, "n0", edges, z)
    meshed = newton.MeshedNetwork(nodes, "n0", edges, {c: 1 / v for c, v in z.items()})
    idx = [3, 7, 11]
    r1, x1 = radial.sensitivity(idx)
    r2, x2 = meshed.sensitivity(idx)
    np.testing.assert_allclose(r2, r1, atol=1e-12)
    np.testing.assert_allclose(x2, x1, atol=1e-12)


def test_newton_rejects_an_unfed_node_and_reports_divergence():
    with pytest.raises(ValueError, match="not connected"):
        newton.MeshedNetwork(["s", "a", "b"], "s", [("A", "s", "a")], {"A": 10.0})
    weak = newton.MeshedNetwork(["s", "a"], "s", [("A", "s", "a")], {"A": 1 / complex(0.5, 0.5)})
    v, converged, _ = weak.solve(np.array([0, 5 + 2j]))                     # far beyond the nose point
    assert not converged


def test_controls_settle_with_the_newton_solver_on_a_ring():
    net = newton.MeshedNetwork(["a", "b", "m", "k"], ["a", "b"],
                               [("A", "a", "m"), ("B", "b", "k"), ("C", "m", "k")],
                               {"A": 1 / complex(0.03, 0.05), "B": 1 / complex(0.04, 0.06), "C": 1 / complex(0.02, 0.03)})
    s = np.array([0, 0, 0.6 + 0.3j, 0.4 + 0.2j])
    d = solver.DERControl("G", "k", "voltage", p_available=0.1, q_max=1.0, v_set_pu=1.0)
    res = solver.solve_with_controls(net, s, ders=[d])
    assert res.converged and res.settled and abs(res.v[3]) == pytest.approx(1.0, abs=1e-6)


def test_solver_request_field():
    from pydantic import ValidationError

    assert PFDataGenRequest(historical_database_id="h1").solver == "sweep"
    assert PFDataGenRequest(historical_database_id="h1", solver="newton").solver == "newton"
    with pytest.raises(ValidationError):
        PFDataGenRequest(historical_database_id="h1", solver="gauss")


def test_closed_tie_between_two_substations_with_newton(tmp_path):
    result, out = _run_two_feeders(tmp_path, _two_feeders(tie_open=False), solver="newton")
    assert result["status"] == "completed", result
    assert all(r[RT_REASON] is None and r[RT_SOLVER] == "newton" for r in out["runtime"])
    br = {r[3]: r for r in _sample(out["branch"], 0)}
    assert br["T"][BR_IN_SERVICE] is True and abs(br["T"][BR_PF]) > 1e3      # the tie carries power
    assert br["T"][BR_PF_LIN] is None                                       # meshed: no LinDistFlow baseline
    buses = {r[3]: r for r in _sample(out["bus"], 0)}
    assert buses["N3"][BUS_VM_LIN_PU] is None and buses["N3"][BUS_VM_PU] > 0.98
    summary = next(s for s in out["summary"] if s[2] == 0)
    assert summary[SUM["p_slack"]] == pytest.approx(br["C1"][BR_PF] + br["D1"][BR_PF], rel=1e-9)
    # Every node's injection balances: the tie's flow equals what M2 gets beyond D1's share.
    assert br["D1"][BR_PT] + br["T"][BR_PT] == pytest.approx(-LOAD.real, rel=1e-6)


def test_newton_and_sweep_agree_end_to_end_on_a_radial_grid(tmp_path):
    _, sweep = _run_two_feeders(tmp_path / "a", _two_feeders(), {"type": "n_minus_k", "k": 1})
    _, nr = _run_two_feeders(tmp_path / "b", _two_feeders(), {"type": "n_minus_k", "k": 1}, solver="newton")
    assert [r[RT_SOLVER] for r in sweep["runtime"]] == ["sweep"] * len(sweep["runtime"])
    vs = {(r[0], r[3]): r[BUS_VM_PU] for r in sweep["bus"]}
    vn = {(r[0], r[3]): r[BUS_VM_PU] for r in nr["bus"]}
    assert vs.keys() == vn.keys()
    for k in vs:
        assert vn[k] == pytest.approx(vs[k], abs=1e-9)
    lin = {(r[0], r[3]): r[BUS_VM_LIN_PU] for r in nr["bus"]}
    assert all(v is not None for v in lin.values())                         # radial: baseline still computed


def test_sparse_and_dense_newton_agree(monkeypatch):
    rng = np.random.default_rng(9)
    nodes, edges, z, y_sh, ratio, shunt, s, gen = _random_radial_case(rng, 25)
    edges.append(("LOOP", nodes[3], nodes[20]))                            # make it meshed
    z["LOOP"] = 0.01 + 0.02j
    net = newton.MeshedNetwork(nodes, "n0", edges, {c: 1 / v for c, v in z.items()}, y_sh, ratio=ratio,
                               node_y_shunt_pu=shunt)
    model = solver.LoadModel(p=(0.3, 0.3, 0.4), q=(0.5, 0.0, 0.5))
    v_dense, ok_d, _ = net.solve(s, 1.0, load_model=model, s_gen_pu=gen)
    r_dense, x_dense = net.sensitivity([2, 9])
    monkeypatch.setattr(newton, "DENSE_MAX_NODES", 0)
    v_sparse, ok_s, _ = net.solve(s, 1.0, load_model=model, s_gen_pu=gen)
    r_sparse, x_sparse = net.sensitivity([2, 9])
    assert ok_d and ok_s
    np.testing.assert_allclose(v_sparse, v_dense, atol=1e-12)
    np.testing.assert_allclose(x_sparse, x_dense, atol=1e-12)
    np.testing.assert_allclose(r_sparse, r_dense, atol=1e-12)


def test_connections_on_loops_treat_substations_as_one_node():
    edges = [("C1", "SUB", "N2"), ("C2", "N2", "N3"), ("D1", "SUB2", "M2"), ("T", "N3", "M2"), ("X", "N2", "X1")]
    assert newton.connections_on_loops(edges, ["C2", "X"]) == set()                 # no cycle in the graph itself
    assert newton.connections_on_loops(edges, ["C2", "X"], ["SUB", "SUB2"]) == {"C2"}  # SUB .. SUB2 is a loop
    par = [("A", "s", "a"), ("B", "s", "a")]
    assert newton.connections_on_loops(par, ["A"]) == {"A"}


def test_regulator_on_a_closed_loop_holds_its_tap(tmp_path):
    topo = _two_feeders(tie_open=False)
    c2 = next(b for b in topo.branches if b.connection_id == "C2")
    c2.kind, c2.regulator_id = "regulated_line", "R1"
    c2.reg_v_set_pu, c2.reg_deadband_pu = 1.05, 0.01                       # would tap up if it regulated
    c2.tap_step_pct, c2.tap_min, c2.tap_max, c2.tap_pos = 0.625, -16, 16, 0
    _, out = _run_two_feeders(tmp_path / "loop", topo, solver="newton")
    reg = next(r for r in _sample(out["branch"], 0) if r[3] == "C2")
    assert reg[BR_TAP_POS] == 0 and reg[BR_TAP] == 1.0

    c2_radial = _two_feeders()                                              # tie open: it regulates again
    r = next(b for b in c2_radial.branches if b.connection_id == "C2")
    r.kind, r.regulator_id, r.reg_v_set_pu, r.reg_deadband_pu = "regulated_line", "R1", 1.05, 0.01
    r.tap_step_pct, r.tap_min, r.tap_max, r.tap_pos = 0.625, -16, 16, 0
    _, out = _run_two_feeders(tmp_path / "radial", c2_radial, solver="newton")
    assert next(x for x in _sample(out["branch"], 0) if x[3] == "C2")[BR_TAP_POS] > 0


# â”€â”€ Phase 2a: load level (axis 1), load shape (axis 2), report â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

from gridarena.pf_datagen import load_level as load_level_mod  # noqa: E402
from gridarena.pf_datagen import load_shape as load_shape_mod  # noqa: E402
from gridarena.pf_datagen import sampling  # noqa: E402
from gridarena.pf_datagen.report import DiversityReport  # noqa: E402
from gridarena.pf_datagen.schemas import LoadLevelConfig, LoadShapeConfig  # noqa: E402


def test_stratified_draws_fill_every_bin_evenly_and_reproducibly():
    v = sampling.draw_values(1003, 0.1, 1.2, "stratified", 10, sampling.stream(5, sampling.STREAM_LOAD_LEVEL))
    counts = np.bincount(sampling.bin_of(v, 0.1, 1.2, 10), minlength=10)
    assert counts.max() - counts.min() <= 1 and counts.sum() == 1003
    assert v.min() >= 0.1 and v.max() <= 1.2
    again = sampling.draw_values(1003, 0.1, 1.2, "stratified", 10, sampling.stream(5, sampling.STREAM_LOAD_LEVEL))
    assert np.array_equal(v, again)
    assert np.all(sampling.draw_values(4, 0.7, 0.7, "uniform", 10, np.random.default_rng(0)) == 0.7)


def test_loadability_search_brackets_both_ways_and_reports_the_limit():
    pattern = np.array([0j, 100 + 30j, 50 + 10j])          # gross demand 150 W

    def evaluate(s):                                      # voltage limit at 1000 W, thermal at 800 W
        g = load_level_mod.gross_demand(s)
        return "thermal" if g > 800 else None

    up = load_level_mod.find_loadability(pattern, evaluate)
    assert up.p_max_w == pytest.approx(800, rel=0.011) and up.p_max_w <= 800 and up.limit == "thermal"
    down = load_level_mod.find_loadability(pattern * 100, evaluate)
    assert down.p_max_w == pytest.approx(800, rel=0.011) and down.limit == "thermal"
    with pytest.raises(load_level_mod.LoadabilityError):
        load_level_mod.find_loadability(pattern, lambda s: "voltage")


def test_apply_levels_hits_targets_and_flags_near_empty_samples():
    samples = [np.array([0j, 300 + 90j, -50 + 0j]), np.array([0j, 30 + 9j, 10 + 0j]), np.array([0j, 0.01 + 0j, 0j])]
    out, levels, scales = load_level_mod.apply_levels(samples, "loadability", np.array([0.5, 1.1, 0.8]), 1000.0)
    assert load_level_mod.gross_demand(out[0]) == pytest.approx(500.0)
    assert load_level_mod.gross_demand(out[1]) == pytest.approx(1100.0)
    assert out[0][1].imag / out[0][1].real == pytest.approx(0.3)          # power factor kept
    assert out[0][2].real < 0                                             # exporting node scaled, not flipped
    assert levels[2] is None and scales[2] == 1.0 and out[2] is not samples[2]
    out_m, levels_m, _ = load_level_mod.apply_levels(samples, "multiplier", np.array([2.0, 0.5, 1.0]))
    assert out_m[0][1] == 2 * samples[0][1] and levels_m == [2.0, 0.5, 1.0]


def _loadability_job(tmp_path, lo, hi, **extra):
    return _run_job(tmp_path, {"type": "none"}, n_timestamps=6,
                    load_level={"mode": "loadability", "min": lo, "max": hi, "sampling": "uniform"}, **extra)


def test_loadability_level_just_below_p_max_is_within_limits_and_just_above_is_not(tmp_path):
    result, out = _loadability_job(tmp_path / "below", 0.99, 0.99)
    m = result["metrics"]
    assert result["status"] == "completed" and m["p_max_w"] > 0 and m["p_max_limit"] in ("voltage", "thermal")
    S = columns.index(columns.SUMMARY)
    assert all(r[S["n_undervoltage"]] == 0 and r[S["max_loading_pct"]] <= 100.0 for r in out["summary"])
    RT = columns.index(columns.RUNTIME)
    assert all(r[RT["load_level"]] == pytest.approx(0.99) for r in out["runtime"])
    p_load = [r[S["p_load"]] for r in out["summary"]]
    assert p_load == pytest.approx([0.99 * m["p_max_w"]] * len(p_load), rel=1e-6)

    # Just above P_max every sample breaks a limit (or fails to solve).
    _, out_above = _loadability_job(tmp_path / "above", 1.05, 1.05)
    summary_by_id = {r[0]: r for r in out_above["summary"]}
    assert out_above["runtime"]
    for rt in out_above["runtime"]:
        if rt[RT["converged"]]:
            s = summary_by_id[rt[0]]
            assert s[S["n_undervoltage"]] > 0 or s[S["max_loading_pct"]] > 100.0


def test_load_level_off_leaves_runtime_columns_empty_and_writes_a_report(tmp_path):
    result, out = _run_job(tmp_path, {"type": "none"})
    RT = columns.index(columns.RUNTIME)
    assert all(r[RT["load_level"]] is None and r[RT["load_scale"]] is None for r in out["runtime"])
    report = json.loads((tmp_path / "run" / "report.json").read_text())
    assert report["outcomes"]["converged"] == result["metrics"]["converged"]
    assert "load_level" not in report and report["voltage_pu"]["counts"]


def test_report_counts_samples_and_violations_per_level_band(tmp_path):
    _, out = _loadability_job(tmp_path, 0.1, 1.2)
    report = json.loads((tmp_path / "run" / "report.json").read_text())
    lvl = report["load_level"]
    assert sum(lvl["samples"]) == len(out["runtime"]) and len(lvl["edges"]) == 11
    assert report["loadability"]["p_max_w"] > 0


def test_load_level_config_defaults_and_validation():
    assert (LoadLevelConfig(mode="loadability").min, LoadLevelConfig(mode="loadability").max) == (0.1, 1.2)
    assert (LoadLevelConfig(mode="multiplier").min, LoadLevelConfig(mode="multiplier").max) == (0.5, 1.5)
    with pytest.raises(ValueError):
        LoadLevelConfig(mode="multiplier", min=2.0, max=1.0)
    with pytest.raises(ValueError):
        LoadShapeConfig(pf_min=0.99, pf_max=0.9)


def test_point_sizes_follow_transformer_rating():
    sizes = load_shape_mod.point_mean_load_w(["A", "B", "C"], {"A": 100.0, "B": 400.0}, 0.4)
    assert sizes == {"A": 40_000.0, "B": 160_000.0, "C": 100_000.0}      # C: average of the others
    assert load_shape_mod.resolve_point_sizing("auto", "LV") == "series"
    assert load_shape_mod.resolve_point_sizing("auto", "Feeder") == "rating"


def _shape_scenarios(reactive="series", seed=1, n_ts=4, period=2, **shape):
    """build_scenarios with a mocked Feeder database: 3 connection points of
    100 / 200 / 400 kVA, series of very different magnitudes, no Q."""
    ts = [f"2024-01-01 0{i}:00:00" for i in range(n_ts)]
    series = {"big": 5e6, "small": 1e3}
    values = {(s, t): (mag * (1 + 0.1 * i), None) for s, mag in series.items() for i, t in enumerate(ts)}
    with patch.object(scenario_builder, "get_historical_source_type", return_value="Feeder"), \
            patch.object(scenario_builder, "get_historical_timeline", return_value=ts), \
            patch.object(scenario_builder, "get_all_mv_connection_points", return_value=["P1", "P2", "P3"]), \
            patch.object(scenario_builder, "get_historical_series_ids", return_value=list(series)), \
            patch.object(scenario_builder, "fetch_series_values", return_value=values):
        return scenario_builder.build_scenarios(
            "mv1", {"SUB": 0, "P1": 1, "P2": 2, "P3": 3}, n_ts, 0.0, "db", period, np.random.default_rng(seed),
            MagicMock(), load_shape=LoadShapeConfig(reactive=reactive, **shape),
            point_kva={"P1": 100.0, "P2": 200.0, "P3": 400.0}, shape_rng=np.random.default_rng(seed + 1),
        )


def test_rating_sizing_makes_load_proportional_to_kva_whatever_the_series():
    sc = _shape_scenarios()
    p = np.array([[s["S"][i].real for i in (1, 2, 3)] for s in sc])
    mean = p.mean(axis=0)
    assert mean == pytest.approx([40e3, 80e3, 160e3], rel=1e-9)          # 0.4 x kVA, series magnitude gone


def test_missing_q_comes_from_a_power_factor_drawn_per_point_per_block():
    sc = _shape_scenarios(n_ts=4, period=2)
    pf = np.array([[abs(s["S"][i].real) / abs(s["S"][i]) for i in (1, 2, 3)] for s in sc])
    assert np.all((pf >= 0.90 - 1e-12) & (pf <= 0.99 + 1e-12))
    assert pf[0] == pytest.approx(pf[1]) and pf[2] == pytest.approx(pf[3])   # constant within a block
    assert not np.allclose(pf[0], pf[2])                                        # redrawn in the next block
    assert all(s["S"][i].imag > 0 for s in sc for i in (1, 2, 3))              # inductive


def test_load_shape_is_reproducible_and_noise_keeps_loads_positive():
    a = _shape_scenarios(seed=3, spatial_sigma=0.0)
    b = _shape_scenarios(seed=3, spatial_sigma=0.0)
    assert all(np.array_equal(x["S"], y["S"]) for x, y in zip(a, b))
    noise = load_shape_mod.LoadNoise(50, local_sigma=0.5, q_sigma=0.5)
    f, fq = zip(*(noise.draw(np.random.default_rng(i)) for i in range(400)))
    f, fq = np.array(f), np.array(fq)
    assert (f > 0).all() and (fq > 0).all()
    assert f.mean() == pytest.approx(1.0, abs=0.03) and fq.mean() == pytest.approx(1.0, abs=0.03)


def test_spatial_noise_correlation_follows_cable_distance():
    branches = [MVBranch("A", "S", "X", "c", 1000.0, 0.2, 0.1, 300.0),
                MVBranch("B", "X", "Y", "c", 2000.0, 0.2, 0.1, 300.0),
                MVBranch("T", "Y", "Y_LV", None, 0.0, 0.0, 0.0, 10.0, kind="transformer"),
                MVBranch("D", "Z", "W", "c", 500.0, 0.2, 0.1, 300.0)]           # separate part
    d = load_shape_mod.cable_distances_km(branches, ["S", "X", "Y", "Y_LV", "Z", "W"], ["X", "Y", "Y_LV", "W"])
    assert d[0, 1] == pytest.approx(2.0) and d[1, 2] == pytest.approx(0.0, abs=1e-6) and np.isinf(d[0, 3])
    noise = load_shape_mod.LoadNoise(4, spatial_sigma=0.3, correlation_km=2.0, distance_km=d)
    log_f = np.log(np.array([noise.draw(np.random.default_rng(i))[0] for i in range(6000)]))
    corr = np.corrcoef(log_f.T)
    assert corr[0, 1] == pytest.approx(np.exp(-1.0), abs=0.04)              # 2 km apart, l = 2 km
    assert corr[1, 2] == pytest.approx(1.0, abs=0.01)                       # behind the same transformer
    assert abs(corr[0, 3]) < 0.04                                           # not connected: independent


def test_report_endpoint_serves_a_finished_runs_report():
    from fastapi.testclient import TestClient

    from gridarena.app import app
    from gridarena.pf_datagen.run_manager import pf_datagen_registry

    client = TestClient(app)
    with patch.object(pf_datagen_registry, "get_run", return_value=MagicMock()), \
            patch.object(pf_datagen_registry, "get_report", return_value={"version": 1}):
        assert client.get("/pf-datagen/runs/abcd1234/report").json() == {"version": 1}
    with patch.object(pf_datagen_registry, "get_run", return_value=MagicMock()), \
            patch.object(pf_datagen_registry, "get_report", return_value=None):
        assert client.get("/pf-datagen/runs/abcd1234/report").status_code == 404


def test_report_load_level_bands_count_violations():
    rep = DiversityReport("loadability", 0.0, 1.0, 2)
    S = columns.index(columns.SUMMARY)
    row = [0] * len(columns.SUMMARY)
    row[S["n_undervoltage"]], row[S["p_slack"]], row[S["p_loss"]] = 1, 100.0, 2.0
    rep.add_result(0.8, None, tuple(row))
    rep.add_result(0.2, "max_iterations")
    rep.add_result(None, None, tuple([0] * len(columns.SUMMARY)))
    d = rep.as_dict()
    assert d["load_level"]["samples"] == [1, 1] and d["load_level"]["undervoltage_rate"] == [None, 1.0]
    assert d["load_level"]["unscaled_samples"] == 1 and d["failure_reasons"] == {"max_iterations": 1}


# ── Phase 2b: generated DER (axis 3), set-points (axis 6) ───────────────────

from collections import Counter  # noqa: E402

from gridarena.pf_datagen.der_sampling import DERSampler  # noqa: E402
from gridarena.pf_datagen.newton import MeshedNetwork  # noqa: E402
from gridarena.pf_datagen.schemas import DERConfig, SetpointsConfig, SlackVoltageRange  # noqa: E402
from gridarena.pf_datagen.setpoints import SetpointSampler  # noqa: E402
from gridarena.pf_datagen.solver import RegulatorControl, ShuntControl  # noqa: E402

#   SUB ── A ── B ── C ── D      (1 km each);  A and C have 400 kVA transformers
_DER_NODES = ["SUB", "A", "B", "C", "D"]


def _der_topology(grid_ders=()):
    branches = [MVBranch(f"L{i}", _DER_NODES[i - 1], _DER_NODES[i], "c", 1000.0, 0.2, 0.1, 300.0)
                for i in range(1, 5)]
    topo = MVGridTopology("g", 20.0, ["SUB"], {n: i for i, n in enumerate(_DER_NODES)}, branches,
                          ders=list(grid_ders))
    topo.node_coords = {n: (41.15, -8.61) for n in _DER_NODES}
    return topo


_GEN = {"database_id": "gen"}


def _groups(*groups, **cfg):
    """A DERConfig generating `groups` (dicts over _GEN: count, capacity_mw, ...)."""
    return DERConfig(groups=[{**_GEN, "capacity_mw": 0.25, **g} for g in (groups or ({},))],
                     **{"source": "replace", **cfg})


def _gen(n, data_type="wind", zero_every=0):
    """A generation profile (pf_datagen/der_profiles.py) over n samples: an
    aggregate at 60 % of its reference maximum, zero every `zero_every` samples."""
    from gridarena.pf_datagen.der_profiles import GenerationProfile
    agg = np.full(n, 0.6e6)
    if zero_every:
        agg[::zero_every] = 0.0
    return GenerationProfile(database_id="gen", data_type=data_type, series_ids=["agg"], values_w=agg[None, :],
                             ratings_w=np.array([agg.max()]), reference_year=2024, reference_max_w=1e6,
                             gen_times=1.7e9 + 900.0 * np.arange(n), step_s=900.0, shift_s=0.0, n_fallback=0)


def _sampler(cfg, topo=None, n=200, p_max=1e6, seed=3, gen=None):
    topo = topo or _der_topology()
    if cfg.layout is None:
        cfg = cfg.model_copy(update={"layout": "per_sample"})
    return DERSampler(cfg, topo, _DER_NODES, set(_DER_NODES), p_max, n, seed, {"A": 400.0, "C": 400.0},
                      generation={"gen": gen if gen is not None else _gen(n)})


def test_generated_der_capacity_is_the_groups():
    s = _sampler(_groups({"count": 3, "capacity_mw": 0.2}, {"count": 1, "capacity_mw": 0.5}))
    for i in range(50):
        smp = s.sample(i, "2024-06-01 12:00:00", {})
        assert [d.s_rated_kva for d in smp.ders] == [200.0, 200.0, 200.0, 500.0]
        assert smp.capacity_w == pytest.approx(1.1e6) and smp.penetration == pytest.approx(1.1)
        assert smp.n_generated == 4 and all(d.der_id in smp.synthetic for d in smp.ders)
        assert all(d.node_id != "SUB" for d in smp.ders)
        assert all(d.der_type == "wind" for d in smp.ders)                  # the database's type


def test_add_keeps_the_grids_own_der_and_grid_mode_adds_nothing():
    own = MVDER("PV0", "B", "pv", 600.0, 600.0, "constant", 0.5, "fixed_pf")
    ev = MVDER("EV0", "C", "ev_hub", 300.0, 300.0, "constant", -0.5, "fixed_pf")
    topo = _der_topology([own, ev])
    grid = _sampler(DERConfig(), topo).sample(0, "2024-06-01 12:00:00", {"PV0": 3e5, "EV0": -1.5e5})
    assert [d.der_id for d in grid.ders] == ["PV0", "EV0"] and grid.n_generated == 0 and grid.penetration is None
    s = _sampler(_groups({"count": 2}, source="add"), topo)
    for i in range(10):
        smp = s.sample(i, "2024-06-01 12:00:00", {"PV0": 3e5, "EV0": -1.5e5})
        # The EV hub is a load: only the 600 kVA PV counts with the plants.
        assert smp.capacity_w == pytest.approx(6e5 + 5e5) and smp.penetration == pytest.approx(1.1)
        assert [d.der_id for d in smp.ders] == ["PV0", "EV0", "GEN-WIND-01", "GEN-WIND-02"]


def test_per_run_layout_keeps_one_der_set_up_for_every_sample():
    cfg = _groups({"count": 4}, layout="per_run")
    s = _sampler(cfg, gen=_gen(200, zero_every=5))
    first = s.sample(0, "2024-06-01 12:00:00", {})
    for i in (1, 7, 150):
        smp = s.sample(i, "2024-06-01 12:00:00", {})
        assert smp.penetration == first.penetration and smp.location == first.location
        assert [(d.der_id, d.node_id, d.s_rated_kva, d.control_mode) for d in smp.ders] == \
            [(d.der_id, d.node_id, d.s_rated_kva, d.control_mode) for d in first.ders]
    # The output still follows the generation database: nothing where it is zero.
    assert all(v == 0 for v in s.sample(5, "2024-06-01 12:00:00", {}).p_available.values())
    assert all(v > 0 for v in s.sample(6, "2024-06-01 12:00:00", {}).p_available.values())
    per_sample = _sampler(cfg.model_copy(update={"layout": "per_sample"}))
    assert len({tuple(d.node_id for d in per_sample.sample(i, "t", {}).ders) for i in range(10)}) > 1


def test_placements_put_capacity_where_intended():
    def mean_position(placement):
        s = _sampler(_groups({"count": 2, "placement": placement}), n=300)
        pos = {n: i for i, n in enumerate(_DER_NODES)}
        cap = np.zeros(len(_DER_NODES))
        for i in range(300):
            for d in s.sample(i, "2024-06-01 12:00:00", {}).ders:
                cap[pos[d.node_id]] += d.s_rated_kva
        return cap / cap.sum()

    uniform, end, load = mean_position("random"), mean_position("feeder_end"), mean_position("load")
    assert uniform[1:] == pytest.approx([0.25] * 4, abs=0.05)
    assert end[4] > end[3] > end[2] > end[1]                   # weighted by distance from the substation
    assert load[2] == 0 and load[4] == 0 and load[1] == pytest.approx(0.5, abs=0.07)   # only A and C have kVA


def test_units_take_the_database_type_and_its_mode_mix():
    s = _sampler(_groups({"count": 4}), n=400, gen=_gen(400, data_type="chp"))
    modes = Counter(d.control_mode for i in range(400) for d in s.sample(i, "2024-06-01 12:00:00", {}).ders)
    assert modes["fixed_pf"] / sum(modes.values()) == pytest.approx(0.7, abs=0.06)
    other = _sampler(_groups({"count": 3}), gen=_gen(200, data_type="other_generation"))
    ders = other.sample(0, "2024-06-01 12:00:00", {}).ders
    assert {d.der_type for d in ders} == {"other"} and {d.control_mode for d in ders} == {"fixed_pf"}
    fixed = _sampler(_groups({"count": 5, "control": "p_v"}), gen=_gen(200, data_type="chp"))
    assert {d.control_mode for d in fixed.sample(3, "t", {}).ders} == {"p_v"}       # the group's own mode
    with pytest.raises(ValueError):
        _groups({"control": "droop"})
    with pytest.raises(ValueError, match="der.groups"):
        DERConfig(source="add")                                    # generated DER need plant groups


def test_both_solvers_hold_each_substation_at_its_own_set_point():
    nodes, roots = ["S1", "S2", "A", "B"], ["S1", "S2"]
    edges = [("L1", "S1", "A"), ("L2", "S2", "B")]
    z = {"L1": 0.01 + 0.02j, "L2": 0.01 + 0.02j}
    s = np.array([0, 0, 0.5 + 0.1j, 0.5 + 0.1j])
    v_root = {"S1": 1.03, "S2": 0.97}
    radial = solver.build_radial_network(nodes, roots, edges, z)
    v_r, ok, _ = radial.solve(np.array([s[nodes.index(n)] for n in radial.node_ids]), v_root)
    meshed = MeshedNetwork(nodes, roots, edges, {k: 1 / v for k, v in z.items()})
    v_m, ok_m, _ = meshed.solve(np.array([s[nodes.index(n)] for n in meshed.node_ids]), v_root)
    assert ok and ok_m
    vr = dict(zip(radial.node_ids, v_r))
    vm = dict(zip(meshed.node_ids, v_m))
    assert abs(vr["S1"]) == pytest.approx(1.03) and abs(vr["S2"]) == pytest.approx(0.97)
    assert all(abs(vr[n] - vm[n]) < 1e-8 for n in nodes)
    assert abs(vr["A"]) > abs(vr["B"])                              # same load, higher set-point


def test_voltage_controlled_capacitor_steps_into_its_band():
    nodes = ["S", "A", "B"]
    net = solver.build_radial_network(nodes, "S", [("L1", "S", "A"), ("L2", "A", "B")],
                                      {"L1": 0.02 + 0.04j, "L2": 0.02 + 0.04j})
    s = np.array([0, 0, 1.0 + 0.4j])
    v0, _, _ = net.solve(s, 1.0)
    assert abs(v0[2]) < 0.97
    bank = ShuntControl("C1", "B", 0.2j, 8, 0, 0, 0.97, 1.03)
    res = solver.solve_with_controls(net, s, 1.0, shunts=[bank])
    assert res.converged and res.settled and res.shunt_steps["C1"] > 0
    assert 0.97 <= abs(res.v[2]) <= 1.03 or res.shunt_steps["C1"] == 8
    plain, _, _ = net.solve(s, 1.0, y_shunt_extra_pu=res.y_shunt_extra)
    assert np.allclose(plain, res.v)                                  # the extra shunt is what was solved


def test_setpoint_sampler_modes():
    subs = ["S1", "S2"]
    cfg = SetpointsConfig(slack_mode="load_dependent", slack_noise=0.0)
    sp = SetpointSampler(cfg, 1.0, subs, 3, 1, load_levels=[0.0, 0.5, None], gross_demand=[1, 2, 4])
    assert [sp.slack_for_output(i) for i in range(3)] == pytest.approx([1.00, 1.02, 1.04])  # None: 4 / peak 4
    per = SetpointSampler(SetpointsConfig(slack_mode="uniform", per_substation=True),
                          SlackVoltageRange(min=0.95, max=1.05), subs, 100, 2)
    draws = [per.slack_for_output(i) for i in range(100)]
    assert all(set(d) == set(subs) and 0.95 <= min(d.values()) and max(d.values()) <= 1.05 for d in draws)
    assert any(abs(d["S1"] - d["S2"]) > 1e-6 for d in draws) and per.slack_for_output(7) == draws[7]
    # slack_mode None: slack_voltage_pu's range redrawn per output from stream [seed, 1], as before.
    legacy = SetpointSampler(SetpointsConfig(), SlackVoltageRange(min=0.95, max=1.05), subs, 10, 9)
    ref = np.random.default_rng([9, 1])
    assert [legacy.slack_for_output(0) for _ in range(3)] == [float(ref.uniform(0.95, 1.05)) for _ in range(3)]
    regs = [RegulatorControl("R", 1.0, 0.02, 0.625, -16, 16, 0)]
    re = SetpointSampler(SetpointsConfig(regulators_resample=True), 1.0, subs, 5, 4).regulators(2, regs)[0]
    assert 0.98 <= re.v_set_pu <= 1.03 and 0.015 <= re.deadband_pu <= 0.025 and -16 <= re.tap_pos <= 16
    tr = MVBranch("T", "A", "A_LV", None, 0.0, 0, 0, 10.0, kind="transformer", tap_min=-2, tap_max=2)
    taps = SetpointSampler(SetpointsConfig(transformer_taps_resample=True), 1.0, subs, 1, 4)
    assert all(-2 <= taps.transformer_taps(v, [tr])["T"] <= 2 for v in range(20))
    with pytest.raises(ValueError):
        PFDataGenRequest(historical_database_id="h", setpoints={"slack_mode": "uniform"})


def test_job_writes_generated_der_and_set_point_columns(tmp_path):
    topo = _feeder()
    topo.node_coords = {n: (41.15, -8.61) for n in FEEDER_NODES}
    topo.shunts = [MVShunt("CB", "N5", 300.0, 3, 0)]
    from gridarena.pf_datagen import der_profiles
    with patch.object(der_profiles, "generation_step_s", return_value=900.0), \
            patch.object(der_profiles, "build_generation_profile", return_value=_gen(6, data_type="chp")):
        result, out = _run_job(
            tmp_path, {"type": "none"}, n_timestamps=6, topology=topo,
            load_level={"mode": "loadability", "min": 0.9, "max": 1.1},
            der={"source": "replace", "groups": [{**_GEN, "count": 2, "capacity_mw": 0.3}]},
            setpoints={"shunt_steps": "voltage", "shunt_band_min": 0.99, "shunt_band_max": 1.05},
        )
    assert result["status"] == "completed"
    RT, DR = columns.index(columns.RUNTIME), columns.index(columns.DER)
    p_max = result["metrics"]["p_max_w"]
    for r in out["runtime"]:
        assert r[RT["der_penetration"]] == pytest.approx(0.6e6 / p_max)
        assert r[RT["der_capacity"]] == pytest.approx(0.6e6) and r[RT["der_units"]] == 2
        assert r[RT["shunt_steps_in"]] is not None
    assert out["der"] and all(d[DR["synthetic"]] and d[DR["der_id"]].startswith("GEN-CHP-") for d in out["der"])
    assert any(r[RT["shunt_steps_in"]] > 0 for r in out["runtime"])    # heavy load: the bank switched in
    report = json.loads((tmp_path / "run" / "report.json").read_text())
    assert report["version"] == 5 and report["der_penetration"]["samples"] == [len(out["runtime"])]
    assert report["der_penetration"]["edges"] == pytest.approx([0.6e6 / p_max] * 2)
    assert report["setpoints"]["shunt_steps_in"]["n"] == result["metrics"]["converged"]


# --- Phase 2c: topology (axis 4) and parameters (axis 5) ---------------------

import itertools  # noqa: E402

import networkx as nx  # noqa: E402

from gridarena.pf_datagen.outages import OutageSampler, assign_variants, bypass  # noqa: E402
from gridarena.pf_datagen.parameters import (  # noqa: E402
    ALPHA_PER_C, CableType, ParameterSampler, build_catalogue, cable_family, effective_branches,
)
from gridarena.pf_datagen.perturbation import build_run_variants, build_variants, perturb_admittances  # noqa: E402
from gridarena.pf_datagen.reconfiguration import Configuration, Reconfigurer, draw_configurations  # noqa: E402
from gridarena.pf_datagen.schemas import (  # noqa: E402
    OutagesConfig, ParametersConfig, ReconfigurationConfig, TemperatureConfig, TopologyPerturbationConfig,
)


#   SUB ─F─ a ─L1─ b ─L2─ c       Two radial feeders and three normally-open
#           ┆L5     ┆T1    ┆T2     ties between them (┆). L2 is a closed switch;
#   SUB2 ─L3─ d ─L4─ e ────┘       with `marked` only it and the ties switch.
def _meshed_grid():
    def line(cid, x, y, **kw):
        return MVBranch(cid, x, y, "cab", 500.0, 0.2, 0.1, 300.0, **kw)

    def tie(cid, x, y):
        return line(cid, x, y, is_switch=True, normally_open=True, in_service=False)
    return [line("F", "SUB", "a"), line("L1", "a", "b"), line("L2", "b", "c", is_switch=True),
            line("L3", "SUB2", "d"), line("L4", "d", "e"), tie("L5", "a", "d"), tie("T1", "b", "e"),
            tie("T2", "c", "e")]


MESH_NODES = ["SUB", "SUB2", "a", "b", "c", "d", "e"]
MESH_SUBS = ["SUB", "SUB2"]


def _in_service(branches, config):
    return ({b.connection_id for b in branches if b.in_service} - config.opened) | config.closed


def _radial_and_covering(branches, config):
    """Every node fed and no loop, with the substations acting as one root."""
    g = nx.Graph()
    g.add_nodes_from(MESH_NODES)
    ends = {b.connection_id: (b.from_node_id, b.to_node_id) for b in branches}
    g.add_edges_from(ends[c] for c in _in_service(branches, config))
    g = nx.contracted_nodes(g, "SUB", "SUB2", self_loops=True)
    return nx.is_tree(g)


def test_reconfiguration_draws_distinct_radial_configurations_keeping_plain_cables_closed():
    branches = _meshed_grid()
    rec = Reconfigurer(branches, MESH_NODES, MESH_SUBS, "all_lines")
    configs = rec.draw(rec.count() - 1, np.random.default_rng(1))      # every one besides the base
    keys = {frozenset(_in_service(branches, c)) for c in configs}
    assert len(keys) == len(configs)
    assert frozenset(b.connection_id for b in branches if b.in_service) not in keys
    for c in configs:
        assert _radial_and_covering(branches, c)
    with pytest.raises(ValueError, match="radial configuration"):
        rec.draw(rec.count(), np.random.default_rng(1))
    # marked: L5 and T1 would close a loop of plain cables, so the only other
    # radial configuration opens L2 and closes T2.
    marked = Reconfigurer(branches, MESH_NODES, MESH_SUBS, "marked")
    assert marked.count() == 2
    (c,) = marked.draw(1, np.random.default_rng(2))
    assert _radial_and_covering(branches, c) and (c.opened, c.closed) == ({"L2"}, {"T2"})


def test_reconfiguration_count_is_exact_and_draws_are_uniform():
    branches = _meshed_grid()
    by_id = {b.connection_id: b for b in branches}
    rec = Reconfigurer(branches, MESH_NODES, MESH_SUBS, "all_lines")
    switchable = sorted(rec.switchable)
    trees = []
    for k in range(len(switchable) + 1):
        for closed in itertools.combinations(switchable, k):
            c = Configuration(1, frozenset(x for x in switchable if x not in closed and by_id[x].in_service),
                              frozenset(x for x in closed if not by_id[x].in_service))
            if _radial_and_covering(branches, c):
                trees.append(frozenset(closed))
    assert rec.count() == len(trees) == 30
    rng = np.random.default_rng(5)
    counts = Counter(frozenset(rec.draw_tree(rng) | set(rec.forced)) for _ in range(30000))
    assert set(counts) == set(trees)
    assert all(850 < n < 1150 for n in counts.values())               # ~1000 each


def test_reconfiguration_never_feeds_a_regulator_from_its_far_side():
    branches = _meshed_grid()
    branches[1].kind = "regulated_line"                               # L1: a -> b, fed from a in the base
    ends = {b.connection_id: (b.from_node_id, b.to_node_id) for b in branches}
    rec = Reconfigurer(branches, MESH_NODES, MESH_SUBS, "all_lines")
    for c in rec.draw(10, np.random.default_rng(3)):
        in_service = _in_service(branches, c)
        if "L1" in in_service:
            g = nx.Graph()
            g.add_nodes_from(MESH_NODES)
            g.add_edges_from(ends[x] for x in in_service)
            dist = nx.multi_source_dijkstra_path_length(g, set(MESH_SUBS), weight=None)
            assert dist["a"] < dist["b"]


def test_meshed_reconfiguration_closes_extra_switches_and_needs_newton():
    branches = _meshed_grid()
    rec = Reconfigurer(branches, MESH_NODES, MESH_SUBS, "marked")
    for c in rec.draw(3, np.random.default_rng(4), allow_meshed=True, extra_closed=1):
        assert c.meshed and not _radial_and_covering(branches, c)
    with pytest.raises(ValueError, match="newton"):
        PFDataGenRequest(historical_database_id="h", reconfiguration={"enabled": True, "allow_meshed": True})
    configs, n = draw_configurations(ReconfigurationConfig(), branches, MESH_NODES, MESH_SUBS,
                                     np.random.default_rng(0))
    assert configs == [Configuration(0)] and n is None


def test_run_variants_reproduce_the_base_configuration_outage_sets():
    branches = _meshed_grid()
    cfg = TopologyPerturbationConfig(type="random", k=2, n_variants=5)
    old = build_variants(cfg, branches, MESH_NODES, MESH_SUBS, np.random.default_rng(9),
                         candidates=[b for b in branches if b.in_service],
                         out_of_service={b.connection_id for b in branches if not b.in_service})
    new = build_run_variants(cfg, [Configuration(0)], branches, MESH_NODES, MESH_SUBS,
                             np.random.default_rng(9), seed=9)
    assert [v.disabled_connection_ids for v in new] == [v.disabled_connection_ids for v in old]
    assert {v.kind for v in new} == {"outage"}
    rec = Reconfigurer(branches, MESH_NODES, MESH_SUBS, "marked")
    both = build_run_variants(TopologyPerturbationConfig(type="n_minus_k", k=1),
                              [Configuration(0)] + rec.draw(1, np.random.default_rng(1)), branches, MESH_NODES,
                              MESH_SUBS, np.random.default_rng(9), seed=9)
    assert Counter(v.kind for v in both) == {"base": 1, "outage": 5, "reconfiguration": 1, "both": 5}
    base_in_service = {b.connection_id for b in branches if b.in_service}
    for v in both:                     # outages only of connections closed in the configuration
        in_service = (base_in_service - v.opened_connection_ids) | v.closed_connection_ids
        assert v.disabled_connection_ids <= in_service
        assert v.active_connection_ids(base_in_service) == in_service - v.disabled_connection_ids


def test_per_sample_outages_hit_at_their_rates_and_bypass_pins_the_tap():
    ders, banks = [f"G{k}" for k in range(10)], ["CB1", "CB2"]
    s = OutageSampler(OutagesConfig(der_trip_p=0.05, shunt_out_p=0.2, regulator_bypass_p=0.5), seed=1)
    draws = [s.sample(i, ders, banks, ["R1"]) for i in range(4000)]
    assert sum(len(d.ders_tripped) for d in draws) / 40000 == pytest.approx(0.05, abs=0.01)
    assert sum(len(d.shunts_out) for d in draws) / 8000 == pytest.approx(0.2, abs=0.02)
    assert sum(len(d.regulators_bypassed) for d in draws) / 4000 == pytest.approx(0.5, abs=0.03)
    assert s.sample(7, ders, banks, ["R1"]) == draws[7]                       # reproducible
    # Changing one probability leaves the other kinds' draws alone.
    s2 = OutageSampler(OutagesConfig(der_trip_p=0.3, shunt_out_p=0.2, regulator_bypass_p=0.5), seed=1)
    assert all(s2.sample(i, ders, banks, ["R1"]).shunts_out == draws[i].shunts_out for i in range(200))
    assert not OutageSampler(OutagesConfig(), seed=1).enabled
    reg = RegulatorControl("R1", 1.0, 0.02, 1.25, -16, 16, 5)
    pinned = bypass([reg], frozenset({"R1"}))[0]
    assert (pinned.tap_min, pinned.tap_max, pinned.tap_pos) == (0, 0, 0)
    assert bypass([reg], frozenset()) == [reg]


def test_variants_per_sample_spreads_samples_evenly():
    out = assign_variants(n_samples=101, n_variants=7, m=3, seed=2)
    per_sample = Counter(i for samples in out for i in samples)
    assert set(per_sample.values()) == {3} and len(per_sample) == 101
    assert all(len(set(s)) == len(s) for s in out)
    sizes = [len(s) for s in out]
    assert max(sizes) - min(sizes) <= 1 and sum(sizes) == 303
    assert out == assign_variants(n_samples=101, n_variants=7, m=3, seed=2)


def test_job_reconfiguration_outages_and_variants_per_sample(tmp_path):
    topo = _feeder()
    # A tie from the end of the feeder back to N2 makes a loop to reconfigure.
    topo.branches.append(MVBranch("TIE", "N5", "N2", "cab", 800.0, 0.2, 0.1, 300.0,
                                  is_switch=True, normally_open=True, in_service=False))
    result, out = _run_job(
        tmp_path, {"type": "n_minus_k", "k": 1}, n_timestamps=6, topology=topo,
        reconfiguration={"enabled": True, "n_configs": 3, "switchable": "all_lines"},
        variants_per_sample=2,
    )
    assert result["status"] == "completed", result
    V, RT = columns.index(columns.VARIANT), columns.index(columns.RUNTIME)
    m = result["metrics"]
    assert m["n_configurations"] == 4 and m["radial_configurations"] == 4
    assert len(out["runtime"]) == 6 * 2 == m["total_output_groups"]          # linear in variants_per_sample
    assert Counter(r[RT["load_scenario_idx"]] for r in out["runtime"]) == {i: 2 for i in range(6)}
    kinds = Counter(v[V["kind"]] for v in out["variants"])
    assert kinds["base"] == 1 and kinds["reconfiguration"] == 3
    for v in out["variants"]:
        if v[V["kind"]] == "reconfiguration":
            assert v[V["closed_ties"]] == ["TIE"] and len(v[V["open_switches"]]) == 1
    assert all(r[RT["converged"]] for r in out["runtime"])
    report = json.loads((tmp_path / "run" / "report.json").read_text())
    rep = report["topology"]
    assert rep["n_configurations"] == 4 and sum(rep["variants_by_kind"].values()) == len(out["variants"])
    assert sum(k["outputs"] for k in rep["outcomes_by_kind"].values()) == 12
    assert rep["nodes_per_substation"]["p100"] == 5


def test_job_trips_der_and_takes_banks_out_per_sample(tmp_path):
    topo = _feeder()
    topo.shunts = [MVShunt("CB", "N5", 300.0, 3, 2)]
    topo.ders = [_der("G1", "N5", s_kva=300.0)]
    result, out = _run_job(tmp_path, {"type": "none"}, n_timestamps=8, topology=topo,
                           outages={"der_trip_p": 0.5, "shunt_out_p": 0.5})
    assert result["status"] == "completed", result
    RT, DR = columns.index(columns.RUNTIME), columns.index(columns.DER)
    for r in out["runtime"]:
        der = next(d for d in out["der"] if d[0] == r[0])
        assert der[DR["in_service"]] == (r[RT["n_der_out"]] == 0)
        assert r[RT["shunt_steps_in"]] == (0 if r[RT["n_shunt_out"]] else 2)
        assert r[RT["n_regulators_bypassed"]] is None
    assert {r[RT["n_der_out"]] for r in out["runtime"]} == {0, 1}
    assert {r[RT["n_shunt_out"]] for r in out["runtime"]} == {0, 1}


def test_cable_family_uses_charging_capacitance_then_x_over_r():
    assert cable_family(0.125, 0.0899, 0.1176) == "underground"       # X/R 0.72, but 0.118 uF/km
    assert cable_family(0.699, 0.360, 0.0101) == "overhead"
    assert cable_family(0.206, 0.096, None) == "underground"          # X/R 0.47
    assert cable_family(0.21, 0.13, None) == "overhead"               # X/R 0.62
    db_types = [CableType("A", 0.206, 0.096, 325.0, 0.109), CableType("B", 0.125, 0.09, 428.0, 0.118)]
    up = ParametersConfig(catalogue=[{"cable_id": "U", "r_ohm_per_km": 0.206, "x_ohm_per_km": 0.096,
                                      "nom_curr_a": 325.0, "cap_uf_per_km": 0.11}]).catalogue
    assert [t.cable_id for t in build_catalogue(db_types, up)] == ["B", "U"]   # same values: U replaces A


def _param_lines():
    return [MVBranch("L1", "SUB", "a", "A", 1000.0, 0.206, 0.096, 325.0, cap_uf_per_km=0.109),
            MVBranch("L2", "a", "b", "OH", 2000.0, 0.699, 0.360, 193.0, cap_uf_per_km=0.0101),
            MVBranch("T", "b", "b_LV", None, 0.0, 0.0, 0.0, 24.0, kind="transformer", z_ohm=complex(2.0, 10.0))]


def test_length_factor_scales_r_x_and_charging_together():
    lines = _param_lines()
    sampler = ParameterSampler(ParametersConfig(length_sigma=0.05), 1, lines)
    draw = sampler.static(0, None)
    eff, lf, n_swaps = effective_branches(lines, draw, None)
    assert n_swaps == 0
    for b0, b in zip(lines[:2], eff[:2]):
        f = lf[b0.connection_id]
        assert 0.95 <= f <= 1.05
        assert 1 / grid_loader.branch_admittance(b) == pytest.approx(f / grid_loader.branch_admittance(b0))
        assert grid_loader.branch_charging_susceptance(b) == pytest.approx(
            f * grid_loader.branch_charging_susceptance(b0))
    assert eff[2] == lines[2]
    # Reproducible, and drawn per variant.
    assert sampler.static(0, None).length_factor == draw.length_factor != sampler.static(1, None).length_factor


def test_temperature_changes_r_only_by_the_coefficient():
    lines = _param_lines()
    sampler = ParameterSampler(ParametersConfig(temperature=TemperatureConfig(min=50, max=50, spread=0)), 1, lines)
    eff, _, _ = effective_branches(lines, None, sampler.temperature(3))
    assert eff[0].r_ohm_per_km == pytest.approx(0.206 * (1 + ALPHA_PER_C * 30))
    assert eff[0].x_ohm_per_km == 0.096 and eff[0].length_m == 1000.0
    assert eff[2].z_ohm == lines[2].z_ohm
    spread = ParameterSampler(ParametersConfig(temperature=TemperatureConfig(min=40, max=40, spread=5)), 1, lines)
    t = spread.temperature(0)
    assert t.temp_c == 40 and all(35 <= x <= 45 for x in t.cable_temp_c.values())


def test_swaps_stay_in_family_and_carry_ratings():
    lines = _param_lines()
    cat = [CableType("A", 0.206, 0.096, 325.0, 0.109), CableType("B", 0.125, 0.090, 428.0, 0.118, 500.0),
           CableType("OH", 0.699, 0.360, 193.0, 0.0101), CableType("OH2", 0.243, 0.329, 357.0, 0.0111)]
    sampler = ParameterSampler(ParametersConfig(swap_p=1.0), 4, lines, cat)
    eff, _, n_swaps = effective_branches(lines, sampler.static(0, None), None)
    assert n_swaps == 2
    assert (eff[0].cable_id, eff[0].nom_curr_a, eff[0].emerg_curr_a, eff[0].r_ohm_per_km) == ("B", 428.0, 500.0, 0.125)
    assert (eff[1].cable_id, eff[1].nom_curr_a, eff[1].cap_uf_per_km) == ("OH2", 357.0, 0.0111)
    assert eff[0].length_m == 1000.0


def test_transformer_impedance_stays_in_its_ranges():
    lines = _param_lines()
    sampler = ParameterSampler(ParametersConfig(transformer_uk_sigma=0.1, transformer_xr_sigma=0.2), 2, lines)
    z0 = lines[2].z_ohm
    for v in range(50):
        eff, _, _ = effective_branches(lines, sampler.static(v, None), None)
        z = eff[2].z_ohm
        assert 0.9 - 1e-12 <= abs(z) / abs(z0) <= 1.1 + 1e-12
        assert 0.8 - 1e-12 <= (z.imag / z.real) / (z0.imag / z0.real) <= 1.2 + 1e-12
        assert eff[0] == lines[0]


def test_admittance_perturbation_is_unchanged_by_the_parameter_path():
    lines = _param_lines()
    overrides = perturb_admittances(lines, 0.2, np.random.default_rng(3))
    eff, lf, _ = effective_branches(lines, None, None, overrides)
    assert lf is None
    for b, new in zip(lines[:2], eff[:2]):
        r, x = overrides[b.connection_id]
        assert grid_loader.branch_admittance(new) == grid_loader.branch_admittance(b, r, x)


def test_job_records_cable_type_length_factor_and_temperature(tmp_path):
    result, out = _run_job(tmp_path, {"type": "random", "k": 1, "n_variants": 2}, n_timestamps=3,
                           parameters={"length_sigma": 0.05, "temperature": {"min": 30, "max": 30, "spread": 0}})
    assert result["status"] == "completed", result
    B, RT = columns.index(columns.BRANCH), columns.index(columns.RUNTIME)
    assert all(r[RT["conductor_temp_c"]] == 30 and r[RT["n_cable_swaps"]] is None for r in out["runtime"])
    for r in out["branch"]:
        f = r[B["length_factor"]]
        assert r[B["cable_type"]] == "cab" and 0.95 <= f <= 1.05
        assert r[B["r_ohm_total"]] == pytest.approx(0.2 * (1 + ALPHA_PER_C * 10) * f)
        assert r[B["x_ohm_total"]] == pytest.approx(0.1 * f)
    by_variant = {}
    for r in out["branch"]:
        by_variant.setdefault(r[2], {})[r[3]] = r[B["length_factor"]]
    assert by_variant[0] != by_variant[1]                            # drawn per variant
    report = json.loads((tmp_path / "run" / "report.json").read_text())
    assert report["parameters"]["conductor_temp_c"]["p50"] == 30


# --- Phase 2c: tie-line suggestions -------------------------------------------

from gridarena.pf_datagen.ties import haversine_m, suggest_ties, validate_ties  # noqa: E402
from gridarena.schemas.mv_grid_schema import MVTieLine  # noqa: E402

_M_PER_DEG_LAT = 111195.0


def _u_feeder():
    """A U-shaped feeder: S - A1 - A2 - A3 runs north, then east and back
    south to B3 - B2 - B1, so A1 and B1 are 300 m apart on the ground but
    2.3 km apart along the cables; a second feeder S - C1 runs west."""
    def node(nid, north_m, east_m, kind="mv"):
        return {"node_id": nid, "coord_lat": 41.0 + north_m / _M_PER_DEG_LAT,
                "coord_lon": -8.0 + east_m / (_M_PER_DEG_LAT * np.cos(np.radians(41.0))), "node_type": kind}
    nodes = [node("S", 0, 0, "substation"), node("A1", 100, 0), node("A2", 600, 0), node("A3", 1100, 0),
             node("B3", 1100, 300), node("B2", 600, 300), node("B1", 100, 300), node("C1", 0, -400)]
    chain = [("S", "A1", 100), ("A1", "A2", 500), ("A2", "A3", 500), ("A3", "B3", 300), ("B3", "B2", 500),
             ("B2", "B1", 500), ("S", "C1", 400)]
    conns = [{"connection_id": f"L{i}", "from_node_id": a, "to_node_id": b, "cable_id": "cab" if i else "big",
              "length": float(l), "status": "active", "is_switch": False, "normally_open": False}
             for i, (a, b, l) in enumerate(chain)]
    return {"nodes": nodes, "connections": conns, "cables": [{"cable_id": "cab"}, {"cable_id": "big"}]}


def test_tie_suggestions_find_close_nodes_far_apart_along_cables():
    grid = _u_feeder()
    out = suggest_ties(grid, max_distance_m=500, min_cable_distance_m=2000, max_ties=5)
    pairs = [frozenset((t["from_node_id"], t["to_node_id"])) for t in out["suggestions"]]
    assert pairs[0] == frozenset(("A1", "B1"))                        # 300 m apart, 2.3 km along cables
    first = out["suggestions"][0]
    assert first["distance_m"] == pytest.approx(300, rel=0.01)
    assert first["cable_distance_m"] == pytest.approx(2300)
    assert first["reason"] == "distant_along_cables"
    assert first["length"] == pytest.approx(1.2 * first["distance_m"], abs=0.1)
    assert first["cable_id"] == "cab" and first["connection_id"] == "TIE01"
    existing = {frozenset((c["from_node_id"], c["to_node_id"])) for c in grid["connections"]}
    assert not set(pairs) & existing                                  # never duplicates a connection
    assert all("S" not in p for p in pairs)                           # substations excluded
    # Spread: no second tie with an end within 500 m of the first one's ends.
    for t in out["suggestions"][1:]:
        assert all(haversine_m(_coords(grid, t[e]), _coords(grid, first[f])) >= 500
                   for e in ("from_node_id", "to_node_id") for f in ("from_node_id", "to_node_id"))
    # A1 - C1 are 412 m apart on different feeders: proposed although only 500 m apart along the cables.
    strict = suggest_ties(grid, max_distance_m=500, min_cable_distance_m=10000, max_ties=5)
    assert [(frozenset((t["from_node_id"], t["to_node_id"])), t["reason"]) for t in strict["suggestions"]] == [
        (frozenset(("A1", "C1")), "different_feeders")]


def _coords(grid, nid):
    n = next(n for n in grid["nodes"] if n["node_id"] == nid)
    return n["coord_lat"], n["coord_lon"]


def test_tie_suggestions_skip_pairs_an_existing_tie_already_shortcuts():
    grid = _u_feeder()
    grid["connections"].append({"connection_id": "TIE01", "from_node_id": "A2", "to_node_id": "B2", "cable_id": "cab",
                                "length": 360.0, "status": "active", "is_switch": True, "normally_open": True})
    out = suggest_ties(grid, max_distance_m=500, min_cable_distance_m=2000, max_ties=5)
    assert all(t["connection_id"] != "TIE01" for t in out["suggestions"])
    # Through the open tie A1 - B1 is 500 + 360 + 500 = 1360 m < 2 km: no longer proposed.
    assert frozenset(("A1", "B1")) not in {frozenset((t["from_node_id"], t["to_node_id"])) for t in out["suggestions"]}


def test_validate_ties_rejects_unknown_nodes_cables_and_duplicates():
    grid = _u_feeder()
    ok = MVTieLine(connection_id="TIE01", from_node_id="A1", to_node_id="B1", cable_id="cab", length=360)
    assert validate_ties(grid, [ok]) == []
    bad = [MVTieLine(connection_id="L1", from_node_id="A1", to_node_id="X", cable_id="none", length=10),
           MVTieLine(connection_id="T2", from_node_id="A2", to_node_id="A1", cable_id="cab", length=10),
           ok, MVTieLine(connection_id="TIE01", from_node_id="A3", to_node_id="B1", cable_id="cab", length=10)]
    errors = " ".join(validate_ties(grid, bad))
    for part in ("node 'X' not found", "cable 'none' not found", "'L1': connection id already used",
                 "already connected", "'TIE01': connection id already used"):
        assert part in errors


# --- Phase 2d: joint design, target mix, splits, groups ----------------------

from gridarena.pf_datagen import design as design_mod  # noqa: E402
from gridarena.pf_datagen import splits as splits_mod  # noqa: E402
from gridarena.pf_datagen import target_mix as tm  # noqa: E402
from gridarena.pf_datagen.group_report import build_group_report  # noqa: E402


def test_lhs_keeps_every_axis_stratified_and_lowers_correlation():
    rng = np.random.default_rng(0)
    u = design_mod.lhs_unit(40, 3, rng)
    for k in range(3):
        assert sorted(np.floor(u[:, k] * 40).astype(int)) == list(range(40))     # one point per 1/n stratum
    ranks = [np.argsort(np.argsort(u[:, k])) for k in range(3)]
    worst = max(abs(np.corrcoef(ranks[i], ranks[j])[0, 1]) for i in range(3) for j in range(i + 1, 3))
    # Independent shuffles of 40 points: the largest of 3 rank correlations is typically ~0.2.
    single = []
    for s in range(50):
        r = np.random.default_rng(s)
        p = [r.permutation(40) for _ in range(3)]
        single.append(max(abs(np.corrcoef(p[i], p[j])[0, 1]) for i in range(3) for j in range(i + 1, 3)))
    assert worst < np.median(single)


def test_factorial_fills_every_cell_and_refuses_too_few_samples():
    u = design_mod.factorial_unit(53, [5, 4], np.random.default_rng(1))
    cells = Counter((int(a * 5), int(b * 4)) for a, b in u)
    assert len(cells) == 20 and set(cells.values()) <= {2, 3} and sum(cells.values()) == 53
    with pytest.raises(ValueError, match="at least as many samples as cells"):
        design_mod.factorial_unit(19, [5, 4], np.random.default_rng(1))
    axes = [design_mod.Axis("level", 0.1, 1.1, 5), design_mod.Axis("penetration", 0.0, 1.0, 4)]
    d = design_mod.build_design("factorial", axes, 40, seed=3)
    assert 0.1 <= d.values["level"].min() and d.values["level"].max() < 1.1
    assert set(Counter(d.cells()).values()) == {2} and d.cells()[0].startswith("L") and ".P" in d.cells()[0]
    assert design_mod.build_design("independent", axes, 40, seed=3) is None


def test_design_spreads_every_variant_over_the_load_range():
    axes = [design_mod.Axis("level", 0.0, 1.0, 10)]
    d = design_mod.build_design("lhs", axes, 60, seed=2)
    out = assign_variants(60, 6, 1, seed=2, order=design_mod.variant_order(d, 60))
    for samples in out:
        lv = sorted(d.values["level"][samples])
        assert len(lv) == 10 and lv[0] < 0.1 and lv[-1] > 0.9      # each variant: the whole range


def test_target_weights_tilt_only_as_far_as_needed():
    strata = ["L0", "L1", "L2", "L3"]
    outcomes = [(c, v) for c, rate in zip(strata, (0.0, 0.1, 0.4, 0.9))
                for v in [True] * int(rate * 20) + [False] * (20 - int(rate * 20))]
    met = tm.choose_weights(strata, outcomes, 0.3, 80, 320)
    assert met.tilt == 0 and met.weights == pytest.approx(dict.fromkeys(strata, 0.25))
    more = tm.choose_weights(strata, outcomes, 0.5, 80, 320)
    assert 0 < more.tilt < tm.MAX_TILT and more.reachable
    assert more.expected_share == pytest.approx(0.5, abs=1e-6)
    assert more.weights["L3"] > more.weights["L2"] > more.weights["L0"] >= tm.FLOOR * 0.25
    never = tm.choose_weights(strata, outcomes, 0.95, 80, 320)
    assert never.tilt == tm.MAX_TILT and not never.reachable
    assert min(never.weights.values()) >= tm.FLOOR * 0.25 - 1e-12             # every stratum stays covered
    labels = tm.allocate(more.weights, 320, np.random.default_rng(0))
    counts = Counter(labels)
    assert sum(counts.values()) == 320
    assert all(abs(counts[c] - more.weights[c] * 320) <= 1 for c in strata)
    w = tm.design_weights(["L0"] * 10 + ["L1"] * 30, ["L0", "L1"])
    assert w[0] == pytest.approx(2.0) and w[-1] == pytest.approx(2 / 3)       # undo the oversampling
    assert sum(w) == pytest.approx(40)


def test_rounds_split_the_rest_randomly():
    rest = list(range(10, 100))
    rs = tm.rounds(rest, seed=4)
    assert len(rs) == tm.ROUNDS and sorted(i for r in rs for i in r) == rest
    assert rs[0] != rest[:30]                                    # not in timeline order


def test_job_target_mix_reaches_its_share_and_weights_undo_it(tmp_path):
    weights = {}
    with patch.object(db, "set_pf_datagen_design_weights",
                      side_effect=lambda run_id, w, conn, cursor: weights.update(w)):
        result, out = _run_job(
            tmp_path, {"type": "none"}, n_timestamps=60,
            load_level={"mode": "loadability", "min": 0.3, "max": 1.3, "bins": 5},
            target_mix={"min_violation_share": 0.5, "violations": "any"},
        )
    assert result["status"] == "completed", result
    RT, SM = columns.index(columns.RUNTIME), columns.index(columns.SUMMARY)
    viol = [((s[SM["n_thermal_violations"]] or 0) + (s[SM["n_undervoltage"]] or 0)) > 0 for s in out["summary"]]
    report = json.loads((tmp_path / "run" / "report.json").read_text())
    t = report["target_mix"]
    assert len(t["rounds"]) == tm.ROUNDS
    assert t["achieved_share"] == pytest.approx(sum(viol) / len(viol))
    assert t["achieved_share"] > 0.4                              # an even design gives about 0.3 here
    assert len(weights) == 60 and all(w > 0 for w in weights.values())
    level = {r[RT["load_scenario_idx"]]: r[RT["load_level"]] for r in out["runtime"]}
    weighted = sum(weights[i] * level[i] for i in level) / sum(weights[i] for i in level)
    assert weighted == pytest.approx(0.8, abs=0.06)              # the even design's mean level
    assert all(r[RT["design_cell"]].startswith("L") for r in out["runtime"])


def test_job_factorial_design_fills_its_strata(tmp_path):
    result, out = _run_job(tmp_path, {"type": "none"}, n_timestamps=8,
                           load_level={"mode": "multiplier", "min": 0.5, "max": 1.5, "bins": 4},
                           design={"mode": "factorial"})
    assert result["status"] == "completed", result
    RT = columns.index(columns.RUNTIME)
    assert Counter(r[RT["design_cell"]] for r in out["runtime"]) == {"L0": 2, "L1": 2, "L2": 2, "L3": 2}
    assert all(r[RT["design_weight"]] is None for r in out["runtime"])
    report = json.loads((tmp_path / "run" / "report.json").read_text())
    assert report["design"]["strata_filled"] == report["design"]["n_strata"] == 4
    with pytest.raises(ValueError, match="needs a load level"):
        PFDataGenRequest(historical_database_id="h", target_mix={"min_violation_share": 0.2})


def test_report_counts_near_duplicate_load_samples():
    r = DiversityReport()
    s = np.array([1e5 + 2e4j, 2e5 + 5e4j, 0j])
    for x in (s, s.copy(), s * 1.5, np.array([2e5 + 2e4j, 1e5 + 5e4j, 0j])):
        r.add_load_sample(x)
    assert r.as_dict()["load"]["near_duplicate_share"] == pytest.approx(0.25)   # only the copy


def test_load_dependent_set_points_follow_redrawn_levels():
    cfg = SetpointsConfig(slack_mode="load_dependent", slack_v0=1.0, slack_v1=1.04, slack_noise=0.0)
    sp = SetpointSampler(cfg, 1.0, ["SUB"], 3, 1, [0.0, 0.5, 1.0], [1.0, 1.0, 1.0])
    assert sp.slack_for_output(1) == pytest.approx(1.02)
    sp.set_levels([1], [1.0])
    assert sp.slack_for_output(1) == pytest.approx(1.04) and sp.slack_for_output(0) == pytest.approx(1.0)


def _split_rows():
    """Six days x 3 hours x 4 variants; variant 0 the base, config_idx = variant // 2."""
    rows = []
    for d in (1, 2, 3, 4, 5, 6):
        for h in range(3):
            i = (d - 1) * 3 + h
            for v in range(4):
                rows.append((i * 4 + v, i, v, f"2024-01-0{d} 0{h}:00:00", v // 2, "base" if v == 0 else "outage"))
    return rows


def test_time_split_keeps_each_day_whole():
    out = splits_mod.split_samples(_split_rows(), "time", 0.5, 0.25, seed=1)
    split_of_day = {}
    for s_ in out["samples"]:
        split_of_day.setdefault(s_["scenario_timestamp"][:10], set()).add(s_["split"])
    assert all(len(v) == 1 for v in split_of_day.values())
    assert set().union(*split_of_day.values()) == {"train", "val", "test"}
    assert sum(out["counts"].values()) == len(_split_rows())


def test_topology_split_keeps_configurations_whole_with_the_base_in_training():
    rows = _split_rows()
    out = splits_mod.split_samples(rows, "topology", 0.5, 0.25, seed=2)
    assert out["unit"] == "configuration" and 0 in out["units"]["train"]
    per_unit = {}
    for s_, r in zip(out["samples"], rows):
        per_unit.setdefault(r[4], set()).add(s_["split"])
    assert all(len(v) == 1 for v in per_unit.values())
    no_reconf = [r[:4] + (0, r[5]) for r in rows]
    out = splits_mod.split_samples(no_reconf, "topology", 0.5, 0.25, seed=2)
    assert out["unit"] == "topology_variant" and 0 in out["units"]["train"]
    with pytest.raises(ValueError):
        splits_mod.split_samples(rows, "random")
    # Three units, three splits: each gets one (the base still in training).
    three = splits_mod.assign_units({0: 120, 1: 120, 2: 120}, 0.7, 0.15, seed=0, forced_train=[0])
    assert three[0] == "train" and sorted(three.values()) == ["test", "train", "val"]


def test_grid_split_is_stratified_by_size():
    sizes = {f"G{i:02d}": 10 + 5 * i for i in range(20)}
    out = splits_mod.split_grids(sizes, 0.7, 0.15, seed=0)
    assert set(out["grids"]) == set(sizes)
    counts = Counter(out["grids"].values())
    assert counts["train"] == 14 and counts["val"] == 3 and counts["test"] == 3
    small = [g for g in sorted(sizes, key=sizes.get)[:10]]
    assert {out["grids"][g] for g in small} >= {"train", "val"} or {out["grids"][g] for g in small} >= {"train", "test"}


def test_group_report_sums_counts_and_lists_every_grid():
    class R:
        def __init__(self, status):
            self.status = status
    hist = {"edges": [0.9, 1.0, 1.1], "counts": [1, 2], "below": 0, "above": 1}
    group = {"group_id": "abcd1234", "runs": [
        {"mv_grid_id": "A", "run_id": "r1", "n_nodes": 40, "n_connection_points": 12},
        {"mv_grid_id": "B", "run_id": "r2", "n_nodes": 60, "n_connection_points": 20},
        {"mv_grid_id": "C", "run_id": "r3", "n_nodes": 30, "n_connection_points": 9}],
        "skipped": [{"mv_grid_id": "D", "reason": "5 nodes (fewer than 10)"}]}
    reports = {"r1": {"outcomes": {"total": 10, "converged": 9, "failed": 1}, "failure_reasons": {"diverged": 1},
                      "voltage_pu": hist, "loading_pct": hist},
               "r2": {"outcomes": {"total": 5, "converged": 5, "failed": 0}, "failure_reasons": {},
                      "voltage_pu": hist, "loading_pct": hist, "loadability": {"p_max_w": 2e6}}}
    out = build_group_report(group, {"r1": R("completed"), "r2": R("completed"), "r3": R("running")}, reports,
                             {"vm_pu_min": None}, {"r1": {"converged": 9, "undervoltage_rate": 0.1,
                                                          "overvoltage_rate": 0.0, "thermal_rate": 0.2}})
    assert out["outcomes"] == {"total": 15, "converged": 14, "failed": 1}
    assert out["voltage_pu"]["counts"] == [2, 4] and out["voltage_pu"]["above"] == 2
    assert not out["complete"] and out["status_counts"] == {"completed": 2, "running": 1}
    assert [g["mv_grid_id"] for g in out["grids"]] == ["A", "B", "C"]
    assert out["grids"][0]["thermal_rate"] == 0.2 and out["grids"][1]["p_max_w"] == 2e6
    assert out["skipped"][0]["mv_grid_id"] == "D"


def test_group_endpoint_skips_small_grids_and_derives_seeds(tmp_path):
    from fastapi.testclient import TestClient

    from gridarena.app import app
    from gridarena.pf_datagen import run_manager

    cursor = MagicMock()
    cursor.fetchall.return_value = [("A", 40, 12), ("B", 8, 5), ("C", 50, 2), ("D", 30, 9)]
    started = []

    def fake_start(mv_grid_id, request, group_id=None):
        started.append((mv_grid_id, request.seed, group_id))
        return f"{len(started):08x}"

    with patch.object(db, "get_db_connection", return_value=(MagicMock(), cursor)), \
            patch.object(run_manager, "GROUPS_DIR", tmp_path), \
            patch.object(run_manager.pf_datagen_registry, "start_run", side_effect=fake_start):
        client = TestClient(app)
        r = client.post("/pf-datagen/group/run", json={"mv_grid_ids": "all",
                                                       "request": {"historical_database_id": "h", "seed": 11}})
        assert r.status_code == 200, r.text
        body = r.json()
        assert [x["mv_grid_id"] for x in body["runs"]] == ["A", "D"]
        assert {s["mv_grid_id"] for s in body["skipped"]} == {"B", "C"}
        assert len({seed for _, seed, _ in started}) == 2 and all(g == body["group_id"] for *_, g in started)
        g = client.get(f"/pf-datagen/groups/{body['group_id']}")
        assert g.status_code == 200 and g.json()["seed"] == 11
        r = client.post("/pf-datagen/group/run", json={"mv_grid_ids": ["B"], "request": {"historical_database_id": "h"}})
        assert r.status_code == 422
        r = client.post("/pf-datagen/group/run", json={"mv_grid_ids": ["Z"], "request": {"historical_database_id": "h"}})
        assert r.status_code == 404


# ── Phase 2f: measurement view (axis 8), `window` timeline ──────────────────

import random as _random  # noqa: E402
from datetime import datetime, timedelta  # noqa: E402

from gridarena.benchmarks.general_noise.noise import DEFAULT_PROFILES, corrupt_measurement_record  # noqa: E402
from gridarena.pf_datagen import measurements as meas_mod  # noqa: E402
from gridarena.pf_datagen import timeline as timeline_mod  # noqa: E402
from gridarena.pf_datagen.records import SampleKey  # noqa: E402
from gridarena.pf_datagen.schemas import MeasurementsConfig  # noqa: E402


def _cp_chain(n=20, switches=("C5", "C12")):
    """SUB - N1 - ... - Nn, a transformer at every node; two lines are switches."""
    from gridarena.pf_datagen.grid_loader import build_transformer_branch

    nodes = ["SUB"] + [f"N{i}" for i in range(1, n + 1)]
    branches = [MVBranch(f"C{i}", nodes[i - 1], nodes[i], "cab", 500.0, 0.2, 0.1, 300.0,
                         is_switch=f"C{i}" in switches) for i in range(1, n + 1)]
    index = {nid: i for i, nid in enumerate(nodes)}
    vn = {}
    for nid in nodes[1:]:
        t, lv, lv_kv = build_transformer_branch(f"T{nid}", nid, 250.0, 15.0, 400.0, 0.01, 0.04, "active",
                                                0, 2.5, -2, 2)
        branches.append(t)
        index[lv] = len(index)
        vn[lv] = lv_kv
    return MVGridTopology("mv1", 15.0, ["SUB"], index, branches, node_vn_kv=vn)


def test_layout_covers_its_shares_and_is_fixed_by_the_seed():
    topo = _cp_chain()
    cfg = MeasurementsConfig(enabled=True)
    layout = meas_mod.build_layout(cfg, topo, seed=1, generated_der=False)
    counts = layout.counts()
    assert counts == {"substation": 1, "feeder_head": 1, "switch": 2, "cable": 2, "connection_point": 6,
                      "pseudo": 14}                         # 30 % of 20 points; 10 % of the 17 other lines
    head = next(m for m in layout.meters if m.reason == "feeder_head")
    assert head.element_id == "C1" and head.end == "from" and head.profile == "easy"
    assert all(m.profile == "medium" for m in layout.meters if m.reason in ("switch", "cable", "connection_point"))
    metered = {m.element_id for m in layout.meters if m.reason == "connection_point"}
    assert metered.isdisjoint(m.element_id for m in layout.meters if m.reason == "pseudo")
    again = meas_mod.build_layout(cfg, topo, seed=1, generated_der=False)
    other = meas_mod.build_layout(cfg, topo, seed=2, generated_der=False)
    assert again.rows() == layout.rows() and other.rows() != layout.rows()
    assert len(layout.rows()[0]) == len(columns.MEASUREMENT_LAYOUT)
    full = meas_mod.build_layout(MeasurementsConfig(enabled=True, cp_coverage=1, cable_coverage=1), topo, 1, True)
    assert full.counts()["connection_point"] == 20 and "pseudo" not in full.counts()
    assert full.counts()["cable"] == 17 and any(m.reason == "der_generated" for m in full.meters)


@pytest.mark.parametrize("name", ["easy", "medium", "hard"])
def test_noise_level_per_profile_and_exact_outlier_flags(name):
    prof = DEFAULT_PROFILES[name]
    errors, flags = [], []
    for k in range(6000):
        value, outlier = meas_mod._corrupt(1000.0, "p", _random.Random(k), prof)
        r = _random.Random(k)
        plain = 1000.0 + r.gauss(0.0, prof["pq_rel_sigma"] * 1000.0)
        assert outlier == (abs(value - plain) > 1e-9)        # the flag is exactly the outlier term
        flags.append(outlier)
        if not outlier:
            errors.append((value - 1000.0) / 1000.0)
    assert np.std(errors) == pytest.approx(prof["pq_rel_sigma"], rel=0.05)
    assert np.mean(flags) == pytest.approx(prof["outlier_prob"], abs=0.004)
    v = [meas_mod._corrupt(15000.0, "v", _random.Random(k), prof)[0] / 15000.0 - 1 for k in range(4000)]
    assert np.median(np.abs(v)) < 1.2 * prof["v_mag_rel_sigma"]
    # The benchmark module's own call, unchanged.
    assert corrupt_measurement_record({"power_active": 5.0}, _random.Random(0), DEFAULT_PROFILES["clean"]) == \
        {"power_active": 5.0}


def _row(cols, **values):
    idx = columns.index(cols)
    out = [None] * len(cols)
    for k, v in values.items():
        out[idx[k]] = v
    return tuple(out)


def _fake_sample(i, energised=True):
    key = SampleKey(f"r-{i}", f"2024-01-01 {i}", 0, i, i)
    bus = [_row(columns.BUS, node_id="N5", vm=15000.0 if energised else 0.0, energised=energised,
                pd=0.0, qd=0.0)]
    branch = [_row(columns.BRANCH, connection_id="T1", from_node="N5", to_node="N5-LV", pf=200e3, qf=50e3,
                   pt=-199e3, qt=-48e3, i_mag=8.0, in_service=True)]
    der = [_row(columns.DER, der_id="GEN-PV-1", node_id="N5", p=1e6, q=-1e5, s_rated=2e6, synthetic=True),
           _row(columns.DER, der_id="GEN-PV-2", node_id="N5", p=2e5, q=0.0, s_rated=5e5, synthetic=True)]
    return key, bus, branch, der


def _fake_sampler(**cfg):
    cfg = MeasurementsConfig(enabled=True, **cfg)
    layout = meas_mod.MeasurementLayout(
        [meas_mod.Meter("connection_point", "N5", "N5", ("p", "q"), "pseudo", None),
         meas_mod.Meter("der", "*", None, ("p", "q", "v"), "der_generated", cfg.profile)],
        cfg.der_min_mva * 1e6, cfg.profile, {"N5": ["T1"]})
    return meas_mod.MeasurementSampler(cfg, layout, 5, PerUnitBases.for_grid(15.0, 1.0),
                                       {"N5": 15.0, "N5-LV": 0.4})


def _readings(sampler, i, energised=True):
    key, bus, branch, der = _fake_sample(i, energised)
    return sampler.readings(key, 0, bus, branch, der)


def test_missing_rate_pseudo_error_and_generated_der_rule():
    sampler = _fake_sampler(missing_p=0.2, pseudo_sigma=0.3)
    rows = [r for i in range(1500) for r in _readings(sampler, i)]
    M = columns.index(columns.MEASUREMENT)
    meter = [r for r in rows if r[M["source"]] == "meter"]
    pseudo = [r for r in rows if r[M["source"]] == "pseudo"]
    assert {r[M["element_id"]] for r in meter} == {"GEN-PV-1"}           # 2 MVA metered, 0.5 MVA not
    assert np.mean([r[M["missing"]] for r in meter]) == pytest.approx(0.2, abs=0.02)
    assert all(r[M["value"]] is None and r[M["value_pu"]] is None for r in meter if r[M["missing"]])
    p = np.array([r[M["value"]] for r in pseudo if r[M["quantity"]] == "p"]) / 200e3 - 1
    assert np.std(p) == pytest.approx(0.3, rel=0.08) and p.min() >= -1.0
    assert not any(r[M["missing"]] or r[M["is_outlier"]] for r in pseudo)
    rep = sampler.report()
    assert rep["missing_rate"] == pytest.approx(0.2, abs=0.02)
    assert rep["pseudo_relative_error"]["p50"] == pytest.approx(0.3 * 0.674, rel=0.1)
    # Reproducible per sample, whatever was solved before it.
    a = _readings(_fake_sampler(), 7)
    b = _fake_sampler()
    _readings(b, 3)
    assert _readings(b, 7) == a


def test_islanded_sensors_read_exactly_zero():
    rows = _readings(_fake_sampler(missing_p=0.0), 0, energised=False)
    M = columns.index(columns.MEASUREMENT)
    assert rows and all(r[M["value"]] == 0.0 and not r[M["is_outlier"]] for r in rows)


def test_job_measurements_follow_the_clean_solution(tmp_path):
    topo = _feeder_with_transformer()
    full = {"enabled": True, "cp_coverage": 1.0, "cable_coverage": 1.0, "missing_p": 0.0}
    _, clean = _run_job_lv(tmp_path / "a", topo,
                           measurements={**full, "profile": "clean", "substation_profile": "clean"})
    M = columns.index(columns.MEASUREMENT)
    assert {(r[0], r[1], r[4]) for r in clean["layout"]} == {
        ("bus", "SUB", "substation"), ("branch", "C1", "feeder_head"), ("branch", "C2", "cable"),
        ("branch", "C3", "cable"), ("branch", "C4", "cable"), ("connection_point", "N5", "connection_point")}
    assert len(clean["measurements"]) == 2 * (1 + 4 * 3 + 3)
    br = {(r[0], r[3]): r for r in clean["branch"]}
    bus = {(r[0], r[3]): r for r in clean["bus"]}
    for r in clean["measurements"]:
        g, el, q, v = r[M["output_group_id"]], r[M["element_id"]], r[M["quantity"]], r[M["value"]]
        if el == "SUB":
            assert v == bus[(g, "SUB")][BUS_VM]
        elif el == "N5":
            expected = {"v": bus[(g, "N5")][BUS_VM], "p": br[(g, "T1")][BR_PF], "q": br[(g, "T1")][BR_QF]}[q]
            assert v == pytest.approx(expected)
        else:
            expected = {"p": br[(g, el)][BR_PF], "q": br[(g, el)][BR_QF], "i": br[(g, el)][BR_I_MAG]}[q]
            assert v == pytest.approx(expected)
            if q == "i":
                assert r[M["value_pu"]] == pytest.approx(v * SQRT3 * 15e3 / 1e6)
    _, noisy = _run_job_lv(tmp_path / "b", topo, measurements=full)
    _, again = _run_job_lv(tmp_path / "c", topo, measurements=full)
    assert noisy["measurements"] == again["measurements"]
    assert [r[M["value"]] for r in noisy["measurements"]] != [r[M["value"]] for r in clean["measurements"]]
    _, off = _run_job_lv(tmp_path / "d", topo)
    assert off["measurements"] == [] and off["layout"] == []


def test_job_open_branches_read_zero_and_the_report_has_measurements(tmp_path):
    result, out = _run_job(tmp_path, {"type": "n_minus_k", "k": 1},
                           measurements={"enabled": True, "cable_coverage": 1.0, "missing_p": 0.0})
    M = columns.index(columns.MEASUREMENT)
    v_c2 = _variant_disabling(out, "C2")
    rows = [r for r in out["measurements"] if r[M["topology_variant_idx"]] == v_c2]
    by_el = {}
    for r in rows:
        by_el.setdefault(r[M["element_id"]], []).append(r[M["value"]])
    assert all(v == 0.0 for el in ("C2", "C3", "C4") for v in by_el[el])
    assert all(v != 0.0 for v in by_el["C1"]) and by_el["SUB"][0] > 0
    report = json.loads((tmp_path / "run" / "report.json").read_text())
    m = report["measurements"]
    assert m["meters"]["cable"] == 3 and m["readings"] == len(out["measurements"]) and m["dead_readings"] > 0
    assert report["timeline"]["mode"] == "spread" and "per_sample_axes" not in report["timeline"]


def test_window_timeline_takes_consecutive_timestamps():
    t0 = datetime(2024, 1, 1)
    ts = [t0 + timedelta(minutes=15 * k) for k in range(1000)]
    spread = timeline_mod.select_timestamps(ts, 10, "spread")
    assert spread == [ts[i] for i in np.linspace(0, 999, 10, dtype=int)]
    w = timeline_mod.select_timestamps(ts, 96, "window", rng=np.random.default_rng([3, 17]))
    assert all(b - a == timedelta(minutes=15) for a, b in zip(w, w[1:])) and len(w) == 96
    assert w == timeline_mod.select_timestamps(ts, 96, "window", rng=np.random.default_rng([3, 17]))
    assert timeline_mod.select_timestamps(ts, 4, "window", start=datetime(2024, 1, 2, 0, 7)) == ts[97:101]
    assert timeline_mod.select_timestamps(ts, 10, "window", start=datetime(2030, 1, 1)) == ts[-10:]
    assert timeline_mod.select_timestamps(ts[:5], 10, "window") == ts[:5]
    s = timeline_mod.timeline_summary("window", [str(t) for t in ts[:4] + ts[10:12]])
    assert s["step_s"] == 900 and s["n_gaps"] == 1 and s["first"] == ts[0].isoformat()


def test_timeline_settings_and_per_sample_axes():
    base = {"historical_database_id": "h"}
    assert PFDataGenRequest(**base).timeline == "spread"
    with pytest.raises(ValueError, match="timeline_start"):
        PFDataGenRequest(**base, timeline_start="2024-01-01T00:00:00")
    req = PFDataGenRequest(**base, timeline="window", timeline_start="2024-01-01T00:00:00")
    assert timeline_mod.per_sample_axes(req) == []
    busy = PFDataGenRequest(**base, timeline="window", load_level={"mode": "multiplier"}, load_noise_sigma=0.05,
                            outages={"der_trip_p": 0.1}, setpoints={"shunt_steps": "random"})
    assert timeline_mod.per_sample_axes(busy) == ["load_level", "load_noise", "capacitor_steps", "outages"]
    group = [{**_GEN, "capacity_mw": 1.0}]
    moving = PFDataGenRequest(**base, timeline="window", der={"source": "add", "groups": group, "layout": "per_sample"})
    assert timeline_mod.per_sample_axes(moving) == ["der_layout"]
    fixed = PFDataGenRequest(**base, timeline="window", der={"source": "add", "groups": group})
    assert fixed.der.layout == "per_run" and timeline_mod.per_sample_axes(fixed) == []      # follows the timeline
    warm = PFDataGenRequest(**base, timeline="window", parameters={"temperature": {"min": 40, "max": 40, "spread": 0}})
    assert timeline_mod.per_sample_axes(warm) == []
    ranged = PFDataGenRequest(**base, timeline="window", parameters={"temperature": {"min": 20, "max": 40}})
    assert timeline_mod.per_sample_axes(ranged) == ["parameters"]
    with pytest.raises(ValueError):
        MeasurementsConfig(profile="extreme")


def test_layout_meters_large_grid_der_unless_replaced():
    topo = _cp_chain()
    topo.ders = [MVDER("G1", "N3", "pv", 2000.0, 2000.0, "solar", 1.0, "fixed_pf"),
                 MVDER("G2", "N4", "pv", 400.0, 400.0, "solar", 1.0, "fixed_pf")]
    cfg = MeasurementsConfig(enabled=True)
    kept = meas_mod.build_layout(cfg, topo, 1, generated_der=False)
    assert [m.element_id for m in kept.meters if m.reason == "der"] == ["G1"]        # 2 MVA, not 0.4 MVA
    replaced = meas_mod.build_layout(cfg, topo, 1, generated_der=True, grid_der=False)
    assert not any(m.reason == "der" for m in replaced.meters)
    assert any(m.reason == "der_generated" for m in replaced.meters)
    rep = _fake_sampler()
    for i in range(50):
        _readings(rep, i)
    assert set(rep.report()["relative_error"]["medium"]) == {"v", "pqi"}


def test_start_run_writes_a_window_start_and_the_job_reads_it_back(tmp_path, monkeypatch):
    run_manager = importlib.import_module("gridarena.pf_datagen.run_manager")
    monkeypatch.setattr(run_manager, "RUNS_DIR", tmp_path)
    registry = run_manager.PFDataGenRunRegistry()
    monkeypatch.setattr(registry, "_pool", MagicMock())
    req = PFDataGenRequest(historical_database_id="h", timeline="window", timeline_start="2024-01-15T06:00",
                           measurements={"enabled": True})
    rid = registry.start_run("mv1", req)
    data = json.loads((tmp_path / rid / "run_config.json").read_text())
    assert data["timeline_start"] == "2024-01-15T06:00:00" and data["config"]["measurements"]["enabled"]
    again = PFDataGenRequest(**{k: data[k] for k in PFDataGenRequest.model_fields if k in data})
    assert again.timeline_start == datetime(2024, 1, 15, 6, 0) and again == req


# ── Synthetic Data section: pages ───────────────────────────────────────────


def test_synthetic_data_pages_render(monkeypatch):
    from fastapi.testclient import TestClient

    from gridarena.app import app
    from gridarena.pf_datagen import run_manager
    from gridarena.pf_datagen.schemas import PFDataGenRunStatus

    cursor = MagicMock()
    cursor.fetchall.return_value = [("G1", "Grid one", 15.0, "north", 20, 19, 8, 0),
                                    ("SYN-rural-0001", None, 20.0, "synthetic", 50, 49, 20, 0)]
    run = PFDataGenRunStatus(run_id="abcdef12", mv_grid_id="G1", status="completed", created_at="2026-10-19T12:00:00",
                             total_scenarios=10, completed_scenarios=9, failed_scenarios=1, group_id="abcdef34",
                             config={"scenario_count": 10})
    group = {"group_id": "abcdef34", "created_at": "2026-10-19T12:00:00", "seed": 3, "runs": [], "skipped": [],
             "config": {"scenario_count": 10}}
    reg = run_manager.pf_datagen_registry
    monkeypatch.setattr(reg, "list_runs", lambda mv_grid_id=None: [run])
    monkeypatch.setattr(reg, "get_run", lambda rid: run if rid == "abcdef12" else None)
    monkeypatch.setattr(reg, "list_groups", lambda: [{"group_id": "abcdef34", "created_at": group["created_at"],
                                                       "n_runs": 2, "n_skipped": 0}])
    monkeypatch.setattr(reg, "get_group", lambda gid: group if gid == "abcdef34" else None)
    with patch.object(db, "get_db_connection", return_value=(MagicMock(), cursor)),             patch("gridarena.settings_store.db_configured", return_value=True):
        client = TestClient(app)
        html = {"accept": "text/html"}
        r = client.get("/pf-datagen/ui", headers=html)
        assert r.status_code == 200 and "Synthetic Data Generation" in r.text
        assert 'href="/pf-datagen/ui/runs/abcdef12"' in r.text and 'href="/pf-datagen/ui/groups/abcdef34"' in r.text
        r = client.get("/pf-datagen/ui/new", headers=html)
        assert r.status_code == 200 and all(f'id="step{k}"' in r.text for k in range(1, 8))
        assert 'value="G1"' in r.text and "pf-datagen-wizard.js" in r.text
        assert 'href="/pf-datagen/ui/synthetic-grids"' in r.text            # nav entry and step 1 link
        r = client.get("/pf-datagen/ui/runs/abcdef12", headers=html)
        assert r.status_code == 200 and 'pfInitRunPage("abcdef12", "completed")' in r.text
        assert client.get("/pf-datagen/ui/runs/ffffffff", headers=html).status_code == 404
        r = client.get("/pf-datagen/ui/groups/abcdef34", headers=html)
        assert r.status_code == 200 and 'pfInitGroupPage("abcdef34")' in r.text
        assert client.get("/pf-datagen/ui/groups/ffffffff", headers=html).status_code == 404
        r = client.get("/pf-datagen/ui/synthetic-grids", headers=html)
        assert r.status_code == 200 and "SYN-rural-0001" in r.text and 'id="synthGenerate"' in r.text
        assert 'href="/mv_grid/ui/G1"' not in r.text
        r = client.get("/mv_grid/ui", headers=html)
        assert r.status_code == 200 and "pane-pfdatagen" not in r.text and "synthPanel" not in r.text


def test_registry_reports_progress_of_active_runs_from_disk(tmp_path, monkeypatch):
    run_manager = importlib.import_module("gridarena.pf_datagen.run_manager")
    monkeypatch.setattr(run_manager, "RUNS_DIR", tmp_path)
    registry = run_manager.PFDataGenRunRegistry()
    monkeypatch.setattr(registry, "_pool", MagicMock())
    rid = registry.start_run("mv1", PFDataGenRequest(historical_database_id="h"))
    cfg = tmp_path / rid / "run_config.json"
    data = json.loads(cfg.read_text())
    data.update(status="running", total_scenarios=800, completed_scenarios=300, n_topology_variants=4)
    cfg.write_text(json.dumps(data))
    r = registry.get_run(rid)                          # the worker's progress, not the start state
    assert (r.status, r.completed_scenarios, r.total_scenarios, r.n_topology_variants) == ("running", 300, 800, 4)
    assert registry.list_runs()[0].completed_scenarios == 300
    cfg.write_text('{"run_id": "' + rid + '", "stat')  # mid-write: keep the last state
    assert registry.get_run(rid).completed_scenarios == 300
    registry._runs[rid].status = "failed"              # a crash marked by the done callback stays
    data.update(status="running")
    cfg.write_text(json.dumps(data))
    assert registry.get_run(rid).status == "failed"


# ── Generated DER from a historical generation database (pf_datagen/der_profiles.py) ──

def test_job_with_a_historical_generation_profile(tmp_path):
    from gridarena.pf_datagen import der_profiles

    agg = np.array([20e6, 40e6])                         # the aggregate at the job's two timestamps
    prof = der_profiles.GenerationProfile(
        database_id="eredes_pv", data_type="pv", series_ids=["agg"], values_w=agg[None, :],
        ratings_w=np.array([40e6]), reference_year=2024, reference_max_w=40e6,
        gen_times=1.7e9 + 3600.0 * np.arange(2), step_s=900.0, shift_s=0.0, n_fallback=0)
    der = {"source": "replace", "layout": "per_run",
           "groups": [{"database_id": "eredes_pv", "count": 1, "capacity_mw": 0.5},
                      {"database_id": "eredes_pv", "count": 1, "capacity_mw": 1.0, "placement": "feeder_end"}]}
    with patch.object(der_profiles, "generation_step_s", return_value=900.0), \
            patch.object(der_profiles, "build_generation_profile", return_value=prof):
        result, out = _run_job(tmp_path, {"type": "none"}, der=der)
    assert result["status"] == "completed", result
    gen_cfg = PFDataGenRequest(historical_database_id="h", der=der).der.generation
    shares, _ = der_profiles.farm_shares(prof, [0, 1], gen_cfg, 7)
    rows = sorted(out["der"], key=lambda r: (r[1], r[3]))
    assert [r[DER_COL["der_type"]] for r in rows] == ["pv"] * 4
    for r in rows:
        k = int(r[3][-2:]) - 1
        i = 0 if r[1].startswith("2024-01-01 00") else 1
        cap = (0.5e6, 1.0e6)[k]
        assert r[DER_COL["p_available"]] == pytest.approx(cap * shares[k, i])
        assert 0 <= r[DER_COL["p"]] <= cap
    report = json.loads((tmp_path / "run" / "report.json").read_text())
    dp_ = report["der_profile"]
    assert [d["database_id"] for d in dp_["databases"]] == ["eredes_pv"] and len(dp_["groups"]) == 2
    assert [(p["der_id"], p["capacity_w"]) for p in dp_["plants"]] == [("GEN-PV-01", 0.5e6), ("GEN-PV-02", 1.0e6)]
    assert dp_["installed_capacity_w"] == 1.5e6 and dp_["rho_per_15min"] == 0.85
    assert "penetration" not in (report.get("design") or {}).get("axes", {})


@pytest.fixture(autouse=True)
def _fresh_reference_means():
    """The shape divisors are cached per process: start every test without them."""
    importlib.import_module("gridarena.pf_datagen.scenario_builder")._REFERENCE_MEANS.clear()
