"""Voltage control as a code-submission benchmark.

The user's script defines::

    predict(snapshot: dict) -> [{"node_id", "phase", "corrected_voltage",
                                 "adjusted_power_active", "adjusted_power_reactive"}, ...]

and is called once per test snapshot: a test grid at a timestamp where at
least one node violates the voltage limit. It returns its corrective actions
in the same format as the reference "solutions" of the training data. The
snapshots are the ones the old /voltage-control-data endpoint served (with
reference solutions preferred, filtered by hardness for the difficulty); the
reference solutions stay here, in ``PreparedInputs.truth``.

Scoring is the old /score formula, per snapshot, with two changes so that a
script can't improve its score by answering less:
  * every measured (node, phase) of the snapshot is scored; one the script
    doesn't return is taken as left unchanged (measured voltage and power);
  * every snapshot counts, including those the script returns nothing for.
"""

from collections import defaultdict
from typing import Any, Dict, List

import gridarena.benchmarks.voltage_control.get_measurement_data as vc
import gridarena.database as db
from gridarena.benchmarks.submissions import BenchmarkTask, PreparedInputs

# Reference-based scoring tunables (unchanged from the old /score endpoint).
V_TOL = 1.0  # volts: "close enough to the reference voltages"
V_REF = 1.0  # volts: how fast the voltage factor decays when missing them
K = 2.0      # steepness

_SOLUTION_QUERY = """
    SELECT node_id, phase, corrected_voltage, adjusted_power_active, adjusted_power_reactive
    FROM "VoltageControlSolutions"
    WHERE grid_id = %s AND datetime = %s
"""


def _number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and x == x and abs(x) != float("inf")


def load_test_snapshots(voltage_limit: float, scenario: str, difficulty: str) -> List[Dict[str, Any]]:
    """The test snapshots for these settings: ``[{"grid_id", "datetime", "measurements", "solutions"}]``."""
    conn, cursor = db.get_db_connection()
    try:
        cursor.execute("SELECT grid_id FROM GridUsage WHERE voltage_control_test = 1")
        grids_for_testing = {row[0] for row in cursor.fetchall()}
        if not grids_for_testing:
            raise Exception("No grids are marked for voltage control testing.")

        fetch = vc.get_all_measurements_over if scenario == "Overvoltages" else vc.get_all_measurements_under
        rows = fetch(cursor, voltage_limit, grids_for_testing, phase_type="test")

        snapshots = defaultdict(list)
        for grid_id, node_id, phase, dt, p_act, p_rea, v_mag, v_ang in rows:
            snapshots[(grid_id, dt)].append({
                "node_id": node_id, "phase": phase,
                "power_active": p_act, "power_reactive": p_rea,
                "voltage_magnitude": v_mag, "voltage_angle": v_ang,
            })

        complexity = {}
        solutions, hardness = {}, {}
        for key, meas in snapshots.items():
            grid_id, dt = key
            if grid_id not in complexity:
                complexity[grid_id] = vc.get_grid_complexity_metrics(cursor, grid_id)
            cursor.execute(_SOLUTION_QUERY, (grid_id, dt))
            solutions[key] = [
                {"node_id": r[0], "phase": r[1], "corrected_voltage": r[2],
                 "adjusted_power_active": r[3], "adjusted_power_reactive": r[4]}
                for r in cursor.fetchall()
            ]
            metrics = vc.compute_snapshot_metrics(
                measurements=meas,
                solutions=solutions[key] or None,
                nominal_voltage=vc.NOMINAL_VOLTAGE,
                band=vc.VOLTAGE_BAND,
                grid_complexity_norm=float(complexity[grid_id]["grid_complexity_norm"]),
                grid_complexity_profile=complexity[grid_id],
            )
            hardness[key] = float(metrics["hardness_score"])
    finally:
        conn.close()

    # Like the old endpoint: prefer snapshots that have reference solutions,
    # then keep the hardness band of the difficulty.
    with_solutions = [k for k in snapshots if solutions[k]]
    population = with_solutions or list(snapshots)
    selected, _ = vc.difficulty_filter_keys({k: hardness[k] for k in population}, difficulty)
    return [
        {"grid_id": k[0], "datetime": k[1], "measurements": snapshots[k], "solutions": solutions[k]}
        for k in sorted(selected, key=lambda k: (k[0], str(k[1])))
    ]


def build_test_inputs(settings: dict, snapshots=None) -> PreparedInputs:
    """One call per snapshot: ``predict({"datetime", "scenario", "voltage_limit",
    "nominal_voltage", "voltage_band", "measurements"})``."""
    if snapshots is None:
        snapshots = load_test_snapshots(settings["voltage_limit"], settings["scenario"], settings["difficulty"])
    if not snapshots:
        raise Exception("No test snapshots match these settings.")

    calls, truth = {}, {}
    for i, snap in enumerate(snapshots):
        dt = snap["datetime"]
        dt_text = dt.isoformat() if hasattr(dt, "isoformat") else str(dt)
        name = f"{i:04d}"
        calls[name] = [{
            "datetime": dt_text,
            "scenario": settings["scenario"],
            "voltage_limit": settings["voltage_limit"],
            "nominal_voltage": vc.NOMINAL_VOLTAGE,
            "voltage_band": vc.VOLTAGE_BAND,
            "measurements": snap["measurements"],
        }]
        truth[name] = {"grid_id": snap["grid_id"], "datetime": dt_text,
                       "measurements": snap["measurements"], "solutions": snap["solutions"]}
    return PreparedInputs(calls=calls, truth=truth)


def score_snapshot(measurements: list, solutions: list, prediction) -> dict:
    """Score one snapshot's corrective actions (the old /score formula, see module docstring)."""
    nominal = vc.NOMINAL_VOLTAGE
    low, high = (1.0 - vc.VOLTAGE_BAND) * nominal, (1.0 + vc.VOLTAGE_BAND) * nominal

    actions, invalid = {}, 0
    for entry in prediction if isinstance(prediction, list) else []:
        if (isinstance(entry, dict) and entry.get("node_id") is not None
                and _number(entry.get("corrected_voltage"))):
            actions[(str(entry["node_id"]), str(entry.get("phase")))] = entry
        else:
            invalid += 1

    ref_v = {(str(s["node_id"]), str(s["phase"])): float(s["corrected_voltage"])
             for s in solutions or [] if s.get("corrected_voltage") is not None}

    node_scores = []
    sum_abs_dp = sum_abs_p = sum_abs_v_err = 0.0
    v_err_count = changed = 0
    for m in measurements:
        if m.get("voltage_magnitude") is None:
            continue
        key = (str(m["node_id"]), str(m["phase"]))
        v_meas = float(m["voltage_magnitude"])
        p_meas = float(m.get("power_active") or 0.0)
        action = actions.get(key)
        v_corr = float(action["corrected_voltage"]) if action else v_meas
        p_adj = action.get("adjusted_power_active") if action else None
        p_corr = float(p_adj) if _number(p_adj) else p_meas
        changed += v_corr != v_meas or p_corr != p_meas

        sum_abs_dp += abs(p_corr - p_meas)
        sum_abs_p += abs(p_meas)
        if key in ref_v:
            sum_abs_v_err += abs(v_corr - ref_v[key])
            v_err_count += 1

        before_dev, after_dev = abs(v_meas - nominal), abs(v_corr - nominal)
        voltage_score = max(min((before_dev - after_dev) / (before_dev + 1e-6), 1.0), -1.0)
        power_penalty = 1.0 - min(1.0, abs(p_corr - p_meas) / abs(p_meas)) if abs(p_meas) > 1e-6 else 1.0
        bonus = 1.0 if (v_meas < low or v_meas > high) and low <= v_corr <= high else 0.0
        node_scores.append(0.7 * max(voltage_score, 0.0) + 0.2 * power_penalty + 0.1 * bonus)

    result = {"nodes": len(node_scores), "changed_nodes": changed, "invalid": invalid}
    if not node_scores:
        return {**result, "mode": "no_measurements", "timestamp_score": 0.0}

    effort_norm = sum_abs_dp / (sum_abs_p + 1e-6)
    effort = vc.effort_factor(effort_norm)
    result["effort_norm"] = round(effort_norm, 6)
    if v_err_count:
        v_mae = sum_abs_v_err / v_err_count
        if v_mae <= V_TOL:
            # Once the reference voltages are matched, only effort matters.
            mode, score = "ref_voltage_then_effort", effort
        else:
            mode = "ref_voltage_miss_penalised"
            score = effort / (1.0 + (v_mae / V_REF) ** K)
        result["reference_voltage_mae"] = round(v_mae, 6)
    else:
        mode, score = "outcome_only_fallback", (sum(node_scores) / len(node_scores)) * effort
    return {**result, "mode": mode, "timestamp_score": round(score, 4)}


def score_predictions(outputs: dict, truth: dict) -> dict:
    """Node-weighted mean of the snapshot scores (0-1, higher is better), per grid and overall."""
    per_snapshot = []
    for name in sorted(truth):
        t = truth[name]
        out = outputs.get(name, {})
        prediction = out.get("result") if out.get("ok") else None
        error = None
        if not out.get("ok"):
            lines = (out.get("error") or "predict() failed").strip().splitlines()
            error = lines[-1] if lines else "predict() failed"
        elif not isinstance(prediction, list):
            error = f"predict() must return a list of node actions, got {type(prediction).__name__}"
            prediction = None
        row = score_snapshot(t["measurements"], t["solutions"], prediction)
        per_snapshot.append({"grid_id": t["grid_id"], "datetime": t["datetime"], **row,
                             "seconds": out.get("seconds"), "error": error})

    def weighted(rows):
        nodes = sum(r["nodes"] for r in rows)
        return round(sum(r["timestamp_score"] * r["nodes"] for r in rows) / nodes, 4) if nodes else 0.0

    by_grid = defaultdict(list)
    for row in per_snapshot:
        by_grid[row["grid_id"]].append(row)
    per_grid = [{
        "grid_id": grid_id,
        "snapshots": len(rows),
        "nodes": sum(r["nodes"] for r in rows),
        "score": weighted(rows),
        "failed_snapshots": sum(1 for r in rows if r["error"]),
        "seconds": round(sum(r["seconds"] or 0.0 for r in rows), 3),
        "error": next((r["error"] for r in rows if r["error"]), None),
    } for grid_id, rows in sorted(by_grid.items())]

    return {
        "score": weighted(per_snapshot),
        "snapshots": len(per_snapshot),
        "nodes": sum(r["nodes"] for r in per_snapshot),
        "failed_snapshots": sum(1 for r in per_snapshot if r["error"]),
        "grids": len(per_grid),
        "runtime_seconds": round(sum(r["seconds"] or 0.0 for r in per_snapshot), 3),
        "per_grid": per_grid,
        "per_snapshot": per_snapshot,
    }


VOLTAGE_CONTROL_TASK = BenchmarkTask(
    name="voltage_control",
    function="predict",
    build_inputs=build_test_inputs,
    score=score_predictions,
)
