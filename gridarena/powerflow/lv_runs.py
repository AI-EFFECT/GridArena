"""Power flow runs on stored LV grids: load the inputs, solve, store, compare.

The inputs come from the grid tables (topology, per-phase cable impedances)
and from "Measurements": the measured active/reactive power of every node and
the measured voltage of the substation node "PT", which is the reference.
The voltages the power flow computes are stored in "PowerFlowResults", apart
from the measured ones; ``comparison`` lines both up per node, phase and
timestamp, so the model can be checked against the meters.
"""

import json
import math
from collections import defaultdict
from typing import Optional

import numpy as np

import gridarena.database as db
from gridarena.powerflow import three_phase as tp

ROOT_NODE = "PT"
MODES = ("unbalanced", "balanced")


class PowerFlowInputError(ValueError):
    """The grid or its measurements can't be solved (shown to the user)."""


def _load_grid(cursor, grid_id):
    """Node ids (root first), connections as index pairs, (n_conn, 3) phase impedances in ohm."""
    cursor.execute('SELECT DISTINCT "NodeId" FROM "Node" WHERE grid_id = %s', (grid_id,))
    node_ids = sorted(row[0] for row in cursor.fetchall())
    if not node_ids:
        raise PowerFlowInputError(f"Grid '{grid_id}' has no nodes.")
    if ROOT_NODE not in node_ids:
        raise PowerFlowInputError(f"Grid '{grid_id}' has no substation node '{ROOT_NODE}' to use as reference.")
    node_ids.remove(ROOT_NODE)
    node_ids.insert(0, ROOT_NODE)
    index = {n: i for i, n in enumerate(node_ids)}

    cursor.execute(
        """
        SELECT c."FromNodeId", c."ToNodeId", c."Length",
               k."RImpReal", k."RImpImag", k."SImpReal", k."SImpImag", k."TImpReal", k."TImpImag"
        FROM "Connection" c
        JOIN "Cable" k ON k."CableId" = c."CableId" AND k.grid_id = c.grid_id
        WHERE c.grid_id = %s
        """,
        (grid_id,),
    )
    rows = cursor.fetchall()
    if not rows:
        raise PowerFlowInputError(f"Grid '{grid_id}' has no connections with cable data.")

    connections, z = [], []
    for from_node, to_node, length, *imp in rows:
        km = float(length) / 1000.0  # impedances are ohm/km, lengths metres
        connections.append((index[from_node], index[to_node]))
        z.append([complex(imp[0], imp[1]) * km, complex(imp[2], imp[3]) * km, complex(imp[4], imp[5]) * km])
    return node_ids, connections, np.array(z, dtype=complex)


def load_radial_grid(cursor, grid_id, neutral_impedance_ratio=0.0):
    """Node ids (root first) and the solvable RadialGrid of a stored grid.

    The default ratio 0 makes the phases independent, as the per-phase power
    flow of the digital twins expects (see three_phase.solve_single_phase).
    Raises PowerFlowInputError, or NotRadialError when the cables form a loop.
    """
    node_ids, connections, z = _load_grid(cursor, grid_id)
    return node_ids, tp.build_radial_grid(len(node_ids), connections, z, neutral_ratio=neutral_impedance_ratio)


def _load_measurements(cursor, grid_id, start_time, end_time):
    query = """
        SELECT node_id, phase, datetime, power_active, power_reactive, voltage_magnitude, voltage_angle
        FROM "Measurements"
        WHERE grid_id = %s
    """
    params = [grid_id]
    if start_time:
        query += " AND datetime >= %s"
        params.append(start_time)
    if end_time:
        query += " AND datetime <= %s"
        params.append(end_time)
    cursor.execute(query + " ORDER BY datetime", params)
    return cursor.fetchall()


def _root_voltages(pt_rows):
    """Reference phase voltages at one timestamp from the PT measurements
    ({phase: (magnitude, angle_deg)}). A phase PT doesn't measure gets the mean
    measured magnitude at its nominal angle relative to R. None if PT has no
    usable voltage at all."""
    usable = {ph: v for ph, v in pt_rows.items() if v[0] is not None}
    if not usable:
        return None
    mean_mag = sum(v[0] for v in usable.values()) / len(usable)
    if "R" in usable and usable["R"][1] is not None:
        r_angle = usable["R"][1]
    else:
        ph, (mag, ang) = next(iter(usable.items()))
        r_angle = (ang or 0.0) - tp.NOMINAL_ANGLES_DEG[tp.PHASES.index(ph)]
    out = []
    for p, phase in enumerate(tp.PHASES):
        mag, ang = usable.get(phase, (mean_mag, None))
        if ang is None:
            ang = r_angle + tp.NOMINAL_ANGLES_DEG[p]
        out.append(mag * np.exp(1j * math.radians(ang)))
    return out


def run(cursor, grid_id: str, mode: str = "unbalanced", neutral_impedance_ratio: float = 1.0,
        start_time: Optional[str] = None, end_time: Optional[str] = None) -> dict:
    """Solve every measurement timestamp of the grid (in the range) and replace
    the grid's stored results. Returns the run summary."""
    if mode not in MODES:
        raise PowerFlowInputError(f"mode must be one of {', '.join(MODES)}.")
    if neutral_impedance_ratio is None or not neutral_impedance_ratio >= 0:
        raise PowerFlowInputError("neutral_impedance_ratio must be a number >= 0.")

    node_ids, connections, z = _load_grid(cursor, grid_id)
    index = {n: i for i, n in enumerate(node_ids)}
    ratio = neutral_impedance_ratio if mode == "unbalanced" else 0.0
    try:
        grid = tp.build_radial_grid(len(node_ids), connections, z, neutral_ratio=ratio, root=0)
    except tp.NotRadialError as e:
        raise PowerFlowInputError(str(e))

    rows = _load_measurements(cursor, grid_id, start_time, end_time)
    if not rows:
        raise LookupError("No measurements found for this grid in the given range.")

    loads = defaultdict(dict)    # datetime -> {(node index, phase index): S}
    pt = defaultdict(dict)       # datetime -> {phase: (magnitude, angle)}
    for node_id, phase, dt, p, q, v_mag, v_ang in rows:
        if node_id not in index or phase not in tp.PHASES:
            continue
        loads[dt][(index[node_id], tp.PHASES.index(phase))] = complex(p or 0.0, q or 0.0)
        if node_id == ROOT_NODE:
            pt[dt][phase] = (v_mag, v_ang)

    timestamps, s_load, v_root = [], [], []
    skipped = 0
    for dt in sorted(loads):
        ref = _root_voltages(pt.get(dt, {}))
        if ref is None:
            skipped += 1
            continue
        s = np.zeros((len(node_ids), 3), dtype=complex)
        for (i, p), value in loads[dt].items():
            s[i, p] = value
        timestamps.append(dt)
        s_load.append(s)
        v_root.append(ref)
    if not timestamps:
        raise PowerFlowInputError(
            f"No timestamp has a voltage measurement at the substation node '{ROOT_NODE}' to use as reference.")

    s_load, v_root = np.array(s_load), np.array(v_root)
    if mode == "balanced":
        s_load, v_root = tp.balanced_inputs(s_load, v_root)
    result = tp.solve(grid, s_load, v_root, n_nodes=len(node_ids))

    out_rows = []
    reachable = set(grid.nodes.tolist())
    for t, dt in enumerate(timestamps):
        stamp, ok = str(dt), bool(result.converged[t])
        for i, node_id in enumerate(node_ids):
            if i not in reachable:
                continue
            for p, phase in enumerate(tp.PHASES):
                v = result.v_phase[t, i, p]
                out_rows.append((node_id, phase, stamp, float(v.real), float(v.imag), ok))
            if ratio > 0:
                vn = result.v_neutral[t, i]
                out_rows.append((node_id, "N", stamp, float(vn.real), float(vn.imag), ok))

    summary = {
        "mode": mode,
        "neutral_impedance_ratio": ratio if mode == "unbalanced" else None,
        "start_time": start_time,
        "end_time": end_time,
        "timestamps": len(timestamps),
        "not_converged": int((~result.converged).sum()),
        "skipped_timestamps": skipped,
        "unreachable_nodes": [node_ids[i] for i in grid.unreachable],
    }
    db.replace_pf_run(cursor, grid_id, summary, out_rows)
    return summary


def get_run(cursor, grid_id: str) -> Optional[dict]:
    cursor.execute(
        """
        SELECT mode, neutral_impedance_ratio, start_time, end_time, timestamps,
               not_converged, skipped_timestamps, unreachable_nodes, created_at
        FROM "PowerFlowRuns" WHERE grid_id = %s
        """,
        (grid_id,),
    )
    row = cursor.fetchone()
    if row is None:
        return None
    keys = ("mode", "neutral_impedance_ratio", "start_time", "end_time", "timestamps",
            "not_converged", "skipped_timestamps", "unreachable_nodes", "created_at")
    run_info = dict(zip(keys, row))
    run_info["unreachable_nodes"] = json.loads(run_info["unreachable_nodes"] or "[]")
    run_info["created_at"] = run_info["created_at"].isoformat() if run_info["created_at"] else None
    return run_info


def _polar(real, imag):
    return math.hypot(real, imag), math.degrees(math.atan2(imag, real))


def comparison(cursor, grid_id: str, phase: Optional[str] = None) -> list:
    """Computed voltages with the measured ones next to them, per node, phase
    and timestamp. ``measured`` is None where no meter measured that node and
    phase at that time (always for phase N, the neutral)."""
    query = """
        SELECT node_id, phase, datetime, voltage_real, voltage_imag, converged
        FROM "PowerFlowResults" WHERE grid_id = %s
    """
    params = [grid_id]
    if phase:
        query += " AND phase = %s"
        params.append(phase)
    cursor.execute(query + " ORDER BY datetime, node_id, phase", params)
    computed = cursor.fetchall()
    if not computed:
        return []

    query = """
        SELECT node_id, phase, datetime, voltage_magnitude, voltage_angle
        FROM "Measurements" WHERE grid_id = %s AND voltage_magnitude IS NOT NULL
    """
    params = [grid_id]
    if phase:
        query += " AND phase = %s"
        params.append(phase)
    cursor.execute(query, params)
    measured = {(n, ph, str(dt)): (mag, ang) for n, ph, dt, mag, ang in cursor.fetchall()}

    out = []
    for node_id, ph, stamp, real, imag, converged in computed:
        mag, ang = _polar(real, imag)
        meas = measured.get((node_id, ph, stamp))
        out.append({
            "timestamp": stamp,
            "node_id": node_id,
            "phase": ph,
            "converged": converged,
            "computed": {"voltage_magnitude": mag, "voltage_angle": ang, "real": real, "imag": imag},
            "measured": None if meas is None else {"voltage_magnitude": meas[0], "voltage_angle": meas[1]},
        })
    return out


def error_summary(results: list) -> list:
    """Per node and phase: how far the computed voltages are from the measured
    ones (computed - measured), over converged timestamps with a measurement.
    The substation is skipped: it is the reference, equal by construction."""
    groups = defaultdict(list)
    for r in results:
        if r["measured"] is None or not r["converged"] or r["node_id"] == ROOT_NODE:
            continue
        m = r["measured"]
        if m["voltage_magnitude"] is None:
            continue
        d_mag = r["computed"]["voltage_magnitude"] - m["voltage_magnitude"]
        d_ang = None
        if m["voltage_angle"] is not None:
            d_ang = (r["computed"]["voltage_angle"] - m["voltage_angle"] + 180.0) % 360.0 - 180.0
        groups[(r["node_id"], r["phase"])].append((d_mag, d_ang))

    summary = []
    for (node_id, ph), diffs in sorted(groups.items()):
        mags = np.array([d for d, _ in diffs])
        angs = np.array([a for _, a in diffs if a is not None])
        summary.append({
            "node_id": node_id,
            "phase": ph,
            "points": len(diffs),
            "mae_voltage_magnitude": round(float(np.abs(mags).mean()), 4),
            "max_abs_voltage_magnitude": round(float(np.abs(mags).max()), 4),
            "bias_voltage_magnitude": round(float(mags.mean()), 4),
            "mae_voltage_angle": round(float(np.abs(angs).mean()), 4) if len(angs) else None,
        })
    return summary
