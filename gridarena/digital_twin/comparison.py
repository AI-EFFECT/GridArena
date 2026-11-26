"""Power-flow comparison: run PF on streamed data and compare against true measurements."""

import logging
import math
import time

import numpy as np

import gridarena.database as db
from gridarena.digital_twin.schemas import StreamedMeasurementBatch
from gridarena.powerflow import lv_runs
from gridarena.powerflow import three_phase as tp

logger = logging.getLogger(__name__)


def run_comparison(
    batch: StreamedMeasurementBatch,
    phase: str,
    voltage_ref: float = 230.0,
) -> dict:
    """Run power flow using streamed measurements and compare against true voltages.

    The power flow is solved on ``phase`` with that phase's cable impedances,
    from the substation node "PT", whatever direction the grid's connections
    are stored in. Returns a metrics dict suitable for insert_digital_twin_result().
    """
    t0 = time.time()
    grid_id = batch.grid_id
    timestamp = batch.timestamp

    phase_measurements = [m for m in batch.measurements if m.phase == phase or m.phase is None]

    metrics = {
        "timestamp": timestamp,
        "measurements_count": len(phase_measurements),
        "powerflow_algorithm": "backward_forward_sweep",
    }

    conn, cursor = db.get_db_connection()
    try:
        node_ids, radial = lv_runs.load_radial_grid(cursor, grid_id)
    except (lv_runs.PowerFlowInputError, tp.NotRadialError) as e:
        logger.error("Power flow failed for grid %s: %s", grid_id, e)
        metrics["convergence_status"] = "failed"
        metrics["details"] = {"error": str(e)}
        metrics["execution_time_ms"] = round((time.time() - t0) * 1000, 2)
        return metrics
    finally:
        conn.close()

    node_id_to_index = {node_id: idx for idx, node_id in enumerate(node_ids)}
    n_nodes = len(node_ids)
    S_array = np.zeros(n_nodes, dtype=complex)
    true_voltages = {}
    missing_count = 0
    invalid_count = 0
    mapped_nodes = set()

    for m in phase_measurements:
        idx = node_id_to_index.get(m.node_id)
        if idx is None:
            invalid_count += 1
            continue
        mapped_nodes.add(m.node_id)

        try:
            p = float(m.power_active)
            q = float(m.power_reactive)
            S_array[idx] = complex(p, q)
        except (ValueError, TypeError):
            invalid_count += 1
            continue

        if m.voltage_magnitude is not None and m.voltage_magnitude > 0:
            true_voltages[m.node_id] = m.voltage_magnitude

    for node_id in node_id_to_index:
        if node_id not in mapped_nodes and node_id != lv_runs.ROOT_NODE:
            missing_count += 1

    reference_voltage = voltage_ref
    pt_measurement = next((m for m in phase_measurements if m.node_id == lv_runs.ROOT_NODE), None)
    if pt_measurement and pt_measurement.voltage_magnitude > 0:
        reference_voltage = pt_measurement.voltage_magnitude

    metrics["missing_measurements"] = missing_count
    metrics["invalid_measurements"] = invalid_count

    try:
        volt_values, converged = tp.solve_single_phase(radial, S_array, phase, reference_voltage)
        metrics["convergence_status"] = "converged" if converged else "not_converged"

        # Nodes not connected to PT have no voltage (NaN) and are left out.
        simulated_voltages = {}
        for node_id, idx in node_id_to_index.items():
            if not np.isnan(volt_values[idx]):
                simulated_voltages[node_id] = float(abs(volt_values[idx]))

        # PT is left out: its voltage is the reference the power flow starts from.
        voltage_errors = []
        for node_id, true_v in true_voltages.items():
            if node_id in simulated_voltages and node_id != lv_runs.ROOT_NODE:
                err = abs(simulated_voltages[node_id] - true_v)
                voltage_errors.append(err)

        if voltage_errors:
            metrics["voltage_mae"] = sum(voltage_errors) / len(voltage_errors)
            metrics["voltage_rmse"] = math.sqrt(sum(e ** 2 for e in voltage_errors) / len(voltage_errors))
            metrics["max_voltage_error"] = max(voltage_errors)

        p_errors = []
        for m in phase_measurements:
            idx = node_id_to_index.get(m.node_id)
            if idx is not None and m.node_id != lv_runs.ROOT_NODE:
                sim_v = volt_values[idx]
                if abs(sim_v) > 0:
                    sim_s = S_array[idx]
                    p_err = abs(sim_s.real - m.power_active)
                    p_errors.append(p_err)

        if p_errors:
            metrics["active_power_error"] = sum(p_errors) / len(p_errors)

        power_by_node = {}
        for m in phase_measurements:
            power_by_node[m.node_id] = {"power_active": m.power_active, "power_reactive": m.power_reactive}

        details = {}
        for node_id, sim_v in simulated_voltages.items():
            entry = {"simulated_voltage": round(sim_v, 4)}
            if node_id in true_voltages:
                entry["true_voltage"] = true_voltages[node_id]
                entry["error"] = round(abs(sim_v - true_voltages[node_id]), 4)
            if node_id in power_by_node:
                entry["power_active"] = power_by_node[node_id]["power_active"]
                entry["power_reactive"] = power_by_node[node_id]["power_reactive"]
            details[node_id] = entry
        metrics["details"] = details

    except Exception as e:
        logger.error("Power flow failed for grid %s: %s", grid_id, e)
        metrics["convergence_status"] = "failed"
        metrics["details"] = {"error": str(e)}

    elapsed_ms = (time.time() - t0) * 1000
    metrics["execution_time_ms"] = round(elapsed_ms, 2)

    return metrics
