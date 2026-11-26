"""Topology discovery as a code-submission benchmark.

The user's script defines::

    predict(grid: dict) -> list[[node_key, node_key], ...]

and is called once per test grid with every node of the grid (also the ones
without measurements, which the topology still has to connect), the
substation node and the measurements per node. It returns the grid's edges.
Node keys are random per submission, so they carry no hint of the wiring;
the true edges stay here, in ``PreparedInputs.truth``.
"""

import hashlib
import random
import secrets
from collections import defaultdict

import gridarena.database as db
from gridarena.benchmarks.general_noise import corrupt_measurement_record, get_profile
from gridarena.benchmarks.submissions import BenchmarkTask, PreparedInputs

SUBSTATION_NODE_ID = "PT"


def _load_test_grids():
    """(grid_ids, nodes per grid, edges per grid, measurement rows) of the test grids."""
    conn, cursor = db.get_db_connection()
    try:
        cursor.execute("SELECT grid_id FROM GridUsage WHERE topology_detection_test = 1")
        grid_ids = sorted(row[0] for row in cursor.fetchall())
        if not grid_ids:
            raise Exception("No grids are marked for topology discovery testing.")

        cursor.execute('SELECT grid_id, "NodeId" FROM "Node" WHERE grid_id = ANY(%s)', (grid_ids,))
        nodes = defaultdict(list)
        for grid_id, node_id in cursor.fetchall():
            nodes[grid_id].append(node_id)

        cursor.execute(
            'SELECT grid_id, "FromNodeId", "ToNodeId" FROM "Connection" WHERE grid_id = ANY(%s)', (grid_ids,),
        )
        edges = defaultdict(list)
        for grid_id, from_node, to_node in cursor.fetchall():
            edges[grid_id].append((from_node, to_node))

        cursor.execute(
            """
            SELECT grid_id, node_id, phase, datetime, power_active, power_reactive,
                   voltage_magnitude, voltage_angle
            FROM "Measurements"
            WHERE grid_id = ANY(%s)
            ORDER BY grid_id, node_id, phase, datetime
            """,
            (grid_ids,),
        )
        measurements = cursor.fetchall()
        return grid_ids, nodes, edges, measurements
    finally:
        conn.close()


def build_test_inputs(difficulty: str, grids=None) -> PreparedInputs:
    """One call per test grid: ``predict({"nodes", "root", "measurements"})``.

    ``grids`` (for tests) replaces the database: ``(grid_ids, nodes, edges, measurements)``
    as returned by ``_load_test_grids``. Noise is seeded per (grid, node, phase,
    difficulty), so every submission at a given difficulty sees the same values."""
    grid_ids, nodes, edges, measurements = grids if grids is not None else _load_test_grids()
    profile = get_profile(difficulty)

    records = defaultdict(lambda: defaultdict(list))  # grid -> node -> records
    rngs = {}
    for grid_id, node_id, phase, dt, p_act, p_rea, v_mag, v_ang in measurements:
        key = (grid_id, node_id, phase)
        if key not in rngs:
            seed = hashlib.sha256(f"{grid_id}:{node_id}:{phase}:{difficulty}".encode()).hexdigest()
            rngs[key] = random.Random(seed)
        record = {
            "datetime": dt.isoformat() if hasattr(dt, "isoformat") else str(dt),
            "phase": phase,
            "power_active": p_act,
            "power_reactive": p_rea,
            "voltage_magnitude": v_mag,
            "voltage_angle": v_ang,
        }
        records[grid_id][node_id].append(corrupt_measurement_record(record, rngs[key], profile))

    shuffler = random.SystemRandom()
    calls, truth = {}, {}
    for i, grid_id in enumerate(g for g in grid_ids if nodes.get(g)):
        node_ids = list(dict.fromkeys(nodes[grid_id]))
        shuffler.shuffle(node_ids)
        keys = {}
        for node_id in node_ids:
            key = secrets.token_hex(6)
            while key in keys.values():
                key = secrets.token_hex(6)
            keys[node_id] = key

        true_edges = sorted({
            tuple(sorted((keys[a], keys[b])))
            for a, b in edges.get(grid_id, []) if a in keys and b in keys and a != b
        })
        name = f"{i:04d}"
        calls[name] = [{
            "nodes": [keys[n] for n in node_ids],
            "root": keys.get(SUBSTATION_NODE_ID),
            "measurements": {keys[n]: recs for n, recs in records[grid_id].items() if n in keys},
        }]
        truth[name] = {"grid_id": grid_id, "nodes": sorted(keys.values()), "edges": [list(e) for e in true_edges]}
    return PreparedInputs(calls=calls, truth=truth)


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def score_predictions(outputs: dict, truth: dict) -> dict:
    """Precision, recall and F1 of the predicted edges (undirected) against the
    true ones, per grid and over all grids (micro-averaged). A grid whose call
    raised predicts no edges."""
    per_grid = []
    tp_all = fp_all = fn_all = 0
    runtime = 0.0

    for name in sorted(truth):
        nodes = set(truth[name]["nodes"])
        true_edges = {tuple(e) for e in truth[name]["edges"]}
        out = outputs.get(name, {})
        runtime += out.get("seconds") or 0.0
        row = {"grid_id": truth[name]["grid_id"], "nodes": len(nodes), "true_edges": len(true_edges),
               "predicted_edges": 0, "invalid": 0, "seconds": out.get("seconds"), "error": None}

        predicted = set()
        prediction = out.get("result") if out.get("ok") else None
        if not out.get("ok"):
            lines = (out.get("error") or "predict() failed").strip().splitlines()
            row["error"] = lines[-1] if lines else "predict() failed"
        elif not isinstance(prediction, list):
            row["error"] = f"predict() must return a list of [node, node] pairs, got {type(prediction).__name__}"
        else:
            for edge in prediction:
                if (isinstance(edge, list) and len(edge) == 2 and all(isinstance(n, str) for n in edge)
                        and edge[0] != edge[1] and edge[0] in nodes and edge[1] in nodes):
                    predicted.add(tuple(sorted(edge)))
                else:
                    row["invalid"] += 1

        tp = len(predicted & true_edges)
        fp = len(predicted - true_edges)
        fn = len(true_edges - predicted)
        precision = tp / (tp + fp) if predicted else 0.0
        recall = tp / (tp + fn) if true_edges else 0.0
        row.update(predicted_edges=len(predicted), correct=tp, extra=fp, missed=fn,
                   precision=round(precision, 4), recall=round(recall, 4), f1=round(_f1(precision, recall), 4))
        per_grid.append(row)
        tp_all, fp_all, fn_all = tp_all + tp, fp_all + fp, fn_all + fn

    precision = tp_all / (tp_all + fp_all) if tp_all + fp_all else 0.0
    recall = tp_all / (tp_all + fn_all) if tp_all + fn_all else 0.0
    return {
        "f1": round(_f1(precision, recall), 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "correct": tp_all,
        "extra": fp_all,
        "missed": fn_all,
        "grids": len(per_grid),
        "failed_grids": sum(1 for r in per_grid if r["error"]),
        "runtime_seconds": round(runtime, 3),
        "per_grid": per_grid,
    }


TOPOLOGY_TASK = BenchmarkTask(
    name="topology_discovery",
    function="predict",
    build_inputs=lambda settings: build_test_inputs(settings["difficulty"]),
    score=score_predictions,
)
