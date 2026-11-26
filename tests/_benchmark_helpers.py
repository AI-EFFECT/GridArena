"""Shared helpers for the code-submission benchmark tests."""

import io
import json
import time

import gridarena.database as db

USAGE_FLAGS = [
    f"{task}_{split}"
    for task in ("phase_detection", "topology_detection", "voltage_control", "state_estimation")
    for split in ("test", "train")
]

CABLE = {
    "cable_id": "TypeLine00001",
    "r_imp_real": 0.01, "r_imp_imag": 0.001,
    "s_imp_real": 0.01, "s_imp_imag": 0.001,
    "t_imp_real": 0.01, "t_imp_imag": 0.001,
    "r_nom_curr": 10000, "s_nom_curr": 10000, "t_nom_curr": 10000,
}


def wait_for(registry, submission_id, benchmark, timeout=60):
    """Poll a registry until the submission is completed or failed."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        record = registry.get(submission_id, benchmark)
        if record["status"] in ("completed", "failed"):
            return record
        time.sleep(0.1)
    raise AssertionError(f"submission {submission_id} did not finish within {timeout} s")


def register_grid(client, grid_id, edges, measurements, usage):
    """Register a grid (nodes from ``edges``) with the given usage flags, plus its
    measurements: ``{node_id: [{datetime, phase, power_active, ...}, ...]}``."""
    node_ids = list(dict.fromkeys(n for edge in edges for n in edge))
    grid = {
        "grid_id": grid_id,
        "nodes": [{"node_id": n, "coord_lat": 38.65 + i / 1000, "coord_lon": -7.98, "coord_error": 1}
                  for i, n in enumerate(node_ids)],
        "connections": [
            {"connection_id": f"Line{i:03d}", "from_node_id": a, "to_node_id": b,
             "length": 500, "cable_id": CABLE["cable_id"]}
            for i, (a, b) in enumerate(edges)
        ],
        "cables": [CABLE],
    }
    flags = {flag: "YES" if flag in usage else "NO" for flag in USAGE_FLAGS}
    resp = client.post("/grid/", params=flags,
                       files={"file": ("grid.json", io.BytesIO(json.dumps(grid).encode()), "application/json")})
    assert resp.status_code == 200, resp.text

    payload = {"grid_id": grid_id,
               "historical": [{"node_id": n, "measurements": m} for n, m in measurements.items()]}
    resp = client.post("/measurements/",
                       files={"file": ("m.json", io.BytesIO(json.dumps(payload).encode()), "application/json")})
    assert resp.status_code == 200, resp.text


def delete_grids(grid_ids):
    """Delete only these grids; ON DELETE CASCADE removes their dependent rows."""
    conn, cursor = db.get_db_connection()
    try:
        cursor.execute("DELETE FROM grids WHERE grid_id = ANY(%s)", (list(grid_ids),))
        conn.commit()
    except Exception:
        conn.rollback()
    finally:
        conn.close()

