"""MV Grid Router — register, view, connect/disconnect LV grids via transformers."""

import html
import json
import uuid
from pathlib import Path
from typing import List

import networkx as nx
from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError

import gridarena.database as db
import gridarena.schemas as schema
from gridarena.database import NotFoundError

from ._upload_guard import ensure_upload_size_ok

router = APIRouter()

# Handlers are plain `def` unless they await something: their database calls
# block, and FastAPI runs plain handlers in a worker thread, so one slow query
# can't freeze the whole server. Async handlers (uploads) move their database
# work to a worker thread with run_in_threadpool. Tables are created once at
# startup (gridarena/startup.py), not per request.

_TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=_TEMPLATES_DIR)


# ── Validation helper ──


def _validate_mv_grid(mv_grid: schema.MVGrid) -> List[str]:
    """Validate internal references in an MV grid. Returns list of error strings."""
    errors = []
    node_ids = {n.node_id for n in mv_grid.nodes}

    if mv_grid.cables:
        cable_ids = {c.cable_id for c in mv_grid.cables}
    else:
        cable_ids = set()

    if mv_grid.connections:
        for c in mv_grid.connections:
            if c.from_node_id not in node_ids:
                errors.append(f"Connection '{c.connection_id}': from_node_id '{c.from_node_id}' not found in nodes.")
            if c.to_node_id not in node_ids:
                errors.append(f"Connection '{c.connection_id}': to_node_id '{c.to_node_id}' not found in nodes.")
            if c.cable_id not in cable_ids:
                errors.append(f"Connection '{c.connection_id}': cable_id '{c.cable_id}' not found in cables.")

    cp_ids = set()
    if mv_grid.connection_points:
        for cp in mv_grid.connection_points:
            if cp.node_id not in node_ids:
                errors.append(f"Connection point '{cp.connection_point_id}': node_id '{cp.node_id}' not found in nodes.")
            if cp.connection_point_id in cp_ids:
                errors.append(f"Duplicate connection_point_id '{cp.connection_point_id}'.")
            cp_ids.add(cp.connection_point_id)

    if mv_grid.transformers:
        for t in mv_grid.transformers:
            if t.connection_point_id not in cp_ids:
                errors.append(f"Transformer '{t.transformer_id}': connection_point_id '{t.connection_point_id}' not found.")
            if t.primary_voltage_kv != mv_grid.nominal_voltage_kv:
                errors.append(f"Transformer '{t.transformer_id}': primary_voltage_kv ({t.primary_voltage_kv}) does not match grid nominal ({mv_grid.nominal_voltage_kv}).")

    if mv_grid.regulators:
        connection_ids = {c.connection_id for c in (mv_grid.connections or [])}
        regulated = set()
        for r in mv_grid.regulators:
            if r.connection_id not in connection_ids:
                errors.append(f"Regulator '{r.regulator_id}': connection_id '{r.connection_id}' not found in connections.")
            elif r.connection_id in regulated:
                errors.append(f"Regulator '{r.regulator_id}': connection '{r.connection_id}' already has a regulator.")
            regulated.add(r.connection_id)

    if mv_grid.shunts:
        shunt_ids = set()
        for sh in mv_grid.shunts:
            if sh.node_id not in node_ids:
                errors.append(f"Shunt '{sh.shunt_id}': node_id '{sh.node_id}' not found in nodes.")
            if sh.shunt_id in shunt_ids:
                errors.append(f"Duplicate shunt_id '{sh.shunt_id}'.")
            shunt_ids.add(sh.shunt_id)

    if mv_grid.ders:
        substations = {n.node_id for n in mv_grid.nodes if n.node_type == "substation"}
        coords = {n.node_id: (n.coord_lat, n.coord_lon) for n in mv_grid.nodes}
        der_ids = set()
        for d in mv_grid.ders:
            if d.node_id not in node_ids:
                errors.append(f"DER '{d.der_id}': node_id '{d.node_id}' not found in nodes.")
            elif d.node_id in substations:
                errors.append(f"DER '{d.der_id}': can't sit on the substation node '{d.node_id}'.")
            elif d.profile == "solar" and None in coords[d.node_id]:
                errors.append(f"DER '{d.der_id}': a solar profile needs coordinates on node '{d.node_id}'.")
            if d.der_id in der_ids:
                errors.append(f"Duplicate der_id '{d.der_id}'.")
            der_ids.add(d.der_id)

    return errors


# Element kinds whose operating settings can be edited: kind -> (schema model, id field, label).
_CONFIG_KINDS = {
    "connections": (schema.MVConnection, "connection_id", "Connection"),
    "transformers": (schema.Transformer, "transformer_id", "Transformer"),
    "regulators": (schema.Regulator, "regulator_id", "Regulator"),
    "shunts": (schema.Shunt, "shunt_id", "Shunt"),
    "ders": (schema.DER, "der_id", "DER"),
}


def _merge_configuration(grid: dict, update: schema.MVConfigurationUpdate):
    """Apply configuration edits to a grid detail (get_mv_grid_detail) in
    memory. Each edited element is re-checked against its full schema, so
    the same rules as an upload hold (tap within range, steps <= n_steps,
    normally_open only on a switch, DER curve limits, ...).

    Returns (changes, errors): changes maps kind -> [{id field, changed
    fields}], keeping only fields that differ from the stored value."""
    changes, errors = {}, []
    for kind, (model, id_field, label) in _CONFIG_KINDS.items():
        current_by_id = {el[id_field]: el for el in grid.get(kind) or []}
        for edit in getattr(update, kind):
            fields = edit.model_dump(exclude_none=True)
            element_id = fields.pop(id_field)
            current = current_by_id.get(element_id)
            if current is None:
                errors.append(f"{label} '{element_id}' not found in this grid.")
                continue
            try:
                model(**{**current, **fields})
            except ValidationError as e:
                for err in e.errors():
                    msg = err["msg"].removeprefix("Value error, ")
                    loc = ".".join(str(p) for p in err["loc"])
                    errors.append(msg if not loc else f"{label} '{element_id}', {loc}: {msg}")
                continue
            diff = {k: v for k, v in fields.items() if current.get(k) != v}
            if diff:
                changes.setdefault(kind, []).append({id_field: element_id, **diff})
    return changes, errors


def _active(status) -> bool:
    return (status or "active").lower() == "active"


def _compute_graph_layout(grid_data: dict) -> dict:
    """Build a NetworkX graph and return node positions + edge list for frontend rendering.

    Each edge carries its connection's state ("closed", "normally_open" for
    an open switch / tie line, "out_of_service" for a status other than
    active), cable, length and regulator; each node carries its transformers,
    shunts and DER, so the graph can draw them."""
    G = nx.Graph()
    for n in grid_data.get("nodes", []):
        nid = n.get("node_id") or n.get("NodeId", "")
        G.add_node(nid)

    regulator_by_conn = {r["connection_id"]: r for r in grid_data.get("regulators") or []}
    graph_edges = []
    for c in grid_data.get("connections") or []:
        fid = c.get("from_node_id") or c.get("FromNodeId", "")
        tid = c.get("to_node_id") or c.get("ToNodeId", "")
        if not (fid and tid):
            continue
        G.add_edge(fid, tid)
        if not _active(c.get("status")):
            state = "out_of_service"
        elif c.get("normally_open"):
            state = "normally_open"
        else:
            state = "closed"
        edge = {"id": c.get("connection_id"), "from": fid, "to": tid, "open": state != "closed", "state": state,
                "is_switch": bool(c.get("is_switch")), "cable_id": c.get("cable_id"), "length": c.get("length")}
        reg = regulator_by_conn.get(c.get("connection_id"))
        if reg:
            edge["regulator"] = {k: reg.get(k) for k in ("regulator_id", "v_set_pu", "deadband_pu", "tap_pos",
                                                         "tap_min", "tap_max", "tap_step_pct")}
        graph_edges.append(edge)

    cp_ids = set()
    for cp in grid_data.get("connection_points", []):
        cpn = cp.get("node_id") or cp.get("NodeId", "")
        cp_ids.add(cpn)

    lv_by_cp, transformers_by_cp = {}, {}
    for t in grid_data.get("transformers", []):
        cpid = t.get("connection_point_id") or t.get("ConnectionPointId", "")
        lv = t.get("lv_grid_id")
        if cpid and lv:
            lv_by_cp.setdefault(cpid, []).append(lv)
        if cpid:
            transformers_by_cp.setdefault(cpid, []).append({
                "transformer_id": t.get("transformer_id"), "rated_power_kva": t.get("rated_power_kva"),
                "tap_pos": t.get("tap_pos") or 0, "active": _active(t.get("status")),
            })

    shunts_by_node, ders_by_node = {}, {}
    for sh in grid_data.get("shunts") or []:
        shunts_by_node.setdefault(sh["node_id"], []).append({
            "shunt_id": sh["shunt_id"], "q_step_kvar": sh["q_step_kvar"], "n_steps": sh.get("n_steps") or 1,
            "step": sh.get("step") if sh.get("step") is not None else 1, "active": _active(sh.get("status")),
        })
    for d in grid_data.get("ders") or []:
        ders_by_node.setdefault(d["node_id"], []).append({
            "der_id": d["der_id"], "type": d.get("type"), "s_rated_kva": d.get("s_rated_kva"),
            "control_mode": d.get("control_mode") or "fixed_pf", "active": _active(d.get("status")),
        })

    has_coords = all(
        (n.get("coord_lat") or n.get("CoordLat")) is not None
        for n in grid_data.get("nodes", [])
    )

    if has_coords:
        pos = {}
        for n in grid_data.get("nodes", []):
            nid = n.get("node_id") or n.get("NodeId", "")
            lon = n.get("coord_lon") or n.get("CoordLon") or 0.0
            lat = n.get("coord_lat") or n.get("CoordLat") or 0.0
            pos[nid] = (float(lon), float(lat))
    else:
        pos = nx.spring_layout(G, seed=42)
        pos = {k: (float(v[0]), float(v[1])) for k, v in pos.items()}

    cp_point_ids = {}
    for cp in grid_data.get("connection_points", []):
        nid = cp.get("node_id") or cp.get("NodeId", "")
        cpid = cp.get("connection_point_id") or cp.get("ConnectionPointId", "")
        cp_point_ids[nid] = cpid

    graph_nodes = []
    for nid in G.nodes():
        x, y = pos.get(nid, (0, 0))
        ntype = "normal"
        nt = None
        for n in grid_data.get("nodes", []):
            if (n.get("node_id") or n.get("NodeId", "")) == nid:
                nt = n.get("node_type") or n.get("NodeType")
                break
        if nid in cp_ids:
            ntype = "connection_point"
        elif nt == "substation":
            ntype = "substation"

        node_entry = {"id": nid, "x": x, "y": y, "type": ntype}
        if nid in cp_point_ids:
            node_entry["connection_point_id"] = cp_point_ids[nid]
            connected = lv_by_cp.get(cp_point_ids[nid], [])
            if connected:
                node_entry["connected_lv"] = connected
            if cp_point_ids[nid] in transformers_by_cp:
                node_entry["transformers"] = transformers_by_cp[cp_point_ids[nid]]
        if nid in shunts_by_node:
            node_entry["shunts"] = shunts_by_node[nid]
        if nid in ders_by_node:
            node_entry["ders"] = ders_by_node[nid]
        graph_nodes.append(node_entry)

    return {"nodes": graph_nodes, "edges": graph_edges}


# ── API Endpoints ──


@router.post("/validate")
async def validate_mv_grid(file: UploadFile = File(...)):
    """Validate an MV grid JSON without saving. Returns parsed grid, graph layout, or errors."""
    try:
        contents = await file.read()
        data = json.loads(contents.decode("utf-8"))
        mv_grid = schema.MVGrid(**data)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format.")
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors())

    errors = _validate_mv_grid(mv_grid)
    if errors:
        return {"valid": False, "errors": errors}

    grid_dump = mv_grid.model_dump()
    graph_data = _compute_graph_layout(grid_dump)

    return {"valid": True, "mv_grid": grid_dump, "graph": graph_data}


@router.post("/")
async def register_mv_grid(file: UploadFile = File(...)):
    """Register a new MV grid from a JSON file."""
    ensure_upload_size_ok(file)
    try:
        contents = await file.read()
        data = json.loads(contents.decode("utf-8"))
        mv_grid = schema.MVGrid(**data)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format.")
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors())

    errors = _validate_mv_grid(mv_grid)
    if errors:
        raise HTTPException(status_code=422, detail={"validation_errors": errors})

    def store():
        conn, cursor = db.get_db_connection()
        try:
            db.insert_mv_grid_data(mv_grid, conn, cursor)
        except Exception as e:
            conn.rollback()
            raise HTTPException(status_code=500, detail=f"Database error: {e}")
        finally:
            conn.close()

    await run_in_threadpool(store)
    return {"message": f"MV Grid '{mv_grid.mv_grid_id}' registered successfully."}


@router.get("/list")
def list_mv_grids():
    conn, cursor = db.get_db_connection()
    try:
        return db.get_mv_grid_list(cursor)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    finally:
        conn.close()


@router.get("/data/{mv_grid_id}")
def get_mv_grid(mv_grid_id: str):
    """Get full MV grid with nodes, connections, connection points, and transformers."""
    conn, cursor = db.get_db_connection()
    try:
        return db.get_mv_grid_detail(mv_grid_id, cursor)
    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{e}")
    finally:
        conn.close()


# ── Synthetic grids (Phase 2, axis 7B; gridarena/synthetic/) ──
# Registered before the /{mv_grid_id} routes so "synthetic" isn't read as a grid id.


@router.get("/synthetic/calibration")
def get_synthetic_calibration():
    """What the generator is calibrated on: per archetype, the stored grids'
    statistics and the fitted parameters."""
    from gridarena.synthetic.calibrate import calibrate

    conn, cursor = db.get_db_connection()
    try:
        fits = calibrate(cursor)
    finally:
        conn.close()
    return {a: f.as_dict() for a, f in fits.items()}


@router.post("/synthetic")
def generate_synthetic_grids(request: schema.SyntheticGridRequest):
    """Generate and store synthetic MV grids; returns their ids and the
    generator report (generated against stored statistics, per archetype)."""
    import numpy as np

    from gridarena.synthetic.batch import choose_cables, generate, next_numbers
    from gridarena.synthetic.calibrate import calibrate
    from gridarena.synthetic.generator import Options

    seed = request.seed if request.seed is not None else int(np.random.SeedSequence().entropy % (2**32))
    opts = Options(
        feeders_per_substation=request.feeders_per_substation, second_substation=request.second_substation,
        ties_between_feeders=request.ties_between_feeders,
        n_nodes=(request.n_nodes_min, request.n_nodes_max) if request.n_nodes_min is not None else None,
        reference=(request.reference_lat, request.reference_lon),
    )
    conn, cursor = db.get_db_connection()
    try:
        fits = calibrate(cursor)
        if not fits:
            raise HTTPException(status_code=422, detail="No stored MV grid to calibrate the generator on.")
        try:
            grids, report = generate(fits, request.archetype, request.count, seed, opts, choose_cables(cursor),
                                     next_numbers(cursor))
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        conn.rollback()                       # end the read transaction before writing
        for grid in grids:
            errors = _validate_mv_grid(grid)
            if errors:
                raise HTTPException(status_code=500, detail={"validation_errors": errors, "grid": grid.mv_grid_id})
            db.insert_mv_grid_data(grid, conn, cursor)
    except HTTPException:
        conn.rollback()
        raise
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    finally:
        conn.close()
    return {"created": [g.mv_grid_id for g in grids], "seed": seed, "report": report,
            "message": f"Generated {len(grids)} synthetic grid(s)."}


@router.delete("/synthetic")
def delete_synthetic_grids():
    """Delete every synthetic grid (ids starting with SYN-), with everything that cascades from them."""
    from gridarena.synthetic.calibrate import SYNTHETIC_PREFIX

    conn, cursor = db.get_db_connection()
    try:
        cursor.execute('SELECT mv_grid_id FROM "MVGrid" WHERE mv_grid_id LIKE %s', (SYNTHETIC_PREFIX + "%",))
        ids = [r[0] for r in cursor.fetchall()]
        for gid in ids:
            db.delete_mv_grid(gid, conn, cursor)
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    finally:
        conn.close()
    return {"deleted": len(ids), "message": f"Deleted {len(ids)} synthetic grid(s)."}


@router.delete("/{mv_grid_id}")
def delete_mv_grid(mv_grid_id: str):
    conn, cursor = db.get_db_connection()
    try:
        db.delete_mv_grid(mv_grid_id, conn, cursor)
    except NotFoundError as e:
        conn.rollback()
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"{e}")
    finally:
        conn.close()
    return {"message": f"MV Grid '{mv_grid_id}' and all associated data deleted."}


@router.patch("/{mv_grid_id}/configuration")
def patch_mv_grid_configuration(mv_grid_id: str, update: schema.MVConfigurationUpdate):
    """Change operating settings of an MV grid's elements: connection status
    and switch state, transformer taps and status, regulator settings,
    shunt steps and status, DER status and control. Every edited element is
    validated first; nothing is written if any edit is invalid."""
    conn, cursor = db.get_db_connection()
    try:
        grid = db.get_mv_grid_detail(mv_grid_id, cursor)
        changes, errors = _merge_configuration(grid, update)
        if errors:
            raise HTTPException(status_code=422, detail={"validation_errors": errors})
        updated = db.update_mv_grid_configuration(mv_grid_id, changes, conn, cursor)
    except HTTPException:
        conn.rollback()
        raise
    except NotFoundError as e:
        conn.rollback()
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    finally:
        conn.close()
    return {"message": f"Updated {updated} element(s) of MV grid '{mv_grid_id}'.", "updated": updated}


@router.get("/{mv_grid_id}/tie-suggestions")
def get_tie_suggestions(
    mv_grid_id: str,
    max_distance_m: float = Query(500.0, gt=0, le=5000, description="Most straight-line distance between the ends."),
    min_cable_distance_m: float = Query(2000.0, ge=0, description="Least distance along the cables, unless the "
                                                                  "ends are on different feeders."),
    max_ties: int = Query(5, ge=1, le=50),
):
    """Propose normally-open ties for reconfiguration (pf_datagen/ties.py).
    Nothing is stored: POST the ones to keep to /mv-grid/{id}/ties."""
    from gridarena.pf_datagen.ties import suggest_ties

    conn, cursor = db.get_db_connection()
    try:
        grid = db.get_mv_grid_detail(mv_grid_id, cursor)
    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    finally:
        conn.close()
    return suggest_ties(grid, max_distance_m, min_cable_distance_m, max_ties)


@router.post("/{mv_grid_id}/ties")
def add_ties(mv_grid_id: str, request: schema.MVTiesRequest):
    """Store tie lines as active, normally-open switches. All are validated
    first; nothing is written if any is invalid."""
    from gridarena.pf_datagen.ties import validate_ties

    conn, cursor = db.get_db_connection()
    try:
        grid = db.get_mv_grid_detail(mv_grid_id, cursor)
        errors = validate_ties(grid, request.ties)
        if errors:
            raise HTTPException(status_code=422, detail={"validation_errors": errors})
        added = db.insert_mv_ties(mv_grid_id, request.ties, conn, cursor)
    except HTTPException:
        conn.rollback()
        raise
    except NotFoundError as e:
        conn.rollback()
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    finally:
        conn.close()
    return {"message": f"Added {added} tie line(s) to MV grid '{mv_grid_id}'.", "added": added}


# ── LV Connection Endpoints ──


@router.get("/lv-grids")
def list_available_lv_grids():
    """List existing LV grids from the database that can be connected."""
    conn, cursor = db.get_db_connection()
    try:
        cursor.execute("SELECT grid_id FROM grids ORDER BY grid_id ASC")
        return [{"grid_id": r[0]} for r in cursor.fetchall()]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    finally:
        conn.close()


@router.post("/{mv_grid_id}/connect/{connection_point_id}")
def connect_lv_grid(mv_grid_id: str, connection_point_id: str, request: schema.ConnectLVRequest):
    """Connect an LV grid to an MV connection point via a transformer bridge."""
    conn, cursor = db.get_db_connection()
    try:
        transformer_id = request.transformer_id or f"TR-{uuid.uuid4().hex[:8]}"
        db.connect_lv_to_mv(
            mv_grid_id=mv_grid_id,
            connection_point_id=connection_point_id,
            lv_grid_id=request.lv_grid_id,
            transformer_id=transformer_id,
            rated_power_kva=request.rated_power_kva or 400.0,
            primary_voltage_kv=request.primary_voltage_kv or 20.0,
            secondary_voltage_v=request.secondary_voltage_v or 230.0,
            imp_real=request.imp_real,
            imp_imag=request.imp_imag,
            name=request.name,
            conn=conn,
            cursor=cursor,
        )
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=400, detail=f"{e}")
    finally:
        conn.close()

    return {"message": f"LV grid '{request.lv_grid_id}' connected to point '{connection_point_id}' via transformer '{transformer_id}'."}


@router.delete("/{mv_grid_id}/disconnect/{connection_point_id}/{lv_grid_id}")
def disconnect_lv_grid(mv_grid_id: str, connection_point_id: str, lv_grid_id: str):
    """Remove the transformer bridge between an MV connection point and an LV grid."""
    conn, cursor = db.get_db_connection()
    try:
        db.disconnect_lv_from_mv(mv_grid_id, connection_point_id, lv_grid_id, conn, cursor)
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=400, detail=f"{e}")
    finally:
        conn.close()
    return {"message": f"LV grid '{lv_grid_id}' disconnected from point '{connection_point_id}'."}


@router.get("/{mv_grid_id}/pv/{lv_grid_id}")
def get_lv_aggregated_pv(mv_grid_id: str, lv_grid_id: str, limit: int = Query(500, ge=1, le=10000)):
    """Get aggregated P/V pairs for a connected LV grid.

    Returns the average power_active and voltage_magnitude per timestamp across all nodes,
    representing the grid-level load profile at the MV/LV boundary.
    """
    conn, cursor = db.get_db_connection()
    try:
        cursor.execute(
            'SELECT 1 FROM "MVTransformer" WHERE mv_grid_id = %s AND lv_grid_id = %s',
            (mv_grid_id, lv_grid_id),
        )
        if cursor.fetchone() is None:
            raise HTTPException(status_code=404, detail=f"LV grid '{lv_grid_id}' is not connected to MV grid '{mv_grid_id}'.")

        pv_pairs = db.get_aggregated_pv(lv_grid_id, cursor, limit=limit)
        if not pv_pairs:
            raise HTTPException(status_code=404, detail=f"No measurement data found for LV grid '{lv_grid_id}'.")

        cursor.execute("SELECT COUNT(DISTINCT node_id) FROM \"Node\" WHERE grid_id = %s", (lv_grid_id,))
        n_nodes = cursor.fetchone()[0]

        return {
            "mv_grid_id": mv_grid_id,
            "lv_grid_id": lv_grid_id,
            "n_nodes": n_nodes,
            "aggregation": "avg_per_timestamp_across_all_nodes",
            "count": len(pv_pairs),
            "pv_pairs": pv_pairs,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    finally:
        conn.close()


# ── UI Endpoints ──


def mv_grid_summaries(cursor) -> list:
    """Every MV grid with its node, connection, connection-point and
    connected-LV counts (the MV Grids page and the Synthetic Data pages)."""
    cursor.execute(
        """
        SELECT g.mv_grid_id, g.name, g.nominal_voltage_kv, g.region,
               COALESCE(n.cnt, 0) AS n_nodes,
               COALESCE(c.cnt, 0) AS n_connections,
               COALESCE(cp.cnt, 0) AS n_cps,
               COALESCE(t.cnt, 0) AS n_connected
        FROM "MVGrid" g
        LEFT JOIN (SELECT mv_grid_id, COUNT(*) AS cnt FROM "MVNode" GROUP BY mv_grid_id) n
            ON n.mv_grid_id = g.mv_grid_id
        LEFT JOIN (SELECT mv_grid_id, COUNT(*) AS cnt FROM "MVConnection" GROUP BY mv_grid_id) c
            ON c.mv_grid_id = g.mv_grid_id
        LEFT JOIN (SELECT mv_grid_id, COUNT(*) AS cnt FROM "MVConnectionPoint" GROUP BY mv_grid_id) cp
            ON cp.mv_grid_id = g.mv_grid_id
        LEFT JOIN (SELECT mv_grid_id, COUNT(*) AS cnt FROM "MVTransformer"
                   WHERE lv_grid_id IS NOT NULL GROUP BY mv_grid_id) t
            ON t.mv_grid_id = g.mv_grid_id
        ORDER BY g.mv_grid_id
        """
    )
    return [{
        "mv_grid_id": r[0], "name": r[1], "nominal_voltage_kv": r[2], "region": r[3],
        "n_nodes": r[4], "n_connections": r[5], "n_cps": r[6], "n_connected": r[7],
    } for r in cursor.fetchall()]


@router.get("/ui", response_class=HTMLResponse, include_in_schema=False)
def mv_grid_ui(request: Request):
    conn, cursor = db.get_db_connection()
    try:
        mv_grids = mv_grid_summaries(cursor)
        cursor.execute("SELECT grid_id FROM grids ORDER BY grid_id ASC")
        lv_grids = [r[0] for r in cursor.fetchall()]
    except Exception:
        mv_grids = []
        lv_grids = []
    finally:
        conn.close()

    total_nodes = sum(g["n_nodes"] for g in mv_grids)
    total_cps = sum(g["n_cps"] for g in mv_grids)
    total_connected = sum(g["n_connected"] for g in mv_grids)

    from gridarena.pf_datagen.run_manager import pf_datagen_registry
    pf_datagen_runs = [r.model_dump() for r in pf_datagen_registry.list_runs()]

    return templates.TemplateResponse(request, "mv_grid/list.html", {
        "request": request,
        "title": "MV Grids",
        "active": "mv_grid",
        "mv_grids": mv_grids,
        "lv_grids": lv_grids,
        "total_nodes": total_nodes,
        "total_cps": total_cps,
        "total_connected": total_connected,
    })


@router.get("/ui/{mv_grid_id}", response_class=HTMLResponse, include_in_schema=False)
def mv_grid_detail_ui(request: Request, mv_grid_id: str):
    conn, cursor = db.get_db_connection()
    try:
        grid = db.get_mv_grid_detail(mv_grid_id, cursor)
        cursor.execute("SELECT grid_id FROM grids ORDER BY grid_id ASC")
        lv_grids = [r[0] for r in cursor.fetchall()]
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))
    finally:
        conn.close()

    graph_data = _compute_graph_layout(grid)

    return templates.TemplateResponse(request, "mv_grid/detail.html", {
        "request": request,
        "title": f"MV Grid: {mv_grid_id}",
        "active": "mv_grid",
        "grid": grid,
        "lv_grids": lv_grids,
        "graph_data": graph_data,
    })
