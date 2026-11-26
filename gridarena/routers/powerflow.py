"""Power Flow Router"""

import logging
from pathlib import Path
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

import gridarena.database as db
from gridarena.powerflow import lv_runs

logger = logging.getLogger(__name__)

router = APIRouter()

_TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=_TEMPLATES_DIR)

Mode = Literal["unbalanced", "balanced"]
ResultPhase = Literal["R", "S", "T", "N"]


@router.post("/{grid_id}/run")
async def run_power_flow(
    grid_id: str,
    mode: Mode = Query("unbalanced"),
    neutral_impedance_ratio: float = Query(1.0, ge=0),
    start_time: Optional[str] = Query(None),
    end_time: Optional[str] = Query(None),
):
    """
    Runs the power flow for every measurement timestamp of a grid (optionally
    within a time range) and replaces the grid's stored results.

    All three phases are solved together. ``unbalanced`` uses each phase's own
    measured loads, cable impedance and substation voltage, with the neutral
    conductor modelled as ``neutral_impedance_ratio`` x the phase impedance
    (0 = ideal neutral). ``balanced`` spreads each node's total power evenly
    over the phases and uses a symmetric substation voltage.

    Parameters:
        grid_id (str): The grid to solve.
        mode (str): "unbalanced" (default) or "balanced".
        neutral_impedance_ratio (float): Neutral impedance relative to the phase
            impedance, unbalanced mode only (default 1.0: same as the phases).
        start_time (str, optional): Lower bound for the timestamp range (ISO 8601).
        end_time (str, optional): Upper bound for the timestamp range (ISO 8601).

    Returns:
        dict: The run summary, eg.::

          {
              "message": "Power flow has been run successfully.",
              "mode": "unbalanced", "neutral_impedance_ratio": 1.0,
              "start_time": null, "end_time": null,
              "timestamps": 336, "not_converged": 0, "skipped_timestamps": 0,
              "unreachable_nodes": []
          }
    """
    conn, cursor = db.get_db_connection()
    try:
        db.create_pf_database(cursor)
        summary = lv_runs.run(cursor, grid_id, mode=mode, neutral_impedance_ratio=neutral_impedance_ratio,
                              start_time=start_time, end_time=end_time)
        conn.commit()
        return {"message": "Power flow has been run successfully.", **summary}
    except LookupError as e:
        conn.rollback()
        raise HTTPException(status_code=404, detail=str(e))
    except lv_runs.PowerFlowInputError as e:
        conn.rollback()
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        conn.rollback()
        logger.exception("Power flow run failed for grid %s", grid_id)
        raise HTTPException(status_code=500, detail=f"Error: {e}")
    finally:
        conn.close()


@router.get("/data/{grid_id}/results")
async def get_power_flow_results(grid_id: str, phase: Optional[ResultPhase] = Query(None)):
    """
    Retrieve the stored power flow results of a grid, with the measured
    voltages next to the computed ones.

    Parameters:
        grid_id (str): The unique identifier of the grid.
        phase (str, optional): Only this phase: "R", "S", "T", or "N" (the
            neutral-to-ground voltage of unbalanced runs). Default: all.

    Returns:
        dict: The run that produced the results, the per-node error of the
        computed voltages against the measured ones, and the results, eg.::

          {
              "grid_id": "grid001",
              "run": {"mode": "unbalanced", "neutral_impedance_ratio": 1.0, ...},
              "reference_node": "PT",
              "errors": [{"node_id": "N1", "phase": "R", "points": 336,
                          "mae_voltage_magnitude": 0.41, ...}],
              "results": [
                  {"timestamp": "2024-08-20 14:00:00", "node_id": "N1", "phase": "R",
                   "converged": true,
                   "computed": {"voltage_magnitude": 229.4, "voltage_angle": -0.3,
                                "real": 229.4, "imag": -1.2},
                   "measured": {"voltage_magnitude": 229.1, "voltage_angle": -0.4}}
              ]
          }

    Raises:
        HTTPException: 404 if the grid has no results, 500 on a database error.
    """
    conn, cursor = db.get_db_connection()
    try:
        try:
            results = lv_runs.comparison(cursor, grid_id, phase)
            run_info = lv_runs.get_run(cursor, grid_id) if results else None
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Database error: {e}") from e

        if not results:
            raise HTTPException(status_code=404, detail="No power flow results found.")

        return {
            "grid_id": grid_id,
            "run": run_info,
            "reference_node": lv_runs.ROOT_NODE,
            "errors": lv_runs.error_summary(results),
            "results": results,
        }
    finally:
        conn.close()


# ---------- HELPERS ----------


def _fetch_grid_data_for_ui(cursor) -> list[dict]:
    cursor.execute(
        """
        SELECT g.grid_id,
               COUNT(DISTINCT n."NodeId") AS n_nodes,
               CASE WHEN EXISTS (
                   SELECT 1 FROM "Measurements" m WHERE m.grid_id = g.grid_id LIMIT 1
               ) THEN TRUE ELSE FALSE END AS has_historical,
               CASE WHEN EXISTS (
                   SELECT 1 FROM "PowerFlowResults" p WHERE p.grid_id = g.grid_id LIMIT 1
               ) THEN TRUE ELSE FALSE END AS has_results
        FROM grids g
        LEFT JOIN "Node" n ON n.grid_id = g.grid_id
        GROUP BY g.grid_id
        ORDER BY g.grid_id ASC
        """
    )
    return [{
        "grid_id": r[0], "n_nodes": r[1],
        "has_historical": r[2], "has_results": r[3],
    } for r in cursor.fetchall()]


# ---------- ROOT PAGE ----------


@router.get("/ui", response_class=HTMLResponse, include_in_schema=False)
def powerflow_ui_root(request: Request) -> HTMLResponse:
    """Power Flow Simulator workspace."""
    conn, cursor = db.get_db_connection()
    try:
        grids = _fetch_grid_data_for_ui(cursor)
    except Exception:
        logger.warning("Failed to load grid list for the Power Flow UI", exc_info=True)
        grids = []
    finally:
        conn.close()

    grids_with_results = sum(1 for g in grids if g["has_results"])

    return templates.TemplateResponse(request, "powerflow/list.html", {
        "request": request,
        "title": "Power Flow Simulator",
        "active": "algorithms",
        "grids": grids,
        "grids_with_results": grids_with_results,
    })


# ---------- RESULTS PAGE ----------


@router.get("/ui/{grid_id}", response_class=HTMLResponse, include_in_schema=False)
def powerflow_ui_results(
    request: Request,
    grid_id: str,
    phase: str = Query("R", description="Phase (R/S/T, or N for the neutral)"),
):
    """Page showing a grid's computed voltages against the measured ones, per phase."""
    if phase not in ("R", "S", "T", "N"):
        phase = "R"
    conn, cursor = db.get_db_connection()
    try:
        results = lv_runs.comparison(cursor, grid_id, phase)
        run_info = lv_runs.get_run(cursor, grid_id)
    finally:
        conn.close()

    context = {
        "request": request,
        "title": f"Power Flow: {grid_id}",
        "active": "algorithms",
        "grid_id": grid_id,
        "phase": phase,
        "run": run_info,
        "reference_node": lv_runs.ROOT_NODE,
        "empty": not results,
    }
    if results:
        context["errors"] = lv_runs.error_summary(results)
        context["results"] = results
        context["not_converged_points"] = sum(1 for r in results if not r["converged"])
    return templates.TemplateResponse(request, "powerflow/results.html", context)
