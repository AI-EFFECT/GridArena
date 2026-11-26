"""MV Power-Flow Data-Generation Router: the JSON API, and the pages of the
Synthetic Data section (/pf-datagen/ui...): an overview of runs and groups,
the New Dataset steps, a page per run and per group, and the synthetic grid
generator. See gridarena/pf_datagen/ for the pipeline itself.
"""

import io
import logging
import os
import tempfile
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from starlette.background import BackgroundTask
from fastapi.templating import Jinja2Templates

import gridarena.database as db
from gridarena.pf_datagen.run_manager import pf_datagen_registry
from gridarena.pf_datagen.schemas import PFDataGenGroupRequest, PFDataGenRequest, PFDataGenStartResponse

logger = logging.getLogger(__name__)

router = APIRouter()

_TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=_TEMPLATES_DIR)

# Handlers are plain `def` unless they await something: their database calls
# block, and FastAPI runs plain handlers in a worker thread, so one slow query
# can't freeze the whole server. Async handlers (uploads) move their database
# work to a worker thread with run_in_threadpool. Tables are created once at
# startup (gridarena/startup.py), not per request.


def _require_mv_grid(mv_grid_id: str, cursor):
    cursor.execute('SELECT 1 FROM "MVGrid" WHERE mv_grid_id = %s', (mv_grid_id,))
    if cursor.fetchone() is None:
        raise HTTPException(status_code=404, detail=f"MV Grid '{mv_grid_id}' not found.")


@router.post("/mv/{mv_grid_id}/run")
def start_pf_datagen_run(mv_grid_id: str, request: PFDataGenRequest):
    conn, cursor = db.get_db_connection()
    try:
        _require_mv_grid(mv_grid_id, cursor)
    finally:
        conn.close()

    run_id = pf_datagen_registry.start_run(mv_grid_id, request)
    return PFDataGenStartResponse(
        run_id=run_id, status="queued",
        message=f"PF data-generation run started for MV grid '{mv_grid_id}'.",
    )


def _group_grids(body: PFDataGenGroupRequest):
    """The grids of a group request that pass its size filter, and the
    skipped ones with the reason. Raises 404 / 422."""
    conn, cursor = db.get_db_connection()
    try:
        cursor.execute(
            """
            SELECT g.mv_grid_id,
                   (SELECT count(*) FROM "MVNode" n WHERE n.mv_grid_id = g.mv_grid_id),
                   (SELECT count(*) FROM "MVConnectionPoint" c WHERE c.mv_grid_id = g.mv_grid_id)
            FROM "MVGrid" g ORDER BY g.mv_grid_id
            """
        )
        stats = {r[0]: (r[1], r[2]) for r in cursor.fetchall()}
    finally:
        conn.close()
    wanted = sorted(stats) if body.mv_grid_ids == "all" else list(dict.fromkeys(body.mv_grid_ids))
    unknown = [g for g in wanted if g not in stats]
    if unknown:
        raise HTTPException(status_code=404, detail=f"MV grid(s) not found: {', '.join(unknown)}.")
    grids, skipped = [], []
    for g in wanted:
        n_nodes, n_cps = stats[g]
        if n_nodes < body.min_nodes:
            skipped.append({"mv_grid_id": g, "reason": f"{n_nodes} nodes (fewer than {body.min_nodes})"})
        elif n_cps < body.min_connection_points:
            skipped.append({"mv_grid_id": g, "reason": f"{n_cps} connection points "
                                                        f"(fewer than {body.min_connection_points})"})
        else:
            grids.append({"mv_grid_id": g, "n_nodes": n_nodes, "n_connection_points": n_cps})
    if not grids:
        raise HTTPException(status_code=422, detail={"message": "No grid passes the size filter.",
                                                     "skipped": skipped})
    return grids, skipped


@router.post("/group/run")
def start_pf_datagen_group(body: PFDataGenGroupRequest):
    """One run per grid, as a group. Grids with fewer than `min_nodes` nodes or
    `min_connection_points` connection points are skipped and listed."""
    grids, skipped = _group_grids(body)
    group_id = pf_datagen_registry.start_group(grids, body.request, skipped)
    group = pf_datagen_registry.get_group(group_id)
    return {"group_id": group_id, "runs": [{"mv_grid_id": r["mv_grid_id"], "run_id": r["run_id"]}
                                           for r in group["runs"]],
            "skipped": skipped, "message": f"Started {len(grids)} run(s) as group '{group_id}'."}


# ── Standalone package (pf_datagen/package.py) ──
#
# The same request as a run, but instead of solving it here GridArena builds
# its inputs (grid, catalogue, load scenarios) and returns a zip with the
# solver code, run.py and how_to_run.md, to run on any machine.


def _fixed_seed(request: PFDataGenRequest) -> int:
    return request.seed if request.seed is not None else int(np.random.SeedSequence().entropy % (2**32))


def _package_response(grids, group_seed=None, skipped=()):
    from gridarena.pf_datagen.package import build_package, new_package_id

    package_id = new_package_id()
    fd, path = tempfile.mkstemp(suffix=".zip", prefix="pf_datagen_package_")
    os.close(fd)
    try:
        build_package(path, grids, db, package_id, group_seed=group_seed, skipped=skipped)
    except Exception as e:
        os.unlink(path)
        logger.error("PF datagen package failed: %s", e, exc_info=True)
        # Inputs that can't be built (no usable series, no LV grids, ...) say why.
        raise HTTPException(status_code=400, detail=f"Could not build the package: {e}")
    return FileResponse(path, media_type="application/zip", filename=f"gridarena_dataset_{package_id}.zip",
                        background=BackgroundTask(os.unlink, path))


@router.post("/mv/{mv_grid_id}/package")
def package_pf_datagen_run(mv_grid_id: str, request: PFDataGenRequest):
    """A standalone package of this run, to solve outside GridArena (zip)."""
    conn, cursor = db.get_db_connection()
    try:
        _require_mv_grid(mv_grid_id, cursor)
    finally:
        conn.close()
    return _package_response([(mv_grid_id, request.model_copy(update={"seed": _fixed_seed(request)}))])


@router.post("/group/package")
def package_pf_datagen_group(body: PFDataGenGroupRequest):
    """A standalone package of a group: one folder per grid that passes the
    size filter, each grid seeded from the group seed as a group run is."""
    from gridarena.pf_datagen.package import grid_seeds

    grids, skipped = _group_grids(body)
    seed = _fixed_seed(body.request)
    seeds = grid_seeds(seed, len(grids))
    return _package_response(
        [(g["mv_grid_id"], body.request.model_copy(update={"seed": s})) for g, s in zip(grids, seeds)],
        group_seed=seed, skipped=skipped,
    )


@router.get("/groups")
def list_pf_datagen_groups():
    return pf_datagen_registry.list_groups()


def _group_or_404(group_id: str) -> dict:
    group = pf_datagen_registry.get_group(group_id)
    if group is None:
        raise HTTPException(status_code=404, detail="Group not found.")
    return group


@router.get("/groups/{group_id}")
def get_pf_datagen_group(group_id: str):
    """The group's runs with their status, the skipped grids and overall progress."""
    group = _group_or_404(group_id)
    runs = []
    done = total = 0
    for entry in group["runs"]:
        r = pf_datagen_registry.get_run(entry["run_id"])
        runs.append({**entry, "status": r.status if r else "unknown",
                     "completed_scenarios": r.completed_scenarios if r else 0,
                     "failed_scenarios": r.failed_scenarios if r else 0,
                     "total_scenarios": r.total_scenarios if r else 0})
        if r:
            done += r.completed_scenarios + r.failed_scenarios
            total += r.total_scenarios
    return {"group_id": group_id, "created_at": group.get("created_at"), "seed": group.get("seed"),
            "runs": runs, "skipped": group.get("skipped", []),
            "progress": {"solved": done, "total": total}}


@router.get("/groups/{group_id}/report")
def get_pf_datagen_group_report(group_id: str):
    """The group's diversity report: summed counts and histograms, per-sample
    percentiles over every completed run, and one row per grid."""
    from gridarena.pf_datagen.group_report import build_group_report

    group = _group_or_404(group_id)
    runs = {e["run_id"]: pf_datagen_registry.get_run(e["run_id"]) for e in group["runs"]}
    runs = {k: v for k, v in runs.items() if v is not None}
    reports = {rid: pf_datagen_registry.get_report(rid) for rid, r in runs.items() if r.status == "completed"}
    reports = {k: v for k, v in reports.items() if v is not None}
    stats, per_run = None, {}
    if reports:
        conn, cursor = db.get_db_connection()
        try:
            stats, per_run = db.get_pf_datagen_group_stats(list(reports), cursor)
        finally:
            conn.close()
    return build_group_report(group, runs, reports, stats, per_run)


@router.get("/groups/{group_id}/splits")
def get_pf_datagen_group_splits(group_id: str, train: float = Query(0.7, gt=0, lt=1),
                                val: float = Query(0.15, ge=0, lt=1), seed: int = 0):
    """Grids to train / val / test, stratified by grid size (pf_datagen/splits.py)."""
    from gridarena.pf_datagen.splits import split_grids

    group = _group_or_404(group_id)
    sizes = {e["mv_grid_id"]: e.get("n_nodes") or 0 for e in group["runs"]}
    samples = {}
    for e in group["runs"]:
        r = pf_datagen_registry.get_run(e["run_id"])
        samples[e["mv_grid_id"]] = r.total_scenarios if r else 0
    try:
        out = split_grids(sizes, train, val, seed, weights=samples)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    out["runs"] = {e["mv_grid_id"]: e["run_id"] for e in group["runs"]}
    return out


@router.get("/runs/{run_id}/splits")
def get_pf_datagen_run_splits(run_id: str, by: Literal["time", "topology"] = "time",
                              train: float = Query(0.7, gt=0, lt=1), val: float = Query(0.15, ge=0, lt=1),
                              seed: int = 0, format: Literal["json", "csv"] = "json"):
    """Leakage-safe split suggestion for one run (pf_datagen/splits.py): whole
    days (`time`) or whole configurations / topology variants (`topology`)."""
    from gridarena.pf_datagen.splits import split_samples

    if pf_datagen_registry.get_run(run_id) is None:
        raise HTTPException(status_code=404, detail="Run not found.")
    conn, cursor = db.get_db_connection()
    try:
        rows = db.get_pf_datagen_split_rows(run_id, cursor)
    finally:
        conn.close()
    try:
        out = split_samples(rows, by, train, val, seed)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    if format == "csv":
        df = pd.DataFrame(out["samples"])
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        buf.seek(0)
        return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv", headers={
            "Content-Disposition": f'attachment; filename="pf_datagen_{run_id}_splits_{by}.csv"'})
    return out


@router.get("/mv/{mv_grid_id}/runs")
def list_pf_datagen_runs_for_grid(mv_grid_id: str):
    return [r.model_dump() for r in pf_datagen_registry.list_runs(mv_grid_id=mv_grid_id)]


@router.get("/runs")
def list_all_pf_datagen_runs():
    """All runs across every MV grid."""
    return [r.model_dump() for r in pf_datagen_registry.list_runs()]


@router.get("/runs/{run_id}")
def get_pf_datagen_run(run_id: str):
    run = pf_datagen_registry.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found.")

    payload = run.model_dump()
    if run.status == "completed":
        conn, cursor = db.get_db_connection()
        try:
            payload["summary"] = db.get_pf_datagen_summary(run_id, cursor)
        finally:
            conn.close()
    return payload


@router.get("/runs/{run_id}/report")
def get_pf_datagen_run_report(run_id: str):
    """The run's diversity report: outcomes, voltage and loading histograms,
    per-sample distributions and, with a load level, coverage and violation
    rates per level band. Written when the run finishes."""
    if pf_datagen_registry.get_run(run_id) is None:
        raise HTTPException(status_code=404, detail="Run not found.")
    report = pf_datagen_registry.get_report(run_id)
    if report is None:
        raise HTTPException(status_code=404, detail="No report for this run yet: it is still running, failed, "
                                                    "or ran before reports existed.")
    return report


@router.get("/runs/{run_id}/results")
def get_pf_datagen_run_results(
    run_id: str,
    table: Literal["bus", "branch", "ybus", "runtime", "summary", "der", "variants", "measurements",
                   "layout"] = Query(...),
    limit: int = Query(500, ge=1, le=5000),
    offset: int = Query(0, ge=0),
):
    run = pf_datagen_registry.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found.")

    conn, cursor = db.get_db_connection()
    try:
        rows, columns = db.get_pf_datagen_results(run_id, table, cursor, limit=limit, offset=offset)
    except Exception as e:
        logger.error("PF datagen results fetch failed for run %s/%s: %s", run_id, table, e, exc_info=True)
        raise HTTPException(status_code=400, detail="Could not fetch results. See server logs for details.")
    finally:
        conn.close()

    return {"run_id": run_id, "table": table, "columns": columns, "rows": rows}


@router.get("/runs/{run_id}/export.csv")
def export_pf_datagen_run(
    run_id: str, table: Literal["bus", "branch", "ybus", "runtime", "summary", "der", "variants", "measurements",
                                "layout"] = Query(...),
):
    run = pf_datagen_registry.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found.")

    conn, cursor = db.get_db_connection()
    try:
        rows, columns = db.get_pf_datagen_all_rows(run_id, table, cursor)
    except Exception as e:
        logger.error("PF datagen export failed for run %s/%s: %s", run_id, table, e, exc_info=True)
        raise HTTPException(status_code=400, detail="Could not export results. See server logs for details.")
    finally:
        conn.close()

    df = pd.DataFrame(rows, columns=columns)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)

    filename = f"pf_datagen_{run_id}_{table}.csv"
    return StreamingResponse(
        iter([buf.getvalue()]), media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ── Synthetic Data pages ──


def _grid_summaries() -> list:
    from gridarena.routers.mv_grid import mv_grid_summaries

    conn, cursor = db.get_db_connection()
    try:
        return mv_grid_summaries(cursor)
    except Exception as e:
        logger.warning("Could not list MV grids: %s", e)
        return []
    finally:
        conn.close()


DEVICE_SUMMARY_MAX_GRIDS = 20


@router.get("/devices")
def grid_devices(grids: str = Query(..., description="Comma-separated MV grid ids")):
    """For the wizard's DER and Set-points steps: per grid (the first
    DEVICE_SUMMARY_MAX_GRIDS), its own DER (count, MW, types) and how many
    substations, capacitor banks / reactors, regulators and MV/LV
    transformers it has."""
    from gridarena.pf_datagen.grid_loader import get_mv_grid_topology

    ids = [g for g in dict.fromkeys(s.strip() for s in grids.split(",")) if g]
    out = []
    conn, cursor = db.get_db_connection()
    try:
        for grid_id in ids[:DEVICE_SUMMARY_MAX_GRIDS]:
            try:
                t = get_mv_grid_topology(grid_id, cursor)
            except Exception as e:
                conn.rollback()
                out.append({"mv_grid_id": grid_id, "error": str(e)})
                continue
            ders = [d for d in t.ders if d.in_service and d.der_type != "ev_hub"]
            types = {}
            for d in ders:
                types[d.der_type] = types.get(d.der_type, 0) + 1
            out.append({
                "mv_grid_id": grid_id,
                "ders": {"count": len(ders), "mw": sum(d.s_rated_kva for d in ders) / 1e3, "types": types},
                "substations": len(t.substation_node_ids),
                "capacitor_banks": sum(1 for s in t.shunts if s.in_service),
                "regulators": sum(1 for b in t.branches if b.kind == "regulated_line"),
                "transformers": sum(1 for b in t.branches if b.kind == "transformer" and b.in_service),
            })
    finally:
        conn.close()
    return {"grids": out, "n_requested": len(ids), "n_summarised": len(out)}


@router.get("/generation-preview")
def generation_preview(
    database_ids: str = Query(..., description="Comma-separated generation databases"),
    load_database_id: str = Query(None),
    load_start: str = Query(None, description="Start of the load window (local clock); none: the load's first"),
    generation_start: str = Query(None),
    keep_time_of_day: bool = Query(True),
    timezone: str = Query("UTC", description="Clock of the generation databases"),
    load_timezone: str = Query("Europe/Lisbon"),
    days: float = Query(3, gt=0, le=14),
    load_aggregation_s: int = Query(None, ge=60, le=86400, description="Average both over intervals of this"),
):
    """The load's first days overlaid with each generation database over the
    window a run would read (pf_datagen/der_profiles.py::generation_preview)."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from gridarena.pf_datagen.der_profiles import generation_preview as preview

    def when(s):
        try:
            return datetime.fromisoformat(s) if s else None
        except ValueError:
            raise HTTPException(status_code=422, detail=f"Not a date and time: '{s}'.")
    for tz in (timezone, load_timezone):
        try:
            ZoneInfo(tz)
        except Exception:
            raise HTTPException(status_code=422, detail=f"Unknown time zone '{tz}'.")
    ids = [d for d in dict.fromkeys(s.strip() for s in database_ids.split(",")) if d]
    conn, cursor = db.get_db_connection()
    try:
        return preview(load_database_id or None, ids, when(load_start), when(generation_start), keep_time_of_day,
                       timezone, load_timezone, days, cursor, load_aggregation_s=load_aggregation_s)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    finally:
        conn.close()


@router.get("/ui", response_class=HTMLResponse, include_in_schema=False)
def pf_datagen_overview_ui(request: Request):
    runs = [r.model_dump() for r in pf_datagen_registry.list_runs()]
    return templates.TemplateResponse(request, "pf_datagen/overview.html", {
        "request": request,
        "title": "Synthetic Data Generation",
        "active": "synthetic_data",
        "runs": runs,
        "groups": pf_datagen_registry.list_groups(),
        "n_active": sum(1 for r in runs if r["status"] in ("queued", "running")),
        "n_completed": sum(1 for r in runs if r["status"] == "completed"),
    })


@router.get("/ui/new", response_class=HTMLResponse, include_in_schema=False)
def pf_datagen_new_ui(request: Request):
    return templates.TemplateResponse(request, "pf_datagen/new.html", {
        "request": request,
        "title": "New Synthetic Dataset",
        "active": "synthetic_data_new",
        "mv_grids": _grid_summaries(),
    })


@router.get("/ui/runs/{run_id}", response_class=HTMLResponse, include_in_schema=False)
def pf_datagen_run_ui(request: Request, run_id: str):
    run = pf_datagen_registry.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found.")
    return templates.TemplateResponse(request, "pf_datagen/run.html", {
        "request": request,
        "title": f"Dataset run {run_id}",
        "active": "synthetic_data",
        "run": run.model_dump(),
    })


@router.get("/ui/groups/{group_id}", response_class=HTMLResponse, include_in_schema=False)
def pf_datagen_group_ui(request: Request, group_id: str):
    group = _group_or_404(group_id)
    return templates.TemplateResponse(request, "pf_datagen/group.html", {
        "request": request,
        "title": f"Dataset group {group_id}",
        "active": "synthetic_data",
        "group": group,
    })


@router.get("/ui/synthetic-grids", response_class=HTMLResponse, include_in_schema=False)
def pf_datagen_synthetic_grids_ui(request: Request):
    from gridarena.synthetic.calibrate import SYNTHETIC_PREFIX

    grids = _grid_summaries()
    return templates.TemplateResponse(request, "pf_datagen/synthetic_grids.html", {
        "request": request,
        "title": "Synthetic Grids",
        "active": "synthetic_grids",
        "synthetic": [g for g in grids if g["mv_grid_id"].startswith(SYNTHETIC_PREFIX)],
        "n_stored": sum(1 for g in grids if not g["mv_grid_id"].startswith(SYNTHETIC_PREFIX)),
    })
