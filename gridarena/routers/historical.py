"""Historical Router with endpoints to register, access and delete historical
databases -- named collections of standalone power/voltage series ("historicals")
that all share one timeline. Each database has a data_type: "load"
(consumption, the default) or "pv" (solar PV production). See
gridarena/schemas/historical_data.py.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal, Optional

import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError

import gridarena.database as db
from gridarena.database.insert_historical_database import HistoricalDataTypeConflict
from gridarena.schemas import HistoricalDatabaseUpload
from gridarena.schemas.types import HistoricalDataType

from ._upload_guard import ensure_upload_size_ok

# Short badge labels, and what each type's power values are (upload message, page text).
DATA_TYPE_LABELS = {
    "load": "Load", "pv": "Solar PV", "chp": "Cogeneration", "wind": "Wind", "hydro": "Hydro",
    "other_generation": "Other generation", "injection_total": "Total injection",
}
DATA_TYPE_DESCRIPTIONS = {
    "load": "consumption", "pv": "solar PV production", "chp": "cogeneration production",
    "wind": "wind production", "hydro": "hydro production",
    "other_generation": "production by other technologies",
    "injection_total": "total generation injected into the grid",
}

router = APIRouter()

# Handlers are plain `def` unless they await something: their database calls
# block, and FastAPI runs plain handlers in a worker thread, so one slow query
# can't freeze the whole server. Async handlers (uploads) move their database
# work to a worker thread with run_in_threadpool. Tables are created once at
# startup (gridarena/startup.py), not per request.

_TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=_TEMPLATES_DIR)

# Safety cap for unbounded list queries (e.g. no start/end filter given) --
# prevents a single request from serializing the entire HistoricalRecords
# table (41.8M+ rows in the real dataset).
MAX_RECORDS_PER_REQUEST = 50_000


@router.post("/")
async def upload_historical_database(
    file: UploadFile = File(...),
    data_type: Annotated[Optional[HistoricalDataType], Form(
        description="What the database's power values are: 'load' (consumption) or a kind of generation "
                    "('pv', 'chp', 'wind', 'hydro', 'other_generation', 'injection_total'). Used when the file "
                    "has no data_type; must match it if the file has one."
    )] = None,
):
    """
    Uploads and inserts a historical database -- a shared timeline plus any
    number of series ("historicals") -- into the database from a JSON file.

    Parameters:
        file (UploadFile): A JSON file containing the database to be uploaded.
        data_type (str, optional): "load" or a kind of generation ("pv",
            "chp", ...); the file's own data_type, or "load", when omitted.

    Raises:
        HTTPException 400: If the file is not valid JSON.
        HTTPException 409: If the database_id exists with a different data_type.
        HTTPException 422: If the file does not conform to the HistoricalDatabaseUpload
            schema (including a value whose datetime isn't in the declared timestamps),
            or its data_type contradicts the data_type field.
        HTTPException 500: If a database error occurs during the insertion.

    Returns:
        dict: A success message with the database ID.
    """
    ensure_upload_size_ok(file)
    try:
        contents = await file.read()
        data = json.loads(contents.decode("utf-8"))
        if data_type is not None and isinstance(data, dict):
            in_file = data.get("data_type")
            if in_file is not None and in_file != data_type:
                raise HTTPException(
                    status_code=422,
                    detail=f"The file declares data_type '{in_file}' but '{data_type}' was selected.",
                )
            data["data_type"] = data_type
        database_upload = HistoricalDatabaseUpload(**data)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format.")
    except ValidationError as e:
        # Without the context: a validator's error carries the raw exception there, which isn't JSON.
        raise HTTPException(status_code=422, detail=e.errors(include_url=False, include_context=False))

    def store():
        conn, cursor = db.get_db_connection()
        try:
            db.insert_historical_data(database_upload, conn, cursor)
        except HistoricalDataTypeConflict as e:
            conn.rollback()
            raise HTTPException(status_code=409, detail=str(e))
        except Exception as e:
            conn.rollback()
            raise HTTPException(status_code=500, detail=f"Database error: {e}")
        finally:
            conn.close()

    # One statement per record: a large upload takes a while, so off the event loop.
    await run_in_threadpool(store)


    n_series = len(database_upload.historicals)
    dt = database_upload.data_type
    kind = f" of {DATA_TYPE_DESCRIPTIONS[dt]}" if dt != "load" else ""
    return {
        "message": f"Historical database '{database_upload.database_id}'{kind} "
                   f"inserted successfully ({n_series} series)."
    }


@router.delete("/{database_id}")
def delete_historical_database(database_id: str):
    """
    Deletes an entire historical database, including its timeline and every
    series/record within it (via ON DELETE CASCADE).

    Parameters:
        database_id (str): The database to delete.

    Raises:
        HTTPException 500: If a database error occurs.

    Returns:
        dict: A message indicating whether a database was deleted.
    """
    conn, cursor = db.get_db_connection()

    try:
        cursor.execute('DELETE FROM "HistoricalDatabase" WHERE database_id = %s', (database_id,))
        deleted_count = cursor.rowcount
        conn.commit()

    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    finally:
        conn.close()


    return {"message": f"{deleted_count} database(s) deleted for '{database_id}'."}


@router.delete("/{database_id}/{series_id}")
def delete_historical_series_records(
    database_id: str,
    series_id: str,
    start: Optional[str] = Query(None, description="Start datetime ISO (inclusive)"),
    end: Optional[str] = Query(None, description="End datetime ISO (exclusive)"),
    phase: Optional[Literal["R", "S", "T"]] = Query(
        None, description="Optional phase filter; omit to delete aggregated + all phases"
    ),
):
    """
    Deletes records for one series within a database. The series and
    database registrations are left in place even if this empties its records.

    Parameters:
        database_id (str): The database the series belongs to.
        series_id (str): The series to delete records from.
        start (str, optional): ISO start datetime (inclusive).
        end (str, optional): ISO end datetime (exclusive).
        phase (str, optional): Filter by phase (R/S/T).

    Raises:
        HTTPException 400: If an invalid datetime format is provided.
        HTTPException 500: If a database error occurs.

    Returns:
        dict: A message indicating how many records were deleted.
    """
    try:
        start_dt = datetime.fromisoformat(start) if start else None
        end_dt = datetime.fromisoformat(end) if end else None

    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid datetime format. Use ISO format (e.g. 2025-07-31T00:00:00)",
        )

    conn, cursor = db.get_db_connection()

    try:
        query = 'DELETE FROM "HistoricalRecords" WHERE database_id = %s AND series_id = %s'
        params: list[object] = [database_id, series_id]

        if phase is not None:
            query += " AND phase = %s"
            params.append(phase)
        if start_dt:
            query += " AND datetime >= %s"
            params.append(start_dt)
        if end_dt:
            query += " AND datetime < %s"
            params.append(end_dt)

        cursor.execute(query, tuple(params))
        deleted_count = cursor.rowcount
        conn.commit()

    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    finally:
        conn.close()


    return {"message": f"{deleted_count} record(s) deleted for series '{series_id}' in database '{database_id}'."}


@router.get("/data/{database_id}/{series_id}")
def get_historical_series_data(
    database_id: str,
    series_id: str,
    start: Optional[str] = Query(None, description="Start datetime (inclusive, ISO format)"),
    end: Optional[str] = Query(None, description="End datetime (exclusive, ISO format)"),
    phase: Optional[Literal["R", "S", "T"]] = Query(None, description="Optional single phase filter"),
):
    """
    Retrieves records for one series within a database.

    Parameters:
        database_id (str): The database the series belongs to.
        series_id (str): The series to retrieve records for.
        start / end (str, optional): ISO datetime range.
        phase (str, optional): Single phase filter.

    Raises:
        HTTPException 400/404/500 as appropriate.

    Returns:
        dict: Series data with records and count.
    """
    try:
        start_dt = datetime.fromisoformat(start) if start else None
        end_dt = datetime.fromisoformat(end) if end else None
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid datetime format. Use ISO format.")

    conn, cursor = db.get_db_connection()

    try:
        cursor.execute(
            'SELECT s.series_id, s.grid_level, s.name, s.description, d.data_type FROM "HistoricalSeries" s '
            'JOIN "HistoricalDatabase" d ON d.database_id = s.database_id '
            "WHERE s.database_id = %s AND s.series_id = %s",
            (database_id, series_id),
        )
        series_row = cursor.fetchone()
        if series_row is None:
            raise HTTPException(
                status_code=404, detail=f"Series '{series_id}' not found in database '{database_id}'."
            )

        sql = """
            SELECT datetime, phase, power_active, power_reactive, voltage_magnitude, voltage_angle
            FROM "HistoricalRecords"
            WHERE database_id = %s AND series_id = %s
        """
        params: list[object] = [database_id, series_id]

        if start_dt:
            sql += " AND datetime >= %s"
            params.append(start_dt)
        if end_dt:
            sql += " AND datetime < %s"
            params.append(end_dt)
        if phase:
            sql += " AND phase = %s"
            params.append(phase)

        sql += " ORDER BY datetime ASC LIMIT %s"
        params.append(MAX_RECORDS_PER_REQUEST + 1)

        cursor.execute(sql, tuple(params))
        rows = cursor.fetchall()

        truncated = len(rows) > MAX_RECORDS_PER_REQUEST
        if truncated:
            rows = rows[:MAX_RECORDS_PER_REQUEST]

        columns = ["datetime", "phase", "power_active", "power_reactive", "voltage_magnitude", "voltage_angle"]
        data = [dict(zip(columns, row)) for row in rows]

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    finally:
        conn.close()

    return {
        "database_id": database_id,
        "data_type": series_row[4],
        "series_id": series_row[0],
        "grid_level": series_row[1],
        "name": series_row[2],
        "description": series_row[3],
        "datetime_range": {"start": start, "end": end},
        "records": data,
        "count": len(data),
        "truncated": truncated,
        **({"truncation_hint": f"Result capped at {MAX_RECORDS_PER_REQUEST} records; narrow with start/end."} if truncated else {}),
    }


def _classify_grid_level(n_mv: int, n_lv: int, n_feeder: int) -> str:
    """Which single grid_level a database's series share, for consumers
    (like PF Data Generation) that require homogeneity. "Mixed" if more
    than one level is present, "Empty" if the database has no series."""
    present = [level for level, n in (("MV", n_mv), ("LV", n_lv), ("Feeder", n_feeder)) if n > 0]
    if len(present) == 1:
        return present[0]
    return "Mixed" if present else "Empty"


def _fetch_database_summaries(cursor) -> list[dict]:
    """Each historical database's series counts per grid level, record count
    and shared timeline (number of timestamps, first and last).

    Record counts are the per-series counts stored in "HistoricalSeries" and
    kept exact by triggers on "HistoricalRecords" (see
    database/create_historical_database.py) -- counting the records table
    itself scans all of it (~90 s on the 41.8M-row import). So this reads
    only small tables and answers in well under a second, uncached."""
    cursor.execute(
        """
        SELECT d.database_id, d.name, d.data_type,
               COALESCE(sc.n_series, 0) AS n_series,
               COALESCE(sc.n_mv, 0) AS n_mv,
               COALESCE(sc.n_lv, 0) AS n_lv,
               COALESCE(sc.n_feeder, 0) AS n_feeder,
               COALESCE(sc.n_records, 0) AS n_records,
               COALESCE(tr.n_timestamps, 0) AS n_timestamps,
               tr.first_ts, tr.last_ts
        FROM "HistoricalDatabase" d
        LEFT JOIN (
            SELECT database_id, COUNT(*) AS n_series,
                   COUNT(*) FILTER (WHERE grid_level = 'MV') AS n_mv,
                   COUNT(*) FILTER (WHERE grid_level = 'LV') AS n_lv,
                   COUNT(*) FILTER (WHERE grid_level = 'Feeder') AS n_feeder,
                   SUM(n_records) AS n_records
            FROM "HistoricalSeries"
            GROUP BY database_id
        ) sc ON sc.database_id = d.database_id
        LEFT JOIN (
            SELECT database_id, COUNT(*) AS n_timestamps, MIN(datetime) AS first_ts, MAX(datetime) AS last_ts
            FROM "HistoricalTimestamp"
            GROUP BY database_id
        ) tr ON tr.database_id = d.database_id
        ORDER BY d.database_id COLLATE "C" ASC
        """
    )
    return [{
        "database_id": r[0],
        "name": r[1],
        "data_type": r[2],
        "n_series": r[3],
        "n_mv": r[4],
        "n_lv": r[5],
        "n_feeder": r[6],
        "n_records": int(r[7]),
        "n_timestamps": r[8],
        "first_ts": str(r[9])[:16] if r[9] else None,
        "last_ts": str(r[10])[:16] if r[10] else None,
    } for r in cursor.fetchall()]


def _fetch_series_summary(database_id: str, cursor) -> list[dict]:
    """A database's series with their stored record counts (see
    _fetch_database_summaries)."""
    cursor.execute(
        """
        SELECT series_id, grid_level, name, n_records, n_power, n_voltage
        FROM "HistoricalSeries"
        WHERE database_id = %s
        ORDER BY series_id COLLATE "C" ASC
        """,
        (database_id,),
    )
    return [{
        "series_id": r[0],
        "grid_level": r[1],
        "name": r[2],
        "n_records": r[3],
        "has_power": r[4] > 0,
        "has_voltage": r[5] > 0,
    } for r in cursor.fetchall()]


# ---------- JSON: DATABASES SUMMARY (for the PF Data Generation source picker) ----------


@router.get("/databases")
def list_historical_databases():
    """All historical databases with their data_type ("load" or "pv" for
    solar PV production), detected grid_level ("MV"/"LV"/"Feeder"/"Mixed"/
    "Empty"), series and record counts and timeline, for populating a
    data-source picker elsewhere in the app (e.g. PF Data Generation, which
    loads this as soon as the MV Grids page opens and only offers "load"
    databases as its load source)."""
    conn, cursor = db.get_db_connection()
    try:
        items = _fetch_database_summaries(cursor)
    except Exception:
        items = []
    finally:
        conn.close()

    for item in items:
        item["grid_level"] = _classify_grid_level(item["n_mv"], item["n_lv"], item["n_feeder"])

    return items


# ---------- UI: ROOT (list of databases) ----------


@router.get("/ui", response_class=HTMLResponse, include_in_schema=False)
def historical_ui_root(request: Request) -> HTMLResponse:
    conn, cursor = db.get_db_connection()
    try:
        items = _fetch_database_summaries(cursor)
    except Exception:
        items = []
    finally:
        conn.close()

    total_records = sum(d["n_records"] for d in items)
    total_series = sum(d["n_series"] for d in items)
    n_generation = sum(d["data_type"] != "load" for d in items)

    return templates.TemplateResponse(request, "historical/list.html", {
        "request": request,
        "title": "Historical",
        "active": "historical",
        "items": items,
        "total_records": total_records,
        "total_series": total_series,
        "n_generation": n_generation,
        "data_type_labels": DATA_TYPE_LABELS,
    })


# ---------- UI: DATABASE DETAIL (list of series within it) ----------


@router.get("/ui/{database_id}", response_class=HTMLResponse, include_in_schema=False)
def historical_ui_database(request: Request, database_id: str) -> HTMLResponse:
    conn, cursor = db.get_db_connection()
    try:
        cursor.execute(
            'SELECT database_id, name, description, data_type FROM "HistoricalDatabase" WHERE database_id = %s',
            (database_id,),
        )
        database_row = cursor.fetchone()
        if database_row is None:
            raise HTTPException(status_code=404, detail=f"Database '{database_id}' not found.")

        series_items = _fetch_series_summary(database_id, cursor)

        cursor.execute(
            'SELECT MIN(datetime), MAX(datetime), COUNT(*) FROM "HistoricalTimestamp" WHERE database_id = %s',
            (database_id,),
        )
        first_ts, last_ts, n_timestamps = cursor.fetchone()
    except HTTPException:
        raise
    finally:
        conn.close()

    return templates.TemplateResponse(request, "historical/database.html", {
        "request": request,
        "title": f"Historical: {database_id}",
        "active": "historical",
        "database_id": database_row[0],
        "name": database_row[1],
        "description": database_row[2],
        "data_type": database_row[3],
        "data_type_label": DATA_TYPE_LABELS.get(database_row[3], database_row[3]),
        "data_type_description": DATA_TYPE_DESCRIPTIONS.get(database_row[3], database_row[3]),
        "series_items": series_items,
        "n_timestamps": n_timestamps or 0,
        "first_ts": str(first_ts)[:16] if first_ts else None,
        "last_ts": str(last_ts)[:16] if last_ts else None,
    })


# ---------- UI: SERIES DETAIL ----------


@router.get("/ui/{database_id}/{series_id}", response_class=HTMLResponse, include_in_schema=False)
def historical_ui_series(
    request: Request,
    database_id: str,
    series_id: str,
    start: str | None = Query(None),
    end: str | None = Query(None),
    phase: list[str] | None = Query(None),
) -> HTMLResponse:
    """Dashboard view for a single series within a database."""
    conn, cursor = db.get_db_connection()

    try:
        cursor.execute(
            'SELECT s.series_id, s.grid_level, s.name, s.description, d.data_type FROM "HistoricalSeries" s '
            'JOIN "HistoricalDatabase" d ON d.database_id = s.database_id '
            "WHERE s.database_id = %s AND s.series_id = %s",
            (database_id, series_id),
        )
        series_row = cursor.fetchone()
        if series_row is None:
            raise HTTPException(status_code=404, detail=f"Series '{series_id}' not found in database '{database_id}'.")

        selected_phases = [p for p in (phase or []) if p in {"R", "S", "T"}]

        query = (
            "SELECT datetime, phase, power_active, power_reactive, voltage_magnitude, voltage_angle "
            'FROM "HistoricalRecords" WHERE database_id = %s AND series_id = %s'
        )
        params: list[object] = [database_id, series_id]

        if start:
            query += " AND datetime >= %s"
            params.append(start)
        if end:
            query += " AND datetime <= %s"
            params.append(end)
        if selected_phases:
            query += " AND phase = ANY(%s)"
            params.append(selected_phases)

        query += " ORDER BY datetime ASC, phase ASC LIMIT %s"
        params.append(MAX_RECORDS_PER_REQUEST)

        df = pd.read_sql_query(query, conn, params=params)
    finally:
        conn.close()

    grid_level = series_row[1]
    name = series_row[2]
    description = series_row[3]
    data_type = series_row[4]
    type_context = {"data_type": data_type, "data_type_label": DATA_TYPE_LABELS.get(data_type, data_type),
                    "data_type_description": DATA_TYPE_DESCRIPTIONS.get(data_type, data_type)}

    if df.empty:
        return templates.TemplateResponse(request, "historical/detail.html", {
            "request": request,
            "title": f"Historical: {series_id}",
            "active": "historical",
            "database_id": database_id,
            "series_id": series_id,
            "grid_level": grid_level,
            "name": name,
            "description": description,
            "empty": True,
            **type_context,
        })

    df["datetime"] = df["datetime"].astype(str)
    df["phase"] = df["phase"].fillna("N/A").astype(str)

    records = df.to_dict(orient="records")
    all_phases = sorted([p for p in df["phase"].unique().tolist() if p != "N/A"])
    has_power = df["power_active"].notna().any()
    has_voltage = df["voltage_magnitude"].notna().any()

    checked_r = not selected_phases or "R" in selected_phases
    checked_s = not selected_phases or "S" in selected_phases
    checked_t = not selected_phases or "T" in selected_phases

    return templates.TemplateResponse(request, "historical/detail.html", {
        "request": request,
        "title": f"Historical: {series_id}",
        "active": "historical",
        "database_id": database_id,
        "series_id": series_id,
        "grid_level": grid_level,
        "name": name,
        "description": description,
        "empty": False,
        "start": start,
        "end": end,
        "checked_r": checked_r,
        "checked_s": checked_s,
        "checked_t": checked_t,
        "records": records,
        "all_phases": all_phases,
        "has_power": bool(has_power),
        "has_voltage": bool(has_voltage),
        "record_count": len(records),
        **type_context,
    })
