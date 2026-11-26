import io
import json
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException, UploadFile
from fastapi.testclient import TestClient

from gridarena.app import app
from gridarena.routers import historical as hist_router
from gridarena.routers.historical import upload_historical_database
import gridarena.database as db


def _make_upload(payload: dict, filename: str = "database.json") -> UploadFile:
    file_bytes = io.BytesIO(json.dumps(payload).encode("utf-8"))
    return UploadFile(filename=filename, file=file_bytes)


@pytest.mark.asyncio
@patch("gridarena.database.insert_historical_data")
@patch("gridarena.database.create_historical_data_tables")
@patch("gridarena.database.get_db_connection")
async def test_upload_database_unit(mock_get_conn, mock_create_db, mock_insert_db):
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_get_conn.return_value = (mock_conn, mock_cursor)

    fake_db = {
        "database_id": "db123",
        "timestamps": ["2025-01-01T00:00:00"],
        "historicals": [
            {"series_id": "s1", "grid_level": "LV", "values": [
                {"datetime": "2025-01-01T00:00:00", "power_active": 10.0}
            ]}
        ],
    }
    upload = _make_upload(fake_db)

    result = await upload_historical_database(upload)

    assert "db123" in result["message"]
    assert "inserted successfully" in result["message"]
    mock_create_db.assert_not_called()   # tables are created at startup, not per request
    mock_insert_db.assert_called_once()
    mock_conn.close.assert_called_once()


@pytest.mark.asyncio
async def test_upload_historical_invalid_json():
    """Should raise 400 for invalid JSON."""
    bad_json = io.BytesIO(b"{invalid json")
    upload = UploadFile(filename="bad.json", file=bad_json)

    with pytest.raises(HTTPException) as exc:
        await hist_router.upload_historical_database(upload)

    assert exc.value.status_code == 400
    assert "Invalid JSON format" in exc.value.detail


@pytest.mark.asyncio
async def test_upload_historical_schema_error_missing_field():
    """Should raise 422 if schema validation fails (missing timestamps)."""
    bad_data = {"database_id": "db_bad", "historicals": []}
    upload = _make_upload(bad_data)

    with pytest.raises(HTTPException) as exc:
        await hist_router.upload_historical_database(upload)

    assert exc.value.status_code == 422


@pytest.mark.asyncio
async def test_upload_historical_schema_error_no_power_or_voltage():
    """Should raise 422 if a value has neither power nor voltage set."""
    bad_data = {
        "database_id": "db_bad",
        "timestamps": ["2025-01-01T00:00:00"],
        "historicals": [
            {"series_id": "s1", "grid_level": "MV", "values": [{"datetime": "2025-01-01T00:00:00"}]}
        ],
    }
    upload = _make_upload(bad_data)

    with pytest.raises(HTTPException) as exc:
        await hist_router.upload_historical_database(upload)

    assert exc.value.status_code == 422


@pytest.mark.asyncio
async def test_upload_historical_schema_error_datetime_not_in_timestamps():
    """Should raise 422 if a value's datetime isn't in the declared timestamps."""
    bad_data = {
        "database_id": "db_bad",
        "timestamps": ["2025-01-01T00:00:00"],
        "historicals": [
            {"series_id": "s1", "grid_level": "MV", "values": [
                {"datetime": "2025-01-01T01:00:00", "power_active": 1.0}
            ]}
        ],
    }
    upload = _make_upload(bad_data)

    with pytest.raises(HTTPException) as exc:
        await hist_router.upload_historical_database(upload)

    assert exc.value.status_code == 422


@pytest.mark.asyncio
async def test_upload_historical_schema_error_duplicate_series_id():
    """Should raise 422 if two historicals in the same upload share a series_id."""
    bad_data = {
        "database_id": "db_bad",
        "timestamps": ["2025-01-01T00:00:00"],
        "historicals": [
            {"series_id": "dup", "grid_level": "MV", "values": [
                {"datetime": "2025-01-01T00:00:00", "power_active": 1.0}
            ]},
            {"series_id": "dup", "grid_level": "LV", "values": [
                {"datetime": "2025-01-01T00:00:00", "power_active": 2.0}
            ]},
        ],
    }
    upload = _make_upload(bad_data)

    with pytest.raises(HTTPException) as exc:
        await hist_router.upload_historical_database(upload)

    assert exc.value.status_code == 422


@pytest.mark.asyncio
@patch("gridarena.database.insert_historical_data", side_effect=Exception("SQL insert error"))
@patch("gridarena.database.create_historical_data_tables")
@patch("gridarena.database.get_db_connection")
async def test_upload_historical_db_error(mock_get_conn, mock_create_tables, mock_insert):
    """Should raise 500 if database error occurs."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_get_conn.return_value = (mock_conn, mock_cursor)

    data = {
        "database_id": "db_fail",
        "timestamps": ["2025-01-01T00:00:00"],
        "historicals": [
            {"series_id": "s1", "grid_level": "LV", "values": [
                {"datetime": "2025-01-01T00:00:00", "power_active": 1.0}
            ]}
        ],
    }
    upload = _make_upload(data)

    with pytest.raises(HTTPException) as exc:
        await hist_router.upload_historical_database(upload)

    assert exc.value.status_code == 500
    assert "Database error" in exc.value.detail
    mock_conn.rollback.assert_called_once()
    mock_conn.close.assert_called_once()


# ---------------------------------------------------------------------
# delete_historical_database() / delete_historical_series_records()
# ---------------------------------------------------------------------
@pytest.mark.asyncio
@patch("gridarena.database.get_db_connection")
async def test_delete_database_success(mock_get_conn):
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.rowcount = 1
    mock_get_conn.return_value = (mock_conn, mock_cursor)

    result = hist_router.delete_historical_database("db_ok")
    assert result == {"message": "1 database(s) deleted for 'db_ok'."}
    mock_cursor.execute.assert_called_once()
    mock_conn.commit.assert_called_once()
    mock_conn.close.assert_called_once()


@pytest.mark.asyncio
@patch("gridarena.database.get_db_connection")
async def test_delete_series_records_success(mock_get_conn):
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.rowcount = 3
    mock_get_conn.return_value = (mock_conn, mock_cursor)

    result = hist_router.delete_historical_series_records("db_ok", "s1", start=None, end=None)
    assert result == {"message": "3 record(s) deleted for series 's1' in database 'db_ok'."}
    mock_cursor.execute.assert_called_once()
    mock_conn.commit.assert_called_once()
    mock_conn.close.assert_called_once()


@pytest.mark.asyncio
async def test_delete_series_records_invalid_datetime():
    with pytest.raises(HTTPException) as exc:
        hist_router.delete_historical_series_records("db_bad", "s1", start="bad_date")
    assert exc.value.status_code == 400
    assert "Invalid datetime format" in exc.value.detail


@pytest.mark.asyncio
@patch("gridarena.database.get_db_connection")
async def test_delete_series_records_with_filters(mock_get_conn):
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.rowcount = 1
    mock_get_conn.return_value = (mock_conn, mock_cursor)

    start = datetime(2025, 1, 1).isoformat()
    end = datetime(2025, 1, 2).isoformat()
    result = hist_router.delete_historical_series_records(
        "db_filters", "s1", phase="R", start=start, end=end
    )

    assert "deleted" in result["message"]
    query = mock_cursor.execute.call_args[0][0]
    assert "phase" in query and "datetime" in query
    mock_conn.commit.assert_called_once()


@pytest.mark.asyncio
@patch("gridarena.database.get_db_connection")
async def test_delete_series_records_db_error(mock_get_conn):
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.execute.side_effect = Exception("delete failed")
    mock_get_conn.return_value = (mock_conn, mock_cursor)

    with pytest.raises(HTTPException) as exc:
        hist_router.delete_historical_series_records("db_fail", "s1", start=None, end=None)
    assert exc.value.status_code == 500
    assert "Database error" in exc.value.detail
    mock_conn.rollback.assert_called_once()
    mock_conn.close.assert_called_once()


@pytest.mark.asyncio
@patch("gridarena.database.get_db_connection")
async def test_list_databases_never_scans_records(mock_get_conn):
    """The PF picker loads this when the MV Grids page opens; counting
    HistoricalRecords took ~90 s and froze the server, so record counts come
    from the stored per-series counts and the records table isn't queried."""
    mock_conn, mock_cursor = MagicMock(), MagicMock()
    mock_cursor.fetchall.return_value = [
        ("db1", "Feeder 2024", "load", 40, 0, 0, 40, 41862017, 1052640, datetime(2023, 3, 31, 22),
         datetime(2025, 3, 31, 21, 59)),
        ("db2", None, "pv", 0, 0, 0, 0, 0, 0, None, None),
    ]
    mock_get_conn.return_value = (mock_conn, mock_cursor)

    items = hist_router.list_historical_databases()

    sql = " ".join(str(c.args[0]) for c in mock_cursor.execute.call_args_list)
    assert '"HistoricalRecords"' not in sql
    assert items[0] == {"database_id": "db1", "name": "Feeder 2024", "data_type": "load", "n_series": 40,
                        "n_mv": 0, "n_lv": 0, "n_feeder": 40, "n_records": 41862017, "n_timestamps": 1052640,
                        "first_ts": "2023-03-31 22:00", "last_ts": "2025-03-31 21:59", "grid_level": "Feeder"}
    assert items[1]["grid_level"] == "Empty" and items[1]["first_ts"] is None
    assert items[1]["data_type"] == "pv"
    mock_conn.close.assert_called_once()


# ---------------------------------------------------------------------
# data_type: "load" (default) or "pv" (solar PV production)
# ---------------------------------------------------------------------
_PV_DB = {
    "database_id": "pv_unit",
    "timestamps": ["2025-06-01T12:00:00"],
    "historicals": [
        {"series_id": "pv1", "grid_level": "LV", "values": [
            {"datetime": "2025-06-01T12:00:00", "power_active": 3500.0}
        ]}
    ],
}


def test_upload_schema_data_type_defaults_to_load_and_accepts_known_types_only():
    from pydantic import ValidationError
    from gridarena.schemas import HistoricalDatabaseUpload
    assert HistoricalDatabaseUpload(**_PV_DB).data_type == "load"
    assert HistoricalDatabaseUpload(**_PV_DB, data_type="pv").data_type == "pv"
    with pytest.raises(ValidationError):
        HistoricalDatabaseUpload(**_PV_DB, data_type="nuclear")


@pytest.mark.asyncio
@patch("gridarena.database.insert_historical_data")
@patch("gridarena.database.get_db_connection")
async def test_upload_data_type_field_sets_the_type(mock_get_conn, mock_insert):
    mock_get_conn.return_value = (MagicMock(), MagicMock())
    result = await upload_historical_database(_make_upload(_PV_DB), data_type="pv")
    assert mock_insert.call_args[0][0].data_type == "pv"
    assert "solar PV production" in result["message"]


@pytest.mark.asyncio
@patch("gridarena.database.insert_historical_data")
@patch("gridarena.database.get_db_connection")
async def test_upload_data_type_field_must_match_the_file(mock_get_conn, mock_insert):
    mock_get_conn.return_value = (MagicMock(), MagicMock())
    with pytest.raises(HTTPException) as exc:
        await upload_historical_database(_make_upload({**_PV_DB, "data_type": "load"}), data_type="pv")
    assert exc.value.status_code == 422
    mock_insert.assert_not_called()


@pytest.mark.asyncio
@patch("gridarena.database.get_db_connection")
async def test_upload_into_a_database_of_another_type_is_a_conflict(mock_get_conn):
    from gridarena.database.insert_historical_database import HistoricalDataTypeConflict
    mock_conn = MagicMock()
    mock_get_conn.return_value = (mock_conn, MagicMock())
    with patch("gridarena.database.insert_historical_data",
               side_effect=HistoricalDataTypeConflict("exists with data_type 'load'")):
        with pytest.raises(HTTPException) as exc:
            await upload_historical_database(_make_upload({**_PV_DB, "data_type": "pv"}))
    assert exc.value.status_code == 409
    mock_conn.close.assert_called_once()


def test_insert_refuses_a_different_data_type_and_writes_nothing():
    from gridarena.database.insert_historical_database import (
        HistoricalDataTypeConflict, insert_historical_data,
    )
    from gridarena.schemas import HistoricalDatabaseUpload
    conn, cursor = MagicMock(), MagicMock()
    cursor.fetchone.return_value = ("load",)
    with pytest.raises(HistoricalDataTypeConflict):
        insert_historical_data(HistoricalDatabaseUpload(**_PV_DB, data_type="pv"), conn, cursor)
    assert cursor.execute.call_count == 1          # only the type lookup
    conn.rollback.assert_called_once()
    conn.commit.assert_not_called()


def test_pf_datagen_refuses_a_pv_database_as_load_source():
    from gridarena.pf_datagen.scenario_builder import get_historical_source_type
    cursor = MagicMock()
    cursor.fetchone.return_value = ("pv",)
    with pytest.raises(Exception, match="generation data"):
        get_historical_source_type("pv_db", cursor)


################################################# INTEGRATION TESTS #####################################################

client = TestClient(app)
HIST_PREFIX = "/historical"

DATABASE_ID = "test_db_001"
NOW = datetime.utcnow().replace(microsecond=0)
TIMESTAMPS = [
    (NOW - timedelta(hours=2)).isoformat(),
    (NOW - timedelta(hours=1)).isoformat(),
    NOW.isoformat(),
]

EXAMPLE_DATABASE = {
    "database_id": DATABASE_ID,
    "name": "Example database",
    "timestamps": TIMESTAMPS,
    "historicals": [
        {
            "series_id": "power_series",
            "grid_level": "LV",
            "values": [
                {"datetime": TIMESTAMPS[0], "phase": "R", "power_active": 12.3, "power_reactive": 4.5},
                {"datetime": TIMESTAMPS[1], "phase": "S", "power_active": 15.1, "power_reactive": 3.3},
                {"datetime": TIMESTAMPS[2], "phase": "R", "power_active": 10.8, "power_reactive": 2.9},
            ],
        },
        {
            "series_id": "voltage_series",
            "grid_level": "MV",
            "values": [
                {"datetime": TIMESTAMPS[0], "voltage_magnitude": 20000.0, "voltage_angle": 0.0},
                {"datetime": TIMESTAMPS[1], "voltage_magnitude": 19980.0, "voltage_angle": -0.1},
            ],
        },
    ],
}


def ensure_historical_tables():
    """Ensure all historical-database tables exist (idempotent)."""
    conn, cursor = db.get_db_connection()
    try:
        db.create_historical_data_tables(cursor)
        conn.commit()
    finally:
        conn.close()


def reset_database_data(database_id: str):
    """Ensure historical tables exist and remove a database entirely (cascades)."""
    ensure_historical_tables()
    conn, cursor = db.get_db_connection()
    try:
        cursor.execute('DELETE FROM "HistoricalDatabase" WHERE database_id = %s', (database_id,))
        conn.commit()
    finally:
        conn.close()


def prepare_example_database():
    """Prepare DATABASE_ID with EXAMPLE_DATABASE data, in a clean state."""
    reset_database_data(DATABASE_ID)
    file_bytes = io.BytesIO(json.dumps(EXAMPLE_DATABASE).encode("utf-8"))
    response = client.post(
        f"{HIST_PREFIX}/", files={"file": ("db.json", file_bytes, "application/json")}
    )
    assert response.status_code == 200, response.text


# ----------------------------
# Upload tests
# ----------------------------


def test_upload_valid_database_with_multiple_series():
    """Should successfully upload a database containing several series sharing one timeline."""
    reset_database_data(DATABASE_ID)

    file_bytes = io.BytesIO(json.dumps(EXAMPLE_DATABASE).encode("utf-8"))
    response = client.post(
        f"{HIST_PREFIX}/", files={"file": ("db.json", file_bytes, "application/json")}
    )

    assert response.status_code == 200, response.text
    data = response.json()
    assert "inserted successfully" in data["message"]
    assert DATABASE_ID in data["message"]
    assert "2 series" in data["message"]


def test_upload_feeder_level_series():
    """grid_level should also accept 'Feeder', not just 'MV'/'LV'."""
    database_id = "feeder_level_db"
    reset_database_data(database_id)

    data = {
        "database_id": database_id,
        "timestamps": [NOW.isoformat()],
        "historicals": [
            {"series_id": "feeder_series", "grid_level": "Feeder", "values": [
                {"datetime": NOW.isoformat(), "power_active": 42.0}
            ]}
        ],
    }
    file_bytes = io.BytesIO(json.dumps(data).encode("utf-8"))
    response = client.post(
        f"{HIST_PREFIX}/", files={"file": ("db.json", file_bytes, "application/json")}
    )
    assert response.status_code == 200, response.text

    get_resp = client.get(f"{HIST_PREFIX}/data/{database_id}/feeder_series")
    assert get_resp.status_code == 200, get_resp.text
    assert get_resp.json()["grid_level"] == "Feeder"


def test_upload_invalid_json():
    ensure_historical_tables()
    bad_json = io.BytesIO(b"{ invalid json }")
    response = client.post(
        f"{HIST_PREFIX}/", files={"file": ("bad.json", bad_json, "application/json")}
    )
    assert response.status_code == 400
    assert "Invalid JSON format" in response.json()["detail"]


def test_upload_schema_validation_error_missing_timestamps():
    ensure_historical_tables()
    incomplete_data = {"database_id": DATABASE_ID, "historicals": []}
    file_bytes = io.BytesIO(json.dumps(incomplete_data).encode("utf-8"))
    response = client.post(
        f"{HIST_PREFIX}/", files={"file": ("db_bad.json", file_bytes, "application/json")}
    )
    assert response.status_code == 422


def test_upload_datetime_not_in_timestamps_rejected():
    ensure_historical_tables()
    bad_data = {
        "database_id": "bad_timeline_db",
        "timestamps": [NOW.isoformat()],
        "historicals": [
            {"series_id": "s1", "grid_level": "LV", "values": [
                {"datetime": (NOW - timedelta(days=1)).isoformat(), "power_active": 1.0}
            ]}
        ],
    }
    file_bytes = io.BytesIO(json.dumps(bad_data).encode("utf-8"))
    response = client.post(
        f"{HIST_PREFIX}/", files={"file": ("db_bad.json", file_bytes, "application/json")}
    )
    assert response.status_code == 422


# ----------------------------
# Retrieval tests
# ----------------------------


def test_get_series_data():
    prepare_example_database()

    response = client.get(f"{HIST_PREFIX}/data/{DATABASE_ID}/power_series")
    assert response.status_code == 200, response.text
    data = response.json()

    assert data["database_id"] == DATABASE_ID
    assert data["series_id"] == "power_series"
    assert data["grid_level"] == "LV"
    assert data["count"] == 3


def test_get_series_data_power_only_has_null_voltage():
    prepare_example_database()

    response = client.get(f"{HIST_PREFIX}/data/{DATABASE_ID}/power_series")
    assert response.status_code == 200, response.text
    data = response.json()
    assert all(rec["voltage_magnitude"] is None for rec in data["records"])
    assert all(rec["power_active"] is not None for rec in data["records"])


def test_get_series_data_voltage_only_has_null_power():
    prepare_example_database()

    response = client.get(f"{HIST_PREFIX}/data/{DATABASE_ID}/voltage_series")
    assert response.status_code == 200, response.text
    data = response.json()
    assert all(rec["power_active"] is None for rec in data["records"])
    assert all(rec["voltage_magnitude"] is not None for rec in data["records"])


def test_get_series_with_phase_filter():
    prepare_example_database()

    response = client.get(f"{HIST_PREFIX}/data/{DATABASE_ID}/power_series?phase=R")
    assert response.status_code == 200, response.text
    data = response.json()
    assert len(data["records"]) == 2
    assert all(rec["phase"] == "R" for rec in data["records"])


def test_get_data_nonexistent_series():
    ensure_historical_tables()
    response = client.get(f"{HIST_PREFIX}/data/nonexistent_db_999/nonexistent_series")
    assert response.status_code == 404


def test_get_data_nonexistent_series_in_existing_database():
    prepare_example_database()
    response = client.get(f"{HIST_PREFIX}/data/{DATABASE_ID}/no_such_series")
    assert response.status_code == 404


# ----------------------------
# Delete tests
# ----------------------------


def test_delete_one_series_keeps_sibling_and_database():
    """Deleting one series' records should not affect its siblings or the database itself."""
    prepare_example_database()

    response = client.delete(f"{HIST_PREFIX}/{DATABASE_ID}/power_series")
    assert response.status_code == 200, response.text
    assert "deleted" in response.json()["message"].lower()

    # Sibling series untouched
    res_sibling = client.get(f"{HIST_PREFIX}/data/{DATABASE_ID}/voltage_series")
    assert res_sibling.status_code == 200
    assert res_sibling.json()["count"] == 2

    # Database page still resolves (registration survives even with 0 records for power_series)
    res_db = client.get(f"{HIST_PREFIX}/ui/{DATABASE_ID}")
    assert res_db.status_code == 200


def test_delete_whole_database_cascades():
    """Deleting a database should remove its series/records entirely."""
    prepare_example_database()

    response = client.delete(f"{HIST_PREFIX}/{DATABASE_ID}")
    assert response.status_code == 200, response.text
    assert "1 database" in response.json()["message"]

    res_series = client.get(f"{HIST_PREFIX}/data/{DATABASE_ID}/power_series")
    assert res_series.status_code == 404

    res_ui = client.get(f"{HIST_PREFIX}/ui/{DATABASE_ID}")
    assert res_ui.status_code == 404


def test_delete_nonexistent_database():
    ensure_historical_tables()
    response = client.delete(f"{HIST_PREFIX}/no_such_database_001")
    assert response.status_code == 200
    assert "0 database" in response.json()["message"]


# ----------------------------
# UI smoke tests
# ----------------------------


def test_ui_list_page_loads():
    response = client.get(f"{HIST_PREFIX}/ui")
    assert response.status_code == 200


def test_ui_database_page_loads_and_lists_series():
    prepare_example_database()
    response = client.get(f"{HIST_PREFIX}/ui/{DATABASE_ID}")
    assert response.status_code == 200


def test_ui_database_page_404_for_missing_database():
    ensure_historical_tables()
    response = client.get(f"{HIST_PREFIX}/ui/no_such_database_for_ui")
    assert response.status_code == 404


def test_ui_series_page_loads_for_existing_series():
    prepare_example_database()
    response = client.get(f"{HIST_PREFIX}/ui/{DATABASE_ID}/power_series")
    assert response.status_code == 200


def test_ui_series_page_404_for_missing_series():
    prepare_example_database()
    response = client.get(f"{HIST_PREFIX}/ui/{DATABASE_ID}/no_such_series_for_ui")
    assert response.status_code == 404


# ----------------------------
# Solar PV production databases (data_type "pv")
# ----------------------------

PV_DATABASE_ID = "test_db_pv_001"
PV_DATABASE = {
    "database_id": PV_DATABASE_ID,
    "name": "Rooftop PV",
    "timestamps": TIMESTAMPS,
    "historicals": [
        {
            "series_id": "pv_plant_1",
            "grid_level": "LV",
            "values": [
                {"datetime": TIMESTAMPS[0], "power_active": 0.0},
                {"datetime": TIMESTAMPS[1], "power_active": 2400.0},
                {"datetime": TIMESTAMPS[2], "power_active": -3.5},   # inverter standby draw is accepted
            ],
        },
    ],
}


def _post(payload, **form):
    file_bytes = io.BytesIO(json.dumps(payload).encode("utf-8"))
    return client.post(f"{HIST_PREFIX}/", files={"file": ("db.json", file_bytes, "application/json")}, data=form)


def test_pv_database_upload_is_identified_everywhere():
    reset_database_data(PV_DATABASE_ID)
    response = _post(PV_DATABASE, data_type="pv")
    assert response.status_code == 200, response.text
    assert "solar PV production" in response.json()["message"]

    listed = {d["database_id"]: d for d in client.get(f"{HIST_PREFIX}/databases").json()}
    assert listed[PV_DATABASE_ID]["data_type"] == "pv"
    assert listed[PV_DATABASE_ID]["grid_level"] == "LV"        # same structure as any other database

    data = client.get(f"{HIST_PREFIX}/data/{PV_DATABASE_ID}/pv_plant_1").json()
    assert data["data_type"] == "pv" and data["count"] == 3

    assert "badge--pv" in client.get(f"{HIST_PREFIX}/ui").text
    assert "badge--pv" in client.get(f"{HIST_PREFIX}/ui/{PV_DATABASE_ID}").text
    series_page = client.get(f"{HIST_PREFIX}/ui/{PV_DATABASE_ID}/pv_plant_1").text
    assert "badge--pv" in series_page and "Solar PV production" in series_page
    reset_database_data(PV_DATABASE_ID)


def test_pv_type_can_come_from_the_file():
    reset_database_data(PV_DATABASE_ID)
    assert _post({**PV_DATABASE, "data_type": "pv"}).status_code == 200
    listed = {d["database_id"]: d for d in client.get(f"{HIST_PREFIX}/databases").json()}
    assert listed[PV_DATABASE_ID]["data_type"] == "pv"
    reset_database_data(PV_DATABASE_ID)


def test_existing_databases_default_to_load():
    prepare_example_database()
    listed = {d["database_id"]: d for d in client.get(f"{HIST_PREFIX}/databases").json()}
    assert listed[DATABASE_ID]["data_type"] == "load"


def test_reupload_with_another_type_is_refused_and_keeps_the_database():
    reset_database_data(PV_DATABASE_ID)
    assert _post(PV_DATABASE, data_type="pv").status_code == 200
    response = _post(PV_DATABASE, data_type="load")
    assert response.status_code == 409, response.text
    listed = {d["database_id"]: d for d in client.get(f"{HIST_PREFIX}/databases").json()}
    assert listed[PV_DATABASE_ID]["data_type"] == "pv"
    assert _post(PV_DATABASE, data_type="pv").status_code == 200     # same type: re-upload still merges
    reset_database_data(PV_DATABASE_ID)


def test_unknown_data_type_is_rejected():
    response = _post(PV_DATABASE, data_type="nuclear")
    assert response.status_code == 422


@pytest.mark.parametrize("data_type, title", [("wind", "Wind production"), ("injection_total",
                                                                              "Total generation injected")])
def test_other_generation_types_are_identified(data_type, title):
    reset_database_data(PV_DATABASE_ID)
    assert _post(PV_DATABASE, data_type=data_type).status_code == 200
    listed = {d["database_id"]: d for d in client.get(f"{HIST_PREFIX}/databases").json()}
    assert listed[PV_DATABASE_ID]["data_type"] == data_type
    assert f"badge--{data_type}" in client.get(f"{HIST_PREFIX}/ui/{PV_DATABASE_ID}").text
    assert title in client.get(f"{HIST_PREFIX}/ui/{PV_DATABASE_ID}/pv_plant_1").text
    reset_database_data(PV_DATABASE_ID)


def test_pf_datagen_refuses_any_generation_database():
    from gridarena.pf_datagen.scenario_builder import get_historical_source_type
    cursor = MagicMock()
    cursor.fetchone.return_value = ("wind",)
    with pytest.raises(Exception, match="generation data"):
        get_historical_source_type("wind_db", cursor)
