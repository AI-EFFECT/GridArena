import io
import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException, UploadFile
from fastapi.testclient import TestClient
from pydantic import ValidationError

from gridarena.app import app
from gridarena.routers import mv_grid as mv_router

client = TestClient(app)
MV_PREFIX = "/mv_grid"


# ── Minimal valid MV grid payload ──

EXAMPLE_MV_GRID = {
    "mv_grid_id": "MV_TEST_01",
    "name": "Test MV Grid",
    "nominal_voltage_kv": 20.0,
    "region": "Test Region",
    "nodes": [
        {"node_id": "MV_SUB", "coord_lat": 41.15, "coord_lon": -8.61, "node_type": "substation"},
        {"node_id": "MV_N1", "coord_lat": 41.151, "coord_lon": -8.609},
        {"node_id": "MV_CP1", "coord_lat": 41.152, "coord_lon": -8.608, "node_type": "connection_point"},
    ],
    "cables": [
        {"cable_id": "MVC_01", "imp_real": 0.1, "imp_imag": 0.05, "nom_curr": 300.0},
    ],
    "connections": [
        {"connection_id": "MVCONN_01", "from_node_id": "MV_SUB", "to_node_id": "MV_N1", "cable_id": "MVC_01", "length": 500},
        {"connection_id": "MVCONN_02", "from_node_id": "MV_N1", "to_node_id": "MV_CP1", "cable_id": "MVC_01", "length": 300},
    ],
    "connection_points": [
        {"connection_point_id": "CP1", "node_id": "MV_CP1", "name": "Connection Point 1"},
    ],
    "transformers": [
        {
            "transformer_id": "TR1",
            "connection_point_id": "CP1",
            "rated_power_kva": 400.0,
            "primary_voltage_kv": 20.0,
            "secondary_voltage_v": 230.0,
            "imp_real": 0.004,
            "imp_imag": 0.04,
        }
    ],
}


# ═══════════════════════════════════════════════════════════════
#  Unit: _validate_mv_grid
# ═══════════════════════════════════════════════════════════════


def test_validate_mv_grid_valid():
    """Should return no errors for a valid MV grid."""
    from gridarena.schemas import MVGrid

    mv_grid = MVGrid(**EXAMPLE_MV_GRID)
    errors = mv_router._validate_mv_grid(mv_grid)
    assert errors == []


def test_validate_mv_grid_bad_connection_node():
    """Should catch connection referencing non-existent node."""
    from gridarena.schemas import MVGrid

    payload = {**EXAMPLE_MV_GRID}
    payload["connections"] = [
        {"connection_id": "BAD", "from_node_id": "MISSING", "to_node_id": "MV_N1", "cable_id": "MVC_01", "length": 100},
    ]
    mv_grid = MVGrid(**payload)
    errors = mv_router._validate_mv_grid(mv_grid)
    assert any("MISSING" in e for e in errors)


def test_validate_mv_grid_bad_cable_ref():
    """Should catch connection referencing non-existent cable."""
    from gridarena.schemas import MVGrid

    payload = {**EXAMPLE_MV_GRID}
    payload["connections"] = [
        {"connection_id": "BAD", "from_node_id": "MV_SUB", "to_node_id": "MV_N1", "cable_id": "NO_CABLE", "length": 100},
    ]
    mv_grid = MVGrid(**payload)
    errors = mv_router._validate_mv_grid(mv_grid)
    assert any("NO_CABLE" in e for e in errors)


def test_validate_mv_grid_bad_cp_node():
    """Should catch connection point referencing non-existent node."""
    from gridarena.schemas import MVGrid

    payload = {**EXAMPLE_MV_GRID}
    payload["connection_points"] = [
        {"connection_point_id": "CP_BAD", "node_id": "GHOST_NODE"},
    ]
    payload["transformers"] = []
    mv_grid = MVGrid(**payload)
    errors = mv_router._validate_mv_grid(mv_grid)
    assert any("GHOST_NODE" in e for e in errors)


def test_validate_mv_grid_duplicate_cp():
    """Should catch duplicate connection_point_id."""
    from gridarena.schemas import MVGrid

    payload = {**EXAMPLE_MV_GRID}
    payload["connection_points"] = [
        {"connection_point_id": "CP1", "node_id": "MV_CP1"},
        {"connection_point_id": "CP1", "node_id": "MV_CP1"},
    ]
    mv_grid = MVGrid(**payload)
    errors = mv_router._validate_mv_grid(mv_grid)
    assert any("Duplicate" in e for e in errors)


def test_validate_mv_grid_transformer_voltage_mismatch():
    """Should catch transformer primary_voltage_kv not matching grid nominal."""
    from gridarena.schemas import MVGrid

    payload = {**EXAMPLE_MV_GRID}
    payload["transformers"] = [
        {
            "transformer_id": "TR_BAD",
            "connection_point_id": "CP1",
            "rated_power_kva": 400.0,
            "primary_voltage_kv": 10.0,
            "secondary_voltage_v": 230.0,
        }
    ]
    mv_grid = MVGrid(**payload)
    errors = mv_router._validate_mv_grid(mv_grid)
    assert any("primary_voltage_kv" in e for e in errors)


# ═══════════════════════════════════════════════════════════════
#  Unit: _compute_graph_layout
# ═══════════════════════════════════════════════════════════════


def test_compute_graph_layout_returns_nodes_and_edges():
    """Should return graph structure with correct count."""
    grid_data = {
        "nodes": [
            {"node_id": "A", "coord_lat": 1.0, "coord_lon": 2.0, "node_type": "substation"},
            {"node_id": "B", "coord_lat": 1.1, "coord_lon": 2.1},
        ],
        "connections": [
            {"connection_id": "C1", "from_node_id": "A", "to_node_id": "B"},
        ],
        "connection_points": [],
        "transformers": [],
    }
    result = mv_router._compute_graph_layout(grid_data)
    assert len(result["nodes"]) == 2
    assert len(result["edges"]) == 1
    assert all("x" in n and "y" in n for n in result["nodes"])


# ═══════════════════════════════════════════════════════════════
#  Unit: validate_mv_grid endpoint
# ═══════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_validate_endpoint_valid():
    """Should return valid=True for a well-formed grid."""
    file_bytes = io.BytesIO(json.dumps(EXAMPLE_MV_GRID).encode("utf-8"))
    upload = UploadFile(filename="mv.json", file=file_bytes)
    result = await mv_router.validate_mv_grid(upload)
    assert result["valid"] is True
    assert "graph" in result


@pytest.mark.asyncio
async def test_validate_endpoint_invalid_json():
    """Should raise 400 for invalid JSON."""
    file_bytes = io.BytesIO(b"{ bad json }")
    upload = UploadFile(filename="bad.json", file=file_bytes)
    with pytest.raises(HTTPException) as exc:
        await mv_router.validate_mv_grid(upload)
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_validate_endpoint_schema_error():
    """Should raise 422 for schema validation failure."""
    file_bytes = io.BytesIO(json.dumps({"mv_grid_id": "X"}).encode("utf-8"))
    upload = UploadFile(filename="bad.json", file=file_bytes)
    with pytest.raises(HTTPException) as exc:
        await mv_router.validate_mv_grid(upload)
    assert exc.value.status_code == 422


@pytest.mark.asyncio
async def test_validate_endpoint_validation_errors():
    """Should return valid=False when internal references are broken."""
    payload = {**EXAMPLE_MV_GRID}
    payload["connections"] = [
        {"connection_id": "BAD", "from_node_id": "MISSING", "to_node_id": "MV_N1", "cable_id": "MVC_01", "length": 100},
    ]
    file_bytes = io.BytesIO(json.dumps(payload).encode("utf-8"))
    upload = UploadFile(filename="mv.json", file=file_bytes)
    result = await mv_router.validate_mv_grid(upload)
    assert result["valid"] is False
    assert len(result["errors"]) > 0


# ═══════════════════════════════════════════════════════════════
#  Unit: register_mv_grid
# ═══════════════════════════════════════════════════════════════


@pytest.mark.asyncio
@patch("gridarena.routers.mv_grid.db.insert_mv_grid_data")
@patch("gridarena.routers.mv_grid.db.create_mv_database")
@patch("gridarena.routers.mv_grid.db.get_db_connection")
async def test_register_mv_grid_success(mock_get_db, mock_create, mock_insert):
    """Should register a valid MV grid successfully."""
    conn = MagicMock()
    cursor = MagicMock()
    mock_get_db.return_value = (conn, cursor)

    file_bytes = io.BytesIO(json.dumps(EXAMPLE_MV_GRID).encode("utf-8"))
    upload = UploadFile(filename="mv.json", file=file_bytes)
    result = await mv_router.register_mv_grid(upload)

    assert "registered successfully" in result["message"]
    mock_create.assert_not_called()   # tables are created at startup, not per request
    mock_insert.assert_called_once()
    conn.close.assert_called_once()


@pytest.mark.asyncio
@patch("gridarena.routers.mv_grid.db.insert_mv_grid_data", side_effect=Exception("SQL error"))
@patch("gridarena.routers.mv_grid.db.create_mv_database")
@patch("gridarena.routers.mv_grid.db.get_db_connection")
async def test_register_mv_grid_db_error(mock_get_db, mock_create, mock_insert):
    """Should raise 500 on DB error."""
    conn = MagicMock()
    cursor = MagicMock()
    mock_get_db.return_value = (conn, cursor)

    file_bytes = io.BytesIO(json.dumps(EXAMPLE_MV_GRID).encode("utf-8"))
    upload = UploadFile(filename="mv.json", file=file_bytes)

    with pytest.raises(HTTPException) as exc:
        await mv_router.register_mv_grid(upload)
    assert exc.value.status_code == 500
    conn.rollback.assert_called_once()
    conn.close.assert_called_once()


# ═══════════════════════════════════════════════════════════════
#  Unit: delete_mv_grid
# ═══════════════════════════════════════════════════════════════


@pytest.mark.asyncio
@patch("gridarena.routers.mv_grid.db.delete_mv_grid")
@patch("gridarena.routers.mv_grid.db.create_mv_database")
@patch("gridarena.routers.mv_grid.db.get_db_connection")
async def test_delete_mv_grid_success(mock_get_db, mock_create, mock_delete):
    """Should delete grid and return confirmation."""
    conn = MagicMock()
    cursor = MagicMock()
    mock_get_db.return_value = (conn, cursor)

    result = mv_router.delete_mv_grid("MV_DEL")
    assert "deleted" in result["message"].lower()
    conn.close.assert_called_once()


@pytest.mark.asyncio
@patch("gridarena.routers.mv_grid.db.delete_mv_grid", side_effect=Exception("FK violation"))
@patch("gridarena.routers.mv_grid.db.create_mv_database")
@patch("gridarena.routers.mv_grid.db.get_db_connection")
async def test_delete_mv_grid_db_error(mock_get_db, mock_create, mock_delete):
    """Should raise 500 if delete fails."""
    conn = MagicMock()
    cursor = MagicMock()
    mock_get_db.return_value = (conn, cursor)

    with pytest.raises(HTTPException) as exc:
        mv_router.delete_mv_grid("MV_BAD")
    assert exc.value.status_code == 500
    conn.rollback.assert_called_once()
    conn.close.assert_called_once()


# ═══════════════════════════════════════════════════════════════
#  Unit: list_mv_grids
# ═══════════════════════════════════════════════════════════════


@pytest.mark.asyncio
@patch("gridarena.routers.mv_grid.db.get_mv_grid_list", return_value=[{"mv_grid_id": "G1"}])
@patch("gridarena.routers.mv_grid.db.create_mv_database")
@patch("gridarena.routers.mv_grid.db.get_db_connection")
async def test_list_mv_grids_success(mock_get_db, mock_create, mock_list):
    conn = MagicMock()
    cursor = MagicMock()
    mock_get_db.return_value = (conn, cursor)

    result = mv_router.list_mv_grids()
    assert len(result) == 1
    conn.close.assert_called_once()


# ═══════════════════════════════════════════════════════════════
#  Unit: get_mv_grid
# ═══════════════════════════════════════════════════════════════


@pytest.mark.asyncio
@patch("gridarena.routers.mv_grid.db.get_mv_grid_detail", return_value={"mv_grid_id": "G1", "nodes": []})
@patch("gridarena.routers.mv_grid.db.create_mv_database")
@patch("gridarena.routers.mv_grid.db.get_db_connection")
async def test_get_mv_grid_success(mock_get_db, mock_create, mock_detail):
    conn = MagicMock()
    cursor = MagicMock()
    mock_get_db.return_value = (conn, cursor)

    result = mv_router.get_mv_grid("G1")
    assert result["mv_grid_id"] == "G1"
    conn.close.assert_called_once()


@pytest.mark.asyncio
@patch("gridarena.routers.mv_grid.db.get_mv_grid_detail", side_effect=Exception("not found"))
@patch("gridarena.routers.mv_grid.db.create_mv_database")
@patch("gridarena.routers.mv_grid.db.get_db_connection")
async def test_get_mv_grid_not_found(mock_get_db, mock_create, mock_detail):
    conn = MagicMock()
    cursor = MagicMock()
    mock_get_db.return_value = (conn, cursor)

    with pytest.raises(HTTPException) as exc:
        mv_router.get_mv_grid("MISSING")
    assert exc.value.status_code == 500
    conn.close.assert_called_once()


# ═══════════════════════════════════════════════════════════════
#  Unit: connect / disconnect LV grid
# ═══════════════════════════════════════════════════════════════


@pytest.mark.asyncio
@patch("gridarena.routers.mv_grid.db.connect_lv_to_mv")
@patch("gridarena.routers.mv_grid.db.create_mv_database")
@patch("gridarena.routers.mv_grid.db.get_db_connection")
async def test_connect_lv_grid_success(mock_get_db, mock_create, mock_connect):
    conn = MagicMock()
    cursor = MagicMock()
    mock_get_db.return_value = (conn, cursor)

    from gridarena.schemas import ConnectLVRequest
    req = ConnectLVRequest(lv_grid_id="LV_01")

    result = mv_router.connect_lv_grid("MV_01", "CP1", req)
    assert "connected" in result["message"].lower()
    mock_connect.assert_called_once()
    conn.close.assert_called_once()


@pytest.mark.asyncio
@patch("gridarena.routers.mv_grid.db.connect_lv_to_mv", side_effect=Exception("already connected"))
@patch("gridarena.routers.mv_grid.db.create_mv_database")
@patch("gridarena.routers.mv_grid.db.get_db_connection")
async def test_connect_lv_grid_error(mock_get_db, mock_create, mock_connect):
    conn = MagicMock()
    cursor = MagicMock()
    mock_get_db.return_value = (conn, cursor)

    from gridarena.schemas import ConnectLVRequest
    req = ConnectLVRequest(lv_grid_id="LV_01")

    with pytest.raises(HTTPException) as exc:
        mv_router.connect_lv_grid("MV_01", "CP1", req)
    assert exc.value.status_code == 400
    conn.rollback.assert_called_once()
    conn.close.assert_called_once()


@pytest.mark.asyncio
@patch("gridarena.routers.mv_grid.db.disconnect_lv_from_mv")
@patch("gridarena.routers.mv_grid.db.get_db_connection")
async def test_disconnect_lv_grid_success(mock_get_db, mock_disconnect):
    conn = MagicMock()
    cursor = MagicMock()
    mock_get_db.return_value = (conn, cursor)

    result = mv_router.disconnect_lv_grid("MV_01", "CP1", "LV_01")
    assert "disconnected" in result["message"].lower()
    conn.close.assert_called_once()


# ═══════════════════════════════════════════════════════════════
#  Unit: configuration edits
# ═══════════════════════════════════════════════════════════════


def _config_grid():
    """A grid detail as get_mv_grid_detail returns it, one element of each editable kind."""
    return {
        "mv_grid_id": "MV_01",
        "connections": [
            {"connection_id": "C1", "from_node_id": "A", "to_node_id": "B", "cable_id": "K", "length": 100,
             "status": "active", "is_switch": False, "normally_open": False},
            {"connection_id": "SW", "from_node_id": "B", "to_node_id": "C", "cable_id": "K", "length": 100,
             "status": "active", "is_switch": True, "normally_open": True},
        ],
        "transformers": [{"transformer_id": "T1", "connection_point_id": "CP1", "lv_grid_id": None,
                          "rated_power_kva": 400.0, "primary_voltage_kv": 20.0, "secondary_voltage_v": 230.0,
                          "imp_real": None, "imp_imag": None, "name": None, "status": "active",
                          "tap_pos": None, "tap_step_pct": None, "tap_min": None, "tap_max": None}],
        "regulators": [{"regulator_id": "R1", "connection_id": "C1", "v_set_pu": 1.0, "deadband_pu": 0.02,
                        "tap_step_pct": 0.625, "tap_min": -16, "tap_max": 16, "tap_pos": 0}],
        "shunts": [{"shunt_id": "S1", "node_id": "B", "q_step_kvar": 150.0, "n_steps": 2, "step": 1,
                    "status": "active", "name": None}],
        "ders": [{"der_id": "D1", "node_id": "B", "type": "wind", "s_rated_kva": 1000.0, "p_max_kw": 1000.0,
                  "profile": "constant", "profile_value": 0.5, "control_mode": "fixed_pf", "cos_phi": 1.0,
                  "excitation": "under", "q_limit_pu": 0.33, "qv_curve": [0.95, 0.99, 1.01, 1.05],
                  "pv_v_start": 1.06, "pv_v_end": 1.09, "v_set_pu": 1.02, "status": "active", "name": None}],
    }


def test_merge_configuration_keeps_only_real_changes():
    from gridarena.schemas import MVConfigurationUpdate

    update = MVConfigurationUpdate(
        connections=[{"connection_id": "SW", "normally_open": False}, {"connection_id": "C1", "status": "active"}],
        transformers=[{"transformer_id": "T1", "tap_pos": -2, "status": "inactive"}],
        regulators=[{"regulator_id": "R1", "v_set_pu": 1.02}],
        shunts=[{"shunt_id": "S1", "step": 2}],
        ders=[{"der_id": "D1", "control_mode": "voltage", "v_set_pu": 1.01}],
    )
    changes, errors = mv_router._merge_configuration(_config_grid(), update)
    assert errors == []
    assert changes == {
        "connections": [{"connection_id": "SW", "normally_open": False}],   # C1 already active: dropped
        "transformers": [{"transformer_id": "T1", "tap_pos": -2, "status": "inactive"}],
        "regulators": [{"regulator_id": "R1", "v_set_pu": 1.02}],
        "shunts": [{"shunt_id": "S1", "step": 2}],
        "ders": [{"der_id": "D1", "control_mode": "voltage", "v_set_pu": 1.01}],
    }


def test_merge_configuration_applies_schema_rules():
    from gridarena.schemas import MVConfigurationUpdate

    update = MVConfigurationUpdate(
        connections=[{"connection_id": "C1", "normally_open": True}],       # not a switch
        transformers=[{"transformer_id": "T1", "tap_pos": 3}],              # default range -2..2
        regulators=[{"regulator_id": "R1", "deadband_pu": 0.5}],            # must be < 0.2
        shunts=[{"shunt_id": "S1", "step": 3}],                              # only 2 steps
        ders=[{"der_id": "NOPE", "status": "inactive"}],
    )
    changes, errors = mv_router._merge_configuration(_config_grid(), update)
    assert changes == {}
    text = " | ".join(errors)
    assert "normally_open needs is_switch" in text
    assert "tap_pos 3 must lie within" in text
    assert "Regulator 'R1', deadband_pu" in text
    assert "step 3 exceeds n_steps 2" in text
    assert "DER 'NOPE' not found" in text


def test_configuration_update_rejects_unknown_fields():
    from gridarena.schemas import MVConfigurationUpdate

    with pytest.raises(ValidationError):
        MVConfigurationUpdate(transformers=[{"transformer_id": "T1", "rated_power_kva": 1000}])
    with pytest.raises(ValidationError):
        MVConfigurationUpdate(shunts=[{"shunt_id": "S1", "status": "broken"}])


@pytest.mark.asyncio
@patch("gridarena.routers.mv_grid.db.update_mv_grid_configuration", return_value=1)
@patch("gridarena.routers.mv_grid.db.get_mv_grid_detail")
@patch("gridarena.routers.mv_grid.db.create_mv_database")
@patch("gridarena.routers.mv_grid.db.get_db_connection")
async def test_patch_configuration_writes_changes(mock_get_db, mock_create, mock_detail, mock_update):
    from gridarena.schemas import MVConfigurationUpdate

    conn, cursor = MagicMock(), MagicMock()
    mock_get_db.return_value = (conn, cursor)
    mock_detail.return_value = _config_grid()
    result = mv_router.patch_mv_grid_configuration(
        "MV_01", MVConfigurationUpdate(shunts=[{"shunt_id": "S1", "step": 0}]))
    assert result["updated"] == 1
    mock_update.assert_called_once_with("MV_01", {"shunts": [{"shunt_id": "S1", "step": 0}]}, conn, cursor)
    conn.close.assert_called_once()


@pytest.mark.asyncio
@patch("gridarena.routers.mv_grid.db.update_mv_grid_configuration")
@patch("gridarena.routers.mv_grid.db.get_mv_grid_detail")
@patch("gridarena.routers.mv_grid.db.create_mv_database")
@patch("gridarena.routers.mv_grid.db.get_db_connection")
async def test_patch_configuration_invalid_writes_nothing(mock_get_db, mock_create, mock_detail, mock_update):
    from gridarena.schemas import MVConfigurationUpdate

    conn, cursor = MagicMock(), MagicMock()
    mock_get_db.return_value = (conn, cursor)
    mock_detail.return_value = _config_grid()
    with pytest.raises(HTTPException) as exc:
        mv_router.patch_mv_grid_configuration(
            "MV_01", MVConfigurationUpdate(shunts=[{"shunt_id": "S1", "step": 5}]))
    assert exc.value.status_code == 422
    mock_update.assert_not_called()
    conn.rollback.assert_called_once()
    conn.close.assert_called_once()


# ═══════════════════════════════════════════════════════════════
#  Unit: get_lv_aggregated_pv
# ═══════════════════════════════════════════════════════════════


@pytest.mark.asyncio
@patch("gridarena.routers.mv_grid.db.get_aggregated_pv", return_value=[{"ts": "t1", "avg_p": 10, "avg_v": 230}])
@patch("gridarena.routers.mv_grid.db.get_db_connection")
async def test_get_lv_aggregated_pv_success(mock_get_db, mock_pv):
    conn = MagicMock()
    cursor = MagicMock()
    mock_get_db.return_value = (conn, cursor)

    cursor.fetchone.side_effect = [(1,), (5,)]

    result = mv_router.get_lv_aggregated_pv("MV_01", "LV_01")
    assert result["lv_grid_id"] == "LV_01"
    assert result["count"] == 1
    conn.close.assert_called_once()


@pytest.mark.asyncio
@patch("gridarena.routers.mv_grid.db.get_db_connection")
async def test_get_lv_aggregated_pv_not_connected(mock_get_db):
    conn = MagicMock()
    cursor = MagicMock()
    mock_get_db.return_value = (conn, cursor)
    cursor.fetchone.return_value = None

    with pytest.raises(HTTPException) as exc:
        mv_router.get_lv_aggregated_pv("MV_01", "LV_MISSING")
    assert exc.value.status_code == 404
    conn.close.assert_called_once()


# ═══════════════════════════════════════════════════════════════
#  Integration tests
# ═══════════════════════════════════════════════════════════════


def test_register_valid_mv_grid():
    """Should register a valid MV grid through the API."""
    file_bytes = io.BytesIO(json.dumps(EXAMPLE_MV_GRID).encode("utf-8"))
    response = client.post(
        f"{MV_PREFIX}/",
        files={"file": ("mv.json", file_bytes, "application/json")},
    )
    assert response.status_code == 200, response.text
    assert "registered successfully" in response.json()["message"]


def test_register_invalid_json():
    """Should fail with 400 for invalid JSON."""
    bad = io.BytesIO(b"{ broken }")
    response = client.post(
        f"{MV_PREFIX}/",
        files={"file": ("bad.json", bad, "application/json")},
    )
    assert response.status_code == 400


def test_validate_mv_grid_endpoint():
    """Should validate an MV grid via API without saving."""
    file_bytes = io.BytesIO(json.dumps(EXAMPLE_MV_GRID).encode("utf-8"))
    response = client.post(
        f"{MV_PREFIX}/validate",
        files={"file": ("mv.json", file_bytes, "application/json")},
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["valid"] is True
    assert "graph" in data
    assert len(data["graph"]["nodes"]) == 3
    assert len(data["graph"]["edges"]) == 2


# ── Unit: tie-line suggestions (Phase 2c) ──


def _tie_grid():
    nodes = [{"node_id": "S", "coord_lat": 41.0, "coord_lon": -8.0, "node_type": "substation"}] + [
        {"node_id": f"N{i}", "coord_lat": 41.0 + 0.0045 * (i % 3), "coord_lon": -8.0 + 0.004 * (i // 3),
         "node_type": "mv"} for i in range(1, 7)]
    order = ["S", "N1", "N2", "N5", "N4", "N3", "N6"]
    conns = [{"connection_id": f"L{i}", "from_node_id": a, "to_node_id": b, "cable_id": "C", "length": 1500.0,
              "status": "active", "is_switch": False, "normally_open": False}
             for i, (a, b) in enumerate(zip(order, order[1:]))]
    return {"nodes": nodes, "connections": conns, "cables": [{"cable_id": "C"}]}


@patch("gridarena.routers.mv_grid.db.insert_mv_ties", return_value=1)
@patch("gridarena.routers.mv_grid.db.get_mv_grid_detail")
@patch("gridarena.routers.mv_grid.db.get_db_connection")
def test_tie_endpoints_validate_before_writing(mock_conn, mock_detail, mock_insert):
    mock_conn.return_value = (MagicMock(), MagicMock())
    mock_detail.return_value = _tie_grid()
    r = client.get(f"{MV_PREFIX}/g1/tie-suggestions", params={"max_ties": 1})
    assert r.status_code == 200, r.text
    assert len(r.json()["suggestions"]) == 1 and r.json()["cable_id"] == "C"
    tie = {"connection_id": "L1", "from_node_id": "N1", "to_node_id": "N4", "cable_id": "C", "length": 400}
    r = client.post(f"{MV_PREFIX}/g1/ties", json={"ties": [tie]})
    assert r.status_code == 422 and "already used" in " ".join(r.json()["detail"]["validation_errors"])
    mock_insert.assert_not_called()
    r = client.post(f"{MV_PREFIX}/g1/ties", json={"ties": [dict(tie, connection_id="TIE01")]})
    assert r.status_code == 200 and r.json()["added"] == 1
    mock_insert.assert_called_once()
    assert client.post(f"{MV_PREFIX}/g1/ties", json={"ties": []}).status_code == 422


def test_list_mv_grids_integration():
    """Should list MV grids including the one we just registered."""
    response = client.get(f"{MV_PREFIX}/list")
    assert response.status_code == 200, response.text
    data = response.json()
    assert isinstance(data, list)
    assert any(g.get("mv_grid_id") == EXAMPLE_MV_GRID["mv_grid_id"] for g in data)


def test_get_mv_grid_detail_integration():
    """Should retrieve the full MV grid detail."""
    grid_id = EXAMPLE_MV_GRID["mv_grid_id"]
    response = client.get(f"{MV_PREFIX}/data/{grid_id}")
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["mv_grid_id"] == grid_id
    assert "nodes" in data


def test_delete_mv_grid_integration():
    """Should delete the MV grid."""
    grid_id = EXAMPLE_MV_GRID["mv_grid_id"]
    response = client.delete(f"{MV_PREFIX}/{grid_id}")
    assert response.status_code == 200, response.text
    assert "deleted" in response.json()["message"].lower()


# ── Synthetic grids (Phase 2e) ──


def _synth_fit():
    from math import log

    from gridarena.synthetic.calibrate import ArchetypeFit
    return ArchetypeFit(archetype="urban", n_grids=5, log_nodes=(log(20), 0.2), log_section=(log(250), 0.2),
                        section_spread=0.4, overhead_shares=[0.0], cp_share=0.7, branching=0.08, depth_ratio=0.8,
                        ratings={250.0: 1.0}, p_lateral=0.1, p_return=0.5,
                        real={k: [1.0] for k in ("n_nodes", "section_m", "depth", "far_km", "branching",
                                                  "cp_share", "overhead_share")})


@patch("gridarena.routers.mv_grid.db.insert_mv_grid_data")
@patch("gridarena.routers.mv_grid.db.get_db_connection")
def test_generate_synthetic_grids_endpoint(mock_conn, mock_insert):
    from gridarena.synthetic.planning import CableType

    mock_conn.return_value = (MagicMock(), MagicMock())
    cables = {"UG": [CableType("SYN-UG-S", 0.206, 0.096, 325.0, 0.109), CableType("SYN-UG-L", 0.125, 0.09, 428.0)]}
    cables["OH"] = cables["UG"]
    with patch("gridarena.synthetic.calibrate.calibrate", return_value={"urban": _synth_fit()}), \
            patch("gridarena.synthetic.batch.choose_cables", return_value=cables), \
            patch("gridarena.synthetic.batch.next_numbers", return_value={"urban": 3, "mixed": 1, "rural": 1}):
        r = client.post(f"{MV_PREFIX}/synthetic", json={"archetype": "urban", "count": 2, "seed": 4})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["created"] == ["SYN-urban-0003", "SYN-urban-0004"] and body["seed"] == 4
        assert body["report"]["archetypes"]["urban"]["generated"] == 2
        assert mock_insert.call_count == 2
        r = client.post(f"{MV_PREFIX}/synthetic", json={"archetype": "rural", "count": 1})
        assert r.status_code == 422 and "archetype 'rural'" in r.json()["detail"]
    assert client.post(f"{MV_PREFIX}/synthetic", json={"n_nodes_min": 50}).status_code == 422


@patch("gridarena.routers.mv_grid.db.delete_mv_grid")
@patch("gridarena.routers.mv_grid.db.get_db_connection")
def test_delete_synthetic_grids_endpoint(mock_conn, mock_delete):
    cursor = MagicMock()
    cursor.fetchall.return_value = [("SYN-urban-0001",), ("SYN-rural-0002",)]
    mock_conn.return_value = (MagicMock(), cursor)
    r = client.delete(f"{MV_PREFIX}/synthetic")
    assert r.status_code == 200 and r.json()["deleted"] == 2
    assert [c.args[0] for c in mock_delete.call_args_list] == ["SYN-urban-0001", "SYN-rural-0002"]
    assert "SYN-" in cursor.execute.call_args_list[0].args[1][0]
