"""Insert and query functions for Medium Voltage grid data."""

from .exceptions import NotFoundError


def insert_mv_grid_data(mv_grid, conn, cursor):
    """Insert a full MV grid (header, nodes, cables, connections, connection points, transformers, regulators,
    shunts, DER)."""
    _insert_mv_grid_entry(mv_grid, cursor)
    _insert_mv_nodes(mv_grid, cursor)
    _insert_mv_cables(mv_grid, cursor)
    _insert_mv_connections(mv_grid, cursor)
    _insert_mv_connection_points(mv_grid, cursor)
    _insert_mv_transformers(mv_grid, cursor)
    _insert_mv_regulators(mv_grid, cursor)
    _insert_mv_shunts(mv_grid, cursor)
    _insert_mv_ders(mv_grid, cursor)
    conn.commit()


def _insert_mv_grid_entry(mv_grid, cursor):
    cursor.execute(
        'INSERT INTO "MVGrid" (mv_grid_id, name, description, nominal_voltage_kv, region) '
        "VALUES (%s, %s, %s, %s, %s)",
        (mv_grid.mv_grid_id, mv_grid.name, mv_grid.description,
         mv_grid.nominal_voltage_kv, mv_grid.region),
    )


def _insert_mv_nodes(mv_grid, cursor):
    q = (
        'INSERT INTO "MVNode" ("NodeId", mv_grid_id, "CoordLat", "CoordLon", "NodeType") '
        "VALUES (%s, %s, %s, %s, %s)"
    )
    for node in mv_grid.nodes:
        cursor.execute(q, (node.node_id, mv_grid.mv_grid_id,
                           node.coord_lat, node.coord_lon, node.node_type))


def _insert_mv_cables(mv_grid, cursor):
    if not mv_grid.cables:
        return
    q = (
        'INSERT INTO "MVCable" ("CableId", mv_grid_id, "ImpReal", "ImpImag", "NomCurr", "CapUfPerKm", "EmergCurr") '
        "VALUES (%s, %s, %s, %s, %s, %s, %s)"
    )
    for cable in mv_grid.cables:
        cursor.execute(q, (cable.cable_id, mv_grid.mv_grid_id,
                           cable.imp_real, cable.imp_imag, cable.nom_curr,
                           cable.cap_uf_per_km, cable.emerg_curr))


def _insert_mv_connections(mv_grid, cursor):
    if not mv_grid.connections:
        return
    q = (
        'INSERT INTO "MVConnection" ("ConnectionId", mv_grid_id, "FromNodeId", "ToNodeId", "CableId", "Length", '
        '"Status", "IsSwitch", "NormallyOpen") '
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
    )
    for conn in mv_grid.connections:
        cursor.execute(q, (conn.connection_id, mv_grid.mv_grid_id,
                           conn.from_node_id, conn.to_node_id, conn.cable_id, conn.length,
                           conn.status, conn.is_switch, conn.normally_open))


def _insert_mv_connection_points(mv_grid, cursor):
    if not mv_grid.connection_points:
        return
    q = (
        'INSERT INTO "MVConnectionPoint" ("ConnectionPointId", mv_grid_id, "NodeId", "Name") '
        "VALUES (%s, %s, %s, %s)"
    )
    for cp in mv_grid.connection_points:
        cursor.execute(q, (cp.connection_point_id, mv_grid.mv_grid_id,
                           cp.node_id, cp.name))


def _insert_mv_transformers(mv_grid, cursor):
    if not mv_grid.transformers:
        return
    q = (
        'INSERT INTO "MVTransformer" '
        '("TransformerId", mv_grid_id, "ConnectionPointId", lv_grid_id, '
        '"RatedPowerKVA", "PrimaryVoltageKV", "SecondaryVoltageV", '
        '"ImpReal", "ImpImag", "Name", "Status", "TapPos", "TapStepPct", "TapMin", "TapMax") '
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
    )
    for t in mv_grid.transformers:
        cursor.execute(q, (
            t.transformer_id, mv_grid.mv_grid_id, t.connection_point_id,
            t.lv_grid_id, t.rated_power_kva, t.primary_voltage_kv,
            t.secondary_voltage_v, t.imp_real, t.imp_imag, t.name, t.status,
            t.tap_pos, t.tap_step_pct, t.tap_min, t.tap_max,
        ))


def _insert_mv_regulators(mv_grid, cursor):
    if not mv_grid.regulators:
        return
    q = (
        'INSERT INTO "MVRegulator" ("RegulatorId", mv_grid_id, "ConnectionId", "VSetPu", "DeadbandPu", '
        '"TapStepPct", "TapMin", "TapMax", "TapPos") '
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
    )
    for r in mv_grid.regulators:
        cursor.execute(q, (
            r.regulator_id, mv_grid.mv_grid_id, r.connection_id, r.v_set_pu, r.deadband_pu,
            r.tap_step_pct, r.tap_min, r.tap_max, r.tap_pos,
        ))


def _insert_mv_shunts(mv_grid, cursor):
    if not mv_grid.shunts:
        return
    q = (
        'INSERT INTO "MVShunt" ("ShuntId", mv_grid_id, "NodeId", "QStepKvar", "NSteps", "Step", "Status", "Name") '
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
    )
    for sh in mv_grid.shunts:
        cursor.execute(q, (
            sh.shunt_id, mv_grid.mv_grid_id, sh.node_id, sh.q_step_kvar, sh.n_steps, sh.step, sh.status, sh.name,
        ))


def _insert_mv_ders(mv_grid, cursor):
    if not mv_grid.ders:
        return
    q = (
        'INSERT INTO "MVDER" ("DerId", mv_grid_id, "NodeId", "DerType", "SRatedKva", "PMaxKw", "Profile", "ProfileValue", "ControlMode", "CosPhi", "Excitation", "QLimitPu", "QvV1", "QvV2", "QvV3", "QvV4", "PvVStart", "PvVEnd", "VSetPu", "Status", "Name") '
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
    )
    for d in mv_grid.ders:
        v1, v2, v3, v4 = d.qv_curve
        cursor.execute(q, (
            d.der_id, mv_grid.mv_grid_id, d.node_id, d.type, d.s_rated_kva, d.p_max_kw, d.profile, d.profile_value,
            d.control_mode, d.cos_phi, d.excitation, d.q_limit_pu, v1, v2, v3, v4,
            d.pv_v_start, d.pv_v_end, d.v_set_pu, d.status, d.name,
        ))


def connect_lv_to_mv(mv_grid_id, connection_point_id, lv_grid_id,
                      transformer_id, rated_power_kva, primary_voltage_kv,
                      secondary_voltage_v, imp_real, imp_imag, name, conn, cursor):
    """Connect an existing LV grid to an MV connection point via a new transformer."""
    cursor.execute(
        'SELECT 1 FROM "MVConnectionPoint" WHERE "ConnectionPointId" = %s AND mv_grid_id = %s',
        (connection_point_id, mv_grid_id),
    )
    if cursor.fetchone() is None:
        raise Exception(f"Connection point '{connection_point_id}' not found in MV grid '{mv_grid_id}'.")

    cursor.execute("SELECT 1 FROM grids WHERE grid_id = %s", (lv_grid_id,))
    if cursor.fetchone() is None:
        raise Exception(f"LV grid '{lv_grid_id}' not found.")

    cursor.execute(
        'SELECT 1 FROM "MVTransformer" WHERE mv_grid_id = %s AND "ConnectionPointId" = %s AND lv_grid_id = %s',
        (mv_grid_id, connection_point_id, lv_grid_id),
    )
    if cursor.fetchone() is not None:
        raise Exception(f"LV grid '{lv_grid_id}' is already connected to point '{connection_point_id}'.")

    cursor.execute(
        'INSERT INTO "MVTransformer" '
        '("TransformerId", mv_grid_id, "ConnectionPointId", lv_grid_id, '
        '"RatedPowerKVA", "PrimaryVoltageKV", "SecondaryVoltageV", '
        '"ImpReal", "ImpImag", "Name", "Status") '
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'active')",
        (transformer_id, mv_grid_id, connection_point_id, lv_grid_id,
         rated_power_kva, primary_voltage_kv, secondary_voltage_v,
         imp_real, imp_imag, name),
    )
    conn.commit()


def disconnect_lv_from_mv(mv_grid_id, connection_point_id, lv_grid_id, conn, cursor):
    """Remove the transformer bridge between an MV connection point and an LV grid."""
    cursor.execute(
        'DELETE FROM "MVTransformer" WHERE mv_grid_id = %s AND "ConnectionPointId" = %s AND lv_grid_id = %s',
        (mv_grid_id, connection_point_id, lv_grid_id),
    )
    if cursor.rowcount == 0:
        raise Exception(f"No connection found between point '{connection_point_id}' and LV grid '{lv_grid_id}'.")
    conn.commit()


# Editable operating settings: element kind -> (table, id column, id field, {field: column}).
_CONFIG_COLUMNS = {
    "connections": ("MVConnection", "ConnectionId", "connection_id",
                    {"status": "Status", "normally_open": "NormallyOpen"}),
    "transformers": ("MVTransformer", "TransformerId", "transformer_id",
                     {"status": "Status", "tap_pos": "TapPos"}),
    "regulators": ("MVRegulator", "RegulatorId", "regulator_id",
                   {"v_set_pu": "VSetPu", "deadband_pu": "DeadbandPu", "tap_pos": "TapPos"}),
    "shunts": ("MVShunt", "ShuntId", "shunt_id", {"step": "Step", "status": "Status"}),
    "ders": ("MVDER", "DerId", "der_id",
             {"status": "Status", "control_mode": "ControlMode", "cos_phi": "CosPhi", "excitation": "Excitation",
              "profile_value": "ProfileValue", "v_set_pu": "VSetPu", "q_limit_pu": "QLimitPu"}),
}


def update_mv_grid_configuration(mv_grid_id, changes, conn, cursor):
    """Write configuration edits in one transaction. `changes` maps an element
    kind (see _CONFIG_COLUMNS) to a list of {id field: ..., field: value}
    dicts holding only the fields to change. Returns the number of elements
    updated; raises NotFoundError (nothing written) if an element is missing."""
    updated = 0
    for kind, edits in changes.items():
        table, id_col, id_field, columns = _CONFIG_COLUMNS[kind]
        for edit in edits:
            fields = [f for f in columns if f in edit]
            if not fields:
                continue
            assignments = ", ".join(f'"{columns[f]}" = %s' for f in fields)
            cursor.execute(
                f'UPDATE "{table}" SET {assignments} WHERE mv_grid_id = %s AND "{id_col}" = %s',
                [edit[f] for f in fields] + [mv_grid_id, edit[id_field]],
            )
            if cursor.rowcount == 0:
                conn.rollback()
                raise NotFoundError(f"{id_field} '{edit[id_field]}' not found in MV grid '{mv_grid_id}'.")
            updated += 1
    conn.commit()
    return updated


def insert_mv_ties(mv_grid_id, ties, conn, cursor):
    """Store tie lines (schemas.MVTieLine) as active, normally-open switches,
    in one transaction. Returns how many were added."""
    for t in ties:
        cursor.execute(
            'INSERT INTO "MVConnection" ("ConnectionId", mv_grid_id, "FromNodeId", "ToNodeId", "CableId", "Length", '
            '"Status", "IsSwitch", "NormallyOpen") VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)',
            (t.connection_id, mv_grid_id, t.from_node_id, t.to_node_id, t.cable_id, t.length, "active", True, True),
        )
    conn.commit()
    return len(ties)


def get_mv_grid_list(cursor):
    cursor.execute('SELECT mv_grid_id, name, nominal_voltage_kv, region FROM "MVGrid" ORDER BY mv_grid_id')
    return [{"mv_grid_id": r[0], "name": r[1], "nominal_voltage_kv": r[2], "region": r[3]}
            for r in cursor.fetchall()]


def get_mv_grid_detail(mv_grid_id, cursor):
    cursor.execute('SELECT mv_grid_id, name, description, nominal_voltage_kv, region FROM "MVGrid" WHERE mv_grid_id = %s', (mv_grid_id,))
    row = cursor.fetchone()
    if row is None:
        raise NotFoundError(f"MV Grid '{mv_grid_id}' not found.")
    grid = {"mv_grid_id": row[0], "name": row[1], "description": row[2], "nominal_voltage_kv": row[3], "region": row[4]}

    cursor.execute('SELECT "NodeId", "CoordLat", "CoordLon", "NodeType" FROM "MVNode" WHERE mv_grid_id = %s', (mv_grid_id,))
    grid["nodes"] = [{"node_id": r[0], "coord_lat": r[1], "coord_lon": r[2], "node_type": r[3]} for r in cursor.fetchall()]

    cursor.execute(
        'SELECT "CableId", "ImpReal", "ImpImag", "NomCurr", "CapUfPerKm", "EmergCurr" '
        'FROM "MVCable" WHERE mv_grid_id = %s ORDER BY "CableId"',
        (mv_grid_id,),
    )
    grid["cables"] = [
        {"cable_id": r[0], "imp_real": r[1], "imp_imag": r[2], "nom_curr": r[3],
         "cap_uf_per_km": r[4], "emerg_curr": r[5]}
        for r in cursor.fetchall()
    ]

    cursor.execute(
        'SELECT "ConnectionId", "FromNodeId", "ToNodeId", "CableId", "Length", "Status", "IsSwitch", "NormallyOpen" '
        'FROM "MVConnection" WHERE mv_grid_id = %s', (mv_grid_id,),
    )
    grid["connections"] = [
        {"connection_id": r[0], "from_node_id": r[1], "to_node_id": r[2], "cable_id": r[3], "length": r[4],
         "status": r[5] or "active", "is_switch": bool(r[6]), "normally_open": bool(r[7])}
        for r in cursor.fetchall()
    ]

    cursor.execute('SELECT "ConnectionPointId", "NodeId", "Name" FROM "MVConnectionPoint" WHERE mv_grid_id = %s', (mv_grid_id,))
    grid["connection_points"] = [{"connection_point_id": r[0], "node_id": r[1], "name": r[2]} for r in cursor.fetchall()]

    cursor.execute(
        'SELECT "TransformerId", "ConnectionPointId", lv_grid_id, "RatedPowerKVA", "PrimaryVoltageKV", '
        '"SecondaryVoltageV", "ImpReal", "ImpImag", "Name", "Status", "TapPos", "TapStepPct", "TapMin", "TapMax" '
        'FROM "MVTransformer" WHERE mv_grid_id = %s', (mv_grid_id,),
    )
    grid["transformers"] = [{
        "transformer_id": r[0], "connection_point_id": r[1], "lv_grid_id": r[2],
        "rated_power_kva": r[3], "primary_voltage_kv": r[4], "secondary_voltage_v": r[5],
        "imp_real": r[6], "imp_imag": r[7], "name": r[8], "status": r[9],
        "tap_pos": r[10], "tap_step_pct": r[11], "tap_min": r[12], "tap_max": r[13],
    } for r in cursor.fetchall()]

    cursor.execute(
        'SELECT "RegulatorId", "ConnectionId", "VSetPu", "DeadbandPu", "TapStepPct", "TapMin", "TapMax", "TapPos" '
        'FROM "MVRegulator" WHERE mv_grid_id = %s ORDER BY "RegulatorId"', (mv_grid_id,),
    )
    grid["regulators"] = [{
        "regulator_id": r[0], "connection_id": r[1], "v_set_pu": r[2], "deadband_pu": r[3],
        "tap_step_pct": r[4], "tap_min": r[5], "tap_max": r[6], "tap_pos": r[7],
    } for r in cursor.fetchall()]

    cursor.execute(
        'SELECT "ShuntId", "NodeId", "QStepKvar", "NSteps", "Step", "Status", "Name" '
        'FROM "MVShunt" WHERE mv_grid_id = %s ORDER BY "ShuntId"', (mv_grid_id,),
    )
    grid["shunts"] = [{
        "shunt_id": r[0], "node_id": r[1], "q_step_kvar": r[2], "n_steps": r[3], "step": r[4],
        "status": r[5], "name": r[6],
    } for r in cursor.fetchall()]

    cursor.execute(
        'SELECT "DerId", "NodeId", "DerType", "SRatedKva", "PMaxKw", "Profile", "ProfileValue", "ControlMode", '
        '"CosPhi", "Excitation", "QLimitPu", "QvV1", "QvV2", "QvV3", "QvV4", "PvVStart", "PvVEnd", "VSetPu", '
        '"Status", "Name" FROM "MVDER" WHERE mv_grid_id = %s ORDER BY "DerId"', (mv_grid_id,),
    )
    grid["ders"] = [{
        "der_id": r[0], "node_id": r[1], "type": r[2], "s_rated_kva": r[3], "p_max_kw": r[4],
        "profile": r[5], "profile_value": r[6], "control_mode": r[7], "cos_phi": r[8], "excitation": r[9],
        "q_limit_pu": r[10], "qv_curve": [r[11], r[12], r[13], r[14]], "pv_v_start": r[15], "pv_v_end": r[16],
        "v_set_pu": r[17], "status": r[18], "name": r[19],
    } for r in cursor.fetchall()]

    return grid


def get_aggregated_pv(lv_grid_id, cursor, limit=500):
    """Return aggregated P/Q/V per timestamp for an LV grid from the Measurements table.

    Returns the average power_active, power_reactive and voltage_magnitude per
    timestamp across all nodes/phases, ordered by time -- a representative
    per-meter profile, used for the "View P/V" chart on the MV grid detail page.
    Not a total load figure: see get_summed_pv() for that (used by the PF
    data-generation pipeline).
    """
    query = """
        SELECT datetime,
               AVG(power_active) AS avg_power,
               AVG(power_reactive) AS avg_reactive,
               AVG(voltage_magnitude) AS avg_voltage
        FROM "Measurements"
        WHERE grid_id = %s AND phase IS NOT NULL
        GROUP BY datetime
        ORDER BY datetime ASC
    """
    params = [lv_grid_id]
    if limit is not None:
        query += " LIMIT %s"
        params.append(limit)

    cursor.execute(query, params)
    return [
        {"datetime": str(r[0]), "avg_power_active": r[1], "avg_power_reactive": r[2], "avg_voltage_magnitude": r[3]}
        for r in cursor.fetchall()
    ]


def get_summed_pv(lv_grid_id, cursor, limit=None):
    """Return total P/Q per timestamp for an LV grid from the Measurements table.

    Returns the sum of power_active and power_reactive per timestamp across
    all nodes/phases, ordered by time -- the LV grid's total real/reactive
    power draw at that instant, i.e. what the MV/LV transformer's secondary
    side actually carries (by power conservation). Used by the PF
    data-generation pipeline as the MV connection-point injection; unlike
    get_aggregated_pv(), this is not sensitive to how many points the LV
    grid happens to have metered. Pass limit=None to fetch the full history.
    """
    query = """
        SELECT datetime,
               SUM(power_active) AS total_power,
               SUM(power_reactive) AS total_reactive
        FROM "Measurements"
        WHERE grid_id = %s AND phase IS NOT NULL
        GROUP BY datetime
        ORDER BY datetime ASC
    """
    params = [lv_grid_id]
    if limit is not None:
        query += " LIMIT %s"
        params.append(limit)

    cursor.execute(query, params)
    return [
        {"datetime": str(r[0]), "total_power_active": r[1], "total_power_reactive": r[2]}
        for r in cursor.fetchall()
    ]
