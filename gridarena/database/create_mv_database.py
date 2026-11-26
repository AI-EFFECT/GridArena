"""Creates the database tables for Medium Voltage grids."""

from .exceptions import NotFoundError


def create_mv_grids_table(cursor):
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS "MVGrid" (
            mv_grid_id TEXT PRIMARY KEY,
            name TEXT,
            description TEXT,
            nominal_voltage_kv DOUBLE PRECISION NOT NULL,
            region TEXT
        );
        """
    )


def create_mv_node_table(cursor):
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS "MVNode" (
            "NodeId" VARCHAR(40) NOT NULL,
            mv_grid_id TEXT NOT NULL,
            "CoordLat" DOUBLE PRECISION,
            "CoordLon" DOUBLE PRECISION,
            "NodeType" VARCHAR(20),
            CONSTRAINT mv_node_pk PRIMARY KEY ("NodeId", mv_grid_id),
            CONSTRAINT mv_node_grid_fk FOREIGN KEY (mv_grid_id)
                REFERENCES "MVGrid"(mv_grid_id) ON DELETE CASCADE
        );
        """
    )


def create_mv_cable_table(cursor):
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS "MVCable" (
            "CableId" VARCHAR(40) NOT NULL,
            mv_grid_id TEXT NOT NULL,
            "ImpReal" DOUBLE PRECISION,
            "ImpImag" DOUBLE PRECISION,
            "NomCurr" DOUBLE PRECISION,
            CONSTRAINT mv_cable_pk PRIMARY KEY ("CableId", mv_grid_id),
            CONSTRAINT mv_cable_grid_fk FOREIGN KEY (mv_grid_id)
                REFERENCES "MVGrid"(mv_grid_id) ON DELETE CASCADE
        );
        """
    )
    # Added after the table first shipped (optional, NULL = not modelled):
    # per-phase charging capacitance (uF/km at 50 Hz) and emergency rating (A).
    cursor.execute('ALTER TABLE "MVCable" ADD COLUMN IF NOT EXISTS "CapUfPerKm" DOUBLE PRECISION;')
    cursor.execute('ALTER TABLE "MVCable" ADD COLUMN IF NOT EXISTS "EmergCurr" DOUBLE PRECISION;')


def create_mv_connection_table(cursor):
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS "MVConnection" (
            "ConnectionId" VARCHAR(40) NOT NULL,
            mv_grid_id TEXT NOT NULL,
            "FromNodeId" VARCHAR(40),
            "ToNodeId" VARCHAR(40),
            "CableId" VARCHAR(40),
            "Length" DOUBLE PRECISION,
            CONSTRAINT mv_connection_pk PRIMARY KEY ("ConnectionId", mv_grid_id),
            CONSTRAINT mv_connection_grid_fk FOREIGN KEY (mv_grid_id)
                REFERENCES "MVGrid"(mv_grid_id) ON DELETE CASCADE,
            CONSTRAINT mv_connection_from_fk FOREIGN KEY ("FromNodeId", mv_grid_id)
                REFERENCES "MVNode"("NodeId", mv_grid_id) ON DELETE CASCADE,
            CONSTRAINT mv_connection_to_fk FOREIGN KEY ("ToNodeId", mv_grid_id)
                REFERENCES "MVNode"("NodeId", mv_grid_id) ON DELETE CASCADE,
            CONSTRAINT mv_connection_cable_fk FOREIGN KEY ("CableId", mv_grid_id)
                REFERENCES "MVCable"("CableId", mv_grid_id) ON DELETE CASCADE
        );
        """
    )
    # Added after the table first shipped (NULL = active, not a switch):
    # status, and switchable / normally-open connections (tie lines).
    for col, col_type in (("Status", "VARCHAR(20)"), ("IsSwitch", "BOOLEAN"), ("NormallyOpen", "BOOLEAN")):
        cursor.execute(f'ALTER TABLE "MVConnection" ADD COLUMN IF NOT EXISTS "{col}" {col_type};')


def create_mv_connection_point_table(cursor):
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS "MVConnectionPoint" (
            "ConnectionPointId" VARCHAR(40) NOT NULL,
            mv_grid_id TEXT NOT NULL,
            "NodeId" VARCHAR(40) NOT NULL,
            "Name" TEXT,
            CONSTRAINT mv_cp_pk PRIMARY KEY ("ConnectionPointId", mv_grid_id),
            CONSTRAINT mv_cp_grid_fk FOREIGN KEY (mv_grid_id)
                REFERENCES "MVGrid"(mv_grid_id) ON DELETE CASCADE,
            CONSTRAINT mv_cp_node_fk FOREIGN KEY ("NodeId", mv_grid_id)
                REFERENCES "MVNode"("NodeId", mv_grid_id) ON DELETE CASCADE
        );
        """
    )


def create_mv_transformer_table(cursor):
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS "MVTransformer" (
            "TransformerId" VARCHAR(40) NOT NULL,
            mv_grid_id TEXT NOT NULL,
            "ConnectionPointId" VARCHAR(40) NOT NULL,
            lv_grid_id TEXT,
            "RatedPowerKVA" DOUBLE PRECISION NOT NULL,
            "PrimaryVoltageKV" DOUBLE PRECISION NOT NULL,
            "SecondaryVoltageV" DOUBLE PRECISION DEFAULT 230.0,
            "ImpReal" DOUBLE PRECISION,
            "ImpImag" DOUBLE PRECISION,
            "Name" TEXT,
            "Status" VARCHAR(20) DEFAULT 'active',
            CONSTRAINT mv_transformer_pk PRIMARY KEY ("TransformerId", mv_grid_id),
            CONSTRAINT mv_transformer_grid_fk FOREIGN KEY (mv_grid_id)
                REFERENCES "MVGrid"(mv_grid_id) ON DELETE CASCADE,
            CONSTRAINT mv_transformer_cp_fk FOREIGN KEY ("ConnectionPointId", mv_grid_id)
                REFERENCES "MVConnectionPoint"("ConnectionPointId", mv_grid_id) ON DELETE CASCADE,
            CONSTRAINT mv_transformer_lv_fk FOREIGN KEY (lv_grid_id)
                REFERENCES grids(grid_id) ON DELETE SET NULL,
            CONSTRAINT mv_transformer_unique_lv UNIQUE (mv_grid_id, "ConnectionPointId", lv_grid_id)
        );
        """
    )
    # Added after the table first shipped (optional, NULL = defaults: position
    # 0, 2.5 % steps, -2..+2): fixed off-load tap; positive raises LV voltage.
    for col, col_type in (("TapPos", "INTEGER"), ("TapStepPct", "DOUBLE PRECISION"),
                          ("TapMin", "INTEGER"), ("TapMax", "INTEGER")):
        cursor.execute(f'ALTER TABLE "MVTransformer" ADD COLUMN IF NOT EXISTS "{col}" {col_type};')


def create_mv_regulator_table(cursor):
    """Step voltage regulators, one per MV connection at most (see
    schemas/mv_grid_schema.py::Regulator)."""
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS "MVRegulator" (
            "RegulatorId" VARCHAR(40) NOT NULL,
            mv_grid_id TEXT NOT NULL,
            "ConnectionId" VARCHAR(40) NOT NULL,
            "VSetPu" DOUBLE PRECISION NOT NULL,
            "DeadbandPu" DOUBLE PRECISION NOT NULL,
            "TapStepPct" DOUBLE PRECISION NOT NULL,
            "TapMin" INTEGER NOT NULL,
            "TapMax" INTEGER NOT NULL,
            "TapPos" INTEGER NOT NULL,
            CONSTRAINT mv_regulator_pk PRIMARY KEY ("RegulatorId", mv_grid_id),
            CONSTRAINT mv_regulator_conn_fk FOREIGN KEY ("ConnectionId", mv_grid_id)
                REFERENCES "MVConnection"("ConnectionId", mv_grid_id) ON DELETE CASCADE,
            CONSTRAINT mv_regulator_unique_conn UNIQUE (mv_grid_id, "ConnectionId")
        );
        """
    )


def create_mv_shunt_table(cursor):
    """Capacitor banks (QStepKvar > 0) and shunt reactors (< 0) at MV nodes
    (see schemas/mv_grid_schema.py::Shunt)."""
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS "MVShunt" (
            "ShuntId" VARCHAR(40) NOT NULL,
            mv_grid_id TEXT NOT NULL,
            "NodeId" VARCHAR(40) NOT NULL,
            "QStepKvar" DOUBLE PRECISION NOT NULL,
            "NSteps" INTEGER NOT NULL,
            "Step" INTEGER NOT NULL,
            "Status" VARCHAR(20),
            "Name" TEXT,
            CONSTRAINT mv_shunt_pk PRIMARY KEY ("ShuntId", mv_grid_id),
            CONSTRAINT mv_shunt_node_fk FOREIGN KEY ("NodeId", mv_grid_id)
                REFERENCES "MVNode"("NodeId", mv_grid_id) ON DELETE CASCADE
        );
        """
    )


def create_mv_der_table(cursor):
    """MV-connected generators, batteries and EV hubs (see
    schemas/mv_grid_schema.py::DER): rating, profile and control settings."""
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS "MVDER" (
            "DerId" VARCHAR(40) NOT NULL,
            mv_grid_id TEXT NOT NULL,
            "NodeId" VARCHAR(40) NOT NULL,
            "DerType" VARCHAR(20) NOT NULL,
            "SRatedKva" DOUBLE PRECISION NOT NULL,
            "PMaxKw" DOUBLE PRECISION NOT NULL,
            "Profile" VARCHAR(20) NOT NULL,
            "ProfileValue" DOUBLE PRECISION NOT NULL,
            "ControlMode" VARCHAR(20) NOT NULL,
            "CosPhi" DOUBLE PRECISION NOT NULL,
            "Excitation" VARCHAR(10) NOT NULL,
            "QLimitPu" DOUBLE PRECISION NOT NULL,
            "QvV1" DOUBLE PRECISION NOT NULL,
            "QvV2" DOUBLE PRECISION NOT NULL,
            "QvV3" DOUBLE PRECISION NOT NULL,
            "QvV4" DOUBLE PRECISION NOT NULL,
            "PvVStart" DOUBLE PRECISION NOT NULL,
            "PvVEnd" DOUBLE PRECISION NOT NULL,
            "VSetPu" DOUBLE PRECISION NOT NULL,
            "Status" VARCHAR(20),
            "Name" TEXT,
            CONSTRAINT mv_der_pk PRIMARY KEY ("DerId", mv_grid_id),
            CONSTRAINT mv_der_node_fk FOREIGN KEY ("NodeId", mv_grid_id)
                REFERENCES "MVNode"("NodeId", mv_grid_id) ON DELETE CASCADE
        );
        """
    )


def create_mv_database(conn, cursor):
    """Ensures all MV tables exist. Idempotent via IF NOT EXISTS."""
    try:
        create_mv_grids_table(cursor)
        create_mv_node_table(cursor)
        create_mv_cable_table(cursor)
        create_mv_connection_table(cursor)
        create_mv_connection_point_table(cursor)
        create_mv_transformer_table(cursor)
        create_mv_regulator_table(cursor)
        create_mv_shunt_table(cursor)
        create_mv_der_table(cursor)
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def delete_mv_grid(mv_grid_id: str, conn, cursor):
    cursor.execute('SELECT 1 FROM "MVGrid" WHERE mv_grid_id = %s', (mv_grid_id,))
    if cursor.fetchone() is None:
        raise NotFoundError(f"MV Grid '{mv_grid_id}' not found.")
    cursor.execute('DELETE FROM "MVGrid" WHERE mv_grid_id = %s', (mv_grid_id,))
    conn.commit()
