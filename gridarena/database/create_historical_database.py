"""Creates the sql database tables for historical databases.

A "database" here is a named collection of standalone historical series
("historicals") that all share the exact same timeline -- see
"HistoricalTimestamp" below, which every "HistoricalRecords" row must
reference. Unlike Measurements (node/grid-scoped, see
create_measurements_database.py), none of this is tied to any registered
grid or node.
"""

from gridarena.schemas.types import HISTORICAL_DATA_TYPES


_DATA_TYPE_CHECK = "CHECK (data_type IN ({}))".format(", ".join(f"'{t}'" for t in HISTORICAL_DATA_TYPES))


def create_historical_database_table(cursor):
    cursor.execute(
        f"""
        CREATE TABLE IF NOT EXISTS "HistoricalDatabase" (
            database_id TEXT PRIMARY KEY,
            name TEXT,
            description TEXT,
            data_type TEXT NOT NULL DEFAULT 'load'
                CONSTRAINT historical_database_data_type_chk {_DATA_TYPE_CHECK}
        );
        """
    )
    # The table may predate data_type ("load" = consumption, the rest kinds of
    # generation, see schemas/types.py), or have a check allowing fewer types.
    # Both are looked up first rather than altered unconditionally: ALTER TABLE
    # takes an exclusive lock even when there is nothing to change.
    cursor.execute(
        "SELECT 1 FROM information_schema.columns "
        "WHERE table_name = 'HistoricalDatabase' AND column_name = 'data_type'"
    )
    if cursor.fetchone() is None:
        cursor.execute(
            'ALTER TABLE "HistoricalDatabase" ADD COLUMN IF NOT EXISTS data_type TEXT NOT NULL DEFAULT \'load\''
        )
    cursor.execute(
        "SELECT pg_get_constraintdef(oid) FROM pg_constraint "
        "WHERE conname = 'historical_database_data_type_chk' AND conrelid = '\"HistoricalDatabase\"'::regclass"
    )
    row = cursor.fetchone()
    if row is None or any(f"'{t}'" not in row[0] for t in HISTORICAL_DATA_TYPES):
        if row is not None:
            cursor.execute('ALTER TABLE "HistoricalDatabase" DROP CONSTRAINT historical_database_data_type_chk')
        cursor.execute(
            f'ALTER TABLE "HistoricalDatabase" ADD CONSTRAINT historical_database_data_type_chk {_DATA_TYPE_CHECK}'
        )


def create_historical_timestamp_table(cursor):
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS "HistoricalTimestamp" (
            database_id TEXT NOT NULL
                REFERENCES "HistoricalDatabase"(database_id) ON DELETE CASCADE,
            datetime TIMESTAMP WITHOUT TIME ZONE NOT NULL,
            PRIMARY KEY (database_id, datetime)
        );
        """
    )


def create_historical_series_table(cursor):
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS "HistoricalSeries" (
            database_id TEXT NOT NULL
                REFERENCES "HistoricalDatabase"(database_id) ON DELETE CASCADE,
            series_id TEXT NOT NULL,
            grid_level TEXT NOT NULL
                CONSTRAINT historical_series_grid_level_chk CHECK (grid_level IN ('MV','LV','Feeder')),
            name TEXT,
            description TEXT,
            PRIMARY KEY (database_id, series_id)
        );
        """
    )
    # The table may already exist from before "Feeder" was added as a valid
    # grid_level -- CREATE TABLE IF NOT EXISTS won't touch its old CHECK
    # constraint, so widen it explicitly (idempotent: no-op once migrated).
    cursor.execute(
        """
        DO $$
        DECLARE
            con record;
        BEGIN
            FOR con IN
                SELECT conname FROM pg_constraint
                WHERE conrelid = '"HistoricalSeries"'::regclass
                  AND contype = 'c'
                  AND conname <> 'historical_series_grid_level_chk'
                  AND pg_get_constraintdef(oid) LIKE '%grid_level%'
            LOOP
                EXECUTE format('ALTER TABLE "HistoricalSeries" DROP CONSTRAINT %I', con.conname);
            END LOOP;

            IF NOT EXISTS (
                SELECT 1 FROM pg_constraint WHERE conname = 'historical_series_grid_level_chk'
            ) THEN
                ALTER TABLE "HistoricalSeries"
                    ADD CONSTRAINT historical_series_grid_level_chk
                    CHECK (grid_level IN ('MV','LV','Feeder'));
            END IF;
        END $$;
        """
    )


def create_historical_records_table(cursor):
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS "HistoricalRecords" (
            record_id BIGSERIAL PRIMARY KEY,
            database_id TEXT NOT NULL,
            series_id TEXT NOT NULL,
            datetime TIMESTAMP WITHOUT TIME ZONE NOT NULL,
            phase TEXT DEFAULT NULL,
            power_active DOUBLE PRECISION,
            power_reactive DOUBLE PRECISION,
            voltage_magnitude DOUBLE PRECISION,
            voltage_angle DOUBLE PRECISION,
            CONSTRAINT historical_records_phase_chk CHECK (phase IN ('R','S','T') OR phase IS NULL),
            CONSTRAINT historical_records_has_value
                CHECK (power_active IS NOT NULL OR voltage_magnitude IS NOT NULL),
            CONSTRAINT historical_records_series_fk
                FOREIGN KEY (database_id, series_id)
                REFERENCES "HistoricalSeries"(database_id, series_id)
                ON DELETE CASCADE,
            CONSTRAINT historical_records_timestamp_fk
                FOREIGN KEY (database_id, datetime)
                REFERENCES "HistoricalTimestamp"(database_id, datetime)
                ON DELETE CASCADE,
            CONSTRAINT historical_records_unique UNIQUE (database_id, series_id, datetime, phase)
        );
        """
    )
    cursor.execute(
        'CREATE INDEX IF NOT EXISTS historical_records_series_time_idx '
        'ON "HistoricalRecords"(database_id, series_id, datetime);'
    )
    # The table may already exist from before this FK had ON DELETE CASCADE --
    # CREATE TABLE IF NOT EXISTS won't touch it, so widen it explicitly here
    # (idempotent: no-op once migrated). Without this, deleting a database
    # with any records fails with a foreign key violation (HistoricalRecords
    # still references the HistoricalTimestamp rows being cascade-deleted).
    cursor.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM pg_constraint
                WHERE conname = 'historical_records_timestamp_fk'
                  AND conrelid = '"HistoricalRecords"'::regclass
                  AND confdeltype <> 'c'
            ) THEN
                ALTER TABLE "HistoricalRecords" DROP CONSTRAINT historical_records_timestamp_fk;
                ALTER TABLE "HistoricalRecords"
                    ADD CONSTRAINT historical_records_timestamp_fk
                    FOREIGN KEY (database_id, datetime)
                    REFERENCES "HistoricalTimestamp"(database_id, datetime)
                    ON DELETE CASCADE;
            END IF;
        END $$;
        """
    )


# ---------- Stored record counts ----------
#
# Counting "HistoricalRecords" per database or series means scanning the whole
# table (~90 s for the 41.8M-row feeder import), so each series keeps its own
# counts -- n_records, and how many of those carry power (n_power) and voltage
# (n_voltage) -- and a database's total is the sum over its series. Postgres
# keeps them exact through statement-level triggers with transition tables,
# so every write path counts: the upload endpoint (one upsert per record),
# record deletes, cascades from deleting a series or database, and bulk
# imports run directly against Postgres (COPY / INSERT ... SELECT).
# Requires PostgreSQL 10+.

_COUNTS_LOCK_KEY = 7420190301  # pg advisory lock: one process migrates at a time

_COUNT_TRIGGER_FUNCTION = """
CREATE OR REPLACE FUNCTION historical_records_update_counts() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    IF TG_OP = 'TRUNCATE' THEN
        UPDATE "HistoricalSeries" SET n_records = 0, n_power = 0, n_voltage = 0;
        RETURN NULL;
    END IF;
    IF TG_OP IN ('DELETE', 'UPDATE') THEN
        UPDATE "HistoricalSeries" s
        SET n_records = s.n_records - d.n, n_power = s.n_power - d.n_power, n_voltage = s.n_voltage - d.n_voltage
        FROM (SELECT database_id, series_id, count(*) AS n, count(power_active) AS n_power,
                     count(voltage_magnitude) AS n_voltage
              FROM old_rows GROUP BY database_id, series_id) d
        WHERE s.database_id = d.database_id AND s.series_id = d.series_id;
    END IF;
    IF TG_OP IN ('INSERT', 'UPDATE') THEN
        UPDATE "HistoricalSeries" s
        SET n_records = s.n_records + d.n, n_power = s.n_power + d.n_power, n_voltage = s.n_voltage + d.n_voltage
        FROM (SELECT database_id, series_id, count(*) AS n, count(power_active) AS n_power,
                     count(voltage_magnitude) AS n_voltage
              FROM new_rows GROUP BY database_id, series_id) d
        WHERE s.database_id = d.database_id AND s.series_id = d.series_id;
    END IF;
    RETURN NULL;
END $$;
"""

# Transition tables allow one event per trigger, hence one trigger per event.
_COUNT_TRIGGERS = {
    "historical_records_counts_ins":
        'AFTER INSERT ON "HistoricalRecords" REFERENCING NEW TABLE AS new_rows FOR EACH STATEMENT',
    "historical_records_counts_upd":
        'AFTER UPDATE ON "HistoricalRecords" REFERENCING OLD TABLE AS old_rows NEW TABLE AS new_rows '
        'FOR EACH STATEMENT',
    "historical_records_counts_del":
        'AFTER DELETE ON "HistoricalRecords" REFERENCING OLD TABLE AS old_rows FOR EACH STATEMENT',
    "historical_records_counts_trunc":
        'AFTER TRUNCATE ON "HistoricalRecords" FOR EACH STATEMENT',
}


def historical_record_counts_need_fill(cursor) -> bool:
    """True if the stored counts aren't set up yet (a counts column or a
    trigger is missing), i.e. ensure_historical_record_counts will run the
    full fill. Assumes the historical tables exist."""
    cursor.execute(
        """SELECT count(*) FROM information_schema.columns
           WHERE table_name = 'HistoricalSeries' AND column_name IN ('n_records', 'n_power', 'n_voltage')"""
    )
    if cursor.fetchone()[0] < 3:
        return True
    cursor.execute(
        "SELECT count(*) FROM pg_trigger WHERE tgrelid = '\"HistoricalRecords\"'::regclass AND tgname = ANY(%s)",
        (list(_COUNT_TRIGGERS),),
    )
    return cursor.fetchone()[0] < len(_COUNT_TRIGGERS)


def ensure_historical_record_counts(conn, cursor) -> bool:
    """Create the historical tables if needed and make sure the stored record
    counts and their triggers are in place; commits. The first time (counts
    columns missing) it also fills the counts from the existing records --
    one full scan of "HistoricalRecords", with writes to it held off
    meanwhile so nothing is missed. Afterwards this is a quick no-op.

    Returns True if it ran that one-time fill."""
    try:
        cursor.execute("SELECT pg_advisory_xact_lock(%s)", (_COUNTS_LOCK_KEY,))
        create_historical_data_tables(cursor)
        cursor.execute(
            """SELECT count(*) FROM information_schema.columns
               WHERE table_name = 'HistoricalSeries' AND column_name IN ('n_records', 'n_power', 'n_voltage')"""
        )
        needs_fill = cursor.fetchone()[0] < 3
        cursor.execute(
            "SELECT tgname FROM pg_trigger WHERE tgrelid = '\"HistoricalRecords\"'::regclass AND NOT tgisinternal"
        )
        existing = {r[0] for r in cursor.fetchall()}
        missing = [name for name in _COUNT_TRIGGERS if name not in existing]
        if not needs_fill and not missing:
            conn.commit()
            return False

        # Hold off writes to the records while the counts and triggers are set up.
        cursor.execute('LOCK TABLE "HistoricalRecords" IN SHARE MODE')
        cursor.execute(
            'ALTER TABLE "HistoricalSeries" '
            "ADD COLUMN IF NOT EXISTS n_records BIGINT NOT NULL DEFAULT 0, "
            "ADD COLUMN IF NOT EXISTS n_power BIGINT NOT NULL DEFAULT 0, "
            "ADD COLUMN IF NOT EXISTS n_voltage BIGINT NOT NULL DEFAULT 0"
        )
        cursor.execute(_COUNT_TRIGGER_FUNCTION)
        for name in missing:
            cursor.execute(f"CREATE TRIGGER {name} {_COUNT_TRIGGERS[name]} "
                           "EXECUTE PROCEDURE historical_records_update_counts()")
        # Fill the counts from scratch -- also when only a trigger was missing,
        # since writes made without it went uncounted.
        cursor.execute('UPDATE "HistoricalSeries" SET n_records = 0, n_power = 0, n_voltage = 0')
        cursor.execute(
            """
            UPDATE "HistoricalSeries" s
            SET n_records = c.n, n_power = c.n_power, n_voltage = c.n_voltage
            FROM (SELECT database_id, series_id, count(*) AS n, count(power_active) AS n_power,
                         count(voltage_magnitude) AS n_voltage
                  FROM "HistoricalRecords" GROUP BY database_id, series_id) c
            WHERE s.database_id = c.database_id AND s.series_id = c.series_id
            """
        )
        conn.commit()
        return True
    except Exception:
        conn.rollback()
        raise


def create_historical_data_tables(cursor):
    """
    Creates the tables required to store historical databases.

    This includes:
    - HistoricalDatabase: Registers each database (its name/description and
      data_type: "load", or a kind of generation such as "pv").
    - HistoricalTimestamp: The shared timeline every series in the database must use.
    - HistoricalSeries: Registers each series ("historical") within a database.
    - HistoricalRecords: Stores the power/voltage values for each series.

    Parameters:
        cursor (psycopg.Cursor): PostgreSQL cursor used to execute SQL commands.
    """

    create_historical_database_table(cursor)
    create_historical_timestamp_table(cursor)
    create_historical_series_table(cursor)
    create_historical_records_table(cursor)
