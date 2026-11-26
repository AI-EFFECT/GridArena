"""Functions to insert historical databases (many series sharing one timeline)."""


class HistoricalDataTypeConflict(ValueError):
    """An upload's data_type differs from the existing database's: series of
    PV production and of load must not be merged into one database."""


def insert_historical_data(database_upload, conn, cursor):
    """
    Inserts a historical database -- its shared timeline and every series
    ("historical") within it -- into HistoricalDatabase, HistoricalTimestamp,
    HistoricalSeries and HistoricalRecords.

    Re-uploading the same database_id refreshes its metadata, adds any new
    timestamps, and upserts each series' metadata and records. Its data_type
    ("load" or "pv") must match the existing database's
    (HistoricalDataTypeConflict otherwise).
    """
    try:
        database_id = database_upload.database_id
        data_type = getattr(database_upload, "data_type", "load")

        cursor.execute(
            'SELECT data_type FROM "HistoricalDatabase" WHERE database_id = %s FOR UPDATE',
            (database_id,),
        )
        existing = cursor.fetchone()
        if existing is not None and existing[0] != data_type:
            raise HistoricalDataTypeConflict(
                f"Database '{database_id}' already exists with data_type '{existing[0]}'; this upload is "
                f"'{data_type}'. Use another database_id, or delete the existing database first."
            )

        cursor.execute(
            """
            INSERT INTO "HistoricalDatabase" (database_id, name, description, data_type)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (database_id) DO UPDATE SET
                name = EXCLUDED.name,
                description = EXCLUDED.description
            """,
            (database_id, database_upload.name, database_upload.description, data_type),
        )

        cursor.executemany(
            """
            INSERT INTO "HistoricalTimestamp" (database_id, datetime)
            VALUES (%s, %s)
            ON CONFLICT (database_id, datetime) DO NOTHING
            """,
            [(database_id, ts) for ts in database_upload.timestamps],
        )

        series_q = (
            'INSERT INTO "HistoricalSeries" ('
            " database_id, series_id, grid_level, name, description"
            ") VALUES (%s,%s,%s,%s,%s) "
            "ON CONFLICT (database_id, series_id) DO UPDATE SET "
            " grid_level = EXCLUDED.grid_level, "
            " name = EXCLUDED.name, "
            " description = EXCLUDED.description"
        )

        records_q = (
            'INSERT INTO "HistoricalRecords" ('
            " database_id, series_id, datetime, phase, "
            " power_active, power_reactive, voltage_magnitude, voltage_angle"
            ") VALUES (%s,%s,%s,%s,%s,%s,%s,%s) "
            "ON CONFLICT ON CONSTRAINT historical_records_unique DO UPDATE SET "
            " power_active = EXCLUDED.power_active, "
            " power_reactive = EXCLUDED.power_reactive, "
            " voltage_magnitude = EXCLUDED.voltage_magnitude, "
            " voltage_angle = EXCLUDED.voltage_angle"
        )

        for historical in database_upload.historicals:
            series_id = historical.series_id
            cursor.execute(
                series_q,
                (database_id, series_id, historical.grid_level, historical.name, historical.description),
            )

            for v in historical.values:
                cursor.execute(
                    records_q,
                    (
                        database_id,
                        series_id,
                        v.datetime,
                        getattr(v, "phase", None),
                        v.power_active,
                        v.power_reactive,
                        v.voltage_magnitude,
                        v.voltage_angle,
                    ),
                )

        conn.commit()

    except HistoricalDataTypeConflict:
        conn.rollback()
        raise
    except Exception as e:
        conn.rollback()
        raise Exception(f"Database error while inserting historical data: {e}")
