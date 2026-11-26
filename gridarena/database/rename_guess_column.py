"""Migration helper for the benchmark guess tables.

These tables were first created with a ``user_id`` column while the routers
read and write ``guess_id``. Databases created before the fix still have the
old column name, so it is renamed in place (constraints follow the rename).
"""


def rename_user_id_to_guess_id(cursor, table: str):
    cursor.execute(
        """
        SELECT 1 FROM information_schema.columns
        WHERE table_name = %s AND column_name = 'user_id'
        """,
        (table,),
    )
    if cursor.fetchone() is not None:
        cursor.execute(f'ALTER TABLE "{table}" RENAME COLUMN user_id TO guess_id')
