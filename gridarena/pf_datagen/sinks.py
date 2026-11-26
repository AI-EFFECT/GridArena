"""Where a PF data-generation run's rows go (pf_datagen/job.py::execute).

`DatabaseSink` inserts them into the MVPF* tables, as GridArena runs always
have. `FileSink` writes one file per table into a directory -- the
standalone package's output (pf_datagen/package.py) -- with the columns in
the export order of pf_datagen/columns.py::EXPORT, the same as the CSV
export of a GridArena run.

Rows arrive as tuples in pf_datagen/columns.py's row order. Every sink has
variants(rows), layout(rows), batch(bus, branch, ybus, runtime, summary, der,
measurements), design_weights({load_scenario_idx: weight}), progress(completed,
failed, total), finish() once every row is in, and close() (always, also
after a failure).
"""

import csv
import importlib.util
import logging
from pathlib import Path

from gridarena.pf_datagen import columns

logger = logging.getLogger(__name__)

TABLES = ("bus", "branch", "ybus", "runtime", "summary", "der", "variants", "measurements", "layout")


class DatabaseSink:
    def __init__(self, db, run_id: str):
        self.db, self.run_id = db, run_id
        self._conn = None

    @property
    def conn(self):
        if self._conn is None:
            self._conn, self._cursor = self.db.get_db_connection()
        return self._conn

    @property
    def cursor(self):
        self.conn
        return self._cursor

    def variants(self, rows):
        self.db.insert_pf_datagen_variants(self.run_id, rows, self.conn, self.cursor)

    def layout(self, rows):
        self.db.insert_pf_datagen_measurement_layout(self.run_id, rows, self.conn, self.cursor)

    def batch(self, bus, branch, ybus, runtime, summary, der, meas):
        self.db.insert_pf_datagen_batch(self.run_id, bus, branch, ybus, runtime, self.conn, self.cursor,
                                        summary_rows=summary, der_rows=der, measurement_rows=meas)

    def design_weights(self, weights: dict):
        self.db.set_pf_datagen_design_weights(self.run_id, weights, self.conn, self.cursor)

    def progress(self, completed, failed, total):
        pass                        # the run page reads it from run_config.json

    def finish(self):
        pass

    def close(self):
        if self._conn is not None:
            self._conn.close()
            self._conn = None


def _cell(v):
    """A CSV cell: lists (the variants' connection / node arrays) joined with
    ';' as in the GridArena export, None as an empty cell."""
    if isinstance(v, (list, tuple)):
        return ";".join(str(x) for x in v)
    return "" if v is None else v


class FileSink:
    """One `<table>.csv` per table in `out_dir`, written as the run goes;
    with fmt="parquet" each is converted to `<table>.parquet` when the run
    ends (needs pyarrow), typed by `column_types` (the database's types, see
    database/insert_pf_datagen_database.py::get_pf_datagen_column_types), so
    a column without a value in this run still gets its real type. A table
    the run produced no rows for (e.g. measurements with the measurement
    view off) still gets its header."""

    def __init__(self, out_dir, run_id: str, fmt: str = "csv", on_progress=None, column_types=None):
        if fmt not in ("csv", "parquet"):
            raise ValueError(f"Unknown output format '{fmt}' (csv or parquet).")
        if fmt == "parquet" and importlib.util.find_spec("pyarrow") is None:
            # Fail before solving, not after.
            raise ImportError("Parquet output needs pyarrow: pip install pyarrow")
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.run_id, self.fmt, self.on_progress = run_id, fmt, on_progress
        self.column_types = column_types or {}       # {table: {column: pyarrow type name}}
        self._files, self._writers, self._order = {}, {}, {}
        self._weights = None
        for table in TABLES:
            pos = columns.index(columns.ROWS[table])
            self._order[table] = [pos[c] for c in columns.EXPORT[table]]
            f = open(self.out / f"{table}.csv", "w", newline="", encoding="utf-8")
            self._files[table] = f
            self._writers[table] = csv.writer(f)
            self._writers[table].writerow(("run_id",) + tuple(columns.EXPORT[table]))

    def _write(self, table, rows):
        order, writer = self._order[table], self._writers[table]
        for row in rows:
            writer.writerow([self.run_id] + [_cell(row[i]) for i in order])

    def variants(self, rows):
        self._write("variants", rows)

    def layout(self, rows):
        self._write("layout", rows)

    def batch(self, bus, branch, ybus, runtime, summary, der, meas):
        for table, rows in (("runtime", runtime), ("bus", bus), ("branch", branch), ("ybus", ybus),
                            ("summary", summary), ("der", der), ("measurements", meas)):
            self._write(table, rows)
        for f in self._files.values():
            f.flush()

    def design_weights(self, weights: dict):
        self._weights = {int(k): w for k, w in weights.items()}

    def progress(self, completed, failed, total):
        if self.on_progress is not None:
            self.on_progress(completed, failed, total)

    def finish(self):
        """Every row is in: close the files, fill the design weights, convert."""
        self.close()
        if self._weights is not None:
            self._fill_design_weights()
        if self.fmt == "parquet":
            self._to_parquet()

    def close(self):
        for f in self._files.values():
            if not f.closed:
                f.close()

    def _fill_design_weights(self):
        """The target mix's weights are known once every sample is solved: fill
        runtime.csv's design_weight column (GridArena does it with an UPDATE)."""
        path, tmp = self.out / "runtime.csv", self.out / "runtime.csv.tmp"
        with open(path, newline="", encoding="utf-8") as src, open(tmp, "w", newline="", encoding="utf-8") as dst:
            reader, writer = csv.reader(src), csv.writer(dst)
            header = next(reader)
            writer.writerow(header)
            i_idx, i_w = header.index("load_scenario_idx"), header.index("design_weight")
            for row in reader:
                w = self._weights.get(int(row[i_idx]))
                if w is not None:
                    row[i_w] = w
                writer.writerow(row)
        tmp.replace(path)

    def _to_parquet(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        from pyarrow import csv as pacsv

        for table in TABLES:
            src = self.out / f"{table}.csv"
            dst = self.out / f"{table}.parquet"
            names = ["run_id"] + list(columns.EXPORT[table])
            types = {"run_id": "string", "output_group_id": "string", "scenario_timestamp": "string",
                     **self.column_types.get(table, {})}
            convert = pacsv.ConvertOptions(
                column_types={n: pa.type_for_alias(t) for n, t in types.items() if n in names},
                strings_can_be_null=True,
            )
            # Streamed block by block, so a long run's tables never sit whole in memory.
            # A column of unknown type is typed from the first block; if a later block
            # disagrees, the table is read whole instead (typed from all of it).
            try:
                reader = pacsv.open_csv(src, read_options=pacsv.ReadOptions(block_size=64 << 20),
                                        convert_options=convert)
                with pq.ParquetWriter(dst, reader.schema) as writer:
                    for batch in reader:
                        writer.write_table(pa.Table.from_batches([batch], schema=reader.schema))
            except pa.ArrowInvalid:
                pq.write_table(pacsv.read_csv(src, convert_options=convert), dst)
            src.unlink()
