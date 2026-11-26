"""Database preparation at startup, run in the background.

Every table the app needs is created (or migrated) here, once, instead of on
each request: request handlers used to run "create if missing" DDL, and
`ALTER TABLE ... ADD COLUMN IF NOT EXISTS` takes an ACCESS EXCLUSIVE lock
even when the column already exists, so a page could wait for any open
transaction that had read the table.

Preparation runs in a thread so the server answers at once. Until it has
finished, app.py's middleware serves a loading page (templates/startup.html)
that polls GET /startup/status for the progress reported here. Normally it
takes well under a second; the exception is the one-time fill of the stored
historical record counts (a full scan of "HistoricalRecords", minutes on a
large database), for which the progress is a time estimate.
"""

import logging
import threading
import time
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)

# Serialises preparation across processes (e.g. two servers on one database).
_PREPARE_LOCK_KEY = 7420190302
# Throughput assumed for the one-time count fill: bytes of "HistoricalRecords"
# heap scanned per second (measured ~140 s for the 20 GB local table, cold cache).
_FILL_BYTES_PER_S = 130e6


class _Step:
    def __init__(self, label: str, run: Callable, connects: bool = False):
        self.label = label
        self.run = run
        self.connects = connects      # opens the connection that holds the preparation lock
        self.status = "pending"       # pending | running | done | failed
        self.detail: Optional[str] = None
        self.estimate_s: Optional[float] = None
        self.started: Optional[float] = None
        self.finished: Optional[float] = None

    def as_dict(self) -> dict:
        elapsed = None
        if self.started is not None:
            elapsed = (self.finished or time.monotonic()) - self.started
        return {"label": self.label, "status": self.status, "detail": self.detail,
                "estimate_s": self.estimate_s, "elapsed_s": elapsed}


class DatabaseStartup:
    """State of the background preparation. `state` is one of:
    not_started (never launched, e.g. under tests -- requests pass through),
    running, ready, failed."""

    def __init__(self):
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._rerun = False
        self.state = "not_started"
        self.error: Optional[str] = None
        self.started: Optional[float] = None
        self.finished: Optional[float] = None
        self.steps: List[_Step] = []

    # -- control ---------------------------------------------------------

    def start(self) -> None:
        """Prepare the database in the background. If preparation is already
        running (e.g. the database was changed again meanwhile), it runs once
        more when the current pass ends."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                self._rerun = True
                return
            self._begin()
            self._thread = threading.Thread(target=self._run, name="db-startup", daemon=True)
            self._thread.start()

    def blocks_requests(self) -> bool:
        return self.state in ("running", "failed")

    def status(self) -> dict:
        with self._lock:
            steps = [s.as_dict() for s in self.steps]
            return {
                "state": self.state,
                "error": self.error,
                "elapsed_s": None if self.started is None else (self.finished or time.monotonic()) - self.started,
                "progress": self._progress(),
                "steps": steps,
            }

    # -- internals -------------------------------------------------------

    def _begin(self) -> None:
        self.state = "running"
        self.error = None
        self.started = time.monotonic()
        self.finished = None
        self.steps = [
            _Step("Connecting to the database", self._step_connect, connects=True),
            _Step("Grid and benchmark tables", _step_core_tables),
            _Step("MV grid tables", _step_mv_tables),
            _Step("PF data generation tables", _step_pf_datagen_tables),
            _Step("Historical tables and record counts", _step_historical),
        ]

    def _progress(self) -> float:
        """0..1. Each step weighs 1, a step with a duration estimate weighs its
        estimate in seconds; the running step counts by its elapsed share of
        the estimate (capped at 95 %), or half if it has none."""
        if self.state == "ready":
            return 1.0
        total = done = 0.0
        for s in self.steps:
            weight = max(1.0, s.estimate_s or 1.0)
            total += weight
            if s.status == "done":
                done += weight
            elif s.status == "running" and s.started is not None:
                if s.estimate_s:
                    done += weight * min(0.95, (time.monotonic() - s.started) / s.estimate_s)
                else:
                    done += weight * 0.5
        return done / total if total else 0.0

    def _run(self) -> None:
        while True:
            self._run_once()
            with self._lock:
                if not self._rerun:
                    return
                self._rerun = False
                self._begin()

    def _run_once(self) -> None:
        import gridarena.database as db

        lock_conn = None
        try:
            for step in self.steps:
                with self._lock:
                    step.status, step.started = "running", time.monotonic()
                if step.connects:
                    lock_conn = step.run(step)
                else:
                    conn, cursor = db.get_db_connection()
                    try:
                        step.run(step, conn, cursor)
                        conn.commit()
                    except Exception:
                        conn.rollback()
                        raise
                    finally:
                        conn.close()
                with self._lock:
                    step.status, step.finished = "done", time.monotonic()
            with self._lock:
                self.state, self.finished = "ready", time.monotonic()
            logger.info("Database ready (%.1f s)", self.finished - self.started)
        except Exception as e:
            logger.error("Preparing the database failed", exc_info=True)
            with self._lock:
                for step in self.steps:
                    if step.status == "running":
                        step.status, step.finished = "failed", time.monotonic()
                self.state, self.finished = "failed", time.monotonic()
                self.error = f"{type(e).__name__}: {e}"
        finally:
            if lock_conn is not None:
                try:
                    lock_conn.execute("SELECT pg_advisory_unlock(%s)", (_PREPARE_LOCK_KEY,))
                    lock_conn.commit()
                except Exception:
                    logger.warning("Could not release the database preparation lock", exc_info=True)
                lock_conn.close()

    def _step_connect(self, step: _Step):
        """Check out a connection and hold the cross-process preparation lock
        on it for the whole run (session-level: it outlives transactions)."""
        import gridarena.database as db

        conn, cursor = db.get_db_connection()
        try:
            cursor.execute("SELECT pg_try_advisory_lock(%s)", (_PREPARE_LOCK_KEY,))
            if not cursor.fetchone()[0]:
                with self._lock:
                    step.detail = "Waiting for another GridArena process that is preparing the same database."
                cursor.execute("SELECT pg_advisory_lock(%s)", (_PREPARE_LOCK_KEY,))
                with self._lock:
                    step.detail = None
            conn.commit()
        except Exception:
            conn.close()
            raise
        return conn


def _step_core_tables(step, conn, cursor):
    import gridarena.database as db

    # LV grid tables first ("grids" is referenced by the MV transformer table);
    # create_grid_database ignores its grid_id argument.
    db.create_grid_database(None, conn, cursor)
    db.create_anon_map_database(cursor)
    db.create_phase_guesses_database(cursor)
    db.create_topology_guesses_database(cursor)
    db.create_voltage_timestamp_map_table(cursor)
    db.create_vc_guesses_database(cursor)
    db.create_voltage_control_solutions(cursor)
    db.create_base_tables(cursor)
    db.create_state_estimates_table(cursor)
    db.create_pf_database(cursor)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS "GridAnonymisedMapping" (
            grid_id TEXT PRIMARY KEY,
            anonymised_grid_key TEXT UNIQUE NOT NULL
        )
    """)


def _step_mv_tables(step, conn, cursor):
    import gridarena.database as db

    db.create_mv_database(conn, cursor)


def _step_pf_datagen_tables(step, conn, cursor):
    import gridarena.database as db

    db.create_pf_datagen_database(conn, cursor)


def _step_historical(step, conn, cursor):
    """Historical tables plus the stored record counts and their triggers. If
    the counts still have to be filled, report the one-time scan and an
    estimate of how long it takes before starting it."""
    import gridarena.database as db

    cursor.execute("""SELECT to_regclass('"HistoricalRecords"') IS NOT NULL""")
    if cursor.fetchone()[0] and db.historical_record_counts_need_fill(cursor):
        cursor.execute("""SELECT pg_relation_size('"HistoricalRecords"'), reltuples::bigint
                          FROM pg_class WHERE oid = '"HistoricalRecords"'::regclass""")
        size, rows = cursor.fetchone()
        step.estimate_s = max(5.0, size / _FILL_BYTES_PER_S)
        step.detail = (f"One-time step: counting the {max(rows, 0):,} historical records "
                       f"({size / 1e9:.1f} GB) to store a count per series.")
    conn.rollback()  # end the read-only check before the step's own transaction
    if db.ensure_historical_record_counts(conn, cursor):
        logger.info("Filled the stored historical record counts (one-time)")


db_startup = DatabaseStartup()
