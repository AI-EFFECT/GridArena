"""Creates the sql database tables for the powerflow results data"""


def create_pf_database(cursor):
    # Computed voltages, one row per node, phase and timestamp. phase is R/S/T
    # (phase-to-neutral voltage) or N (neutral-to-ground voltage, unbalanced runs).
    # They are kept apart from the measured voltages in "Measurements".
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS "PowerFlowResults" (
            grid_id TEXT NOT NULL,
            node_id TEXT NOT NULL,
            phase TEXT NOT NULL,
            datetime TEXT NOT NULL,
            voltage_real DOUBLE PRECISION NOT NULL,
            voltage_imag DOUBLE PRECISION NOT NULL,
            PRIMARY KEY (grid_id, node_id, phase, datetime),
            FOREIGN KEY (grid_id) REFERENCES grids(grid_id) ON DELETE CASCADE,
            FOREIGN KEY (node_id, grid_id)
                REFERENCES "Node"("NodeId", grid_id) ON DELETE CASCADE
        );
        """
    )
    cursor.execute(
        """ALTER TABLE "PowerFlowResults" ADD COLUMN IF NOT EXISTS converged BOOLEAN NOT NULL DEFAULT TRUE"""
    )

    # The run that produced a grid's current results: each run replaces the
    # grid's previous results, so there is at most one row per grid.
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS "PowerFlowRuns" (
            grid_id TEXT PRIMARY KEY REFERENCES grids(grid_id) ON DELETE CASCADE,
            mode TEXT NOT NULL CHECK (mode IN ('balanced', 'unbalanced')),
            neutral_impedance_ratio DOUBLE PRECISION,
            start_time TEXT,
            end_time TEXT,
            timestamps INTEGER NOT NULL,
            not_converged INTEGER NOT NULL,
            skipped_timestamps INTEGER NOT NULL,
            unreachable_nodes TEXT NOT NULL DEFAULT '[]',
            created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
