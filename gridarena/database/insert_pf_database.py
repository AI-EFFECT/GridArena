"""Functions to insert powerflow data on the databases"""

import json


def replace_pf_run(cursor, grid_id, run, rows):
    """Replace a grid's power flow results with a new run.

    run:  dict with mode, neutral_impedance_ratio, start_time, end_time,
          timestamps, not_converged, skipped_timestamps, unreachable_nodes (list).
    rows: iterable of (node_id, phase, datetime, voltage_real, voltage_imag, converged).
    """
    cursor.execute('DELETE FROM "PowerFlowResults" WHERE grid_id = %s', (grid_id,))
    cursor.executemany(
        """
        INSERT INTO "PowerFlowResults"
        (grid_id, node_id, phase, datetime, voltage_real, voltage_imag, converged)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """,
        [(grid_id, *row) for row in rows],
    )
    cursor.execute(
        """
        INSERT INTO "PowerFlowRuns"
        (grid_id, mode, neutral_impedance_ratio, start_time, end_time,
         timestamps, not_converged, skipped_timestamps, unreachable_nodes, created_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
        ON CONFLICT (grid_id) DO UPDATE SET
            mode = EXCLUDED.mode,
            neutral_impedance_ratio = EXCLUDED.neutral_impedance_ratio,
            start_time = EXCLUDED.start_time,
            end_time = EXCLUDED.end_time,
            timestamps = EXCLUDED.timestamps,
            not_converged = EXCLUDED.not_converged,
            skipped_timestamps = EXCLUDED.skipped_timestamps,
            unreachable_nodes = EXCLUDED.unreachable_nodes,
            created_at = EXCLUDED.created_at
        """,
        (grid_id, run["mode"], run["neutral_impedance_ratio"], run["start_time"], run["end_time"],
         run["timestamps"], run["not_converged"], run["skipped_timestamps"],
         json.dumps(run["unreachable_nodes"])),
    )
