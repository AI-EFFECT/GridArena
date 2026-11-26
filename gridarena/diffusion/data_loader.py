"""Load power measurement data from the database for diffusion model training."""

import logging
from collections import Counter, defaultdict
from typing import List, Optional, Tuple

import numpy as np

import gridarena.database as db

logger = logging.getLogger(__name__)


def build_daily_snapshots(
    rows,
    p_min: Optional[float] = None,
    p_max: Optional[float] = None,
    require_even_nodes: bool = True,
) -> Tuple[np.ndarray, int, int]:
    """Turn raw (grid_id, node_id, phase, datetime, power_active) rows into
    daily snapshots of shape (n_days, n_series, n_daily_timesteps).

    * Each (grid, node, phase) is its own series, so several grids that reuse
      node names, or the three phases of one node, never overwrite each other.
    * Days are calendar dates. Only days with the dominant time-of-day pattern
      and a value for every series are kept: a gap is dropped, never filled
      with zeros, and a partial first day cannot shift the days after it.
    * ``p_min`` / ``p_max`` exclude whole days that contain an out-of-range
      value (instead of punching holes into the series).
    """
    if not rows:
        raise ValueError("No measurement data found with the given filters.")

    series = sorted({(r[0], r[1], r[2]) for r in rows})
    timestamps = sorted({r[3] for r in rows})
    s_idx = {s: i for i, s in enumerate(series)}
    t_idx = {t: i for i, t in enumerate(timestamps)}

    matrix = np.full((len(timestamps), len(series)), np.nan, dtype=np.float64)
    for grid_id, node_id, phase, dt, p_active in rows:
        if p_active is not None:
            matrix[t_idx[dt], s_idx[(grid_id, node_id, phase)]] = p_active

    # Drop series that carry no power (e.g. the transformer node).
    keep_cols = np.nanmax(np.abs(np.nan_to_num(matrix, nan=0.0)), axis=0) > 1e-8
    matrix = matrix[:, keep_cols]
    if matrix.shape[1] == 0:
        raise ValueError("All series were zero after filtering.")

    # Group timestamps by calendar date.
    by_date = defaultdict(list)
    for i, ts in enumerate(timestamps):
        by_date[ts.date()].append(i)
    pattern_of = {d: tuple(timestamps[i].time() for i in idx) for d, idx in by_date.items()}
    pattern, _ = Counter(pattern_of.values()).most_common(1)[0]
    n_daily_timesteps = len(pattern)

    days, dropped = [], Counter()
    for d in sorted(by_date):
        if pattern_of[d] != pattern:
            dropped["incomplete"] += 1
            continue
        block = matrix[by_date[d], :]  # (n_daily, n_series)
        if np.isnan(block).any():
            dropped["missing values"] += 1
            continue
        if (p_min is not None and block.min() < p_min) or (p_max is not None and block.max() > p_max):
            dropped["power range"] += 1
            continue
        days.append(block.T)

    if dropped:
        logger.info("Dropped days: %s", dict(dropped))
    if not days:
        raise ValueError(
            f"No complete day left ({n_daily_timesteps} timesteps/day, all series present). "
            f"Dropped: {dict(dropped)}"
        )

    snapshots = np.stack(days)  # (n_days, n_series, n_daily_timesteps), chronological
    n_nodes = snapshots.shape[1]

    # The legacy UNet2D backbone requires an even node count — drop last series if
    # odd. Custom backbones (require_even_nodes=False) keep every series.
    if require_even_nodes and n_nodes % 2 != 0:
        logger.info("Odd node count (%d) — dropping last node for UNet2D compatibility", n_nodes)
        snapshots = snapshots[:, :-1, :]
        n_nodes -= 1

    return snapshots, n_nodes, n_daily_timesteps


def load_power_data_from_db(
    grid_ids: List[str],
    phase: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    p_min: Optional[float] = None,
    p_max: Optional[float] = None,
    require_even_nodes: bool = True,
) -> Tuple[np.ndarray, int, int]:
    """Query Measurements table and return daily snapshot matrices.

    Args:
        require_even_nodes: if True (the legacy UNet2D path), an odd series count
            has its last series dropped so the 2D-conv downsampling works. The
            custom permutation-aware backbones never downsample the node axis,
            so callers using them should pass False to keep every series.

    Returns:
        power_snapshots: np.ndarray of shape (n_days, n_nodes, n_daily_timesteps)
        n_nodes: int — number of (grid, node, phase) series
        n_daily_timesteps: int — measurements per day
    Raises:
        ValueError: if no matching data or no complete day.
    """
    conn, cursor = db.get_db_connection()
    try:
        sql = """
            SELECT grid_id, node_id, phase, datetime, power_active
            FROM "Measurements"
            WHERE grid_id = ANY(%s)
        """
        params: list = [grid_ids]

        if phase is not None:
            sql += " AND phase = %s"
            params.append(phase)
        else:
            sql += " AND phase IS NOT NULL"

        if start is not None:
            sql += " AND datetime >= %s"
            params.append(start)
        if end is not None:
            sql += " AND datetime < %s"
            params.append(end)

        sql += " ORDER BY datetime ASC, grid_id ASC, node_id ASC, phase ASC"

        cursor.execute(sql, params)
        rows = cursor.fetchall()
    finally:
        conn.close()

    if not rows:
        raise ValueError(
            f"No measurement data found for grids={grid_ids} with the given filters."
        )

    power_snapshots, n_nodes, n_daily_timesteps = build_daily_snapshots(
        rows, p_min=p_min, p_max=p_max, require_even_nodes=require_even_nodes,
    )

    logger.info(
        "Loaded power data: %d days x %d series x %d timesteps/day from %d grids",
        power_snapshots.shape[0], n_nodes, n_daily_timesteps, len(grid_ids),
    )

    return power_snapshots, n_nodes, n_daily_timesteps
