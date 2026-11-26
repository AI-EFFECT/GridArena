"""Builds PF scenarios for an MV grid from a Historical database (see
gridarena/routers/historical.py), with a randomized series-to-point assignment
that's redrawn on a fixed period.

The database's series must all share one grid_level -- that determines the
population strategy:
- "MV": every MV connection point gets its own randomly-drawn MV series;
  that series' own values are the point's injection directly. No dependency
  on any real LV grid being connected.
- "LV": for every MV connection point that has a real LV grid attached
  (get_connected_load_points), that LV grid's node count worth of LV series
  are drawn as "virtual node slots"; the point's injection is their sum --
  mirroring insert_mv_database.py::get_summed_pv()'s "sum across nodes"
  convention, just fed by synthetic per-node curves instead of real ones.
- "Feeder": one Feeder series per point (distinct if the database has
  enough, else with replacement). With `rating` sizing every connection
  point; with `series` sizing only the points with an LV grid attached.
  With load_shape.feeders_per_point `by_kva` a point sums instead a number
  of distinct series (no feeder twice within a point) drawn from its
  transformer's kVA band, once per run; the series are redrawn per block.

The scenario timestamp axis is the database's own declared timeline
(HistoricalTimestamp): scenario_count timestamps spread over it, as before,
or a window of consecutive ones (pf_datagen/timeline.py). With
`aggregation_s` (the request's load_aggregation_s) the timeline is the
clock-aligned intervals of that length instead, each series' value in one
the mean of its records there.
Every reassignment_period_timesteps consecutive scenarios (in timestamp
order) share one random assignment; crossing into the next block of that
many scenarios redraws it.

Phase 2 axis 2 (pf_datagen/load_shape.py) then shapes each point's load: with
`rating` point sizing (the default for MV and Feeder data) a series gives only
the shape and the point's transformer rating its size, and MV / Feeder series
populate every connection point, LV grid attached or not; missing reactive
power comes from a drawn power factor; and log-normal noise (spatially
correlated along the cables, plus local and Q-only parts) multiplies the
result. Load-level scaling (axis 1) happens afterwards, in the job.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from gridarena.pf_datagen.load_shape import (
    LoadNoise, cable_distances_km, combination_scale, draw_feeder_counts, draw_power_factors, point_mean_load_w,
    reference_timestamps, resolve_point_sizing, series_matrix, series_scales, tan_phi,
)
from gridarena.pf_datagen.schemas import LoadShapeConfig
from gridarena.pf_datagen.timeline import (
    aggregate_timeline, average_intervals, interpolated_window, interval_start, median_step_s, select_timestamps,
)


def get_connected_load_points(mv_grid_id: str, cursor) -> List[dict]:
    """MV connection-point nodes that have an LV grid attached via a transformer.

    Returns a list of {"node_id": ..., "lv_grid_id": ...}. Used by the "LV"
    and "Feeder" source types to determine which points to populate (and,
    for "LV", each point's node count).
    """
    cursor.execute(
        """
        SELECT cp."NodeId", t.lv_grid_id
        FROM "MVTransformer" t
        JOIN "MVConnectionPoint" cp
            ON cp."ConnectionPointId" = t."ConnectionPointId" AND cp.mv_grid_id = t.mv_grid_id
        WHERE t.mv_grid_id = %s AND t.lv_grid_id IS NOT NULL
        """,
        (mv_grid_id,),
    )
    return [{"node_id": r[0], "lv_grid_id": r[1]} for r in cursor.fetchall()]


def get_all_mv_connection_points(mv_grid_id: str, cursor) -> List[str]:
    """Every connection point in the grid, regardless of whether it has a
    real LV grid attached. Used by the "MV" source type."""
    cursor.execute('SELECT "NodeId" FROM "MVConnectionPoint" WHERE mv_grid_id = %s', (mv_grid_id,))
    return [r[0] for r in cursor.fetchall()]


def get_historical_source_type(database_id: str, cursor) -> str:
    """The single grid_level shared by every series in the database ("MV",
    "LV", or "Feeder"). Raises if the database is empty, mixes levels or
    holds generation (a data_type other than "load", e.g. "pv")."""
    cursor.execute('SELECT data_type FROM "HistoricalDatabase" WHERE database_id = %s', (database_id,))
    row = cursor.fetchone()
    if row is not None and row[0] != "load":
        raise Exception(
            f"Historical database '{database_id}' holds generation data (data_type '{row[0]}'), not load -- "
            "PF data generation sources its load from a 'load' database."
        )
    cursor.execute('SELECT DISTINCT grid_level FROM "HistoricalSeries" WHERE database_id = %s', (database_id,))
    levels = {r[0] for r in cursor.fetchall()}
    if not levels:
        raise Exception(f"Historical database '{database_id}' has no series.")
    if len(levels) > 1:
        raise Exception(
            f"Historical database '{database_id}' mixes grid levels ({sorted(levels)}) -- "
            "PF data generation requires every series in the database to share one grid_level."
        )
    return levels.pop()


def get_historical_timeline(database_id: str, cursor) -> List:
    cursor.execute('SELECT datetime FROM "HistoricalTimestamp" WHERE database_id = %s ORDER BY datetime', (database_id,))
    return [r[0] for r in cursor.fetchall()]


def get_historical_series_ids(database_id: str, grid_level: str, cursor) -> List[str]:
    """series_id list for this database/grid_level, restricted to series
    that actually have power data -- a voltage-only series can't supply a
    P/Q injection even if it's tagged the right level. Uses the stored
    per-series counts (kept exact by triggers, see
    database/create_historical_database.py): checking the records themselves
    scanned the whole table, ~90 s on the 41.8M-row feeder database."""
    cursor.execute(
        'SELECT series_id FROM "HistoricalSeries" '
        "WHERE database_id = %s AND grid_level = %s AND n_power > 0 ORDER BY series_id",
        (database_id, grid_level),
    )
    return [r[0] for r in cursor.fetchall()]


def fetch_series_values(database_id: str, series_ids: List[str], timestamps: List,
                        cursor) -> Dict[Tuple[str, object], Tuple[float, Optional[float]]]:
    """Batched lookup: {(series_id, datetime): (power_active, power_reactive)}
    for exactly the given series/timestamps -- bounds memory/time to what's
    actually used, not the database's full history.

    Values are three-phase totals (W / VAr), the MV convention: a series' row
    with phase IS NULL is taken as the total when it exists, otherwise its
    per-phase (R/S/T) rows are summed. Picking one or the other, never adding
    both, keeps a series that stores an aggregate alongside its phases from
    being double-counted. Missing reactive power stays None (not 0): the
    caller fills it from a power factor."""
    if not series_ids or not timestamps:
        return {}
    cursor.execute(
        """
        SELECT series_id, datetime,
               COALESCE(SUM(power_active) FILTER (WHERE phase IS NULL), SUM(power_active)),
               COALESCE(SUM(power_reactive) FILTER (WHERE phase IS NULL), SUM(power_reactive))
        FROM "HistoricalRecords"
        WHERE database_id = %s AND series_id = ANY(%s) AND datetime = ANY(%s)
        GROUP BY series_id, datetime
        """,
        (database_id, series_ids, timestamps),
    )
    return {(r[0], r[1]): (r[2] or 0.0, r[3]) for r in cursor.fetchall()}


# Shape divisors over a database (scale_over `database`), kept for the process: they depend only on
# the database, so the runs of a group or package read them once. Keyed by the database and its
# timeline's length and ends, so a database re-uploaded with other timestamps is read again.
_REFERENCE_MEANS: Dict[tuple, Dict[str, float]] = {}


def reference_means(database_id: str, timeline: List, series_ids: List[str], cursor,
                    size_by: str = "mean") -> Dict[str, float]:
    """The shape divisor of each series (mean |P|, or its peak with size_by
    `peak`) over the database's reference sample
    (load_shape.py::reference_timestamps), every series of `series_ids` read
    at once and cached -- the records too, so both divisors cost one read."""
    entry = _reference_entry(database_id, timeline, series_ids, cursor)
    if size_by not in entry:
        entry[size_by] = series_scales(entry["values"], series_ids, entry["ref"], size_by)
    return entry[size_by]


def _reference_entry(database_id: str, timeline: List, series_ids: List[str], cursor) -> dict:
    key = (database_id, len(timeline), str(timeline[0]), str(timeline[-1]), tuple(sorted(series_ids)))
    if key not in _REFERENCE_MEANS:
        ref = reference_timestamps(timeline)
        _REFERENCE_MEANS[key] = {"ref": ref, "values": fetch_series_values(database_id, list(series_ids), ref, cursor)}
    return _REFERENCE_MEANS[key]


def reference_matrix(database_id: str, timeline: List, series_ids: List[str], cursor) -> Tuple[Dict[str, int], np.ndarray]:
    """The reference sample of reference_means as a series x timestamp matrix
    (load_shape.series_matrix), for the divisor of a sum of series; cached with it."""
    entry = _reference_entry(database_id, timeline, series_ids, cursor)
    if "matrix" not in entry:
        entry["matrix"] = series_matrix(entry["values"], sorted(series_ids), entry["ref"])
    return entry["matrix"]


def fetch_interval_means(database_id: str, series_ids: List[str], lo, hi, step_s: float,
                         cursor) -> Dict[Tuple[str, object], Tuple[float, Optional[float]]]:
    """{(series_id, interval start): (mean P, mean Q)} over the records from
    `lo` to `hi` (inclusive), averaged per clock-aligned interval of step_s
    in the database -- for a long consecutive window, where listing every
    record's timestamp (fetch_series_values) would be slow. Same values as
    timeline.py::average_intervals over fetch_series_values: three-phase
    totals per record, the mean over the records present, Q only when every
    one has it."""
    if not series_ids:
        return {}
    cursor.execute(
        """
        SELECT series_id, date_bin(%s::interval, datetime, TIMESTAMP '2000-01-01 00:00:00') AS start,
               AVG(COALESCE(p, 0.0)), CASE WHEN COUNT(q) = COUNT(*) THEN AVG(q) END
        FROM (
            SELECT series_id, datetime,
                   COALESCE(SUM(power_active) FILTER (WHERE phase IS NULL), SUM(power_active)) AS p,
                   COALESCE(SUM(power_reactive) FILTER (WHERE phase IS NULL), SUM(power_reactive)) AS q
            FROM "HistoricalRecords"
            WHERE database_id = %s AND series_id = ANY(%s) AND datetime >= %s AND datetime <= %s
            GROUP BY series_id, datetime) r
        GROUP BY series_id, start
        """,
        (f"{int(step_s)} seconds", database_id, list(series_ids), lo, hi),
    )
    return {(r[0], r[1]): (float(r[2]), None if r[3] is None else float(r[3])) for r in cursor.fetchall()}


def _lv_node_count(lv_grid_id: str, cursor) -> int:
    cursor.execute('SELECT COUNT(*) FROM "Node" WHERE grid_id = %s', (lv_grid_id,))
    return cursor.fetchone()[0] or 1


def build_scenarios(
    mv_grid_id: str, node_id_to_index: Dict[str, int], scenario_count: int,
    load_noise_sigma: float, historical_database_id: str, reassignment_period_timesteps: int,
    rng: np.random.Generator, cursor, *, load_shape: Optional[LoadShapeConfig] = None,
    point_kva: Optional[Dict[str, float]] = None, branches=None,
    shape_rng: Optional[np.random.Generator] = None, timeline: str = "spread", timeline_start=None,
    window_rng: Optional[np.random.Generator] = None, finer_step_s: Optional[float] = None,
    aggregation_s: Optional[float] = None,
) -> List[dict]:
    """Returns a list of {"timestamp": str, "S": np.ndarray complex (n_nodes,)}.

    `rng` draws the series assignment; `shape_rng` the power factors and the
    noise (Phase 2 axis 2, its own stream). `timeline` / `timeline_start` /
    `window_rng` choose the timestamps (pf_datagen/timeline.py). `point_kva` (connection-point node
    -> active transformer kVA) sizes the points under `rating` sizing;
    `branches` give the cable distances for spatially correlated noise.
    `finer_step_s`: the step of a denser series the run is matched with (a
    generation database); a `window` then steps at it and the load is
    interpolated linearly between the timeline's timestamps.
    `aggregation_s`: average the load over clock-aligned intervals of this
    length first (pf_datagen/timeline.py::aggregate_timeline); a window
    start then takes the interval that holds it."""
    load_shape = load_shape or LoadShapeConfig()
    shape_rng = shape_rng if shape_rng is not None else np.random.default_rng()
    source_type = get_historical_source_type(historical_database_id, cursor)
    sizing = resolve_point_sizing(load_shape.point_sizing, source_type)

    all_timestamps = get_historical_timeline(historical_database_id, cursor)
    if not all_timestamps:
        raise Exception(f"Historical database '{historical_database_id}' has no declared timeline.")

    db_timestamps = all_timestamps
    members = None                      # aggregated: interval start -> the database timestamps in it
    if aggregation_s:
        all_timestamps, members = aggregate_timeline(all_timestamps, aggregation_s)
        if not all_timestamps:
            raise Exception(f"Historical database '{historical_database_id}' has no complete "
                            f"{aggregation_s:g}-s interval to average the load over.")
        if timeline_start is not None:
            timeline_start = interval_start(timeline_start, aggregation_s)

    brackets = None
    if (timeline == "window" and finer_step_s and len(all_timestamps) > 1
            and finer_step_s < 0.99 * median_step_s(all_timestamps)):
        scenario_timestamps, brackets = interpolated_window(all_timestamps, scenario_count, finer_step_s,
                                                            timeline_start, window_rng)
    else:
        scenario_timestamps = select_timestamps(all_timestamps, scenario_count, timeline, timeline_start,
                                                window_rng)

    def read(stamps):
        """{(series, t): (P, Q)} at timeline timestamps `stamps` (interval means when aggregated)."""
        series = list(all_used_series)
        if members is None:
            return fetch_series_values(historical_database_id, series, stamps, cursor)
        if timeline == "window":
            # Consecutive intervals: one range read, averaged in the database.
            lo, hi = min(members[t][0] for t in stamps), max(members[t][-1] for t in stamps)
            wanted = set(stamps)
            return {k: v for k, v in fetch_interval_means(historical_database_id, series, lo, hi, aggregation_s,
                                                          cursor).items() if k[1] in wanted}
        raw = fetch_series_values(historical_database_id, series, [m for t in stamps for m in members[t]], cursor)
        return average_intervals(raw, series, {t: members[t] for t in stamps})

    # ── Population set + per-point pool size (LV only: node count; others: 1) ──
    # With rating sizing, MV and Feeder series populate every connection point:
    # the size comes from the point's transformers, not from an attached LV grid.
    if source_type == "MV" or (source_type == "Feeder" and sizing == "rating"):
        target_points = get_all_mv_connection_points(mv_grid_id, cursor)
        point_pool_size = {p: 1 for p in target_points}
    else:
        load_points = get_connected_load_points(mv_grid_id, cursor)
        if not load_points:
            raise Exception(
                f"MV grid '{mv_grid_id}' has no connected LV grids -- required for "
                f"grid_level='{source_type}' sourcing with point sizing '{sizing}' (use point sizing "
                "'rating', or a grid_level='MV' database, if you don't have real LV connections)."
            )
        target_points = [lp["node_id"] for lp in load_points]
        if source_type == "LV":
            point_pool_size = {lp["node_id"]: _lv_node_count(lp["lv_grid_id"], cursor) for lp in load_points}
        else:  # "Feeder"
            point_pool_size = {p: 1 for p in target_points}
    target_points = [p for p in target_points if p in node_id_to_index]

    series_ids = get_historical_series_ids(historical_database_id, source_type, cursor)
    if not series_ids:
        raise Exception(
            f"No usable '{source_type}' series (with power data) found in historical "
            f"database '{historical_database_id}'."
        )

    # ── Chunk scenarios by reassignment period (every N scenarios, in timestamp
    # order), draw one assignment per chunk ──
    chunk_of_index = [i // reassignment_period_timesteps for i in range(len(scenario_timestamps))]
    chunks = sorted(set(chunk_of_index))

    # Feeders per point (by_kva): a count per point for the run, from its transformer's kVA band.
    combine = source_type == "Feeder" and load_shape.feeders_per_point == "by_kva"
    n_feeders = (draw_feeder_counts(target_points, point_kva or {}, load_shape.feeder_bands, len(series_ids), rng)
                 if combine else {})

    chunk_assignments: Dict[int, Dict[str, List[str]]] = {}
    all_used_series = set()
    for c in chunks:
        assignment: Dict[str, List[str]] = {}
        if combine:
            for p in target_points:      # distinct series within a point; points may share series
                assignment[p] = rng.choice(series_ids, size=n_feeders[p], replace=False).tolist()
        elif source_type == "Feeder":
            replace = len(series_ids) < len(target_points)
            chosen = rng.choice(series_ids, size=len(target_points), replace=replace)
            for p, s in zip(target_points, chosen):
                assignment[p] = [s]
        else:
            for p in target_points:
                n = point_pool_size[p]
                chosen = rng.choice(series_ids, size=n, replace=True)
                assignment[p] = chosen.tolist()
        chunk_assignments[c] = assignment
        all_used_series.update(s for slots in assignment.values() for s in slots)

    if brackets is None:
        values = read(scenario_timestamps)
    else:
        # Interpolated window: read the timeline timestamps around each one and blend.
        around = sorted({t for t0, t1, _ in brackets.values() for t in (t0, t1)})
        raw = read(around)
        values = {}
        for t, (t0, t1, w) in brackets.items():
            for s in all_used_series:
                a, b = raw.get((s, t0)), raw.get((s, t1))
                if a is None or b is None:
                    continue
                q = None if a[1] is None or b[1] is None else (1 - w) * a[1] + w * b[1]
                values[(s, t)] = ((1 - w) * a[0] + w * b[0], q)

    # ── Axis 2: sizing, reactive power, noise ──
    means = {}
    if sizing == "rating" and load_shape.scale_over == "database":
        means = reference_means(historical_database_id, db_timestamps, series_ids, cursor, load_shape.size_by)
    elif sizing == "rating":
        means = series_scales(values, all_used_series, scenario_timestamps, load_shape.size_by)
    size_w = point_mean_load_w(target_points, point_kva or {}, load_shape.utilisation) if sizing == "rating" else {}
    # A summed point (by_kva) is sized as one series: by the divisor of the sum, per block.
    sum_scale: Dict[int, Dict[str, float]] = {}
    if combine and sizing == "rating":
        if load_shape.scale_over == "database":
            row, M = reference_matrix(historical_database_id, db_timestamps, series_ids, cursor)
        else:
            row, M = series_matrix(values, sorted(all_used_series), scenario_timestamps)
        sum_scale = {c: {p: combination_scale(row, M, a[p], load_shape.size_by) for p in target_points}
                     for c, a in chunk_assignments.items()}
    pf_by_chunk = draw_power_factors(len(chunks), target_points, load_shape.pf_min, load_shape.pf_max, shape_rng)
    distances = (cable_distances_km(branches, list(node_id_to_index), target_points)
                 if load_shape.spatial_sigma > 0 and branches is not None else None)
    noise = LoadNoise(len(target_points), local_sigma=load_noise_sigma, spatial_sigma=load_shape.spatial_sigma,
                      correlation_km=load_shape.correlation_km, distance_km=distances, q_sigma=load_shape.q_sigma)
    always_pf = load_shape.reactive == "pf"

    n_nodes = len(node_id_to_index)
    scenarios = []
    for i, ts in enumerate(scenario_timestamps):
        S = np.zeros(n_nodes, dtype=complex)
        assignment = chunk_assignments[chunk_of_index[i]]
        pf = pf_by_chunk[chunk_of_index[i]]          # chunks are numbered 0, 1, 2, ...
        f_pq, f_q = noise.draw(shape_rng) if noise.active else (None, None)
        for j, point in enumerate(target_points):
            tan_p = tan_phi(pf[point])
            p_sum, q_sum = 0.0, 0.0
            for s in assignment[point]:
                p, q = values.get((s, ts), (0.0, None))
                if sizing == "rating" and not combine:
                    m = means.get(s, 0.0)
                    p, q = (p / m, None if q is None else q / m) if m > 0 else (0.0, None)
                if always_pf or q is None:
                    q = p * tan_p
                p_sum += p
                q_sum += q
            if sizing == "rating" and combine:
                m = sum_scale[chunk_of_index[i]][point]
                k = size_w[point] / m if m > 0 else 0.0
                p_sum, q_sum = p_sum * k, q_sum * k
            elif sizing == "rating":
                k = size_w[point] / len(assignment[point])
                p_sum, q_sum = p_sum * k, q_sum * k
            if f_pq is not None:
                p_sum, q_sum = p_sum * f_pq[j], q_sum * f_pq[j] * f_q[j]
            S[node_id_to_index[point]] = complex(p_sum, q_sum)
        scenarios.append({"timestamp": str(ts), "S": S})

    return scenarios
