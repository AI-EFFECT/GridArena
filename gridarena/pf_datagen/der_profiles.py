"""Generated DER, driven by historical generation databases -- the only
source of generated DER. Each plant group (der.groups,
schemas.py::DERGroup) follows one database; the settings they share are in
der.generation (schemas.py::DERGenerationConfig).

Two halves, like the load scenarios:

- `build_generation_profile` (GridArena side, reads the database, called by
  pf_datagen/inputs.py::prepare_inputs once per database the groups use):
  the database's values at each scenario's timestamp. The scenario
  timestamps are local clock time in the request's time zone; the
  database's are in its own clock (`generation.timezone`, UTC for the
  E-REDES imports). Both are compared as
  instants. With `generation_start` the generation window starts there
  instead of at the load's first instant -- by default shifted by whole days,
  so the day/night cycle of the generation matches the load's clock. A
  generation value between two of the database's timestamps is interpolated
  linearly (the generation is the sparser series, e.g. 15-min data under a
  1-min load). With the load aggregated (`average_s`, the request's
  load_aggregation_s) a database with a finer step is instead averaged over
  each interval: its points at the interval's start and every step after
  it. A timestamp in a gap of the data (missing timestamps or
  records) takes the same clock time on the nearest day that has data, up to
  MAX_FALLBACK_DAYS away -- missing data is never read as zero production.
  The result (`GenerationProfile`) is stored with the run's inputs.

- `farm_shares` (solve side, no database): for a group reading its database
  as `aggregated`, the output share of each synthetic plant k (numbered
  across all groups) at every scenario,
      share_k(t) = clip(f(t) * a_k * (1 + eps_k(t)), 0, 1),
  where f(t) = P(t) / max(P over the reference year) is the aggregate's shape
  (one denominator for every plant), a_k ~ U[performance_min,
  performance_max] is drawn once per plant, and eps_k is an AR(1) process,
      eps_k(j) = rho_s * eps_k(j-1) + sigma * sqrt(1 - rho_s^2) * z_k(j),
  eps_k(0) ~ N(0, sigma), advanced at every step of the database's own
  interval over the generation window (night included), so the plants share
  the aggregate's daily, seasonal and weather pattern but differ by a
  persistent random adjustment. sigma is its stationary std; rho is given
  per 15 minutes and rho_s = rho ** (step / 15 min). A plant of capacity C
  delivers C * share_k(t) -- zero whenever f(t) is zero, never above C. The
  plants are synthetic profiles derived from an aggregate pattern, not
  reconstructions of real plants; their total is not forced to match the
  aggregate. For an `individual` group each series is one plant and its
  values are used as recorded.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import numpy as np

from gridarena.pf_datagen.sampling import STREAM_DER_HISTORICAL, sample_stream

MAX_FALLBACK_DAYS = 31          # how far a gap in the generation data may borrow the same clock time from
GAP_FACTOR = 1.5                # two timestamps further apart than this x the step are a gap, not a step
DAY_S = 86400.0
STEP_15MIN_S = 900.0

# The DER type a generated unit gets from the database's data_type.
DER_TYPE_OF_DATA_TYPE = {"pv": "pv", "wind": "wind", "chp": "chp", "hydro": "hydro",
                         "other_generation": "other", "injection_total": "generation"}


@dataclass
class GenerationProfile:
    database_id: str
    data_type: str
    series_ids: List[str]
    values_w: np.ndarray               # (n_series, n_scenarios): each series at each scenario, W
    ratings_w: np.ndarray              # (n_series,): each series' maximum over the database
    reference_year: Optional[int]
    reference_max_w: Optional[float]   # max of the aggregate (sum of the series) over the reference year
    gen_times: np.ndarray              # (n_scenarios,): the generation instant read for each scenario, epoch s
    step_s: float                      # the database's own interval
    shift_s: float                     # generation instant - load instant (before any gap fallback)
    n_fallback: int                    # scenarios whose instant fell in a gap and borrowed another day's
    first: str = ""
    last: str = ""
    info: Dict = field(default_factory=dict)

    @property
    def aggregate_w(self) -> np.ndarray:
        return self.values_w.sum(axis=0)

    @property
    def der_type(self) -> str:
        return DER_TYPE_OF_DATA_TYPE.get(self.data_type, self.data_type)

    def summary(self) -> dict:
        return {"database_id": self.database_id, "data_type": self.data_type, "der_type": self.der_type,
                "n_series": len(self.series_ids), "step_s": self.step_s, "shift_days": self.shift_s / DAY_S,
                "first": self.first, "last": self.last, "n_fallback": self.n_fallback,
                "reference_year": self.reference_year, "reference_max_w": self.reference_max_w, **self.info}


# ── Time ──

def to_epoch(ts: datetime, tz: str) -> float:
    """A naive clock time in `tz` as an instant (s). Ambiguous times (the
    repeated autumn hour) take the first occurrence."""
    if tz.upper() == "UTC":
        return ts.replace(tzinfo=timezone.utc).timestamp()
    return ts.replace(tzinfo=ZoneInfo(tz)).timestamp()


def from_epoch(t: float, tz: str) -> datetime:
    utc = datetime.fromtimestamp(t, tz=timezone.utc)
    return (utc if tz.upper() == "UTC" else utc.astimezone(ZoneInfo(tz))).replace(tzinfo=None)


def median_step_s(epochs: np.ndarray) -> float:
    d = np.diff(np.sort(epochs))
    d = d[d > 0]
    if not len(d):
        raise ValueError("A timeline needs at least two distinct timestamps to have a step.")
    return float(np.median(d))


def window_shift_s(scenario_epochs: np.ndarray, start_epoch: Optional[float], keep_time_of_day: bool) -> float:
    """generation instant - load instant: the generation window starts at
    `start_epoch` where the load's starts; whole days if keep_time_of_day."""
    if start_epoch is None:
        return 0.0
    shift = start_epoch - float(np.min(scenario_epochs))
    return round(shift / DAY_S) * DAY_S if keep_time_of_day else shift


def locate(targets: np.ndarray, grid: np.ndarray, usable: np.ndarray, step_s: float,
           max_days: int = MAX_FALLBACK_DAYS) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """For each target instant, the two grid points to interpolate between
    (j0, j1, weight of j1) and the instant actually read. A target is
    readable when it sits on a usable grid point, or between two usable
    neighbours at most GAP_FACTOR x step apart; otherwise it takes the same
    clock time 1, 2, ... days away (nearest first, earlier before later)
    until one is readable. Raises if none within max_days."""
    n = len(grid)
    j0 = np.zeros(len(targets), dtype=np.int64)
    j1 = np.zeros(len(targets), dtype=np.int64)
    w = np.zeros(len(targets))
    used = np.zeros(len(targets))

    def readable(t):
        j = int(np.searchsorted(grid, t, side="right")) - 1
        if j >= 0 and grid[j] == t:
            return (j, j, 0.0) if usable[j] else None
        if j < 0 or j + 1 >= n or not (usable[j] and usable[j + 1]):
            return None
        if grid[j + 1] - grid[j] > GAP_FACTOR * step_s:
            return None
        return j, j + 1, (t - grid[j]) / (grid[j + 1] - grid[j])

    for i, t in enumerate(targets):
        hit = readable(t)
        d = 0
        while hit is None and d < max_days:
            d += 1
            for cand in (t - d * DAY_S, t + d * DAY_S):
                hit = readable(cand)
                if hit is not None:
                    t = cand
                    break
        if hit is None:
            raise ValueError(
                f"No generation data within {max_days} days of {datetime.fromtimestamp(targets[i], tz=timezone.utc):%Y-%m-%d %H:%M} UTC "
                f"(the database covers {datetime.fromtimestamp(grid[0], tz=timezone.utc):%Y-%m-%d} to "
                f"{datetime.fromtimestamp(grid[-1], tz=timezone.utc):%Y-%m-%d} UTC): set der.generation.generation_start "
                "inside the data.")
        j0[i], j1[i], w[i] = hit
        used[i] = t
    return j0, j1, w, used


# ── Database side ──

def _fetch_power(database_id: str, series_ids: List[str], timestamps: List[datetime], cursor) -> Dict:
    """{(series_id, datetime): power_active (W, three-phase total)}, absent when
    the record or its power is missing (never 0 for missing)."""
    if not timestamps:
        return {}
    cursor.execute(
        """
        SELECT series_id, datetime,
               COALESCE(SUM(power_active) FILTER (WHERE phase IS NULL), SUM(power_active))
        FROM "HistoricalRecords"
        WHERE database_id = %s AND series_id = ANY(%s) AND datetime = ANY(%s)
        GROUP BY series_id, datetime
        """,
        (database_id, series_ids, timestamps),
    )
    return {(r[0], r[1]): float(r[2]) for r in cursor.fetchall() if r[2] is not None}


def _series_maxima(database_id: str, series_ids: List[str], cursor) -> Dict[str, float]:
    cursor.execute(
        """
        SELECT series_id, MAX(p) FROM (
            SELECT series_id, datetime, COALESCE(SUM(power_active) FILTER (WHERE phase IS NULL), SUM(power_active)) AS p
            FROM "HistoricalRecords" WHERE database_id = %s AND series_id = ANY(%s)
            GROUP BY series_id, datetime) s
        GROUP BY series_id
        """,
        (database_id, series_ids),
    )
    return {r[0]: float(r[1]) for r in cursor.fetchall() if r[1] is not None}


def _reference_max(database_id: str, series_ids: List[str], year: int, cursor) -> Optional[float]:
    """Max over the year of the aggregate (the series summed per timestamp)."""
    cursor.execute(
        """
        SELECT MAX(total) FROM (
            SELECT datetime, SUM(p) AS total FROM (
                SELECT series_id, datetime,
                       COALESCE(SUM(power_active) FILTER (WHERE phase IS NULL), SUM(power_active)) AS p
                FROM "HistoricalRecords"
                WHERE database_id = %s AND series_id = ANY(%s) AND datetime >= %s AND datetime < %s
                GROUP BY series_id, datetime) s
            GROUP BY datetime) t
        """,
        (database_id, series_ids, datetime(year, 1, 1), datetime(year + 1, 1, 1)),
    )
    row = cursor.fetchone()
    return None if row is None or row[0] is None else float(row[0])


def generation_database(database_id: str, cursor) -> Tuple[str, List[str], List[datetime]]:
    """The database's data_type, its series with power data and its timeline;
    raises unless it is a generation database with both."""
    cursor.execute('SELECT data_type FROM "HistoricalDatabase" WHERE database_id = %s', (database_id,))
    row = cursor.fetchone()
    if row is None:
        raise ValueError(f"Historical database '{database_id}' does not exist.")
    if row[0] == "load":
        raise ValueError(f"Historical database '{database_id}' holds load, not generation: der.groups need a "
                         "generation database (PV, wind, hydro, ...).")
    cursor.execute('SELECT series_id FROM "HistoricalSeries" WHERE database_id = %s AND n_power > 0 '
                   'ORDER BY series_id', (database_id,))
    series_ids = [r[0] for r in cursor.fetchall()]
    if not series_ids:
        raise ValueError(f"Historical database '{database_id}' has no series with power data.")
    cursor.execute('SELECT datetime FROM "HistoricalTimestamp" WHERE database_id = %s ORDER BY datetime',
                   (database_id,))
    timeline = [r[0] for r in cursor.fetchall()]
    if len(timeline) < 2:
        raise ValueError(f"Historical database '{database_id}' needs at least two timestamps.")
    return row[0], series_ids, timeline


def generation_step_s(database_id: str, timezone_name: str, cursor) -> float:
    """The database's own interval (median spacing of its timeline)."""
    _, _, timeline = generation_database(database_id, cursor)
    return median_step_s(np.array([to_epoch(t, timezone_name) for t in timeline]))


def default_reference_year(timeline: Sequence[datetime]) -> int:
    """The year with the most timestamps (the later one on a tie)."""
    counts: Dict[int, int] = {}
    for t in timeline:
        counts[t.year] = counts.get(t.year, 0) + 1
    return max(counts, key=lambda y: (counts[y], y))


def build_generation_profile(cfg, database_id: str, aggregated: bool, scenario_timestamps: Sequence,
                             load_timezone: str, cursor, average_s: Optional[float] = None) -> GenerationProfile:
    """One generation database's values at each scenario (see the module
    docstring). `cfg`: schemas.py::DERGenerationConfig; `aggregated`: some
    group reads it as aggregated, so it needs the reference maximum;
    `scenario_timestamps`: the load scenarios' timestamps (str or datetime,
    local in load_timezone); `average_s`: the length of the interval each
    one starts (aggregated load), averaged over when finer than it."""
    data_type, series_ids, timeline = generation_database(database_id, cursor)
    tz = cfg.timezone
    grid = np.array([to_epoch(t, tz) for t in timeline])
    order = np.argsort(grid, kind="stable")
    grid, timeline = grid[order], [timeline[k] for k in order]
    step = median_step_s(grid)

    stamps = [t if isinstance(t, datetime) else datetime.fromisoformat(str(t)) for t in scenario_timestamps]
    scen = np.array([to_epoch(t, load_timezone) for t in stamps])
    start = to_epoch(cfg.generation_start.replace(tzinfo=None), tz) if cfg.generation_start is not None else None
    shift = window_shift_s(scen, start, cfg.keep_time_of_day)
    # Points read per scenario: its instant, or every step over its interval.
    n_avg = int(round(average_s / step)) if average_s and step < 0.99 * average_s else 1
    targets = (scen[:, None] + shift + np.arange(n_avg)[None, :] * step).ravel()

    # Locate every target, fetch the grid points it needs, and redo those whose
    # points turn out to lack a record for some series (a gap in the records
    # rather than in the timeline), until every one reads complete data.
    usable = np.ones(len(grid), dtype=bool)
    values: Dict = {}
    fetched = np.zeros(len(grid), dtype=bool)
    for _ in range(20):
        j0, j1, w, used = locate(targets, grid, usable, step)
        need = sorted({int(j) for j in np.concatenate([j0, j1]) if not fetched[j]})
        if need:
            values.update(_fetch_power(database_id, series_ids, [timeline[j] for j in need], cursor))
            fetched[need] = True
        missing = [j for j in set(np.concatenate([j0, j1]).tolist())
                   if any((s, timeline[j]) not in values for s in series_ids)]
        if not missing:
            break
        usable[missing] = False
    else:
        raise ValueError(f"Historical database '{database_id}': too many gaps in the records around the run's "
                         "timestamps.")

    vals = np.zeros((len(series_ids), len(scen)))
    for k, s in enumerate(series_ids):
        v0 = np.array([values[(s, timeline[j])] for j in j0])
        v1 = np.array([values[(s, timeline[j])] for j in j1])
        vals[k] = ((1.0 - w) * v0 + w * v1).reshape(len(scen), n_avg).mean(axis=1)

    maxima = _series_maxima(database_id, series_ids, cursor)
    ratings = np.array([maxima.get(s, 0.0) for s in series_ids])
    ref_year, ref_max = None, None
    if aggregated:
        ref_year = cfg.reference_year or default_reference_year(timeline)
        ref_max = _reference_max(database_id, series_ids, ref_year, cursor)
        if ref_max is None or not np.isfinite(ref_max) or ref_max <= 0:
            raise ValueError(f"Historical database '{database_id}': no positive maximum in reference year "
                             f"{ref_year} to normalise the profile by (set der.generation.reference_year).")
    n_fallback = int(np.sum(np.any((np.abs(used - targets) > 1e-6).reshape(len(scen), n_avg), axis=1)))
    used = used.reshape(len(scen), n_avg)[:, 0]          # each scenario's first point read
    return GenerationProfile(
        database_id=database_id, data_type=data_type, series_ids=series_ids, values_w=vals, ratings_w=ratings,
        reference_year=ref_year, reference_max_w=ref_max, gen_times=used, step_s=step, shift_s=shift,
        n_fallback=n_fallback, first=from_epoch(float(used.min()), tz).isoformat(),
        last=from_epoch(float(used.max()), tz).isoformat(),
        info={"timezone": tz, **({"averaged_over_s": float(average_s), "points_averaged": n_avg} if n_avg > 1 else {})},
    )


PREVIEW_MAX_POINTS = 1500


def _aggregate_between(database_id: str, lo: datetime, hi: datetime, cursor) -> Tuple[List[datetime], np.ndarray]:
    """The database's total power (series summed) per timestamp in [lo, hi)."""
    cursor.execute(
        """
        SELECT datetime, SUM(p) FROM (
            SELECT series_id, datetime,
                   COALESCE(SUM(power_active) FILTER (WHERE phase IS NULL), SUM(power_active)) AS p
            FROM "HistoricalRecords"
            WHERE database_id = %s AND datetime >= %s AND datetime < %s
            GROUP BY series_id, datetime) s
        GROUP BY datetime ORDER BY datetime
        """,
        (database_id, lo, hi),
    )
    rows = [(r[0], r[1]) for r in cursor.fetchall() if r[1] is not None]
    return [r[0] for r in rows], np.array([float(r[1]) for r in rows])


def _thin(times: List, values: np.ndarray) -> Tuple[List, np.ndarray]:
    if len(times) <= PREVIEW_MAX_POINTS:
        return times, values
    idx = np.linspace(0, len(times) - 1, PREVIEW_MAX_POINTS).round().astype(int)
    return [times[k] for k in idx], values[idx]


def _interval_means(times: List[datetime], values: np.ndarray, step_s: Optional[float]
                    ) -> Tuple[List[datetime], np.ndarray]:
    """Mean per clock-aligned interval of step_s (labelled by its start); as given when none or not finer."""
    if not step_s or len(times) < 2 or median_step_s(np.array([t.timestamp() for t in times])) >= 0.99 * step_s:
        return times, values
    from gridarena.pf_datagen.timeline import interval_start
    sums: Dict[datetime, List[float]] = {}
    for t, v in zip(times, values):
        sums.setdefault(interval_start(t, step_s), []).append(float(v))
    starts = sorted(sums)
    return starts, np.array([np.mean(sums[s]) for s in starts])


def generation_preview(load_database_id: Optional[str], database_ids: Sequence[str], load_start: Optional[datetime],
                       generation_start: Optional[datetime], keep_time_of_day: bool, gen_timezone: str,
                       load_timezone: str, days: float, cursor, load_aggregation_s: Optional[float] = None) -> dict:
    """For the wizard: the load's first `days` from `load_start` (default: the
    load database's first timestamp) and each generation database over the
    window it would be read at (same shift as build_generation_profile),
    both on the load's clock and normalised by their maximum in the window,
    so day/night alignment shows. Values are the databases' totals (series
    summed); gaps are left out here (a run borrows the nearest day). Each
    database also gives its series' ratings (maxima), what an individual
    group installs. With `load_aggregation_s` both are averaged over its
    intervals where finer, as a run would."""
    out = {"load": None, "generation": [], "days": days}
    if load_start is None and load_database_id:
        cursor.execute('SELECT MIN(datetime) FROM "HistoricalTimestamp" WHERE database_id = %s', (load_database_id,))
        row = cursor.fetchone()
        load_start = row[0] if row else None
    if load_start is None:
        raise ValueError("Choose a load database (step 2), or a window start, to preview against.")
    load_start = load_start.replace(tzinfo=None)
    load_end = load_start + timedelta(days=days)
    start_epoch = to_epoch(load_start, load_timezone)
    if load_database_id:
        lt, lv = _aggregate_between(load_database_id, load_start, load_end, cursor)
        load_step = median_step_s(np.array([t.timestamp() for t in lt])) if len(lt) > 1 else None
        lt, lv = _thin(*_interval_means(lt, lv, load_aggregation_s))
        peak = float(np.max(np.abs(lv))) if len(lv) else 0.0
        out["load"] = {"database_id": load_database_id, "step_s": load_step, "times": [t.isoformat() for t in lt],
                       "values": (lv / peak).tolist() if peak > 0 else lv.tolist(), "max_w": peak}
    gen_start_epoch = (to_epoch(generation_start.replace(tzinfo=None), gen_timezone)
                       if generation_start is not None else None)
    shift = window_shift_s(np.array([start_epoch]), gen_start_epoch, keep_time_of_day)
    out["load_start"] = load_start.isoformat()
    out["shift_days"] = shift / DAY_S
    for database_id in database_ids:
        data_type, series_ids, timeline = generation_database(database_id, cursor)
        grid = np.array([to_epoch(t, gen_timezone) for t in timeline])
        lo = from_epoch(start_epoch + shift, gen_timezone)
        hi = from_epoch(start_epoch + shift + days * DAY_S, gen_timezone)
        gt, gv = _aggregate_between(database_id, lo, hi, cursor)
        gt, gv = _thin(*_interval_means(gt, gv, load_aggregation_s))
        peak = float(np.max(gv)) if len(gv) else 0.0
        # On the load's clock: each generation instant moved back by the shift.
        local = [from_epoch(to_epoch(t, gen_timezone) - shift, load_timezone).isoformat() for t in gt]
        ratings = [r for r in _series_maxima(database_id, series_ids, cursor).values() if r > 0]
        out["generation"].append({
            "database_id": database_id, "data_type": data_type, "der_type": DER_TYPE_OF_DATA_TYPE.get(data_type),
            "n_series": len(series_ids), "step_s": median_step_s(grid), "first": timeline[0].isoformat(),
            "last": timeline[-1].isoformat(), "window_first": lo.isoformat(), "window_last": hi.isoformat(),
            "times": local, "values": (gv / peak).tolist() if peak > 0 else gv.tolist(), "max_w": peak,
            "n_producing": len(ratings), "ratings_w_total": float(sum(ratings)),
        })
    return out


# ── Solve side ──

def ar1(rng: np.random.Generator, n: int, rho: float, sigma: float,
        state: Optional[float] = None) -> Tuple[np.ndarray, float]:
    """n steps of eps(j) = rho eps(j-1) + sigma sqrt(1-rho^2) z(j). Without a
    `state` the first value is drawn from N(0, sigma) (the stationary
    distribution); with one, the process continues from it. Chunks drawn in
    order from the same generator give the same sequence as one call."""
    if n <= 0:
        return np.zeros(0), state if state is not None else 0.0
    out = np.empty(n)
    k = 0
    prev = state
    if prev is None:
        out[0] = prev = rng.normal(0.0, sigma)
        k = 1
    z = rng.standard_normal(n - k)
    if n > k:
        from scipy.signal import lfilter
        # y(j) = x(j) + rho y(j-1), started from y(-1) = prev.
        out[k:] = lfilter([1.0], [1.0, -rho], sigma * np.sqrt(1.0 - rho * rho) * z, zi=[rho * prev])[0]
    return out, float(out[-1])


@dataclass
class FarmParameters:
    farm_index: int
    performance: float
    rho_step: float
    sigma: float


def farm_shares(profile: GenerationProfile, farms: Sequence[int], cfg, seed: int
                ) -> Tuple[np.ndarray, List[FarmParameters]]:
    """(len(farms), n_scenarios) output shares of the synthetic plants
    numbered `farms` (each its own stream), built from an aggregated profile,
    and each plant's parameters. `cfg`: schemas.py::DERGenerationConfig."""
    f = profile.aggregate_w / profile.reference_max_w
    t = profile.gen_times
    t0 = float(t.min())
    n_steps = int(round((float(t.max()) - t0) / profile.step_s)) + 1
    grid = t0 + np.arange(n_steps) * profile.step_s
    rho_step = cfg.rho ** (profile.step_s / STEP_15MIN_S)
    shares = np.zeros((len(farms), len(t)))
    params = []
    for row, k in enumerate(farms):
        rng = sample_stream(seed, STREAM_DER_HISTORICAL, int(k))
        a = float(rng.uniform(cfg.performance_min, cfg.performance_max))
        eps, _ = ar1(rng, n_steps, rho_step, cfg.sigma)
        eps_t = np.interp(t, grid, eps)
        shares[row] = np.clip(f * a * (1.0 + eps_t), 0.0, 1.0)
        params.append(FarmParameters(int(k), a, rho_step, cfg.sigma))
    return shares, params


# ── Stored with the run's inputs ──

_META = ("database_id", "data_type", "series_ids", "reference_year", "reference_max_w", "step_s", "shift_s",
         "n_fallback", "first", "last", "info")


def profiles_to_files(profiles: Dict[str, GenerationProfile], directory) -> None:
    """generation.json (each profile's metadata, in order) and generation_<k>.npz."""
    import json
    from pathlib import Path
    d = Path(directory)
    metas = []
    for k, profile in enumerate(profiles.values()):
        np.savez_compressed(d / f"generation_{k}.npz", values_w=profile.values_w, ratings_w=profile.ratings_w,
                            gen_times=profile.gen_times)
        metas.append({key: getattr(profile, key) for key in _META})
    with open(d / "generation.json", "w") as fh:
        json.dump({"profiles": metas}, fh, indent=1)


def profiles_from_files(directory) -> Optional[Dict[str, GenerationProfile]]:
    """The profiles saved by profiles_to_files, by database_id (None: no generated DER)."""
    import json
    from pathlib import Path
    d = Path(directory)
    if not (d / "generation.json").exists():
        return None
    with open(d / "generation.json") as fh:
        metas = json.load(fh)["profiles"]
    out = {}
    for k, meta in enumerate(metas):
        with np.load(d / f"generation_{k}.npz") as z:
            out[meta["database_id"]] = GenerationProfile(
                values_w=z["values_w"].copy(), ratings_w=z["ratings_w"].copy(), gen_times=z["gen_times"].copy(),
                **meta)
    return out
