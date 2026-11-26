"""Which historical timestamps a run uses (Phase 2, `timeline`).

- `spread` (the default, as before Phase 2): `scenario_count` timestamps
  evenly spread over the database's whole timeline.
- `window`: `scenario_count` consecutive timestamps, for forecasting
  datasets. The window starts at `timeline_start` (the first timestamp at or
  after it), or at a start drawn from stream 17 when none is given; a start
  too close to the end moves back so the window keeps its length.

Either way a timeline no longer than `scenario_count` is used whole.
When generated DER follow a generation database with a finer step than the
load's, a `window` steps at the generation's interval instead and the load
is interpolated linearly between its own timestamps (`interpolated_window`):
the sparser series is always the interpolated one.
With `load_aggregation_s` the load is first averaged over clock-aligned
intervals of that length (`aggregate_timeline`), each labelled with its
start; the intervals are then the timeline, and the run steps at them.
`per_sample_axes` lists the settings that draw something new for every
sample, which break the time continuity a `window` is for.
"""

from bisect import bisect_right
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np


def select_timestamps(all_timestamps: List, count: int, mode: str = "spread",
                      start: Optional[datetime] = None, rng: Optional[np.random.Generator] = None) -> List:
    n = len(all_timestamps)
    if n <= count:
        return list(all_timestamps)
    if mode == "spread":
        idx = sorted(set(np.linspace(0, n - 1, count, dtype=int).tolist()))
        return [all_timestamps[i] for i in idx]
    last_start = n - count
    if start is not None:
        first = next((i for i, ts in enumerate(all_timestamps) if ts >= start), n)
        i0 = min(first, last_start)
    else:
        rng = rng if rng is not None else np.random.default_rng()
        i0 = int(rng.integers(0, last_start + 1))
    return list(all_timestamps[i0:i0 + count])


def median_step_s(timestamps: List) -> float:
    d = np.diff([t.timestamp() if isinstance(t, datetime) else float(t) for t in sorted(timestamps)])
    d = d[d > 0]
    return float(np.median(d)) if len(d) else 0.0


MIN_INTERVAL_COVERAGE = 0.5     # an aggregation interval holding less of its timestamps than this is a gap


def _as_datetime(t) -> datetime:
    return t if isinstance(t, datetime) else datetime.fromisoformat(str(t))


def interval_start(ts, step_s: float) -> datetime:
    """Start of the clock-aligned interval of `step_s` (dividing a day) that
    holds `ts`: intervals start at midnight and every step_s after it."""
    ts = _as_datetime(ts)
    midnight = ts.replace(hour=0, minute=0, second=0, microsecond=0)
    return midnight + timedelta(seconds=((ts - midnight).total_seconds() // step_s) * step_s)


def aggregate_timeline(all_timestamps: List, step_s: float) -> Tuple[List[datetime], Dict[datetime, List]]:
    """The clock-aligned intervals of `step_s` over a timeline: their starts
    (in order) and the timeline's timestamps in each (as given, to key the
    records by). An interval holding fewer than MIN_INTERVAL_COVERAGE of the
    timestamps it should (step_s / the timeline's own step) is a gap and
    left out. Raises if step_s is finer than the timeline's own step."""
    if not all_timestamps:
        return [], {}
    own = median_step_s([_as_datetime(t) for t in all_timestamps]) if len(all_timestamps) > 1 else 0.0
    if own and step_s < 0.99 * own:
        raise ValueError(f"load_aggregation_s ({step_s:g} s) is finer than the load database's own step "
                         f"({own:g} s): it can only make the load coarser.")
    members: Dict[datetime, List] = {}
    for t in all_timestamps:
        members.setdefault(interval_start(t, step_s), []).append(t)
    need = MIN_INTERVAL_COVERAGE * step_s / own if own else 1
    kept = {k: v for k, v in members.items() if len(v) >= need - 1e-9}
    return sorted(kept), kept


def average_intervals(raw: Dict, series_ids, members: Dict) -> Dict:
    """{(series, interval start): (mean P, mean Q)} from the records of the
    timestamps in each interval (`raw`: {(series, timestamp): (P, Q)}). The
    mean is over the records present; Q is None unless every one of them has
    it. An interval with no record of a series stays absent."""
    out = {}
    for s in series_ids:
        for start, stamps in members.items():
            recs = [raw[(s, t)] for t in stamps if (s, t) in raw]
            if not recs:
                continue
            p = sum(r[0] for r in recs) / len(recs)
            q = None if any(r[1] is None for r in recs) else sum(r[1] for r in recs) / len(recs)
            out[(s, start)] = (p, q)
    return out


def interpolated_window(all_timestamps: List, count: int, step_s: float, start: Optional[datetime] = None,
                        rng: Optional[np.random.Generator] = None, gap_factor: float = 1.5
                        ) -> Tuple[List[datetime], Dict[datetime, Tuple[datetime, datetime, float]]]:
    """A `window` of `count` timestamps step_s apart -- finer than the
    timeline's own step (another series, e.g. a generation database, is the
    denser one) -- starting at one of the timeline's timestamps (the first at
    or after `start`, or drawn from `rng`) and fitting inside it. Returns the
    timestamps and, for each, the two timeline timestamps around it with the
    weight of the later one, for linear interpolation. Raises if the window
    crosses a gap in the timeline (neighbours more than gap_factor x its
    step apart)."""
    ts = sorted(all_timestamps)
    secs = np.array([(t - ts[0]).total_seconds() for t in ts])
    own_step = median_step_s(ts)
    span = (count - 1) * step_s
    last_start = int(np.searchsorted(secs, secs[-1] - span, side="right")) - 1
    if last_start < 0:
        raise ValueError(f"The load timeline ({ts[0]} to {ts[-1]}) is shorter than a window of {count} steps of "
                         f"{step_s:g} s.")
    if start is not None:
        first = next((i for i, t in enumerate(ts) if t >= start), len(ts))
        i0 = min(first, last_start)
    else:
        rng = rng if rng is not None else np.random.default_rng()
        i0 = int(rng.integers(0, last_start + 1))
    targets = [ts[i0] + timedelta(seconds=k * step_s) for k in range(count)]
    brackets = {}
    for t in targets:
        x = (t - ts[0]).total_seconds()
        j = bisect_right(secs, x) - 1
        if secs[j] == x:
            brackets[t] = (ts[j], ts[j], 0.0)
            continue
        if j + 1 >= len(ts) or secs[j + 1] - secs[j] > gap_factor * own_step:
            raise ValueError(f"The load timeline has a gap around {t} ({ts[j]} to "
                             f"{ts[j + 1] if j + 1 < len(ts) else 'end'}): choose another window start.")
        brackets[t] = (ts[j], ts[j + 1], (x - secs[j]) / (secs[j + 1] - secs[j]))
    return targets, brackets


def timeline_summary(mode: str, timestamps: List[str]) -> dict:
    """First and last timestamp, the typical step and the gaps (steps longer
    than 1.5 x the typical one) of the timestamps a run used."""
    out = {"mode": mode, "n": len(timestamps), "first": None, "last": None, "step_s": None, "n_gaps": 0}
    if not timestamps:
        return out
    try:
        ts = sorted(datetime.fromisoformat(str(t)) for t in timestamps)
    except ValueError:                       # not ISO timestamps: only their range
        names = sorted(str(t) for t in timestamps)
        out["first"], out["last"] = names[0], names[-1]
        return out
    out["first"], out["last"] = ts[0].isoformat(), ts[-1].isoformat()
    if len(ts) > 1:
        steps = np.diff([t.timestamp() for t in ts])
        step = float(np.median(steps))
        out["step_s"] = step
        out["n_gaps"] = int((steps > 1.5 * step).sum())
    return out


def per_sample_axes(request) -> List[str]:
    """Settings of a request that draw new values for every sample."""
    axes = []
    if request.load_level.mode != "off":
        axes.append("load_level")
    if request.load_noise_sigma > 0 or request.load_shape.spatial_sigma > 0 or request.load_shape.q_sigma > 0:
        axes.append("load_noise")
    if request.der.source != "grid" and request.der.layout == "per_sample":
        axes.append("der_layout")
    if request.der.resample_grid_settings:
        axes.append("der_settings")
    sp = request.setpoints
    ranged = not isinstance(request.slack_voltage_pu, (int, float))
    if (sp.slack_mode == "uniform" or (sp.slack_mode is None and ranged)
            or (sp.slack_mode == "load_dependent" and sp.slack_noise > 0)):
        axes.append("busbar_setpoint")
    if sp.shunt_steps == "random":
        axes.append("capacitor_steps")
    if sp.regulators_resample:
        axes.append("regulators")
    out = request.outages
    if out.der_trip_p > 0 or out.shunt_out_p > 0 or out.regulator_bypass_p > 0:
        axes.append("outages")
    par = request.parameters
    t = par.temperature
    varying_temperature = t is not None and (t.max > t.min or t.spread > 0)
    if varying_temperature or (par.draw == "per_sample" and (
            par.length_sigma > 0 or par.swap_p > 0 or par.transformer_uk_sigma > 0 or par.transformer_xr_sigma > 0)):
        axes.append("parameters")
    if request.target_mix.min_violation_share is not None:
        axes.append("target_mix")
    return axes
