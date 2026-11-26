"""State estimation as a code-submission benchmark.

The user's script defines::

    predict(task: dict) -> {series_key: {"voltage_magnitude": float, "voltage_angle": float}}

and is called once per estimation task: one test grid at one timestamp, where
some series (node, phase) are measured at that timestamp ("known") and the
rest are not ("unknown") and must be estimated. Each test grid yields
GRIDARENA_SE_TASKS_PER_GRID tasks (default 3). Timestamps and the known/unknown
split are seeded per grid and settings, so every submission with the same
settings gets exactly the same tasks; series keys are random per task. The
true voltages stay here, in ``PreparedInputs.truth``.
"""

import hashlib
import math
import os
import random
import secrets
from collections import defaultdict
from datetime import timedelta

import gridarena.database as db
from gridarena.benchmarks.general_noise import corrupt_measurement_record, get_profile
from gridarena.benchmarks.general_noise.noise import _ANGLE_UNIT
from gridarena.benchmarks.state_estimation.helper import _pick_unknown_count, get_observability_profile
from gridarena.benchmarks.submissions import BenchmarkTask, PreparedInputs


def tasks_per_grid() -> int:
    return int(os.getenv("GRIDARENA_SE_TASKS_PER_GRID", "3"))


def _seeded(*parts) -> random.Random:
    return random.Random(hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest())


def _load_test_measurements():
    conn, cursor = db.get_db_connection()
    try:
        cursor.execute("SELECT grid_id FROM GridUsage WHERE state_estimation_test = 1")
        grid_ids = sorted(row[0] for row in cursor.fetchall())
        if not grid_ids:
            raise Exception("No grids are marked for state estimation testing.")
        cursor.execute(
            """
            SELECT grid_id, node_id, phase, datetime, power_active, power_reactive,
                   voltage_magnitude, voltage_angle
            FROM "Measurements"
            WHERE grid_id = ANY(%s)
            ORDER BY grid_id, node_id, phase, datetime
            """,
            (grid_ids,),
        )
        rows = cursor.fetchall()
        if not rows:
            raise Exception("No measurement data found for the state estimation test grids.")
        return rows
    finally:
        conn.close()


def _record(dt, p_act, p_rea, v_mag, v_ang, with_datetime=True) -> dict:
    record = {"power_active": p_act, "power_reactive": p_rea,
              "voltage_magnitude": v_mag, "voltage_angle": v_ang}
    if with_datetime:
        record = {"datetime": dt.isoformat() if hasattr(dt, "isoformat") else str(dt), **record}
    return record


def build_test_inputs(noise_difficulty: str, observability: str, measurements=None) -> PreparedInputs:
    """One call per (test grid, timestamp) task: ``predict({"timestamp", "known", "unknown", "history"})``.

    Tasks are drawn from the later half of each grid's timestamps, so there is
    history to learn from. Like the old interactive benchmark, the known series
    have history up to the timestamp, the unknown ones only up to
    ``unknown_history_hours`` (observability profile) before it."""
    if measurements is None:
        measurements = _load_test_measurements()
    noise = get_profile(noise_difficulty)
    obs = get_observability_profile(observability)
    unknown_gap = timedelta(hours=float(obs["unknown_history_hours"]))
    known_gap = timedelta(seconds=1)

    series = defaultdict(lambda: defaultdict(list))  # grid -> (node, phase) -> rows
    for grid_id, node_id, phase, dt, p_act, p_rea, v_mag, v_ang in measurements:
        series[grid_id][(node_id, phase)].append((dt, p_act, p_rea, v_mag, v_ang))

    calls, truth = {}, {}
    for grid_id in sorted(series):
        grid_series = series[grid_id]
        by_time = {s: {r[0]: r for r in rows} for s, rows in grid_series.items()}
        timestamps = sorted({t for rows in by_time.values() for t in rows})
        candidates = timestamps[len(timestamps) // 2:]
        picked = sorted(_seeded(grid_id, observability, "tasks").sample(
            candidates, min(tasks_per_grid(), len(candidates))))

        for ts in picked:
            at_ts = {s: rows[ts] for s, rows in by_time.items() if ts in rows}
            split_rng = _seeded(grid_id, ts, observability, "split")
            order = sorted(at_ts)
            split_rng.shuffle(order)
            unknown = set(order[:_pick_unknown_count(len(order), observability, split_rng)])

            keys = {}
            for s in order:
                key = secrets.token_hex(5)
                while key in keys.values():
                    key = secrets.token_hex(5)
                keys[s] = key

            known, history, values = {}, {}, {}
            for s in order:
                node_id, phase = s
                gap = unknown_gap if s in unknown else known_gap
                rng = _seeded(grid_id, node_id, phase, noise_difficulty, ts, "history")
                history[keys[s]] = [corrupt_measurement_record(_record(*r), rng, noise)
                                    for r in grid_series[s] if r[0] < ts - gap]
                if s in unknown:
                    values[keys[s]] = [at_ts[s][3], at_ts[s][4]]
                else:
                    rng = _seeded(grid_id, node_id, phase, noise_difficulty, ts, "known")
                    known[keys[s]] = corrupt_measurement_record(
                        _record(*at_ts[s], with_datetime=False), rng, noise)

            name = f"{len(calls):04d}"
            calls[name] = [{
                "timestamp": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
                "known": known,
                "unknown": sorted(values),
                "history": history,
            }]
            truth[name] = {"grid_id": grid_id, "values": values}
    return PreparedInputs(calls=calls, truth=truth)


def _number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _angle_diff(a: float, b: float) -> float:
    """a - b wrapped to half a turn, in the measurements' angle unit."""
    half = math.pi if _ANGLE_UNIT == "rad" else 180.0
    return ((a - b + half) % (2 * half)) - half


class _Errors:
    """Running error totals, for one grid or for the whole submission."""

    def __init__(self):
        self.sq = []          # (dV)^2 of every estimate, penalties included
        self.sin = []         # |sin(dθ)| of every estimate, penalties included
        self.valid_sq = []    # (dV)^2 of usable estimates only
        self.valid_ang = []   # |dθ| of usable estimates only
        self.missing = self.invalid = 0

    def add(self, estimate, true_v: float, true_a: float) -> None:
        if not isinstance(estimate, dict):
            self.missing += 1
        elif not (_number(estimate.get("voltage_magnitude")) and _number(estimate.get("voltage_angle"))):
            self.invalid += 1
        else:
            dv = estimate["voltage_magnitude"] - true_v
            da = _angle_diff(estimate["voltage_angle"], true_a)
            self.sq.append(dv * dv)
            self.sin.append(abs(math.sin(da if _ANGLE_UNIT == "rad" else math.radians(da))))
            self.valid_sq.append(dv * dv)
            self.valid_ang.append(abs(da))
            return
        # A missing or unusable estimate is scored as 0 V with the worst angle.
        self.sq.append(true_v * true_v)
        self.sin.append(1.0)

    def merge(self, other: "_Errors") -> None:
        for name in ("sq", "sin", "valid_sq", "valid_ang"):
            getattr(self, name).extend(getattr(other, name))
        self.missing += other.missing
        self.invalid += other.invalid

    def summary(self) -> dict:
        n, valid = len(self.sq), len(self.valid_sq)
        return {
            "score": round(sum(self.sq) / n + sum(self.sin) / n, 6) if n else None,
            "rmse_voltage_magnitude": round(math.sqrt(sum(self.valid_sq) / valid), 6) if valid else None,
            "mae_voltage_angle": round(sum(self.valid_ang) / valid, 6) if valid else None,
            "estimates": n,
            "missing": self.missing,
            "invalid": self.invalid,
        }


def score_predictions(outputs: dict, truth: dict) -> dict:
    """``score`` = mean (dV)^2 + mean |sin(dθ)| over every unknown series of every
    task (the old /state-score formula; lower is better). RMSE of the voltage
    magnitude and MAE of the angle are over the usable estimates only."""
    by_grid = defaultdict(lambda: {"errors": _Errors(), "tasks": 0, "failed_tasks": 0,
                                   "seconds": 0.0, "error": None})
    runtime = 0.0

    for name in sorted(truth):
        grid = by_grid[truth[name]["grid_id"]]
        out = outputs.get(name, {})
        seconds = out.get("seconds") or 0.0
        grid["tasks"] += 1
        grid["seconds"] += seconds
        runtime += seconds

        prediction = out.get("result") if out.get("ok") else None
        if not out.get("ok"):
            lines = (out.get("error") or "predict() failed").strip().splitlines()
            error = lines[-1] if lines else "predict() failed"
        elif not isinstance(prediction, dict):
            error = f"predict() must return a dict, got {type(prediction).__name__}"
            prediction = None
        else:
            error = None
        if error:
            grid["failed_tasks"] += 1
            grid["error"] = grid["error"] or error

        for key, (true_v, true_a) in truth[name]["values"].items():
            grid["errors"].add((prediction or {}).get(key), true_v, true_a)

    total = _Errors()
    per_grid = []
    for grid_id in sorted(by_grid):
        grid = by_grid[grid_id]
        total.merge(grid["errors"])
        per_grid.append({
            "grid_id": grid_id,
            "tasks": grid["tasks"],
            **grid["errors"].summary(),
            "failed_tasks": grid["failed_tasks"],
            "seconds": round(grid["seconds"], 3),
            "error": grid["error"],
        })

    return {
        **total.summary(),
        "tasks": len(truth),
        "failed_tasks": sum(r["failed_tasks"] for r in per_grid),
        "grids": len(per_grid),
        "runtime_seconds": round(runtime, 3),
        "per_grid": per_grid,
    }


STATE_ESTIMATION_TASK = BenchmarkTask(
    name="state_estimation",
    function="predict",
    build_inputs=lambda settings: build_test_inputs(settings["noise_difficulty"], settings["observability"]),
    score=score_predictions,
)
