"""Phase identification as a code-submission benchmark.

The user's script defines::

    predict(measurements: dict[str, list[dict]]) -> dict[str, str]

and is called once per test grid with every measurement series of that grid
(so methods that compare series against each other work). The keys are random
per submission and per grid, and the series are shuffled, so neither key nor
order carries any information across submissions. The true phases stay here,
in ``PreparedInputs.truth``.
"""

import hashlib
import random
import secrets
from collections import defaultdict

import gridarena.database as db
from gridarena.benchmarks.phase_identification.get_measurement_data import (
    corrupt_measurement_record,
    get_all_measurements,
    get_corruption_profile,
)
from gridarena.benchmarks.submissions import BenchmarkTask, PreparedInputs

PHASES = ("R", "S", "T")


def _load_test_measurements():
    conn, cursor = db.get_db_connection()
    try:
        return get_all_measurements(cursor, task_type="phase_detection", phase_type="test")
    finally:
        conn.close()


def build_test_inputs(difficulty: str, measurements=None) -> PreparedInputs:
    """One call per test grid: ``predict({key: [record, ...], ...})``.

    Noise is seeded per (grid, node, phase, difficulty), so every submission
    at a given difficulty sees exactly the same measurement values."""
    if measurements is None:
        measurements = _load_test_measurements()
    profile = get_corruption_profile(difficulty=difficulty)

    series = defaultdict(list)  # (grid_id, node_id, phase) -> records
    rngs = {}
    for grid_id, node_id, phase, dt, p_act, p_rea, v_mag, v_ang in measurements:
        key = (grid_id, node_id, phase)
        if key not in rngs:
            seed = hashlib.sha256(f"{grid_id}:{node_id}:{phase}:{difficulty}".encode()).hexdigest()
            rngs[key] = random.Random(seed)
        record = {
            "datetime": dt.isoformat() if hasattr(dt, "isoformat") else str(dt),
            "power_active": p_act,
            "power_reactive": p_rea,
            "voltage_magnitude": v_mag,
            "voltage_angle": v_ang,
        }
        series[key].append(corrupt_measurement_record(record, rngs[key], profile))

    by_grid = defaultdict(list)
    for (grid_id, node_id, phase), records in series.items():
        by_grid[grid_id].append((phase, records))

    shuffler = random.SystemRandom()
    calls, truth = {}, {}
    for i, grid_id in enumerate(sorted(by_grid)):
        entries = by_grid[grid_id]
        shuffler.shuffle(entries)
        payload, labels = {}, {}
        for phase, records in entries:
            key = secrets.token_hex(6)
            while key in payload:
                key = secrets.token_hex(6)
            payload[key] = records
            labels[key] = phase
        name = f"{i:04d}"
        calls[name] = [payload]
        truth[name] = {"grid_id": grid_id, "labels": labels}
    return PreparedInputs(calls=calls, truth=truth)


def score_predictions(outputs: dict, truth: dict) -> dict:
    """Accuracy over every test series of every grid. A series counts as wrong
    when its key is missing from the prediction, its value isn't R/S/T, or the
    call for its grid raised."""
    confusion = {t: {p: 0 for p in (*PHASES, "other")} for t in PHASES}
    per_grid = []
    total_correct = total_series = 0
    runtime = 0.0

    for name in sorted(truth):
        labels = truth[name]["labels"]
        out = outputs.get(name, {})
        runtime += out.get("seconds") or 0.0
        row = {
            "grid_id": truth[name]["grid_id"],
            "series": len(labels),
            "correct": 0,
            "missing": 0,
            "invalid": 0,
            "unknown_keys": 0,
            "seconds": out.get("seconds"),
            "error": None,
        }

        prediction = out.get("result") if out.get("ok") else None
        if not out.get("ok"):
            lines = (out.get("error") or "predict() failed").strip().splitlines()
            row["error"] = lines[-1] if lines else "predict() failed"
        elif not isinstance(prediction, dict):
            row["error"] = f"predict() must return a dict, got {type(prediction).__name__}"
            prediction = None

        for key, true_phase in labels.items():
            guess = prediction.get(key) if prediction is not None else None
            if prediction is not None and key not in prediction:
                row["missing"] += 1
            elif prediction is not None and guess not in PHASES:
                row["invalid"] += 1
            confusion[true_phase][guess if guess in PHASES else "other"] += 1
            if guess == true_phase:
                row["correct"] += 1
        if prediction is not None:
            row["unknown_keys"] = len(set(prediction) - set(labels))

        row["accuracy"] = round(row["correct"] / row["series"], 4) if row["series"] else 0.0
        per_grid.append(row)
        total_correct += row["correct"]
        total_series += row["series"]

    return {
        "accuracy": round(total_correct / total_series, 4) if total_series else 0.0,
        "correct": total_correct,
        "total": total_series,
        "grids": len(per_grid),
        "failed_grids": sum(1 for r in per_grid if r["error"]),
        "runtime_seconds": round(runtime, 3),
        "confusion": confusion,
        "per_grid": per_grid,
    }


PHASE_TASK = BenchmarkTask(
    name="phase_identification",
    function="predict",
    build_inputs=lambda settings: build_test_inputs(settings["difficulty"]),
    score=score_predictions,
)
