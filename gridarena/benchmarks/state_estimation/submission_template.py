"""GridArena -- State Estimation submission.

Fill in ``predict`` with your estimator and upload this file on the State
Estimation page (or POST it to /state_estimation_benchmark/submissions).
GridArena runs it on hidden estimation tasks and scores the voltages it
returns.

THE CONTRACT
------------
``predict`` is called once per estimation task: one test grid at one
timestamp, where some measurement series (one per node and phase) are known
at that timestamp and the others must be estimated::

    task = {
        "timestamp": "2025-01-01T12:00:00",
        "known": {                          # series measured at the timestamp
            "3f9c1a2b7d": {"power_active": 1.243, "power_reactive": 0.081,
                           "voltage_magnitude": 229.7, "voltage_angle": -0.42},
            ...
        },
        "unknown": ["a07be3c95f", ...],     # series to estimate
        "history": {                        # every series, known and unknown
            "3f9c1a2b7d": [
                {"datetime": "2025-01-01T11:45:00", "power_active": 1.198,
                 "power_reactive": 0.079, "voltage_magnitude": 229.8,
                 "voltage_angle": -0.40},
                ...                         # sorted by datetime, before the
            ],                              # timestamp (unknown series stop
            ...                             # a few hours earlier)
        },
    }

and must return an estimate for every unknown series::

    {"a07be3c95f": {"voltage_magnitude": 228.9, "voltage_angle": -121.3}, ...}

Values may be None in the inputs (missing measurements). Keys are random per
task. A missing or non-numeric estimate is scored as 0 V with the worst
angle, so always return one. The score is mean (dV)^2 + mean |sin(dθ)| over
all unknown series of all tasks: lower is better.

The records have the same fields as the training data (GET
/training-state-data). Noise follows the noise difficulty you submit with;
how many series are unknown, and how much of their history is hidden,
follows the observability.

RUNTIME
-------
* The server's Python environment: the standard library plus numpy, pandas,
  scipy, networkx, torch, ... (see requirements.txt). Nothing can be installed.
* No network access. Everything the model needs must ship with the submission.
* Code at module level runs once, before the first ``predict`` call: load your
  model weights there.
* To ship weights or helper modules, upload a .zip with this file renamed to
  ``submission.py`` at its root next to them. Paths are relative to that folder.
* The whole run (all tasks) must finish within the server's time limit
  (15 minutes by default). Anything you print ends up in the submission log.

Run this file directly to check it against training data you downloaded::

    python state_estimation_submission.py training_state_data_clean.json
"""

import numpy as np

# ── Load your model here (runs once) ────────────────────────────────────────
# Example for a PyTorch model shipped in the .zip next to this file:
#
#   from pathlib import Path
#   import torch
#   MODEL = torch.jit.load(Path(__file__).parent / "model.pt")
#   MODEL.eval()


def _mean(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else 0.0


def predict(task):
    """Return {series_key: {"voltage_magnitude": ..., "voltage_angle": ...}} for every unknown series."""
    # TODO: replace this baseline with your estimator. It repeats each series'
    # last historical value, or the average of the known series when the
    # series has no history.
    known = task["known"].values()
    fallback_v = _mean(m["voltage_magnitude"] for m in known)
    fallback_a = _mean(m["voltage_angle"] for m in known)

    estimates = {}
    for key in task["unknown"]:
        past = [r for r in task["history"].get(key, []) if r["voltage_magnitude"] is not None]
        last = past[-1] if past else None
        estimates[key] = {
            "voltage_magnitude": last["voltage_magnitude"] if last else fallback_v,
            "voltage_angle": last["voltage_angle"] if last and last["voltage_angle"] is not None else fallback_a,
        }
    return estimates


# ── Local check (not run by GridArena) ──────────────────────────────────────

def _local_check(training_file, tasks_per_grid=3, hidden_fraction=0.5, unknown_history_hours=6):
    """Score ``predict`` on a training-data download, on tasks built the way
    GridArena builds the test tasks (roughly: medium observability)."""
    import json
    import math
    import random
    from datetime import datetime, timedelta

    with open(training_file, encoding="utf-8") as f:
        grids = json.load(f)["grids"]

    rng = random.Random(0)
    sq, sin = [], []
    for grid_id, series in grids.items():
        by_time = {key: {r["datetime"]: r for r in records} for key, records in series.items()}
        timestamps = sorted({t for records in by_time.values() for t in records})
        for ts in rng.sample(timestamps[len(timestamps) // 2:], min(tasks_per_grid, len(timestamps) // 2)):
            present = sorted(k for k, records in by_time.items() if ts in records)
            unknown = set(rng.sample(present, max(1, int(len(present) * hidden_fraction)))) if len(present) > 1 else set()
            cutoff = datetime.fromisoformat(ts) - timedelta(hours=unknown_history_hours)
            fields = ("power_active", "power_reactive", "voltage_magnitude", "voltage_angle")

            def strip(r):
                return {k: v for k, v in r.items() if k != "phase"}

            task = {
                "timestamp": ts,
                "known": {k: {f: by_time[k][ts][f] for f in fields} for k in present if k not in unknown},
                "unknown": sorted(unknown),
                "history": {
                    k: [strip(r) for r in series[k]
                        if r["datetime"] < ts and (k not in unknown or datetime.fromisoformat(r["datetime"]) < cutoff)]
                    for k in present
                },
            }
            estimates = predict(task)
            for k in unknown:
                true = by_time[k][ts]
                est = estimates.get(k) or {"voltage_magnitude": 0.0, "voltage_angle": true["voltage_angle"] + 90}
                sq.append((est["voltage_magnitude"] - true["voltage_magnitude"]) ** 2)
                sin.append(abs(math.sin(math.radians(est["voltage_angle"] - true["voltage_angle"]))))
        print(f"{grid_id}: done")

    if sq:
        print(f"score {np.mean(sq) + np.mean(sin):.6f} (lower is better)   RMSE |V| {math.sqrt(np.mean(sq)):.4f}")
    else:
        print("no tasks could be built")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        sys.exit(f"usage: python {sys.argv[0]} <training_state_data.json>")
    _local_check(sys.argv[1])
