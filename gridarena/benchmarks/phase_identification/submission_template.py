"""GridArena -- Phase Identification submission.

Fill in ``predict`` with your model and upload this file on the Phase
Identification page (or POST it to /phase_identification_benchmark/submissions).
GridArena runs it on the hidden test grids and scores the phases it returns.

THE CONTRACT
------------
``predict`` is called once per test grid, with every measurement series of
that grid::

    measurements = {
        "3f9c1a2b7d4e": [                       # random key, one per series
            {
                "datetime": "2025-01-01T00:00:00",
                "power_active": 1.243,          # float, or None if missing
                "power_reactive": 0.081,
                "voltage_magnitude": 229.7,
                "voltage_angle": -0.42,
            },
            ...                                 # sorted by datetime
        ],
        ...
    }

and must return the phase of every series::

    {"3f9c1a2b7d4e": "R", "a07be3c95f11": "T", ...}     # values: "R", "S" or "T"

A key that is missing or has another value counts as a wrong prediction. The
records have the same fields as the training data (GET /training-phase-data),
without "phase", and the same noise level as the difficulty you submit with.

RUNTIME
-------
* The server's Python environment: the standard library plus numpy, pandas,
  scipy, networkx, torch, ... (see requirements.txt). Nothing can be installed.
* No network access. Everything the model needs must ship with the submission.
* Code at module level runs once, before the first ``predict`` call: load your
  model weights there.
* To ship weights or helper modules, upload a .zip with this file renamed to
  ``submission.py`` at its root next to them. Paths are relative to that folder.
* The whole run (all grids) must finish within the server's time limit
  (15 minutes by default). Anything you print ends up in the submission log.

Run this file directly to check it against training data you downloaded::

    python phase_identification_submission.py training_phase_data_clean.json
"""

import numpy as np

PHASES = ("R", "S", "T")

# ── Load your model here (runs once) ────────────────────────────────────────
# Example for a PyTorch model shipped in the .zip next to this file:
#
#   from pathlib import Path
#   import torch
#   MODEL = torch.jit.load(Path(__file__).parent / "model.pt")
#   MODEL.eval()

_rng = np.random.default_rng(0)


def predict(measurements):
    """Return {series_key: "R" | "S" | "T"} for every key in ``measurements``."""
    predictions = {}
    for key, records in measurements.items():
        # TODO: replace this random baseline with your model. For example, to
        # get the voltage profile of a series as an array:
        #   v = np.array([r["voltage_magnitude"] for r in records], dtype=float)
        predictions[key] = str(_rng.choice(PHASES))
    return predictions


# ── Local check (not run by GridArena) ──────────────────────────────────────

def _local_check(training_file):
    """Score ``predict`` on a training-data download, the way GridArena scores
    the test grids: one call per grid, phases removed from the input."""
    import json

    with open(training_file, encoding="utf-8") as f:
        grids = json.load(f)["grids"]

    total = correct = 0
    for grid_id, nodes in grids.items():
        # Test series are one (node, phase) pair each; training data groups
        # the records by node only, so split them by phase here.
        inputs, labels = {}, {}
        for node_id, records in nodes.items():
            for record in records:
                key = f"{node_id}:{record['phase']}"
                inputs.setdefault(key, []).append({k: v for k, v in record.items() if k != "phase"})
                labels[key] = record["phase"]

        predictions = predict(inputs)
        hits = sum(predictions.get(key) == phase for key, phase in labels.items())
        print(f"{grid_id}: {hits}/{len(labels)} correct")
        total += len(labels)
        correct += hits

    print(f"accuracy: {correct / total:.2%}" if total else "no series found")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        sys.exit(f"usage: python {sys.argv[0]} <training_phase_data.json>")
    _local_check(sys.argv[1])
