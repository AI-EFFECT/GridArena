"""GridArena -- Voltage Control submission.

Fill in ``predict`` with your controller and upload this file on the Voltage
Control page (or POST it to /voltage_control_benchmark/submissions).
GridArena runs it on the hidden test snapshots and scores the corrective
actions it returns.

THE CONTRACT
------------
``predict`` is called once per test snapshot: a grid at a timestamp where at
least one node violates the voltage limit::

    snapshot = {
        "datetime": "2025-07-01T13:00:00",
        "scenario": "Overvoltages",       # or "Undervoltages"
        "voltage_limit": 241.5,           # the limit you submitted with (V)
        "nominal_voltage": 230.0,         # V
        "voltage_band": 0.05,             # allowed band: nominal +/- 5 %
        "measurements": [
            {"node_id": "NODE0001", "phase": "R",
             "power_active": 3.2, "power_reactive": 0.4,
             "voltage_magnitude": 247.9, "voltage_angle": -0.6},
            ...                           # every node and phase of the grid
        ],
    }

and must return its corrective actions, in the same format as the reference
"solutions" of the training data (GET /training-scenarios)::

    [
        {"node_id": "NODE0001", "phase": "R",
         "corrected_voltage": 241.0,          # voltage after your action (V)
         "adjusted_power_active": 2.1,        # optional: new P (unchanged if omitted)
         "adjusted_power_reactive": 0.4},     # optional
        ...
    ]

A (node, phase) you don't return is scored as left unchanged. Per snapshot,
the score rewards matching the reference corrected voltages (within 1 V), and
then the least total change in active power; snapshots without a reference
are scored on how far your voltages move back towards nominal. The overall
score (0-1, higher is better) is the node-weighted mean over all snapshots.

RUNTIME
-------
* The server's Python environment: the standard library plus numpy, pandas,
  scipy, networkx, torch, ... (see requirements.txt). Nothing can be installed.
* No network access. Everything the model needs must ship with the submission.
* Code at module level runs once, before the first ``predict`` call: load your
  model weights there.
* To ship weights or helper modules, upload a .zip with this file renamed to
  ``submission.py`` at its root next to them. Paths are relative to that folder.
* The whole run (all snapshots) must finish within the server's time limit
  (15 minutes by default). Anything you print ends up in the submission log.

Run this file directly to check it against training data you downloaded::

    python voltage_control_submission.py training_voltage_control_data.json
"""

# ── Load your model here (runs once) ────────────────────────────────────────
# Example for a PyTorch model shipped in the .zip next to this file:
#
#   from pathlib import Path
#   import torch
#   MODEL = torch.jit.load(Path(__file__).parent / "model.pt")
#   MODEL.eval()


def predict(snapshot):
    """Return the corrective actions for one snapshot."""
    # TODO: replace this baseline with your controller. It takes no action:
    # every node keeps its measured voltage and power.
    actions = []
    for m in snapshot["measurements"]:
        actions.append({
            "node_id": m["node_id"],
            "phase": m["phase"],
            "corrected_voltage": m["voltage_magnitude"],
            "adjusted_power_active": m["power_active"],
            "adjusted_power_reactive": m["power_reactive"],
        })
    return actions


# ── Local check (not run by GridArena) ──────────────────────────────────────

def _local_check(training_file):
    """Run ``predict`` on a training-scenarios download and compare its
    voltages with the reference solutions (mean absolute error, V)."""
    import json

    with open(training_file, encoding="utf-8") as f:
        data = json.load(f)

    nominal, band = 230.0, 0.05
    low, high = nominal * (1 - band), nominal * (1 + band)
    for grid_id, snapshots in data["grids"].items():
        errors, violations = [], 0
        for dt, snap in snapshots.items():
            measurements = [{k: v for k, v in m.items() if k != "datetime"} for m in snap["measurements"]]
            actions = predict({
                "datetime": dt,
                "scenario": data["scenario"],
                "voltage_limit": data["voltage_limit"],
                "nominal_voltage": nominal,
                "voltage_band": band,
                "measurements": measurements,
            })
            corrected = {(a["node_id"], a["phase"]): a["corrected_voltage"] for a in actions}
            for m in measurements:
                v = corrected.get((m["node_id"], m["phase"]), m["voltage_magnitude"])
                violations += v is not None and not (low <= v <= high)
            for ref in snap["solutions"]:
                v = corrected.get((ref["node_id"], ref["phase"]))
                if v is not None and ref["corrected_voltage"] is not None:
                    errors.append(abs(v - ref["corrected_voltage"]))
        mae = f"{sum(errors) / len(errors):.3f} V" if errors else "n/a (no reference solutions)"
        print(f"{grid_id}: {len(snapshots)} snapshots, voltage MAE vs reference {mae}, "
              f"{violations} node voltages still outside the band")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        sys.exit(f"usage: python {sys.argv[0]} <training_voltage_control_data.json>")
    _local_check(sys.argv[1])
