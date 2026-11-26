"""GridArena -- Topology Discovery submission.

Fill in ``predict`` with your method and upload this file on the Topology
Discovery page (or POST it to /topology_discovery_benchmark/submissions).
GridArena runs it on the hidden test grids and scores the edges it returns.

THE CONTRACT
------------
``predict`` is called once per test grid::

    grid = {
        "nodes": ["3f9c1a2b7d4e", "a07be3c95f11", ...],   # every node of the grid,
                                                          # random keys, in no order
        "root": "a07be3c95f11",       # the substation (secondary of the MV/LV
                                      # transformer), or None if the grid has none
        "measurements": {             # only nodes with a meter appear here
            "3f9c1a2b7d4e": [
                {
                    "datetime": "2025-01-01T00:00:00",
                    "phase": "R",
                    "power_active": 1.243,          # float, or None if missing
                    "power_reactive": 0.081,
                    "voltage_magnitude": 229.7,
                    "voltage_angle": -0.42,
                },
                ...                                 # sorted by phase, then datetime
            ],
            ...
        },
    }

and must return the grid's connections as pairs of node keys::

    [["a07be3c95f11", "3f9c1a2b7d4e"], ["3f9c1a2b7d4e", "5d21e0c7b8aa"], ...]

Edges are undirected. A pair with an unknown key or the same key twice is
ignored. The score is the F1 of your edges against the true ones, so both
missing an edge and adding a wrong one cost points. Nodes without
measurements (junctions, the substation) still have to be connected.

The records have the same fields as the training data (GET
/training-topology-data), with the same noise level as the difficulty you
submit with.

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

    python topology_discovery_submission.py training_topology_data_clean.json
"""

import numpy as np

_rng = np.random.default_rng(0)

# ── Load your model here (runs once) ────────────────────────────────────────
# Example for a PyTorch model shipped in the .zip next to this file:
#
#   from pathlib import Path
#   import torch
#   MODEL = torch.jit.load(Path(__file__).parent / "model.pt")
#   MODEL.eval()


def predict(grid):
    """Return the grid's edges as a list of [node_key, node_key] pairs."""
    # TODO: replace this baseline with your method. It builds a random radial
    # (tree) network from the substation: every node hangs off a random node
    # that is already connected. For example, to get a node's voltage profile:
    #   v = np.array([r["voltage_magnitude"] for r in grid["measurements"].get(key, [])], dtype=float)
    nodes = list(grid["nodes"])
    root = grid["root"] if grid["root"] is not None else nodes[0]
    others = [n for n in nodes if n != root]
    _rng.shuffle(others)

    connected = [root]
    edges = []
    for node in others:
        parent = connected[_rng.integers(len(connected))]
        edges.append([parent, node])
        connected.append(node)
    return edges


# ── Local check (not run by GridArena) ──────────────────────────────────────

def _local_check(training_file):
    """Score ``predict`` on a training-data download, the way GridArena scores
    the test grids: one call per grid, F1 over the edges."""
    import json

    with open(training_file, encoding="utf-8") as f:
        data = json.load(f)

    tp_all = fp_all = fn_all = 0
    for grid_id, index_of in data["node to corresponding index"].items():
        node_of = {i: n for n, i in index_of.items()}
        true_edges = {frozenset((node_of[a], node_of[b])) for a, b in data["topology (index)"][grid_id]}
        grid = {
            "nodes": list(index_of),
            "root": "PT" if "PT" in index_of else None,
            "measurements": data["grids"].get(grid_id, {}),
        }

        predicted = {frozenset(e) for e in predict(grid) if len(set(e)) == 2}
        tp = len(predicted & true_edges)
        fp, fn = len(predicted) - tp, len(true_edges) - tp
        print(f"{grid_id}: {tp} of {len(true_edges)} edges found, {fp} wrong")
        tp_all, fp_all, fn_all = tp_all + tp, fp_all + fp, fn_all + fn

    precision = tp_all / (tp_all + fp_all) if tp_all + fp_all else 0.0
    recall = tp_all / (tp_all + fn_all) if tp_all + fn_all else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    print(f"precision {precision:.2%}  recall {recall:.2%}  F1 {f1:.2%}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        sys.exit(f"usage: python {sys.argv[0]} <training_topology_data.json>")
    _local_check(sys.argv[1])
