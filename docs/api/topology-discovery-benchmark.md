# Topology Discovery Benchmark

This benchmark tests a participant's ability to infer a grid's electrical
topology (its edge list) from historical per-node power and voltage
measurements. Training data is downloadable, with the real topology included
(`GET /training-topology-data`). The test grids are not: instead the
participant downloads a submission template (`GET /submission-template`),
fills in its `predict` function with their method, and uploads it
(`POST /submissions`). GridArena runs the script on the hidden test grids in a
sandbox and scores the edges it returns (`GET /submissions/{submission_id}`).

**Base path:** `/topology_discovery_benchmark`

Also serves a browser UI at `/topology_discovery_benchmark/ui`.

## Endpoints

### `GET /topology_discovery_benchmark/training-topology-data`

Returns historical measurements plus the real topology for every grid marked
`GridUsage.topology_detection_train = 1`, keyed by real `grid_id`. Each
node's index in `"topology (index)"` and `"conn_data"` refers to its position
in `"node to corresponding index"` for that grid; node `PT` (the
transformer/root node), if present, is always index `0`.

**Query parameters**

| Name | Type | Required | Default | Description |
|---|---|---|---|---|
| `difficulty` | string | no | `clean` | One of `clean`, `easy`, `medium`, `hard` — see [difficulty/noise profiles](index.md#difficulty-and-noise-profiles). |

**Response** `200`

```json
{
  "difficulty": "clean",
  "profile": {
    "pq_rel_sigma": 0.0,
    "v_mag_rel_sigma": 0.0,
    "v_ang_rel_sigma": 0.0,
    "outlier_prob": 0.0,
    "outlier_scale": 0.0
  },
  "grids": {
    "grid001": {
      "N1": [
        {
          "datetime": "2024-01-15T13:00:00",
          "power_active": 1.243,
          "power_reactive": 0.081,
          "voltage_magnitude": 229.7,
          "voltage_angle": -0.42,
          "phase": "R"
        }
      ]
    }
  },
  "node to corresponding index": {
    "grid001": { "PT": 0, "N1": 1, "N2": 2 }
  },
  "topology (index)": {
    "grid001": [[0, 1], [1, 2]]
  },
  "conn_data": {
    "grid001": [["CBL1", 100.0], ["CBL1", 120.0]]
  }
}
```

**Errors**

| Status | Cause |
|---|---|
| `500` | No measurements found for the training grids, or no nodes/connections found for one of them, or another database error — all surfaced as `500` because the underlying helpers raise a generic `Exception` rather than an `HTTPException`. |

---

### `GET /topology_discovery_benchmark/submission-template`

Downloads `topology_discovery_submission.py`: the blank submission script. It documents the
contract below, has a simple baseline in `predict` so it runs as-is, and can
be run locally on a training download (`python topology_discovery_submission.py <training file>.json`)
to check it the way the test data is scored.

**The contract.** The script defines `predict(grid)`. It is called once per
test grid (every grid with `GridUsage.topology_detection_test = 1`):

```python
grid = {
    "nodes": ["3f9c1a2b7d4e", "a07be3c95f11", ...],  # every node, also unmetered ones
    "root": "a07be3c95f11",                          # the substation ("PT"), or None
    "measurements": {                                # metered nodes only
        "3f9c1a2b7d4e": [
            {"datetime": "2025-01-01T00:00:00", "phase": "R",
             "power_active": 1.243, "power_reactive": 0.081,
             "voltage_magnitude": 229.7, "voltage_angle": -0.42},
            ...
        ],
        ...
    },
}
```

and returns the grid's edges as `[[node_key, node_key], ...]` (undirected).
Node keys are random per submission and the node list is shuffled, so they
carry no hint of the wiring. Noise is applied per the submission's
`difficulty`, seeded per `(grid_id, node_id, phase, difficulty)`.

**Runtime.** Code at module level runs once, before the first `predict` call
(load model weights there). The script can import what is installed on the
server (numpy, pandas, scipy, networkx, torch, ...; see `requirements.txt`)
and has no network access. See [Code submissions](index.md#code-submissions)
for how runs are sandboxed and their limits.

---

### `POST /topology_discovery_benchmark/submissions`

Uploads a script for evaluation (`multipart/form-data`). The upload is checked
for syntax errors and a top-level `predict` before it is queued; evaluation
then runs in the background.

**Form fields**

| Name | Type | Required | Default | Description |
|---|---|---|---|---|
| `file` | file | yes | — | A `.py` file, or a `.zip` with `submission.py` at its root (or inside a single top-level folder) next to its weights / helper modules. Max 200 MB. |
| `submission_name` | string | yes | — | Label for this attempt, e.g. `alice_v1`: 1–64 letters, digits, `_`, `-` or `.`. |
| `difficulty` | string | no | `clean` | Noise level of the test measurements — one of `clean`, `easy`, `medium`, `hard` (see [difficulty/noise profiles](index.md#difficulty-and-noise-profiles)). |

```bash
curl -X POST http://localhost:8000/topology_discovery_benchmark/submissions \
     -F file=@topology_discovery_submission.py \
     -F submission_name=alice_v1 -F difficulty=medium
```

**Response** `202`: the submission record.

```json
{
  "submission_id": "3f9c1a2b7d4e",
  "benchmark": "topology_discovery",
  "submission_name": "alice_v1",
  "settings": {"difficulty": "medium"},
  "filename": "topology_discovery_submission.py",
  "status": "queued",
  "created_at": "2026-10-25T13:31:32.847502+00:00",
  "started_at": null,
  "completed_at": null,
  "score": null,
  "error": null
}
```

**Errors**

| Status | Cause |
|---|---|
| `413` | The upload is larger than 200 MB. |
| `422` | Not a `.py`/`.zip`; a `.zip` without `submission.py`, with unsafe paths or symbolic links, or larger than 1 GB unpacked; a syntax error; no top-level `predict` (or an `async def` one); or an invalid `submission_name` or setting. |

---

### `GET /topology_discovery_benchmark/submissions/{submission_id}`

Returns the submission record with its current `status`: `queued`,
`running`, `completed` or `failed`, plus `log_tail` (the last 8000
characters of what the script printed, and tracebacks). Poll it until it is
`completed` or `failed`.

**`score`** (once completed)

```json
{
  "f1": 0.8116,
  "precision": 0.8434,
  "recall": 0.7821,
  "correct": 140, "extra": 26, "missed": 39,
  "grids": 3,
  "failed_grids": 0,
  "runtime_seconds": 1.42,
  "per_grid": [
    {"grid_id": "DEMO_LV_TEST", "nodes": 61, "true_edges": 60, "predicted_edges": 58,
     "correct": 49, "extra": 9, "missed": 11, "invalid": 0,
     "precision": 0.8448, "recall": 0.8167, "f1": 0.8305, "seconds": 0.47, "error": null}
  ]
}
```

- Precision = correct / predicted edges, recall = correct / true edges, both
  over all test grids together; `f1` is the headline metric. (The old `GET /score`
  reported only recall as "accuracy", which returning every possible pair maxed out.)
- Edges are compared undirected and de-duplicated. A pair with an unknown key,
  the same key twice, or not of the form `[key, key]` is counted in `invalid`
  and ignored.
- When `predict` raises for a grid, that grid's `error` holds the exception and
  it predicts no edges; the other grids are still scored.

A submission is `failed` (with `error` set and `score` null) when the script
cannot be loaded (e.g. an import error), the run exceeds the time limit, or
no grid is marked `topology_detection_test = 1`.

**Errors**

| Status | Cause |
|---|---|
| `404` | No submission of this benchmark with that id. |

---

### `GET /topology_discovery_benchmark/submissions`

Lists this benchmark's submission records (without logs), newest first.

| Name | Type | Required | Default | Description |
|---|---|---|---|---|
| `submission_name` | string | no | — | Only submissions with this name. |

**Response** `200`: `{"submissions": [ ...records... ]}`
