# Phase Identification Benchmark

This benchmark tests a participant's ability to infer which grid phase (`R`,
`S`, or `T`) each meter/node is connected to, from historical power and
voltage measurements. Training data is downloadable, with the true `phase`
label included (`GET /training-phase-data`). The test data is not: instead
the participant downloads a submission template (`GET /submission-template`),
fills in its `predict` function with their method, and uploads it
(`POST /submissions`). GridArena runs the script on the hidden test grids in a
sandbox and scores the phases it returns (`GET /submissions/{submission_id}`).

**Base path:** `/phase_identification_benchmark`

Also serves a browser UI at `/phase_identification_benchmark/ui`.

## Endpoints

### `GET /phase_identification_benchmark/training-phase-data`

Returns historical measurements for every grid marked for phase-detection
training (`GridUsage.phase_detection_train = 1`), keyed by real `grid_id` →
`node_id`. Each record includes the true `phase` label, since this data is
meant for training, not for scoring. Noise is applied per the requested
`difficulty` and is deterministic per `(grid_id, node_id, phase, difficulty)`.

**Query parameters**

| Name | Type | Required | Default | Description |
|---|---|---|---|---|
| `difficulty` | string | no | `clean` | One of `clean`, `easy`, `medium`, `hard` — see [difficulty/noise profiles](index.md#difficulty-and-noise-profiles). |

**Response** `200`

```json
{
  "difficulty": "medium",
  "profile": {
    "pq_rel_sigma": 0.03,
    "v_mag_rel_sigma": 0.005,
    "v_ang_rel_sigma": 0.06,
    "outlier_prob": 0.008,
    "outlier_scale": 12.0
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
  }
}
```

**Errors**

| Status | Cause |
|---|---|
| `500` | No grid is marked `phase_detection_train = 1`, or no measurements exist for the training grids, or another database error. Both "no data" cases raise a generic `Exception` internally that the router wraps as `500` rather than `404`. |

---

### `GET /phase_identification_benchmark/submission-template`

Downloads `phase_identification_submission.py`: the blank submission script.
It documents the contract below, has a random baseline in `predict` so it runs
as-is, and can be run locally on a training download
(`python phase_identification_submission.py training_phase_data_clean.json`)
to score it the same way the test grids are scored.

**The contract.** The script defines `predict(measurements)`. It is called
once per test grid with every measurement series of that grid:

```python
measurements = {
    "3f9c1a2b7d4e": [                 # random key, one per (node, phase) series
        {
            "datetime": "2025-01-01T00:00:00",
            "power_active": 1.243,    # float, or None if missing
            "power_reactive": 0.081,
            "voltage_magnitude": 229.7,
            "voltage_angle": -0.42,
        },
        ...                           # sorted by datetime
    ],
    ...
}
```

and returns `{series_key: "R" | "S" | "T"}` for every key. The keys are drawn
at random for every submission and the series are shuffled, so neither carries
information across submissions. Noise is applied per the submission's
`difficulty`, seeded per `(grid_id, node_id, phase, difficulty)`, so every
submission at a given difficulty sees the same values.

**Runtime.** Code at module level runs once, before the first `predict` call
(load model weights there). The script can import what is installed on the
server (numpy, pandas, scipy, networkx, torch, ...; see `requirements.txt`)
and has no network access. The whole run must finish within
`GRIDARENA_SUBMISSION_TIMEOUT` seconds (default 900).

---

### `POST /phase_identification_benchmark/submissions`

Uploads a script for evaluation (`multipart/form-data`). The upload is checked
for syntax errors and a top-level `predict` before it is queued; evaluation
then runs in the background.

**Form fields**

| Name | Type | Required | Default | Description |
|---|---|---|---|---|
| `file` | file | yes | — | A `.py` file, or a `.zip` with `submission.py` at its root (or inside a single top-level folder) next to its weights / helper modules. Max 200 MB. |
| `submission_name` | string | yes | — | Label for this attempt, e.g. `alice_v1`: 1–64 letters, digits, `_`, `-` or `.`. |
| `difficulty` | string | no | `clean` | Noise level of the test data — one of `clean`, `easy`, `medium`, `hard` (see [difficulty/noise profiles](index.md#difficulty-and-noise-profiles)). |

```bash
curl -X POST http://localhost:8000/phase_identification_benchmark/submissions \
     -F file=@phase_identification_submission.py \
     -F submission_name=alice_v1 -F difficulty=medium
```

**Response** `202`

```json
{
  "submission_id": "3f9c1a2b7d4e",
  "benchmark": "phase_identification",
  "submission_name": "alice_v1",
  "difficulty": "medium",
  "filename": "phase_identification_submission.py",
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
| `422` | Not a `.py`/`.zip`; a `.zip` without `submission.py`, with unsafe paths or symbolic links, or larger than 1 GB unpacked; a syntax error; no top-level `predict` (or an `async def` one); or an invalid `submission_name` / `difficulty`. |

---

### `GET /phase_identification_benchmark/submissions/{submission_id}`

Returns the submission record with its current `status`: `queued`,
`running`, `completed` or `failed`. Poll it until it is `completed` or
`failed`.

**Response** `200` (completed; other fields as for `POST /submissions`)

```json
{
  "submission_id": "3f9c1a2b7d4e",
  "status": "completed",
  "score": {
    "accuracy": 0.8333,
    "correct": 10,
    "total": 12,
    "grids": 1,
    "failed_grids": 0,
    "runtime_seconds": 0.41,
    "confusion": {
      "R": {"R": 4, "S": 0, "T": 0, "other": 0},
      "S": {"R": 1, "S": 3, "T": 0, "other": 0},
      "T": {"R": 0, "S": 1, "T": 3, "other": 0}
    },
    "per_grid": [
      {
        "grid_id": "DEMO_LV_TEST",
        "series": 12,
        "correct": 10,
        "accuracy": 0.8333,
        "missing": 0,
        "invalid": 0,
        "unknown_keys": 0,
        "seconds": 0.41,
        "error": null
      }
    ]
  },
  "error": null,
  "log_tail": "[harness] input 0000: ok in 0.41 s\n"
}
```

- `accuracy` = `correct / total` over every series of every test grid. A key
  missing from the returned dict (`missing`) or with a value other than
  `R`/`S`/`T` (`invalid`) counts as wrong; keys that are not in the input
  (`unknown_keys`) are ignored.
- When `predict` raises for a grid, that grid's `error` holds the exception
  and all its series count as wrong; the other grids are still scored.
- `confusion` counts true phase (rows) against predicted phase; `other` is a
  missing or invalid prediction.
- `log_tail` is the last 8000 characters of what the script printed, plus
  tracebacks.

A submission is `failed` (with `error` set and `score` null) when the script
cannot be loaded (e.g. an import error), the run exceeds the time limit, or no
grid is marked `phase_detection_test = 1`.

**Errors**

| Status | Cause |
|---|---|
| `404` | No phase identification submission with that id. |

---

### `GET /phase_identification_benchmark/submissions`

Lists submission records (without logs), newest first.

**Query parameters**

| Name | Type | Required | Default | Description |
|---|---|---|---|---|
| `submission_name` | string | no | — | Only submissions with this name. |

**Response** `200`

```json
{"submissions": [{"submission_id": "3f9c1a2b7d4e", "status": "completed", "score": {"accuracy": 0.8333}}]}
```

---

See [Code submissions](index.md#code-submissions) for how runs are sandboxed and their limits.
