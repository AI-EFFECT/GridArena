# State Estimation Benchmark

This benchmark asks a participant to estimate the voltage magnitude and
angle of "unknown" series at a grid snapshot, given the current measurements
of the "known" series plus their history. Training data is downloadable, with
every series fully measured under masked grid/node identifiers
(`GET /training-state-data`). The test tasks are not: instead the participant
downloads a submission template (`GET /submission-template`), fills in its
`predict` function with their estimator, and uploads it (`POST /submissions`).
GridArena runs the script on hidden estimation tasks in a sandbox and scores
the voltages it returns (`GET /submissions/{submission_id}`).

**Base path:** `/state_estimation_benchmark`

Also serves a browser UI at `/state_estimation_benchmark/ui`.

## Endpoints

### `GET /state_estimation_benchmark/training-state-data`

Returns every measurement for every grid marked
`GridUsage.state_estimation_train = 1`, grouped by masked grid ID → masked
node ID, with noise applied per `noise_difficulty`. Grid/node masks
(`GridMask`/`NodeMask`) are created on demand the first time a real
grid/node is seen and reused afterwards — see
[anonymisation pattern](index.md#anonymisation-pattern).

**Query parameters**

| Name | Type | Required | Default | Description |
|---|---|---|---|---|
| `noise_difficulty` | string | no | `clean` | One of `clean`, `easy`, `medium`, `hard` — see [difficulty/noise profiles](index.md#difficulty-and-noise-profiles). |
| `observability` | string | no | `medium` | Accepted but unused by this endpoint — `observability` only affects the known/unknown split created by `GET /state-data`. |

**Response** `200`

```json
{
  "noise_difficulty": "clean",
  "noise_profile": {
    "pq_rel_sigma": 0.0,
    "v_mag_rel_sigma": 0.0,
    "v_ang_rel_sigma": 0.0,
    "outlier_prob": 0.0,
    "outlier_scale": 0.0
  },
  "grids": {
    "8f2a1c90b3": {
      "4e7b6a1c2d": [
        {
          "datetime": "2024-01-15T13:00:00",
          "phase": "R",
          "power_active": 1.243,
          "power_reactive": 0.081,
          "voltage_magnitude": 229.7,
          "voltage_angle": -0.42
        }
      ]
    }
  }
}
```

**Errors**

| Status | Cause |
|---|---|
| `500` | Database error. Unlike the other three benchmarks' "no data" endpoints, no grid marked for training doesn't raise an error here — the `grid_id = ANY(%s)` query with an empty list simply matches no rows, so the response is still `200` with `"grids": {}`. |

---

### `GET /state_estimation_benchmark/submission-template`

Downloads `state_estimation_submission.py`: the blank submission script. It documents the
contract below, has a simple baseline in `predict` so it runs as-is, and can
be run locally on a training download (`python state_estimation_submission.py <training file>.json`)
to check it the way the test data is scored.

**The contract.** The script defines `predict(task)`. It is called once per
estimation task: one test grid (`GridUsage.state_estimation_test = 1`) at one
timestamp. Each test grid yields `GRIDARENA_SE_TASKS_PER_GRID` tasks
(default 3) at timestamps from the later half of its data, so there is
history to learn from.

```python
task = {
    "timestamp": "2025-01-01T12:00:00",
    "known": {                                   # series measured at the timestamp
        "3f9c1a2b7d": {"power_active": 1.243, "power_reactive": 0.081,
                       "voltage_magnitude": 229.7, "voltage_angle": -0.42},
        ...
    },
    "unknown": ["a07be3c95f", ...],              # series to estimate
    "history": {                                 # every series, before the timestamp
        "3f9c1a2b7d": [{"datetime": "2025-01-01T11:45:00", "power_active": 1.198,
                        "power_reactive": 0.079, "voltage_magnitude": 229.8,
                        "voltage_angle": -0.40}, ...],
        ...
    },
}
```

and returns `{series_key: {"voltage_magnitude": ..., "voltage_angle": ...}}`
for every unknown key. A series is one (node, phase); keys are random per
task. As in the old interactive flow, the number of unknown series and the
gap before the timestamp where their history stops follow the
`observability` profile (`clean`: none unknown; `easy`: up to 1/3, 1 h;
`medium`: 1/3–2/3, 6 h; `hard`: over 2/3, 12 h). Timestamps and the
known/unknown split are seeded per grid and settings, and noise per series
and task, so every submission with the same settings gets exactly the same
tasks.

**Runtime.** Code at module level runs once, before the first `predict` call
(load model weights there). The script can import what is installed on the
server (numpy, pandas, scipy, networkx, torch, ...; see `requirements.txt`)
and has no network access. See [Code submissions](index.md#code-submissions)
for how runs are sandboxed and their limits.

---

### `POST /state_estimation_benchmark/submissions`

Uploads a script for evaluation (`multipart/form-data`). The upload is checked
for syntax errors and a top-level `predict` before it is queued; evaluation
then runs in the background.

**Form fields**

| Name | Type | Required | Default | Description |
|---|---|---|---|---|
| `file` | file | yes | — | A `.py` file, or a `.zip` with `submission.py` at its root (or inside a single top-level folder) next to its weights / helper modules. Max 200 MB. |
| `submission_name` | string | yes | — | Label for this attempt, e.g. `alice_v1`: 1–64 letters, digits, `_`, `-` or `.`. |
| `noise_difficulty` | string | no | `clean` | Noise level of the test measurements — one of `clean`, `easy`, `medium`, `hard` (see [difficulty/noise profiles](index.md#difficulty-and-noise-profiles)). |
| `observability` | string | no | `medium` | How many series are unknown and how much of their history is hidden — one of `clean`, `easy`, `medium`, `hard`. |

```bash
curl -X POST http://localhost:8000/state_estimation_benchmark/submissions \
     -F file=@state_estimation_submission.py \
     -F submission_name=alice_v1 \
     -F noise_difficulty=easy -F observability=medium
```

**Response** `202`: the submission record.

```json
{
  "submission_id": "3f9c1a2b7d4e",
  "benchmark": "state_estimation",
  "submission_name": "alice_v1",
  "settings": {"noise_difficulty": "easy", "observability": "medium"},
  "filename": "state_estimation_submission.py",
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

### `GET /state_estimation_benchmark/submissions/{submission_id}`

Returns the submission record with its current `status`: `queued`,
`running`, `completed` or `failed`, plus `log_tail` (the last 8000
characters of what the script printed, and tracebacks). Poll it until it is
`completed` or `failed`.

**`score`** (once completed)

```json
{
  "score": 0.418702,
  "rmse_voltage_magnitude": 0.612,
  "mae_voltage_angle": 0.351,
  "estimates": 120, "missing": 0, "invalid": 0,
  "tasks": 9, "failed_tasks": 0, "grids": 3,
  "runtime_seconds": 0.81,
  "per_grid": [
    {"grid_id": "DEMO_LV_TEST", "tasks": 3, "score": 0.37, "rmse_voltage_magnitude": 0.58,
     "mae_voltage_angle": 0.33, "estimates": 40, "missing": 0, "invalid": 0,
     "failed_tasks": 0, "seconds": 0.27, "error": null}
  ]
}
```

- `score` = mean (ΔV)² + mean |sin Δθ| over every unknown series of every
  task — the old `/state-score` formula, lower is better. Angles are compared
  wrapped, in the measurements' unit (`LVGPLAY_ANGLE_UNIT`, degrees by default).
  (The old endpoint paired predictions and truths by row order and took the sine
  of degrees as radians; both are fixed.)
- A missing estimate (`missing`) or one without numeric magnitude and angle
  (`invalid`) is scored as 0 V with the worst angle, so always return one.
- `rmse_voltage_magnitude` and `mae_voltage_angle` are over the usable estimates
  only. `score` is `null` when there was nothing to estimate (`observability = clean`).

A submission is `failed` (with `error` set and `score` null) when the script
cannot be loaded (e.g. an import error), the run exceeds the time limit, or
no grid is marked `state_estimation_test = 1`.

**Errors**

| Status | Cause |
|---|---|
| `404` | No submission of this benchmark with that id. |

---

### `GET /state_estimation_benchmark/submissions`

Lists this benchmark's submission records (without logs), newest first.

| Name | Type | Required | Default | Description |
|---|---|---|---|---|
| `submission_name` | string | no | — | Only submissions with this name. |

**Response** `200`: `{"submissions": [ ...records... ]}`
