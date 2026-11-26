# Voltage Control Benchmark

This benchmark asks a participant to correct voltage violations — over- or
under-voltage snapshots — by adjusting node voltages and loads. Training data
is downloadable, including the reference solutions and precomputed metrics
(`GET /training-scenarios`). The test snapshots are not: instead the
participant downloads a submission template (`GET /submission-template`),
fills in its `predict` function with their controller, and uploads it
(`POST /submissions`). GridArena runs the script on the hidden test snapshots
in a sandbox and scores its corrective actions
(`GET /submissions/{submission_id}`).

**Base path:** `/voltage_control_benchmark`

Also serves a browser UI at `/voltage_control_benchmark/ui`.

## Endpoints

### `GET /voltage_control_benchmark/training-scenarios`

Returns training snapshots — `(grid_id, datetime)` combinations with at
least one voltage violation, for grids marked
`GridUsage.voltage_control_train = 1` — together with their reference
`VoltageControlSolutions` (if any) and precomputed scoring metrics. Real
`grid_id`, `node_id`, and `datetime` are all exposed; nothing is hidden in
this endpoint, since it's training data.

`difficulty` here works differently from the other three benchmarks: it
doesn't corrupt values, it filters *which* snapshots come back. Each
snapshot gets a `hardness_score` — for snapshots with a reference solution,
`0.45 × severity_norm + 0.35 × effort_norm + 0.10 × residual_rate + 0.10 ×
grid_complexity_norm`; for snapshots without one, `0.55 × severity_norm +
0.30 × violation_rate + 0.15 × grid_complexity_norm`. Snapshots are then
bucketed by tercile of `hardness_score` among the candidate population:
`easy` = bottom 33%, `medium` = middle third, `hard` = top 33% (`clean` =
every matching snapshot, unfiltered). Snapshots that have a reference
solution are preferred as the population the terciles are computed over; if
none of the matching snapshots have one, all of them are used instead
(surfaced as `"preference": "no_solutions_found_fallback_all"`).

**Query parameters**

| Name | Type | Required | Default | Description |
|---|---|---|---|---|
| `voltage_limit` | float | yes | — | Threshold voltage in volts. |
| `Scenario` | string | yes | — | `Overvoltages` (snapshots where `voltage_magnitude > voltage_limit`) or `Undervoltages` (`voltage_magnitude < voltage_limit`). |
| `difficulty` | string | no | `clean` | `clean`, `easy`, `medium`, or `hard` — hardness-tercile filter, see above. |

**Response** `200`

```json
{
  "difficulty": "hard",
  "difficulty_profile": {
    "method": "quantiles_33_66",
    "t1": 0.081,
    "t2": 0.203,
    "preference": "solutions_only",
    "preferred_population": 12
  },
  "scenario": "Overvoltages",
  "voltage_limit": 253.0,
  "grids": {
    "grid001": {
      "2024-01-15T13:00:00": {
        "measurements": [
          {
            "node_id": "N1",
            "phase": "R",
            "datetime": "2024-01-15T13:00:00",
            "power_active": 1.243,
            "power_reactive": 0.081,
            "voltage_magnitude": 255.4,
            "voltage_angle": -0.42
          }
        ],
        "solutions": [
          {
            "node_id": "N1",
            "phase": "R",
            "corrected_voltage": 231.8,
            "adjusted_power_active": 0.912,
            "adjusted_power_reactive": 0.075,
            "created_at": "2024-01-10T09:00:00"
          }
        ],
        "metrics": {
          "n_nodes": 1,
          "n_violations": 1,
          "violation_rate": 1.0,
          "sum_deviation_volts": 2.4,
          "max_deviation_volts": 2.4,
          "severity_norm": 0.0104,
          "effort_norm": 0.2664,
          "residual_violations": 0,
          "residual_rate": 0.0,
          "grid_complexity_norm": 0.42,
          "grid_complexity_profile": {
            "grid_n_nodes": 20,
            "grid_n_connections": 19,
            "grid_edge_density": 1.0,
            "grid_complexity_raw": 0.7,
            "grid_complexity_norm": 0.42
          },
          "hardness_score": 0.211
        }
      }
    }
  }
}
```

**Errors**

| Status | Cause |
|---|---|
| `500` | No snapshot found above/below `voltage_limit` for the training grids, or another database error — the "no snapshots" case raises a generic `Exception` internally that the router wraps as `500`. |

---

### `GET /voltage_control_benchmark/submission-template`

Downloads `voltage_control_submission.py`: the blank submission script. It documents the
contract below, has a simple baseline in `predict` so it runs as-is, and can
be run locally on a training download (`python voltage_control_submission.py <training file>.json`)
to check it the way the test data is scored.

**The contract.** The script defines `predict(snapshot)`. It is called once
per test snapshot: a grid with `GridUsage.voltage_control_test = 1` at a
timestamp where at least one measurement is above (`Overvoltages`) or below
(`Undervoltages`) `voltage_limit`. Snapshots with reference solutions are
preferred and then filtered by hardness for the `difficulty`, exactly as the
old `/voltage-control-data` endpoint selected them.

```python
snapshot = {
    "datetime": "2025-07-01T13:00:00",
    "scenario": "Overvoltages",
    "voltage_limit": 241.5,
    "nominal_voltage": 230.0,
    "voltage_band": 0.05,
    "measurements": [
        {"node_id": "NODE0001", "phase": "R", "power_active": 3.2, "power_reactive": 0.4,
         "voltage_magnitude": 247.9, "voltage_angle": -0.6},
        ...
    ],
}
```

and returns its corrective actions, in the format of the training data's
reference `solutions`:

```python
[{"node_id": "NODE0001", "phase": "R", "corrected_voltage": 241.0,
  "adjusted_power_active": 2.1, "adjusted_power_reactive": 0.4}, ...]
```

`adjusted_power_*` are optional (unchanged when omitted).

**Runtime.** Code at module level runs once, before the first `predict` call
(load model weights there). The script can import what is installed on the
server (numpy, pandas, scipy, networkx, torch, ...; see `requirements.txt`)
and has no network access. See [Code submissions](index.md#code-submissions)
for how runs are sandboxed and their limits.

---

### `POST /voltage_control_benchmark/submissions`

Uploads a script for evaluation (`multipart/form-data`). The upload is checked
for syntax errors and a top-level `predict` before it is queued; evaluation
then runs in the background.

**Form fields**

| Name | Type | Required | Default | Description |
|---|---|---|---|---|
| `file` | file | yes | — | A `.py` file, or a `.zip` with `submission.py` at its root (or inside a single top-level folder) next to its weights / helper modules. Max 200 MB. |
| `submission_name` | string | yes | — | Label for this attempt, e.g. `alice_v1`: 1–64 letters, digits, `_`, `-` or `.`. |
| `voltage_limit` | number | yes | — | Voltage threshold (V) that selects the snapshots with violations. |
| `Scenario` | string | no | `Overvoltages` | `Overvoltages` or `Undervoltages`. |
| `difficulty` | string | no | `clean` | Filters the snapshots by hardness — one of `clean` (all), `easy`, `medium`, `hard` (hardness terciles). |

```bash
curl -X POST http://localhost:8000/voltage_control_benchmark/submissions \
     -F file=@voltage_control_submission.py \
     -F submission_name=alice_v1 \
     -F voltage_limit=241.5 -F Scenario=Overvoltages -F difficulty=clean
```

**Response** `202`: the submission record.

```json
{
  "submission_id": "3f9c1a2b7d4e",
  "benchmark": "voltage_control",
  "submission_name": "alice_v1",
  "settings": {"voltage_limit": 241.5, "scenario": "Overvoltages", "difficulty": "clean"},
  "filename": "voltage_control_submission.py",
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

### `GET /voltage_control_benchmark/submissions/{submission_id}`

Returns the submission record with its current `status`: `queued`,
`running`, `completed` or `failed`, plus `log_tail` (the last 8000
characters of what the script printed, and tracebacks). Poll it until it is
`completed` or `failed`.

**`score`** (once completed)

```json
{
  "score": 0.7134,
  "snapshots": 24, "nodes": 960, "failed_snapshots": 0, "grids": 2,
  "runtime_seconds": 0.31,
  "per_grid": [
    {"grid_id": "DEMO_LV_TEST", "snapshots": 12, "nodes": 480, "score": 0.69,
     "failed_snapshots": 0, "seconds": 0.15, "error": null}
  ],
  "per_snapshot": [
    {"grid_id": "DEMO_LV_TEST", "datetime": "2025-07-01T13:00:00", "nodes": 40,
     "changed_nodes": 3, "invalid": 0, "mode": "ref_voltage_then_effort",
     "timestamp_score": 0.8571, "effort_norm": 0.025, "reference_voltage_mae": 0.42,
     "seconds": 0.01, "error": null}
  ]
}
```

Each snapshot is scored with the old `GET /score` formula:

- **With reference solutions:** if the mean absolute error between the returned
  and the reference `corrected_voltage` is within 1 V, the snapshot score is the
  effort factor `1 / (1 + effort / 0.15)`, where effort = Σ|ΔP| / Σ|P| (mode
  `ref_voltage_then_effort`); otherwise it is the effort factor divided by
  `1 + MAE²` (mode `ref_voltage_miss_penalised`).
- **Without:** the mean per-node outcome score (0.7 × voltage moved towards
  nominal + 0.2 × power kept + 0.1 × violation fixed) times the effort factor
  (mode `outcome_only_fallback`).

`score` is the node-weighted mean of the snapshot scores (0–1, higher is
better). Two changes from the old endpoint stop a script from scoring better
by answering less: every measured (node, phase) of a snapshot is scored — one
the script doesn't return counts as unchanged — and every snapshot counts,
including those the script returns nothing for. Actions are matched on
(`node_id`, `phase`); an entry without a numeric `corrected_voltage` is counted
in `invalid` and ignored. Like before, `corrected_voltage` is taken as reported:
it is not checked with a power flow.

A submission is `failed` (with `error` set and `score` null) when the script
cannot be loaded (e.g. an import error), the run exceeds the time limit, or
no test snapshot matches the settings.

**Errors**

| Status | Cause |
|---|---|
| `404` | No submission of this benchmark with that id. |

---

### `GET /voltage_control_benchmark/submissions`

Lists this benchmark's submission records (without logs), newest first.

| Name | Type | Required | Default | Description |
|---|---|---|---|---|
| `submission_name` | string | no | — | Only submissions with this name. |

**Response** `200`: `{"submissions": [ ...records... ]}`
