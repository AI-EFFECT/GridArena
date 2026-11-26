# API Reference — Overview

This section documents the JSON API exposed by each of gridarena's 15 FastAPI
routers. It's a handwritten companion to the auto-generated, always-accurate
[Swagger UI](../getting-started.md#interactive-api-documentation) — use these
pages for the workflow/context and Swagger for exact live schemas.

## Base URL and routing

Each router is mounted under a fixed path prefix in `gridarena/app.py`:

| Prefix | Router | OpenAPI tag |
|---|---|---|
| `/grid` | [Grid](grid.md) | Grid |
| `/mv_grid` | [MV Grid](mv-grid.md) | MV Grid |
| `/measurements` | [Measurements](measurements.md) | Measurements |
| `/historical` | [Historical](historical.md) | Historical |
| `/powerflow` | [Power Flow](powerflow.md) | Algorithm |
| `/phase_identification_benchmark` | [Phase Identification Benchmark](phase-identification-benchmark.md) | Phase Identification Benchmark |
| `/topology_discovery_benchmark` | [Topology Discovery Benchmark](topology-discovery-benchmark.md) | Topology Discovery Benchmark |
| `/state_estimation_benchmark` | [State Estimation Benchmark](state-estimation-benchmark.md) | State Estimation Benchmark |
| `/voltage_control_benchmark` | [Voltage Control Benchmark](voltage-control-benchmark.md) | Voltage Control Benchmark |
| `/digital-twins` | [Digital Twin](digital-twin.md) | Digital Twin |
| `/diffusion` | [Diffusion Models](diffusion.md) | Diffusion Models |
| `/pf-datagen` | [PF Data Generation](pf-datagen.md) | PF Data Generation |
| `/rl` | [Reinforcement Learning](rl.md) | Reinforcement Learning |
| `/chat` | [Chat](chat.md) | Chat |
| `/offline-scenarios` | [Offline Scenarios](offline-scenarios.md) | Offline Scenarios |

Requests and responses are JSON (`application/json`) except file uploads,
which use `multipart/form-data`.

## Units

Every stored or exchanged electrical quantity uses plain SI units, with no
k/M prefixes:

| Quantity | Unit | Sign / convention |
|---|---|---|
| `power_active` | W | Load convention: positive = consumption, negative = generation (e.g. PV). |
| `power_reactive` | VAr | Same load convention. |
| `voltage_magnitude` | V | LV: phase-to-neutral. MV: line-to-line. |
| `voltage_angle` | degrees | — |
| Cable impedance | Ω/km | Per phase. |
| Cable nominal current | A | — |

What a power value covers depends on where it is stored:

- **LV** (measurements, LV-level historical series): a row tagged with a
  `phase` (`R`/`S`/`T`) is that phase's power, and the LV power flow
  solves the three phases against phase-to-neutral references (e.g. 230 V)
  — see [Power Flow](powerflow.md#models).
- **MV** (MV- and Feeder-level historical series, MV connection-point
  injections): power is the **three-phase total**, and the MV nominal
  voltage (`nominal_voltage_kv`) is line-to-line. Where a series holds
  per-phase rows instead, the three phases are summed to get the total (see
  [PF Data Generation](pf-datagen.md#units-and-three-phase-convention)).

The exceptions are a few user-facing inputs that state their own unit,
namely the [RL](rl.md) environment's power range (kW), and display-only
labels such as an MV grid's `nominal_voltage_kv` or a transformer's
`rated_power_kva`.

## Authentication

There is currently **no authentication or authorization** — every endpoint is
open. The `submission_name` of a benchmark submission is a caller-supplied
label, not a verified credential; it groups a participant's attempts, it
doesn't restrict who can see them. Treat any deployment of this API
accordingly (e.g. put it behind a reverse proxy or VPN if it shouldn't be
publicly reachable).

If route-level API-key auth is added later, note that the [Chat](chat.md)
assistant's tool calls go through router functions **directly, in-process**,
never over HTTP — a route-level dependency would not cover that path, and the
WebSocket itself would need its own check (the document-assistant WebSocket
already has a minimal placeholder for this, gated on the `LVGPLAY_API_KEY`
env var; see chat.md).

## Error format

Errors follow FastAPI's standard shape:

```json
{ "detail": "Grid 'grid001' not found." }
```

or, for request-validation failures (HTTP 422, raised by Pydantic), a list of
per-field errors:

```json
{
  "detail": [
    {
      "type": "missing",
      "loc": ["query", "grid_id"],
      "msg": "Field required",
      "input": null
    }
  ]
}
```

| Status | Meaning |
|---|---|
| `200` | Success. |
| `400` | Malformed request (e.g. a required file was not attached). |
| `404` | The referenced resource (grid, run, digital twin, ...) doesn't exist. |
| `413` | An uploaded file exceeded the size limit (see below). |
| `422` | Request validation failed (bad JSON shape, out-of-range value, unknown key). |
| `500` | Unexpected server-side error. The response `detail` is a generic message; full details are logged server-side, not returned to the client. |

## Resource-safety limits

A few platform-wide limits apply across routers to bound worst-case request
cost:

| Limit | Value | Applies to |
|---|---|---|
| Upload size | 200 MB | Every file-upload endpoint (grid, MV grid, measurements, historical, chat custom-agent documents). Exceeding it returns `413`. |
| Row cap per request | 50,000 rows | `GET` endpoints that could otherwise return an entire table (historical records, measurements). Once the cap is hit, the response is truncated and includes `"truncated": true` plus a `"truncation_hint"` telling the caller to narrow the request with `start`/`end`. |
| Code submission | 1 GB unpacked, 2,000 files; 900 s per run | Benchmark script uploads (`.zip` limits) and their evaluation (`GRIDARENA_SUBMISSION_TIMEOUT`). See [Code submissions](#code-submissions). |
| Chat message length | 8,000 characters | `POST` chat messages; longer messages get a friendly rejection instead of being sent to the LLM backend. |

## Difficulty and noise profiles

All four benchmarks accept a platform-controlled `difficulty` query parameter
(`clean`, `easy`, `medium`, `hard`) that determines how much synthetic noise
is applied to measurement values before they're returned to a participant.
The exact profile is echoed back in the response body as `"profile"`, so a
client never has to hardcode these values:

| Difficulty | `pq_rel_sigma` | `v_mag_rel_sigma` | `v_ang_rel_sigma` | `outlier_prob` | `outlier_scale` |
|---|---|---|---|---|---|
| `clean` | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| `easy` | 0.01 | 0.002 | 0.02 | 0.002 | 10.0 |
| `medium` | 0.03 | 0.005 | 0.06 | 0.008 | 12.0 |
| `hard` | 0.07 | 0.010 | 0.15 | 0.020 | 15.0 |

Noise is deterministic and reproducible: it's generated from a `random.Random`
seeded with a SHA-256 hash of `(grid_id, key, difficulty)`, so repeated calls
with the same parameters return identical corrupted values, but different
`difficulty` values produce independent noise draws.

## Code submissions

All four benchmarks work the same way. Training data is downloadable from the
grids marked for training. The test data is not: the participant downloads a
submission template (`GET /<benchmark>/submission-template`), fills in its
`predict` function with any method (from a heuristic to a PyTorch model), and
uploads it (`POST /<benchmark>/submissions`, a `.py` or a `.zip` with
`submission.py` and its weights). GridArena runs the script on the hidden test
data and scores what it returns; the participant polls
`GET /<benchmark>/submissions/{submission_id}` for the status, score and log,
and `GET /<benchmark>/submissions` lists past attempts.

Each submission is stored under `gridarena/benchmarks/submissions/runs/<submission_id>/`
(`run.json`, `log.txt` and the uploaded `code/`). To evaluate it, the server
builds the test inputs from the database, writes them next to the code, and
runs `harness.py` in a sandbox, which imports `submission.py`, calls `predict`
once per input and writes the results back. The ground truth never leaves the
server process, identifiers in the inputs are random per submission, and the
test inputs are deleted once the run is scored. Runs queue on
`GRIDARENA_SUBMISSION_WORKERS` workers (default 1); a run that was queued or
running when the server stopped is marked `failed` on restart.

The sandbox is set with `GRIDARENA_SUBMISSION_SANDBOX`:

| Backend | Isolation | Use for |
|---|---|---|
| `subprocess` (default) | A child Python process with a stripped environment (no database DSN or API keys). It still runs as the server's OS user, with that user's file and network access. | Trusted submitters: local or in-house deployments. |
| `docker` | A throw-away container from `GRIDARENA_SUBMISSION_IMAGE` (default `gridarena`, this repo's image) with no network, a read-only root filesystem, `GRIDARENA_SUBMISSION_MEMORY` (4g) / `GRIDARENA_SUBMISSION_CPUS` (2) limits and only the run directory mounted. | Any deployment open to untrusted users. Needs the `docker` CLI on the server; when the server itself runs in a container, the run directory must be at the same path on the host. |

The whole run must finish within `GRIDARENA_SUBMISSION_TIMEOUT` seconds
(default 900).
