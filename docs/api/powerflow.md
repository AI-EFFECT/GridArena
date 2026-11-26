# Power Flow

Endpoints to run gridarena's power flow on a registered LV grid's stored
measurements and to read back the results, with the measured voltages next
to the computed ones. See [Architecture](../architecture.md) for how this
fits alongside the other routers.

**Base path:** `/powerflow`

Also serves a browser UI at `/powerflow/ui` and `/powerflow/ui/{grid_id}`.

## Inputs and outputs

For each measurement timestamp, the power flow takes:

- the grid's radial topology (`Node`, `Connection`) and each cable's
  **per-phase** impedance (`Cable`, Ω/km × length);
- the measured active/reactive power of every node on every phase
  (`Measurements`, load convention: positive = consumption);
- the measured voltage (magnitude and angle) of the substation node `"PT"`
  on each phase, as the reference. A phase PT doesn't measure gets the mean
  measured magnitude at its nominal angle (0°, −120°, +120° relative to R); a
  timestamp where PT measures no voltage at all is skipped.

It computes the complex voltage of every node on every phase. These
**computed** voltages are stored in `PowerFlowResults`, never in
`Measurements`: the **measured** voltages of the other nodes are not inputs,
so comparing the two checks the grid model (topology, cable data, phase
assignment) against the meters.

## Models

| `mode` | What it solves |
|---|---|
| `unbalanced` (default) | Three-phase, four-wire. Each phase keeps its own loads, cable impedance and PT voltage. Loads are connected phase-to-neutral, so unequal phase currents return through the neutral conductor, modelled with impedance `neutral_impedance_ratio` × the mean phase impedance of each cable and grounded at PT only. The neutral's voltage drop shifts the neutral potential along the feeder, and with it every phase-to-neutral voltage — also on lightly loaded phases. `neutral_impedance_ratio = 1` suits cables with a full-size neutral (e.g. 4x95); `0` is an ideal neutral, which makes the three phases independent. |
| `balanced` | Each node's total power (sum of its phases) spread evenly over the three phases, with a symmetric PT voltage (mean measured magnitude, R-phase angle, 120° apart). |

There are no mutual (phase-to-phase) impedances: the grid data holds per-phase
self impedances only. The solver is a backward/forward sweep (in matrix form,
over all timestamps at once) for radial grids: the topology is traversed from
`"PT"`, so the stored direction and order of the connections don't matter,
parallel connections between the same two nodes are combined, a loop is
rejected, and nodes not connected to PT get no voltage. A timestamp converges
when no voltage changes by more than 1 µV between two iterations (at most 200).

## Endpoints

### `POST /powerflow/{grid_id}/run`

Solves every distinct measurement timestamp of the grid (optionally within a
range) and **replaces** the grid's stored results and run record.

!!! warning "Synchronous"
    The solve runs inside the request. It is vectorised over timestamps, so
    a long range costs mostly the database reads and writes.

**Path parameters**

| Name | Type | Description |
|---|---|---|
| `grid_id` | string | The grid to run power flow for. |

**Query parameters**

| Name | Type | Required | Default | Description |
|---|---|---|---|---|
| `mode` | `"unbalanced"` \| `"balanced"` | no | `unbalanced` | The model, see [Models](#models). |
| `neutral_impedance_ratio` | number ≥ 0 | no | `1.0` | Unbalanced mode only: neutral impedance relative to the phase impedance. |
| `start_time` | string (ISO datetime) | no | — | Lower bound on measurement timestamps to include (inclusive). |
| `end_time` | string (ISO datetime) | no | — | Upper bound on measurement timestamps to include (inclusive). |

**Response** `200`

```json
{
  "message": "Power flow has been run successfully.",
  "mode": "unbalanced",
  "neutral_impedance_ratio": 1.0,
  "start_time": null,
  "end_time": null,
  "timestamps": 336,
  "not_converged": 0,
  "skipped_timestamps": 0,
  "unreachable_nodes": []
}
```

`neutral_impedance_ratio` is `null` for a balanced run.

**Errors**

| Status | Cause |
|---|---|
| `404` | No measurements for the grid (in the range, if given). |
| `422` | Invalid `mode` or `neutral_impedance_ratio`; the grid has no nodes, no node `"PT"`, no connections with cable data, or a loop; or no timestamp has a PT voltage. |
| `500` | Any other failure, such as a database error; `detail` is `"Error: {message}"`. |

---

### `GET /powerflow/data/{grid_id}/results`

Returns the grid's stored results, the computed and measured voltage side by
side, plus the run that produced them and the per-node error of the
computed voltages against the measured ones.

**Path parameters**

| Name | Type | Description |
|---|---|---|
| `grid_id` | string | The grid to retrieve results for. |

**Query parameters**

| Name | Type | Required | Default | Description |
|---|---|---|---|---|
| `phase` | `"R"` \| `"S"` \| `"T"` \| `"N"` | no | all | Only this phase. `N` is the neutral-to-ground voltage of unbalanced runs. |

**Response** `200`

```json
{
  "grid_id": "grid001",
  "run": {
    "mode": "unbalanced", "neutral_impedance_ratio": 1.0,
    "start_time": null, "end_time": null,
    "timestamps": 336, "not_converged": 0, "skipped_timestamps": 0,
    "unreachable_nodes": [], "created_at": "2026-10-25T16:02:11.412907"
  },
  "reference_node": "PT",
  "errors": [
    {"node_id": "N1", "phase": "R", "points": 336, "mae_voltage_magnitude": 0.412,
     "max_abs_voltage_magnitude": 1.37, "bias_voltage_magnitude": -0.08, "mae_voltage_angle": 0.21}
  ],
  "results": [
    {
      "timestamp": "2024-08-20 14:00:00",
      "node_id": "N1",
      "phase": "R",
      "converged": true,
      "computed": {"voltage_magnitude": 229.4, "voltage_angle": -0.3, "real": 229.4, "imag": -1.2},
      "measured": {"voltage_magnitude": 229.1, "voltage_angle": -0.4}
    }
  ]
}
```

- `computed` is the power flow's voltage (R/S/T: phase-to-neutral; N:
  neutral-to-ground), `measured` the meter reading for the same node, phase
  and timestamp — `null` where no meter measured it (always for N).
- `errors` lists, per node and phase, computed − measured over the converged
  timestamps where both exist: mean absolute error, largest absolute error
  and mean (bias) of the magnitude in V, and mean absolute angle error in
  degrees. `"PT"` is left out: its voltage is the power flow's input.
- `run` is `null` for results computed before runs were recorded.

**Errors**

| Status | Cause |
|---|---|
| `404` | No power flow results stored for `grid_id` (and `phase`, if given). |
| `500` | Database error while querying. |
