# Data Model

gridarena uses PostgreSQL. Every table is created idempotently at first use via
`CREATE TABLE IF NOT EXISTS` in `gridarena/database/create_*.py` — there is no
separate migration tool. This page summarizes the tables grouped by domain;
see the linked source file for exact column definitions.

Unless noted otherwise, deleting a `grids` row (`DELETE FROM grids WHERE
grid_id = ...`, e.g. via `DELETE /grid/{grid_id}`) cascades through every
table below that references it.

## Grid topology

Source: `gridarena/database/create_grid_database.py`

| Table | Key | Purpose |
|---|---|---|
| `grids` | `grid_id` (PK) | Catalog of registered LV grid IDs. Every other grid-scoped table has a `grid_id` foreign key back to this table, `ON DELETE CASCADE`. |
| `Node` | (`NodeId`, `grid_id`) | A grid's nodes, with optional lat/lon coordinates. |
| `Cable` | (`CableId`, `grid_id`) | Cable/conductor types (per-phase impedance and nominal current) available to a grid. |
| `Connection` | (`ConnectionId`, `grid_id`) | Edges between two `Node`s, referencing a `Cable` type and a length. |
| `GridUsage` | `grid_id` (PK) | Per-benchmark train/test flags for a grid (phase detection, topology detection, state estimation, voltage control). |

```text
grids ──┬──< Node ──┬──< Connection >── Cable
        │           └── (FromNodeId/ToNodeId reference Node)
        └──< GridUsage
```

## MV grids

Source: `gridarena/database/create_mv_database.py`

| Table | Key | Purpose |
|---|---|---|
| `MVGrid` | `mv_grid_id` (PK) | Catalog of registered MV grids (name, nominal voltage, region). |
| `MVNode` | (`NodeId`, `mv_grid_id`) | An MV grid's nodes. |
| `MVCable` | (`CableId`, `mv_grid_id`) | MV cable types: per-phase R/X (Ω/km), continuous rating, and optional charging capacitance (`CapUfPerKm`, µF/km) and emergency rating (`EmergCurr`, A). |
| `MVConnection` | (`ConnectionId`, `mv_grid_id`) | Edges between `MVNode`s, with optional `Status`, `IsSwitch` and `NormallyOpen` (tie lines are normally-open switches). |
| `MVConnectionPoint` | (`ConnectionPointId`, `mv_grid_id`) | A point on the MV grid where an LV grid can be connected via a transformer. |
| `MVTransformer` | (`TransformerId`, `mv_grid_id`) | An MV/LV transformer at a connection point. `lv_grid_id` references `grids(grid_id)` **`ON DELETE SET NULL`** — deleting the downstream LV grid unlinks the transformer rather than deleting it. Impedance (`ImpReal`/`ImpImag`) is in p.u. on the transformer rating; optional fixed off-load tap (`TapPos`, `TapStepPct`, `TapMin`, `TapMax`). |
| `MVRegulator` | (`RegulatorId`, `mv_grid_id`) | A step voltage regulator on an `MVConnection` (at most one per connection, deleted with it): set-point and deadband (p.u.), tap step (%), range and starting position. |
| `MVShunt` | (`ShuntId`, `mv_grid_id`) | A capacitor bank (`QStepKvar` > 0) or shunt reactor (< 0) at an `MVNode` (deleted with it): kvar per step at nominal voltage, `NSteps`, steps switched in (`Step`) and `Status`. |
| `MVDER` | (`DerId`, `mv_grid_id`) | A generator, battery or EV hub at an `MVNode` (deleted with it): type, rating, profile (`solar` / `constant`) and control settings (mode, cos φ, Q limit, Q(V) and P(V) curves, voltage set-point). |

## Measurements and historical data

Two independent, differently-scoped systems:

**Node/grid-scoped measurements** — source: `create_measurements_database.py`

| Table | Key | Purpose |
|---|---|---|
| `HistoricalNodes` | (`node_id`, `grid_id`) | Links a node to a grid for measurement purposes. |
| `Measurements` | `measurement_id` (PK) | Timestamped power/voltage measurements for a `HistoricalNodes` entry, optionally per-phase. |

**Standalone historical series** (not tied to any grid or node) — source:
`create_historical_database.py`

| Table | Key | Purpose |
|---|---|---|
| `HistoricalDatabase` | `database_id` (PK) | A named collection of series sharing one timeline, with a `data_type`: `load` (consumption, the default) or a kind of generation: `pv`, `chp`, `wind`, `hydro`, `other_generation`, `injection_total`. |
| `HistoricalTimestamp` | (`database_id`, `datetime`) | The shared timeline every series in a database must use. |
| `HistoricalSeries` | (`database_id`, `series_id`) | A single series (`grid_level` ∈ `MV`/`LV`/`Feeder`), with its stored record counts: `n_records`, and how many of those carry power (`n_power`) and voltage (`n_voltage`). |
| `HistoricalRecords` | `record_id` (PK) | Power and/or voltage values for a series at a timestamp/phase. |

The record counts in `HistoricalSeries` save scanning `HistoricalRecords`,
which can hold tens of millions of rows. Statement-level triggers on
`HistoricalRecords` (`historical_records_counts_ins` / `_upd` / `_del` /
`_trunc`, function `historical_records_update_counts`) keep them exact on
every write, including bulk imports made directly against Postgres. A
database's total is the sum over its series. The app adds the columns and
triggers at startup. The first time, it fills the counts with one full
scan, holding off writes to `HistoricalRecords` meanwhile. Requires
PostgreSQL 10 or later.

## Power flow

Source: `create_pf_database.py`

| Table | Key | Purpose |
|---|---|---|
| `PowerFlowResults` | (`grid_id`, `node_id`, `phase`, `datetime`) | Computed voltage (real/imaginary) per node, phase and timestamp, and whether that timestamp `converged`. `phase` is `R`/`S`/`T` (phase-to-neutral) or `N` (neutral-to-ground, unbalanced runs). Kept apart from the measured voltages in `Measurements`. |
| `PowerFlowRuns` | `grid_id` | The run behind a grid's current results: model, neutral impedance ratio, time range, timestamps solved / not converged / skipped, unconnected nodes. Each run replaces the grid's previous results. |

## Digital Twin

Source: `create_digital_twin_database.py`

| Table | Key | Purpose |
|---|---|---|
| `DigitalTwin` | `digital_twin_id` (PK) | A twin's configuration: source, field mapping, power-flow config, status, last run/error, and its most recent measurement batch. |
| `DigitalTwinEvent` | `event_id` (PK) | Append-only event log for a twin (e.g. fetch/publish/error events). |
| `DigitalTwinResult` | `result_id` (PK) | Per-run comparison metrics (voltage MAE/RMSE, power error, convergence status) between the twin's power flow and observed data. |

## Offline Scenarios

Source: `create_offline_scenario_database.py`

| Table | Key | Purpose |
|---|---|---|
| `OfflineScenario` | `scenario_id` (PK) | A replay-based (non-live) scenario derived from a digital twin (`source_dt_id`), with a modified grid snapshot / connection changes / power limits. |
| `OfflineScenarioResult` | `result_id` (PK) | Per-timestamp simulation result for a scenario. |

## PF Data Generation

Source: `create_pf_datagen_database.py`

Bulk power-flow scenario output tables, keyed by an opaque `run_id` with **no**
foreign key back to any run-metadata table — run configuration/status instead
lives in `run_config.json` under `gridarena/pf_datagen/runs/<run_id>/` (see
[Architecture › Background job subsystems](architecture.md#background-job-subsystems)).

| Table | Purpose |
|---|---|
| `MVPFScenarioBus` | Per-node results (`pd`, `qd`, `vm`, `va`, bus type, plus `pd_pu`/`qd_pu`/`vm_pu`, and the run's voltage band `vm_min`/`vm_max` with a `v_violation` flag, the LinDistFlow `vm_lin`, the bus's voltage level `vn_kv`, the 1 p.u. reference demand `pd_ref`/`qd_ref`, the shunt elements `gs`/`bs` and the DER injection `pg`/`qg`) for one sample, including each MV/LV transformer's LV-side bus — every node of the grid, with `energised` false for islanded ones. |
| `MVPFScenarioBranch` | Per-connection results (flows at both ends, losses, current, loading %, impedance and charging susceptance, normal and emergency thermal checks, LinDistFlow flows, branch `kind` / `tap` / `tap_pos`, a regulator's set-point and admittance components, the cable type used and the drawn length factor, plus per-unit twins) for one sample, transformers included — every connection, with `in_service` false for disabled ones. |
| `MVPFScenarioYbus` | Y-bus matrix entries (`g`, `b` and their per-unit `g_pu`, `b_pu`) of the energised network for one sample. |
| `MVPFScenarioRuntime` | Solver used, solve time, iterations, control rounds, convergence status and failure reason for one sample (converged or not), plus its load level and load scale factor, DER penetration and installed capacity, capacitor steps switched in, the per-sample outages (DER tripped, banks out, regulators bypassed), conductor temperature and cable swaps, and its design stratum and target-mix weight. |
| `MVPFScenarioSummary` | One row per converged sample: energised voltage range and under/over-voltage counts, highest loading and thermal-violation counts, and totals of load, shunt reactive power, DER generation and curtailment, losses and the power the substation supplies (with per-unit twins). |
| `MVPFScenarioDER` | One row per DER per converged sample, the grid's own and any generated for the sample (`synthetic`): available and solved P/Q, curtailment, the reactive-power bound and whether it was reached (generator convention, with per-unit twins). |
| `MVPFMeasurement` | With the measurement view on: one row per reading of a converged sample (long format): the element (`element_type`, `element_id`, `node_id`), `quantity` (`v` / `p` / `q` / `i`), `value` and `value_pu` (null when `missing`), `source` (`meter` or `pseudo`) and `is_outlier`. |
| `MVPFMeasurementLayout` | One row per meter of a run's layout, drawn once per run: the element, the `quantities` it reads, why it is there (`reason`: substation, feeder_head, connection_point, switch, cable, der, der_generated or pseudo), its noise `profile`, `source` and, on a branch, the end it reads (`meter_end`). |
| `MVPFTopologyVariant` | One row per topology variant of a run: its `kind` (base / reconfiguration / outage / both), configuration (`config_idx`, the switches it opens and ties it closes), disabled connections (and how many are transformers) and the nodes that leaves islanded. |

The four sample tables carry `scenario` / `load_scenario_idx` /
`topology_variant_idx` identifiers — see
[PF Data Generation › Samples, topology variants and outages](api/pf-datagen.md#samples-topology-variants-and-outages).

## Benchmarks

Benchmark submissions are Python scripts, evaluated against ground truth read
from the grid tables (`Measurements`, `Node`, `Connection`, ...) at run time.
Submissions and their scores are kept on disk
(`gridarena/benchmarks/submissions/runs/`), not in the database — see
[API Reference › Code submissions](api/index.md#code-submissions).

Tables the benchmarks still use:

| Table | Key | Purpose |
|---|---|---|
| `GridMask` | `grid_id` (PK) | State estimation training data: maps a grid to an opaque `masked_id`. |
| `NodeMask` | (`grid_id`, `node_id`, `phase`) | State estimation training data: maps a real node/phase to an opaque `masked_node_id`. |
| `VoltageControlSolutions` | `solution_id` (PK) | Reference solutions (corrected voltage, adjusted P/Q) for a grid snapshot: served with the training scenarios and used to score submissions. Created by `vc_solutions.py`. |

Tables of the previous download-and-upload-results flow. Startup still creates
them so that existing rows are kept, but nothing reads or writes them any more:

| Table | Previously |
|---|---|
| `AnonymisedMapping`, `UserGuesses` | Phase identification: persistent `(node_id, phase)` → key mapping, and guessed phases. |
| `GridAnonymisedMapping`, `TopologyUserGuesses` | Topology discovery: anonymised grid keys, and guessed edge lists. |
| `EstimationTasks`, `StateEstimates` | State estimation: per-user known/unknown splits, and submitted estimates. |
| `VoltageTimestampMapping`, `VCUserGuesses` | Voltage control: anonymised snapshot keys, and guessed voltages/loads. |
