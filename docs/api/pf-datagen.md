# PF Data Generation

Endpoints for bulk-generating MV power-flow scenarios (per-bus, per-branch,
per-generator, Y-bus, solve-runtime and per-sample summary results, and
optionally noisy measurements of them) from a historical database's load
series, optionally with topology and admittance perturbations. Runs as a
background job — see
[Architecture › Background job subsystems](../architecture.md#background-job-subsystems)
for the shared `run_id` / `queued`→`running`→`completed`/`failed` /
`runs/<run_id>/` pattern also used by [RL](rl.md) and
[Diffusion Models](diffusion.md). Output rows land in the `MVPFScenario*`
tables — see [Data Model › PF Data Generation](../database-schema.md#pf-data-generation).

**Base path:** `/pf-datagen`

**Pages.** The router also serves the **Synthetic Data** section of the UI
(the nav group of that name). These pages are not part of the API schema.

| Page | What it shows |
|---|---|
| `/pf-datagen/ui` | Overview: every run and group, with status and progress; refreshes while a run is active |
| `/pf-datagen/ui/new` | New Dataset: seven steps (Grids, Load, DER & set-points, Topology & parameters, Sampling design, Measurements, Review & start). Each step checks its settings before moving on; the review lists any problem with a link to its step. Start opens the new run, or the group when several grids are selected |
| `/pf-datagen/ui/runs/{run_id}` | A run: live progress while it runs, then its summary, diversity report, charts, CSV exports, split suggestions and settings |
| `/pf-datagen/ui/groups/{group_id}` | A group: its runs' progress, the combined report with a row per grid, and its settings |
| `/pf-datagen/ui/synthetic-grids` | The [synthetic grid generator](mv-grid.md#post-mv_gridsynthetic) and the stored synthetic grids |

!!! note "`run_id` format is validated before touching the filesystem"
    `run_id`s are always server-generated: 8 lowercase hex characters. As in
    [RL](rl.md) and [Diffusion Models](diffusion.md),
    `PFDataGenRunRegistry.get_run()` checks a `run_id` against that pattern
    before building `RUNS_DIR / run_id / "run_config.json"`. It uses a full
    match, so a trailing newline is rejected too. A malformed `run_id`
    behaves as "not found" (`404`).

## Units and three-phase convention

The MV network is treated as balanced. Every injection is a **three-phase
total** in W / VAr (load convention), and the grid's `nominal_voltage_kv` is
the **line-to-line** voltage at the substation. For each series and
timestamp, the injection comes from the historical database like this:

- if the series has a row with `phase` omitted (`NULL`), that row is the
  three-phase total;
- otherwise its `R`/`S`/`T` rows are summed.

So per-phase LV series contribute their full three-phase load. The two
sources are never added together.

Each scenario is solved by one of two MV solvers, chosen with the request's
`solver`. Both work in per unit on the run's bases, separately from the
solver LV power flow uses, and share one model and one control loop:

- **`sweep`** (default, `gridarena/pf_datagen/solver.py`): a backward/forward
  sweep. Fast, but radial only: each energised part must be a tree fed by
  one substation.
- **`newton`** (`gridarena/pf_datagen/newton.py`): polar Newton-Raphson on
  the Y-bus, every substation a reference bus at angle 0 and every other bus
  PQ. It solves any topology: loops, closed tie lines, parts fed by several
  substations. On a radial grid it gives the sweep's voltages (to about
  1e-11 p.u.). Tolerance 1e-10 p.u. on the power mismatch, at most 30
  iterations.

There is no automatic fallback: with `sweep`, a meshed variant fails (see
below), so a dataset is never a mix of the two. The runtime table's `solver`
column records which one ran. Details shared by both:

- **Pi-model branches.** Each connection is its cable's series impedance plus
  half its charging susceptance at each end,
  `B = 2π · 50 Hz · cap_uf_per_km · length` (see
  [MV Grid](mv-grid.md) cable fields). A cable without `cap_uf_per_km` has
  no charging.
- **Order and direction don't matter.** Branch orientation and processing
  order come from a search outwards from the substations. The sweep solves a
  radial forest: each energised part fed by exactly one substation. A loop,
  or a part fed by two substations (e.g. through a closed tie line), can't
  be solved radially, so with `sweep` every sample of that variant fails with
  `solver_error: MeshedNetworkError: ...`; use `newton` for those.
- **Switches.** Normally-open switches (tie lines) and inactive connections
  are out of service (`in_service = false`) in every sample; branch rows mark
  switches with `is_switch`.
- **Transformers and regulators** are branches with an off-nominal ratio
  (see [below](#transformers-and-voltage-regulators)).
- **Capacitor banks and reactors** (the grid's `shunts`, see
  [MV Grid](mv-grid.md)) are constant admittances at their node, sized so
  that `step × q_step_kvar` is generated at 1 p.u.; their reactive power
  scales with V². An inactive shunt contributes nothing.
- **Loads** are constant power by default. The request's `load_model` makes
  them a ZIP mix: the historical P/Q is the demand at 1 p.u., and each
  iteration re-evaluates it at the current voltage as
  `P = P_ref (z·V² + i·V + p)` (same for Q with its own fractions).
- **Generators (DER)** inject P/Q at their node, set by their profile and
  control mode (see [below](#generators-der)).
- **Substation busbars.** Every substation is an ideal source (`REF`) at the
  request's `slack_voltage_pu` (default 1.0 p.u.; one value per sample,
  shared by all substations) and angle 0, as if fed from one HV grid. No
  upstream grid or HV/MV transformer is modelled.
- **Convergence.** For the sweep, tolerance is 1e-10 p.u. on the voltage
  update, with at most 1000 iterations (`max_iterations` beyond that).

Results are reported on the per-phase equivalent (`S/3` against
`V_LL/√3`, cable values per phase; see `gridarena/pf_datagen/three_phase.py`),
converted back to the input conventions. Each bus is on its own voltage
level (`vn_kv`), and branch impedances, susceptances and SI Y-bus entries are
referred to the MV side:

| Output | Column(s) | Unit / meaning |
|---|---|---|
| Bus demand | `pd`, `qd` | W, VAr — three-phase total drawn at the solved voltage |
| Reference demand | `pd_ref`, `qd_ref` | W, VAr — the demand at 1 p.u., as fed in. Equal to `pd`/`qd` under constant power and on islanded buses |
| Bus generation | `pg`, `qg` | W, VAr — generator convention (positive = injected): the bus's DER as solved. `bus_type` is `PV` while a voltage-controlling DER holds the bus at its set-point |
| Bus shunts | `gs`, `bs` | W, VAr at 1 p.u. (MATPOWER/datakit convention): the active power drawn and the reactive power generated by the bus's capacitor banks / reactors — `bs > 0` for a capacitor. Cable charging stays on the branch (`b_s_total`) |
| Bus voltage | `vm`, `va` | V line-to-line on the bus's own level; degrees relative to the substation |
| Bus level | `vn_kv` | kV line-to-line: the MV nominal, or about 0.4 for a transformer's LV-side bus |
| Voltage check | `vm_min`, `vm_max`, `v_violation` | The run's voltage band (p.u.); `v_violation` = an energised bus outside it |
| Branch flow, from end | `pf`, `qf` | W, VAr — three-phase total entering at the from end |
| Branch flow, to end | `pt`, `qt` | W, VAr — three-phase total entering at the to end (negative when power is delivered there) |
| Branch losses | `p_loss`, `q_loss` | W, VAr — `pf + pt`, `qf + qt`. `q_loss` includes the charging, so it is negative on a lightly loaded cable |
| Branch type | `kind`, `tap`, `tap_pos`, `is_switch`, `v_set_pu` | `line`, `regulated_line` or `transformer`; the ratio used (1.0 for plain lines), the tap position (`null` for plain lines), whether the connection is a switch, and a regulator's set-point (`null` otherwise, and for a bypassed regulator) |
| Branch parameters | `cable_type`, `length_factor` | The cable type the sample used (a [swap](#reconfiguration-outages-and-parameters) may change it; `null` for transformers), and the drawn length factor (`null` when lengths aren't drawn) |
| Branch current | `i_mag` | A — lines: the larger of the two end currents (they differ when the cable has charging); transformers: the MV-side current |
| Branch loading | `loading_pct` | % — `i_mag / thermal_rating_a × 100` (`null` without a rating). For a transformer the rating is its rated MV-side current, so this is essentially S / S<sub>rated</sub> |
| Branch impedance | `r_ohm_total`, `x_ohm_total` | Ω per phase, for the whole length; transformers referred to the MV side |
| Branch charging | `b_s_total` | S per phase, for the whole length (`0` without `cap_uf_per_km`) |
| Thermal check | `thermal_rating_a`, `thermal_violation` | `i_mag` compared with the cable's `nom_curr`, or the transformer's rated current |
| Emergency check | `emergency_rating_a`, `loading_emergency_pct`, `thermal_violation_emergency` | `i_mag` compared with the cable's `emerg_curr`; all `null` when the cable has none |
| LinDistFlow baseline | bus `vm_lin`; branch `pf_lin`, `qf_lin` | V line-to-line; W, VAr at the from end. `null` when the energised network isn't radial (a `newton` sample with a loop) |
| Branch admittance | `yff_r`, `yff_i`, `yft_r`, `yft_i`, `ytf_r`, `ytf_i`, `ytt_r`, `ytt_i` | p.u. (as in datakit): `Yff = y + jb/2`, `Yft = Ytf = −y/n`, `Ytt = y/n² + jb/2` |
| Y-bus | `g`, `b` | S per phase, referred to the MV side, including the charging halves and capacitor banks / reactors on the diagonal, and the transformer ratios |
| Set-point | runtime `v_slack_pu` | p.u. — the substation busbar voltage used for the sample |

The **LinDistFlow** columns are a cheap linear approximation (Baran & Wu):
lossless branch flows, and a voltage-squared drop linear in P and Q, with
charging and capacitor banks taken at 1 p.u. and loads and DER at the
demand and injection of the exact solution. It's the MV counterpart of the DC power flow
baseline datakit reports, which assumes R ≪ X and doesn't hold for MV
cables. The emergency check matters most in outage variants, where loading
is usually judged against the emergency rather than the continuous rating.

The voltage band comes from the request's `voltage_limits` and defaults to
0.90–1.10 p.u. (EN 50160's ±10 %; in Portugal, ERSE's Regulamento da
Qualidade de Serviço, art. 25.º, applies NP EN 50160 to AT, MT and BT). Islanded buses never get
`v_violation = true`; their `energised = false` already explains the
missing voltage. All of these columns can also be derived from the voltages,
impedances and ratings above. They're computed once here so that every
consumer applies the per-phase / √3 conventions the same way.

### Per-unit columns

Each SI value above also has a per-unit twin. These are the columns to use
for ML features, because they don't depend on the grid's voltage level or
size:

| Table | Per-unit columns | Base |
|---|---|---|
| `bus` | `pd_pu`, `qd_pu`, `pd_ref_pu`, `qd_ref_pu`, `gs_pu`, `bs_pu`, `pg_pu`, `qg_pu` | `S_base` |
| `der` | `s_rated_pu`, `p_available_pu`, `p_pu`, `q_pu`, `p_curtailed_pu`, `q_max_pu` | `S_base` |
| `bus` | `vm_pu`, `vm_lin_pu` | `V_base` |
| `branch` | `pf_pu`, `qf_pu`, `pt_pu`, `qt_pu`, `p_loss_pu`, `q_loss_pu`, `pf_lin_pu`, `qf_lin_pu` | `S_base` |
| `branch` | `r_pu`, `x_pu` | `Z_base` |
| `branch` | `b_pu` | `Y_base` |
| `ybus` | `g_pu`, `b_pu` | `Y_base` |

The bases are three-phase system bases:

- `V_base` = the line-to-line nominal voltage of the bus's level: the grid's
  `nominal_voltage_kv` for MV buses, `√3 × secondary_voltage_v` for a
  transformer's LV-side bus. Impedance and admittance columns use the MV
  `V_base`;
- `S_base` = the request's `s_base_mva` (default 1 MVA);
- `Z_base = V_base² / S_base` and `Y_base = 1 / Z_base`.

`Z_base` is the same whether you read it line-to-line or per phase, so
per-phase ohms and siemens convert directly. Angles (`va`), currents
(`i_mag`, with `loading_pct` as the scale-free alternative), the voltage band
(already in p.u.) and flags have no per-unit column. A completed run records its bases under
`metrics.per_unit_bases`. Rows written before these columns existed have
them set to `null`.

## Transformers and voltage regulators

**MV/LV transformers** (from the grid's `transformers`, see [MV Grid](mv-grid.md))
are modelled as branches:

- **Branch and LV bus.** Each transformer becomes a branch named after it,
  of `kind` `transformer`. It runs from its connection point's MV node to a
  new LV-side bus `<transformer_id>_LV`, whose `vn_kv` is
  `√3 × secondary_voltage_v / 1000`.
- **Impedance and tap.** The impedance comes from `imp_real`/`imp_imag`
  (p.u. on the transformer's rating; typical defaults when missing). The
  fixed tap gives the ratio `1 + tap_pos × tap_step_pct / 100` at the LV end,
  so a positive position raises the LV voltage.
- **Where the load sits.** The load assigned to a connection point with a
  transformer moves to the transformer's LV bus. With several transformers
  at one point, it's split in proportion to their ratings. A point without a
  transformer keeps its load on the MV node.
- **Inactive transformers** (`status` other than `active`) are
  `in_service = false` in every sample, so their LV bus shows as islanded,
  including in the base variant. They aren't listed as disabled in the
  `variants` table.
- **Not perturbed.** Transformers aren't topology-perturbation candidates,
  and admittance perturbation leaves their impedance alone.

**Step voltage regulators** (from the grid's `regulators`):

- **Placement.** A regulator sits at the to end of its connection, which
  becomes `kind = regulated_line`.
- **Control.** In each sample, the solver moves the regulator's tap to bring
  the to-node voltage within `v_set_pu ± deadband_pu / 2`, or stops at the
  tap limit. The chosen position is reported in `tap_pos`, and the ratio in
  `tap`.
- **Unsettled control.** If the taps are still moving after the round limit,
  the sample fails with `control_not_settled`.
- **On a loop** (only possible with `newton`), a regulator holds its
  starting tap instead of regulating. In a closed loop between stiff
  sources, any tap away from neutral drives a circulating current around the
  loop (tens of MW on an MV feeder), which is why regulators aren't operated
  that way. The two substations count as joined through the upstream grid,
  so a path from one substation to the other is a loop too.

Branches with a ratio (transformers and regulated lines) must point away from
the substation. Otherwise every sample of the variant fails with a
`solver_error`.

## Generators (DER)

Generators, batteries and EV hubs on MV nodes come from the grid's `ders`
(see [MV Grid](mv-grid.md)). Everything about them uses the **generator
convention**: positive P/Q is injected into the grid.

**Available power** comes from each unit's `profile`, capped at its rating:

- `solar`: `p_max_kw` × the clear-sky irradiance share × the day's
  clearness. The clear-sky share is Haurwitz's global horizontal irradiance
  (`1098 cos z · exp(−0.057 / cos z)` W/m²) over 1000 W/m², capped at 1, with
  the solar zenith `z` from the NOAA solar-position equations at the DER
  node's coordinates. Timestamps are read as local clock time in the
  request's `timezone`. The clearness factor is drawn once per calendar day,
  shared by every PV unit, uniformly from the request's `pv_clearness` range
  (default 0.3–1.0), on its own random stream.
- `constant`: `profile_value` × `p_max_kw` in every sample. A negative value
  consumes (a charging battery, an EV hub).

**Control** (`control_mode`) then sets the injection in each sample:

| Mode | Active power | Reactive power |
|---|---|---|
| `fixed_pf` | available | `cos_phi`, absorbing when `excitation` is `under` (the default), injecting when `over` |
| `q_v` | available | Q(V) droop through `qv_curve`: full capacitive Q at or below v1, none between v2 and v3, full inductive Q at or above v4 |
| `p_v` | curtailed linearly from 100 % at `pv_v_start` to 0 at `pv_v_end` | as `fixed_pf`, on the curtailed P |
| `voltage` | available | whatever holds the node at `v_set_pu`; at its limit the node becomes PQ |

Reactive power always stays inside the rating circle `√(S² − P²)`; outside
`fixed_pf` it is also limited to `q_limit_pu × s_rated_kva` (default 0.33,
cos φ 0.95 at full output).

The controls and the voltage regulators are settled together: after each
solve, the regulators move their taps, and the DER take one joint Newton
step, using the R/X sensitivities between their nodes (the shared-path
impedance on a radial network, the impedance matrix with the substations
held fixed on a meshed one), so units on one feeder don't fight each
other. A unit whose error changes sign between rounds
has its step halved. If anything is still moving after 100 rounds, the
sample fails with `control_not_settled`. The runtime table records the
rounds used in `control_rounds`.

An inactive DER, or one on an islanded node, injects nothing and is
`in_service = false`. DER aren't topology-perturbation candidates.

The **`der`** table has one row per DER per converged sample:

| Column | Meaning |
|---|---|
| `der_id`, `node_id`, `der_type`, `control_mode` | The unit, as defined in the grid, or as generated for the sample |
| `in_service` | `false` when inactive or islanded |
| `s_rated` | VA — rating |
| `p_available` | W — what the profile offers this sample, reported even when out of service |
| `p`, `q` | W, VAr — the solved injection |
| `p_curtailed` | W — `p_available − p`, held back by P(V) control |
| `q_max`, `at_q_limit` | VAr — the reactive-power bound used, and whether the unit ended at it |
| `synthetic` | `true` for a unit [generated for the sample](#generated-der-and-set-points) (ids `GEN-PV-01`, …) |

## Load: shape and level

Each sample's load starts from the historical database: one timestamp, with
series assigned to the MV connection points, reassigned every
`reassignment_period_timesteps` samples. How a point's load is then built
(`load_shape`, `gridarena/pf_datagen/load_shape.py`):

1. **Sizing** (`point_sizing`):
    - `series`: values as recorded. Right for LV databases, where a point's
      load is the sum of as many LV series as its LV grid has nodes.
    - `rating`: a series supplies only its **shape** (its values ÷ its
      average magnitude, mean |P|). The point's size comes from its
      transformers: average load magnitude = the sum of their kVA ×
      `utilisation` (default 0.4); points without one get the average size. MV and Feeder series then populate **every**
      connection point, LV grid attached or not.
    - `auto` (default): `rating` for MV and Feeder databases, `series` for LV.

    The divisor is the average *magnitude*, so a series that exports at
    times (e.g. a feeder with rooftop PV) keeps a sensible scale. For a
    series that never goes negative it is simply its mean.

    `size_by` sets what `utilisation` measures:
    - `mean` (default): the average loading, mean |P| ÷ kVA.
    - `peak`: the peak loading, the 99th percentile of |P| ÷ kVA. For 99 % of
      the time a point's load, and any export, stays within utilisation ×
      kVA. A series that is much peakier than its average, or that exports
      a lot (a feeder with much rooftop PV), is then bounded by the rating
      too. Typical values are 0.5–1.0; the average loading then comes out at
      the peak ÷ the series' peak-to-average ratio (median 2.8 on
      feeder_measurements).

    `scale_over` sets where that average (or peak) is taken:
    - `database` (default): over the whole database, from a fixed sample of
      2,000 of its timestamps that is the same for every run. `utilisation`
      is then the point's average over the database. A short `window` keeps
      the level of its time of day and season: a night window is lighter than
      an evening peak.
    - `run`: over the run's own timestamps, as before November 2026. Every
      run averages exactly `utilisation`, whatever its time. On a short
      window this erases the time of day, and it magnifies a series whose
      values sit near zero there.

    `feeders_per_point` (Feeder databases only) sets how many series a
    point takes:
    - `one` (default): one series per point, distinct across the grid
      while the database has enough. With 40 feeders, every grid draws from
      the same 40 curves.
    - `by_kva`: a point sums several **distinct** series. No feeder appears
      twice within a point, but points may share feeders. The count is drawn
      once per run, uniformly from the band of `feeder_bands` that covers the
      point's transformer kVA, capped at the database's series. A point above
      the last band uses the last band. The series themselves are redrawn
      every reassignment block. Under `rating` sizing the sum is sized as one
      series, divided by its own mean or P99, so `utilisation` keeps its
      meaning. Summing smooths a point's curve and makes points more alike,
      as transformer-level load is.

    | Transformer (kVA, up to) | 100 | 160 | 250 | 400 | 630 | 1000 | 2000 | 3150 |
    |---|---|---|---|---|---|---|---|---|
    | Feeders (default bands) | 1–2 | 1–3 | 2–4 | 3–6 | 4–8 | 6–12 | 10–20 | 16–32 |

    The default bands suit feeder_measurements, whose LV feeders peak at
    about 40–120 kW. Each band is `{"max_kva", "min", "max"}`, with bands in
    increasing `max_kva` order.
2. **Reactive power** (`reactive`): `series` (default) keeps the historical
   Q and fills **missing** Q from a power factor; `pf` always uses the power
   factor. The power factor is inductive, drawn uniformly from
   [`pf_min`, `pf_max`] (default 0.90–0.99) once per point per reassignment
   block: Q = P · tan(acos(pf)).
3. **Noise**: each point's load is multiplied by a spatial × local factor,
   and its Q by an extra factor. All are log-normal with mean 1, so a load
   never changes sign:
    - spatial (`spatial_sigma`, `correlation_km`): correlated along the
      cables, correlation exp(−distance / `correlation_km`); parts of the
      grid not connected to each other are independent;
    - local (`load_noise_sigma`): independent per point;
    - Q only (`q_sigma`): lets the power factor vary.

The load is then scaled to a **load level** (`load_level`,
`gridarena/pf_datagen/load_level.py`). Every load in a sample is multiplied
by one factor (P and Q together); all topology variants of a sample share it.

| `mode` | Level | Default range |
|---|---|---|
| `off` (default) | Historical load, unscaled | — |
| `multiplier` | A factor on top of the historical load | 0.5–1.5 |
| `loadability` | A fraction of **P_max**: each sample is rescaled so its gross demand is level × P_max | 0.1–1.2 |

- **Demand** is measured as gross demand, the sum of positive P, over the
  buses the base topology energises. Load behind an out-of-service
  transformer, which is never served, doesn't count.
- **P_max** is the largest gross demand the grid carries within its limits.
  It's found at the start of the run by a bisection (within 1 %) on the run's
  average load pattern. Search conditions: base topology, no perturbation,
  DER off, capacitor banks as configured, the run's load model, the busbar
  set-point (middle of its range if drawn), regulators controlling. It stops
  at the first of:
    - no converged, settled solution (`convergence`);
    - a bus below `voltage_limits.vm_min` (`voltage`);
    - a branch above 100 % of its normal rating (`thermal`).

  If even a tiny load breaks a limit, the run fails with an error saying so.
- **Drawing levels:** `sampling` is `stratified` (default: the range is split
  into `bins` equal bands and samples are spread evenly over them) or
  `uniform`. Levels come from their own random stream, so turning them on
  doesn't change the run's other draws.
- **Near-empty samples:** a sample whose gross demand is below 1 % of the
  run's median isn't rescaled; its `load_level` is `null`.

Levels above 1 are deliberate: they produce stressed samples beyond the
grid's limits, labelled by their violation flags, or by a failure reason if
they don't converge.

## Generated DER and set-points

**Generated DER** (`der`, `gridarena/pf_datagen/der_sampling.py`) add
generation to every sample, to cover reverse flow and voltage rise. They are
listed as **plant groups** (`groups`). Each group follows a
[historical database](historical.md) of generation (`data_type` `pv`,
`wind`, `hydro`, `chp`, `other_generation` or `injection_total`,
`gridarena/pf_datagen/der_profiles.py`), whose type and time series its
plants take. Different groups can follow different databases, e.g. a PV
group and a wind group in the same run.

| `source` | DER in a sample |
|---|---|
| `grid` (default) | The grid's own DER only (no `groups` needed) |
| `add` | The grid's own (in service) plus the generated plants |
| `replace` | The generated plants only |

| `groups[]` field | Default | Meaning |
|---|---|---|
| `database_id` | required | The generation database. A `load` database is refused. |
| `interpretation` | `aggregated` | `aggregated`: the data are the summed output of many plants. `individual`: each series is one plant. |
| `count_min`, `count_max` | 1, 1 | Aggregated: the range of plants per grid. Each run (each grid of a group) draws its own number, uniformly, from its seed. `count` sets both, for a fixed number. 0 is allowed: a grid can then have none of the group. |
| `capacity_mw` | required (aggregated) | Aggregated: each plant's capacity. |
| `placement` | `random` | Where the group's plants go, below. |
| `control` | `mixed` | `fixed_pf`, `q_v`, `p_v` or `voltage` for every plant of the group, or `mixed`: each plant's mode drawn from its type's mix. |

- **Capacity.** It is fixed per run: the drawn number × `capacity_mw` per
  aggregated group, and one plant per producing series, rated at the series'
  maximum, per individual group. In a [group of grids](#post-pf-datagengrouprun)
  every grid gets the same groups, but draws its own number of plants (and
  its own placement and plant variation). The **installed penetration**
  (installed DER capacity ÷ P_max, see [load level](#load-shape-and-level))
  is therefore the same in every sample. It is reported, not drawn; P_max is
  still searched whenever DER are generated. With `add`, the grid's own
  capacity counts too. EV hubs, being loads, don't count as capacity.
- **Placement**, per group. Eligible nodes are MV nodes other than
  substations, energised in the base topology. A transformer's LV-side bus
  (`<transformer_id>_LV`) is never eligible: an MV plant there would feed
  through its MV/LV transformer.
    - `random`: every eligible node equally;
    - `load`: weighted by the node's transformer kVA;
    - `clustered`: weighted by exp(−cable distance / `cluster_km`) from a
      random centre (one per group and sample);
    - `feeder_end`: weighted by cable distance from the substation.
- **Control mode**, with the [DER control](#generators-der) defaults. With
  `mixed`, each plant draws its mode from its type's mix (fixed cos φ for a
  type without one):
    - PV: Q(V) 0.4, fixed cos φ 0.4, P(V) 0.2;
    - wind: Q(V) 0.5, fixed cos φ 0.5;
    - CHP and hydro: fixed cos φ 0.7, voltage 0.3;
    - battery (the grid's own): fixed cos φ.

  cos φ is drawn from [`cos_phi_min`, `cos_phi_max`] (absorbing), the
  voltage set-point from [`v_set_min`, `v_set_max`].
- **Available power:** from the group's generation database at the sample's
  time (below).

Plants are named `GEN-<TYPE>-<nn>`, numbered per type in group order. A
request that generates DER with the earlier fields (`historical`,
`penetration_*`, `units_*`, `locations`, `modes`, ...) is refused with a
message pointing to `groups`.

`resample_grid_settings` redraws the grid's own DER control modes and
settings per sample from the same type mixes. The grid's own DER keep their
own profiles; only generated plants follow a generation database. Each
sample draws from its own random streams, so a placement doesn't depend on
the order samples are solved in. `layout` decides whether the plants move:

- `per_sample`: new nodes (and drawn modes) every sample;
- `per_run`: sample 0's placement for the whole run, as a time series needs;
- omitted: follows the timeline, `per_run` for `window` and `per_sample` for
  `spread`.

The plants' output follows the databases over time either way.

### The generation databases

The settings every group's database is read and turned into plants with,
in `der.generation`:

| `generation` field | Default | Meaning |
|---|---|---|
| `generation_start` | none | Where the generation window starts (database clock), matched to the load's first timestamp. None: the same instants as the load. One shift for every database, so all technologies stay on one calendar. |
| `keep_time_of_day` | `true` | Round the window shift to whole days, so day and night line up. |
| `timezone` | `UTC` | Clock of the databases' timestamps (the E-REDES imports are UTC). The load's clock is the request's `timezone`. |
| `reference_year` | best-covered year | Aggregated: the year whose maximum normalises a profile (none: each database's own best-covered year). |
| `performance_min`, `performance_max` | 0.90, 1.00 | Aggregated: range of each plant's fixed performance factor. |
| `rho`, `sigma` | 0.85, 0.08 | Aggregated: correlation between 15-minute steps and stationary std of each plant's random adjustment. |

**Aggregated** data become synthetic plants that keep the aggregate's
daily, seasonal and weather pattern but differ from each other. The
aggregate (the database's series summed) is normalised once,
f(t) = P(t) / max(P over the reference year), the same denominator for every
plant. Plant k of capacity C_k delivers

    P_k(t) = clip(C_k · f(t) · a_k · (1 + ε_k(t)), 0, C_k)

where a_k ~ U[`performance_min`, `performance_max`] is drawn once per plant,
and ε_k is a persistent random adjustment:

    ε_k(j) = ρ_s · ε_k(j−1) + σ · √(1 − ρ_s²) · z_k(j),    ε_k(0) ~ N(0, σ)

It advances at every step of the database's own interval over the generation
window, night included. σ is the stationary standard deviation, and
ρ_s = `rho`^(step / 15 min). Output is zero whenever f(t) is zero and never
above C_k. Plants aren't rescaled afterwards, and their total isn't forced to
match the aggregate. These are synthetic plant profiles derived from an
aggregate pattern, not measurements of real plants. Aggregated plants are
numbered 0, 1, ... across all groups, and plant k's parameters come from its
own random stream (seed, 18, k).

**Individual** data: each series is one plant, rated at the series' maximum
over the database (series that never produce are left out). Its value is
used as recorded; the count and `capacity_mw` don't apply.

**Time.** Load and generation are compared as instants: the load in the
request's `timezone`, the generation in `generation.timezone`. Whichever
series is sparser is interpolated linearly:

- **Generation sparser** (15-min data under a 1-min load): the generation is
  interpolated between its two timestamps around each scenario.
- **Generation denser:** a `window` run steps at the finest generation
  interval, and the load is interpolated between its own timestamps.

To average instead of interpolate, set `load_aggregation_s` (see
[Timeline](#measurement-view-and-timeline)), e.g. 900 for 15-min data. The
run then steps at those intervals, each labelled with its start. A
generation database with a finer step is averaged over each interval, using
its points at the start and every step after it. One with the same step is
read at the interval start, with no interpolation. A coarser one is still
interpolated at the start.

A scenario whose generation instant falls in a gap of the data (missing
timestamps or records) takes the same clock time on the nearest day that has
data, up to 31 days away. Missing data is never read as zero production.

The report's `der_profile` records:

- per database: its type, step, the window shift in days, how many scenarios
  borrowed another day (`n_fallback`), the reference year and maximum;
- the groups as requested, with the number of plants each drew (`plants`),
  the installed and generated capacity and the installed penetration;
- per plant: its group, rating and database, and (aggregated) its
  performance factor and ρ per step, or (individual) its series;
- aggregated: ρ, σ and the seed.

The generation values are part of a run's inputs (`generation.json` /
`generation_<k>.npz`, one per database), so a
[standalone package](#post-pf-datagenmvmv_grid_idpackage) solves offline
with them. What they are read with is locked (`der.generation`):
`generation_start`, `keep_time_of_day`, `timezone`, `reference_year` and
the databases the groups follow, with whether each is read as aggregated.

**Set-points** (`setpoints`, `gridarena/pf_datagen/setpoints.py`):

- **Busbar voltage** (`slack_mode`):
    - none (default): `slack_voltage_pu` as before, a fixed value or a range
      redrawn for every output;
    - `fixed`: `slack_voltage_pu`'s value;
    - `uniform`: one draw per sample from `slack_voltage_pu`'s range,
      `stratified` (default) or `uniform`;
    - `load_dependent`: `slack_v0 + (slack_v1 − slack_v0) × load level ±
      slack_noise` (default 1.00 → 1.04 ± 0.01). Without a load level it
      uses the sample's demand relative to the run's peak.

  `per_substation` draws each substation separately; the solvers then hold
  each substation at its own set-point.
- **Capacitor banks and reactors** (`shunt_steps`):
    - `fixed` (default): the grid's step;
    - `random`: 0…n_steps per bank and sample;
    - `voltage`: each bank switches its own steps, one per control round,
      to keep its node within [`shunt_band_min`, `shunt_band_max`]
      (default 0.97–1.03), alongside the regulators and DER.

  Bus `gs`/`bs` and the Y-bus show the steps as solved.
- **Regulators** (`regulators_resample`): set-point
  ([`reg_v_set_min`, `reg_v_set_max`], default 0.98–1.03), deadband
  ([`reg_deadband_min`, `reg_deadband_max`], default 0.015–0.025) and
  starting tap drawn per sample. The set-point used is in the branch table's
  `v_set_pu`.
- **Transformer taps** (`transformer_taps_resample`): each MV/LV
  transformer's off-load tap drawn per topology variant, within its range.
  With `transformer_tap_steps` N, within ±N steps of nominal (tap 0) as
  well; a range that doesn't reach that far takes its end nearest nominal.

## Reconfiguration, outages and parameters

**Reconfiguration** (`reconfiguration`, `gridarena/pf_datagen/reconfiguration.py`)
draws `n_configs` random radial configurations besides the base one. Each
closes a different spanning set of switchable connections, and all are
distinct.

- **What can switch** (`switchable`): `marked` (default) uses the grid's
  switches and normally-open ties only. `all_lines` lets any MV line open, as
  if every section had a disconnector. Only active connections can be closed.
- **How configurations are drawn:** connections that can't switch are always
  closed and are merged first, along with all the substations into one root.
  A radial configuration is then a spanning tree of what remains, drawn
  uniformly with Wilson's algorithm. Their number is counted exactly
  (matrix-tree theorem; run metric `radial_configurations`). Asking for more
  than exist fails the run with an error that says so.
- **Regulators:** a configuration that would feed a regulator from its far
  side is redrawn, so the draw stays uniform over those that remain. The
  count then includes some that can't be used.
- **Meshed configurations:** `allow_meshed` closes `extra_closed` more open
  switches on top of each radial configuration. These loops need `solver`
  `newton`, and the request is rejected without it.
- **Grids without switches** have only their base configuration. The MV grid
  page can [suggest ties](mv-grid.md#get-mv_gridmv_grid_idtie-suggestions);
  `all_lines` also works.

A topology variant is then **a configuration plus an outage set**: the
outage sets of `topology_perturbation` are drawn for each configuration, over
the connections closed in it. Configuration 0 (the base) draws its outages as
before, so runs without reconfiguration reproduce exactly.

**Outages** (`outages`, `gridarena/pf_datagen/outages.py`):

- `transformers`: MV/LV transformers join the outage candidates; an outage
  islands its LV side.
- Per sample, shared by all its variants (probabilities per unit, 0 by
  default; 0.05 is a typical value):
    - `der_trip_p`: each DER unit trips, the grid's own and generated alike;
    - `shunt_out_p`: each capacitor bank or reactor is out (no steps in);
    - `regulator_bypass_p`: each regulator is bypassed (held at tap 0).

**`variants_per_sample`:** `all` (default) solves every sample on every
variant. A number m solves each sample on m distinct variants, spread so that
every variant gets about the same number of samples, so the cost grows
linearly with m. With many variants, the base variant then gets only its
share of samples.

**Parameters** (`parameters`, `gridarena/pf_datagen/parameters.py`) change the
branches a sample is solved on, never the stored grid. All are off by
default.

| Setting | Effect | Typical value |
|---|---|---|
| `length_sigma` | Each line's length × a factor in 1 ± σ, so R, X and charging scale together | 0.05 |
| `temperature` `{min, max, spread}` | One conductor temperature per sample in [min, max], plus a per-cable offset within ± spread. Changes R only: R<sub>T</sub> = R<sub>20</sub> (1 + 0.004 (T − 20)); stored R is taken to be at 20 °C, for aluminium and copper alike | 20–70 °C, ± 5 |
| `swap_p` | Probability that a line's cable type is replaced by another of its family, ratings included | 0.1 |
| `transformer_uk_sigma`, `transformer_xr_sigma` | MV/LV transformer impedance magnitude × 1 ± σ<sub>uk</sub>, X/R × 1 ± σ<sub>X/R</sub> | 0.1, 0.2 |
| `draw` | When length, swaps and transformers are drawn: `per_variant` (default) or `per_sample`. Temperature is always per sample | |
| `catalogue` | Extra cable types for swaps: `{cable_id, r_ohm_per_km, x_ohm_per_km, nom_curr_a, cap_uf_per_km, emerg_curr_a}` | |

**Cable catalogue for swaps:**

- **Contents:** the cable types of every grid in the database, one per
  distinct (R, X, rating), plus the uploaded `catalogue`. An uploaded type
  replaces a database one with the same values.
- **Families:** a type with at least 0.05 µF/km of charging capacitance is
  underground; one with less is overhead. Without a capacitance, X/R decides,
  with underground below 0.49.

`admittance_perturbation` works as before and combines with these.

With temperature or `per_sample` draws, the network is rebuilt for every
sample. On the showcase grid that made a run about 2.8 times slower.

## Design, target mix, groups and splits

**Joint design** (`design.mode`, `gridarena/pf_datagen/design.py`). Each
per-sample axis is already stratified on its own: the load level and the
busbar set-point in `uniform` mode. (Generated DER have a fixed capacity, so
no penetration axis.) The design decides
how the axes pair up across samples.

| `mode` | Pairing |
|---|---|
| `independent` (default) | Each axis shuffled on its own; in a short run some combinations can come up more often than others |
| `lhs` | One Latin hypercube: every axis stratified into n strata, keeping the pairing with the smallest rank correlation between axes (best of 50) |
| `factorial` | Every combination of the axes' bins filled equally; the run fails up front if there are fewer samples than combinations |

The busbar set-point joins the design only in `uniform` mode with one value
for all substations. With a design and `variants_per_sample`, samples go to
variants in load-level order, so every variant gets samples across the
whole load range. Each sample's stratum is in the runtime table's
`design_cell`, e.g. `L3.S7`: load-level bin 3, busbar set-point bin 7, using
each axis's own `bins`.

**Target mix** (`target_mix`, `gridarena/pf_datagen/target_mix.py`) makes a
run reach a minimum share of samples with a violation.

- **Settings:** `min_violation_share`, and `violations`, which is `voltage`
  (an under- or overvoltage), `thermal` or `any` (the default). It needs a
  load level.
- **Pilot:** the run first solves a random `pilot_share` of its samples
  (default 0.2) and measures the violation rate of each stratum. The strata
  are the load-level bins.
- **Rounds:** the remaining samples are drawn in three rounds. Before each
  round the rates are re-estimated from everything solved so far, shrunk
  slightly towards the run's overall rate. The strata are weighted by
  rate<sup>k</sup>, with the smallest tilt k that reaches the target.
- **Coverage floor:** every stratum keeps at least 20 % of its even share.
  If the target is out of reach even so, the run uses the strongest tilt and
  the report says so.
- **Weights:** oversampling changes the distribution, so every runtime row
  gets a `design_weight`: the stratum's even share divided by its actual
  share in the run. Weighted statistics estimate what an even design would
  give.

**Run groups** (axis 7A). [`POST /pf-datagen/group/run`](#post-pf-datagengrouprun)
starts one run per grid with a shared `group_id`.

- **Size filter:** grids with fewer than `min_nodes` nodes (10) or
  `min_connection_points` connection points (3) are skipped and listed.
- **Seeds:** each grid gets its own seed, derived from the group seed and the
  grid's position in the group.
- **Group report:** [`GET /pf-datagen/groups/{group_id}/report`](#get-pf-datagengroupsgroup_idreport)
  sums the runs' counts and histograms. It computes the per-sample
  percentiles over all the group's summary rows, and adds one row per grid.

**Split suggestions** keep related samples in the same split, so a test set
holds data the model hasn't effectively seen:

- `time`: whole calendar days of the historical timestamps, with every
  variant and every hour of a day together;
- `topology`: whole configurations when the run reconfigured, otherwise
  whole topology variants. The base topology always goes to training;
- groups: whole grids, stratified by node count.

Units are shuffled with the seed and fill train, validation and test in
turn, so split sizes follow the ratios (default 70 / 15 / 15) in samples.
When there are few units, every split with a positive ratio still gets at
least one. See
[`GET /pf-datagen/runs/{run_id}/splits`](#get-pf-datagenrunsrun_idsplits).

## Measurement view and timeline

**Measurement view** (`measurements`, `gridarena/pf_datagen/measurements.py`).
Off by default. When on, every converged sample also gets the readings a
realistic set of meters would give. Failed samples get none.

**The meter layout** is drawn once per run, from the seed, and stored in
the `layout` table, one row per meter:

| Meter (`reason`) | Readings | Noise profile |
|---|---|---|
| `substation`: every substation busbar | V | `substation_profile` |
| `feeder_head`: every branch with a substation at one end, metered at that end | P, Q, I | `substation_profile` |
| `connection_point`: a `cp_coverage` share of the connection points (default 0.30) | V at the MV node; P, Q through the point's transformers, where its load sits | `profile` |
| `switch`: every switch, open or closed | P, Q, I at the from end | `profile` |
| `cable`: a `cable_coverage` share of the other lines (default 0.10) | P, Q, I at the from end | `profile` |
| `der`: the grid's own DER of at least `der_min_mva` (default 1 MVA), unless `der.source` is `replace` | P, Q, V at its node | `profile` |
| `der_generated`: a rule, element id `*` | P, Q, V of every generated DER (axis 3) of at least `der_min_mva`, in the samples it appears in | `profile` |
| `pseudo`: every connection point not metered | P, Q pseudo-measurements | — |

**Readings** go in the `measurements` table, one row per reading:

| Column | Content |
|---|---|
| sample keys | As in the other tables |
| `element_type`, `element_id`, `node_id` | `bus`, `branch`, `connection_point` or `der`; its id; the node it measures at |
| `quantity` | `v`, `p`, `q` or `i` |
| `value`, `value_pu` | The reading in SI (V line-to-line, W, var, A) and per unit. Current's base is S_base / (√3 · the node's base voltage) |
| `source` | `meter` or `pseudo` |
| `missing` | True for a missing reading, whose values are null |
| `is_outlier` | True when the reading got an outlier error |

- **Noise:** each reading goes through the benchmarks' shared
  `corrupt_measurement_record` (`benchmarks/general_noise`) with a profile:
  `clean`, `easy`, `medium` (the meters' default) or `hard`. `easy` is the
  substation's default. Each profile adds relative Gaussian noise and rare
  outliers. Current uses the P/Q error level.
- **Missing:** a reading is missing with probability `missing_p` (0.02).
- **Pseudo-measurements:** the true value × max(0, 1 + N(0, `pseudo_sigma`)),
  with `pseudo_sigma` 0.30 by default. They are never missing.
- **Dead meters:** a meter on a de-energised node, or on a branch that is
  open or not energised at both ends, reads exactly 0.
- **Reproducibility:** each sample's noise, missing readings and
  pseudo-measurements come from streams seeded by the sample. A seed
  reproduces them whatever the solving order.

**Timeline** (`timeline`, `gridarena/pf_datagen/timeline.py`). Which
timestamps of the historical database a run uses:

- `spread` (the default): `scenario_count` timestamps evenly spread over the
  database's whole timeline, as before.
- `window`: `scenario_count` consecutive timestamps.
  - It starts at the first timestamp at or after `timeline_start`, or at a
    start drawn from the seed when `timeline_start` is omitted.
  - A start too close to the end moves back, so the window keeps its
    length.
  - `timeline_start` needs `window`.

A timeline no longer than `scenario_count` is used whole either way. The
report's `timeline` gives the first and last timestamp, the typical step,
the gaps (steps longer than 1.5 × the typical one) and `load_aggregation_s`.

**Load aggregation** (`load_aggregation_s`, seconds, default none). Averages
the load over clock-aligned intervals before it becomes samples:

- Intervals start at midnight and every `load_aggregation_s` after it, so
  the value must divide a day (300, 600, 900, 1800, 3600, ...). Each interval
  is labelled with its start, and its value is the mean of the records in
  [start, start + interval).
- The intervals replace the database's timestamps: `scenario_count`, `spread`
  and `window` all count intervals. A `timeline_start` takes the interval
  that holds it.
- An interval holding less than half of the timestamps it should is a gap
  and is left out. A series' value is the mean of the records it has there.
  Q is averaged only when every one of those records has it; otherwise it
  comes from the drawn power factor.
- The interval can't be finer than the database's own step: the run fails with that message.
- Generation databases with a finer step are averaged over the same
  intervals (see **Time** in [Generated DER and set-points](#generated-der-and-set-points)).

**Forecasting datasets.** For a time series, use `window` and turn off the
settings that draw something new for every sample:

- the load level and load noise;
- generated DER and their settings;
- a drawn busbar set-point, random capacitor steps and redrawn regulators;
- per-sample outages, per-sample parameters and temperature;
- the target mix.

Under `window`, the report lists any of these still on, as
`timeline.per_sample_axes`. Topology variants are fine: every timestamp is
solved on each.

## Samples, topology variants and outages

A run builds its list of **topology variants** once, then solves every
historical timestamp on every variant (or on `variants_per_sample` of them).
A **sample** is one (timestamp × variant) pair.

**Outage sets:** each variant is a [configuration](#reconfiguration-outages-and-parameters)
with some of its closed connections switched off. Open switches are never
outage candidates, and transformers are only with `outages.transformers`.

- `none` gives the configuration alone.
- `n_minus_k` includes the configuration alone, as the empty outage set.
- `random` never includes it: every random variant has at least one outage.
  Random variants are also **distinct**, so no two disable the same set of
  connections. A repeated draw is redrawn, and the run fails up front if
  `n_variants` exceeds the number of distinct sets the grid allows (see the
  [request](#post-pf-datagenmvmv_grid_idrun)).

Each sample is identified by:

| Column | Meaning |
|---|---|
| `scenario` | Integer, unique per sample within a run: `load_scenario_idx × n_variants + topology_variant_idx` |
| `load_scenario_idx` | Position of the sample's timestamp in the run's list (0-based) |
| `topology_variant_idx` | Which variant (see the `variants` table) |
| `scenario_timestamp` | The historical timestamp the injections came from |
| `output_group_id` | String form, `"<run_id>-<timestamp>-<variant>"` |

**Every converged sample has a row for every bus and every branch of the
base grid**, so all samples of a run share one node and edge set. Outages
are flags, not missing rows:

- **Islanded bus** (cut off from every substation): `energised = false`;
  `vm`, `va` and `vm_pu` are `0`. `pd`/`qd` keep the bus's reference demand,
  so summing them gives the load the outage disconnected.
- **Disabled branch:** `in_service = false`, zero flow, no thermal
  violation. Its impedance is still reported.
- **In-service branch with no supply** (both ends islanded):
  `in_service = true` with zero flow.
- **Only the substations left energised** (e.g. the head branches out): still a
  valid sample, with `converged = true`, `n_iter = 0`, `solve_time_ms = 0`
  and no Y-bus rows.

With admittance perturbation enabled, R/X are redrawn once per variant for
every branch, in service or not.

Every sample, converged or not, has a `runtime` row:

| Column | Meaning |
|---|---|
| `converged` | `true` when the sample has a valid solution |
| `n_iter` | Solver iterations, over all control rounds; `null` if the solver raised |
| `control_rounds` | Outer rounds the regulator and DER controls needed (1 without controls); `null` if the solver raised |
| `solver` | `sweep` or `newton` |
| `load_level`, `load_scale` | The sample's [load level](#load-shape-and-level) and the factor applied to its loads; `null` when the load level is off, and `load_level` `null` for a near-empty sample left unscaled |
| `der_penetration` | The installed DER penetration, installed capacity ÷ P_max (fixed by `der.groups`; `null` with `der.source` `grid`) |
| `der_capacity` | VA — installed DER capacity in service: the grid's own (with `add`) plus generated |
| `der_location`, `der_units` | The groups' placements (e.g. `random+feeder_end`) and number of generated plants (`null` / 0 when none were added) |
| `shunt_steps_in` | Capacitor-bank / reactor steps switched in, as solved (`null` without shunts) |
| `n_der_out`, `n_shunt_out`, `n_regulators_bypassed` | Per-sample [outages](#reconfiguration-outages-and-parameters): DER units tripped, banks out, regulators bypassed (`null` when that probability is 0) |
| `conductor_temp_c` | The sample's conductor temperature (`null` without `parameters.temperature`) |
| `n_cable_swaps` | Lines whose cable type was swapped (`null` with `swap_p` 0) |
| `design_cell` | The sample's stratum, e.g. `L3.S7`: bins of load level (`L`) and busbar set-point (`S`), whichever are drawn (`null` with none) |
| `design_weight` | With a [target mix](#design-target-mix-groups-and-splits): the weight that undoes its oversampling (`null` otherwise) |
| `solve_time_ms` | Wall time of the solve |
| `failure_reason` | `null` when converged, otherwise one of the values below |

The `failure_reason` values are:

- `max_iterations`: ran out of iterations with finite voltages;
- `diverged`: voltages went NaN or infinite;
- `control_not_settled`: regulator taps or DER controls were still moving at the round limit;
- `solver_error: <type>: <message>`: the solver raised (truncated to 300
  characters).

The text before the first `:` is the category. A non-converged sample has
no bus, branch or Y-bus rows.

The **`variants`** table lists each variant once:

| Column | Meaning |
|---|---|
| `topology_variant_idx` | Variant number, as used in the sample tables |
| `n_disabled`, `disabled_connections` | Connections switched off by the outage set |
| `n_islanded`, `islanded_nodes` | Nodes that leaves without supply |
| `kind` | `base` (base configuration, no outage), `reconfiguration`, `outage` or `both` |
| `config_idx` | The configuration (0 = base) |
| `open_switches`, `closed_ties` | Connections the configuration opens / closes, relative to the base |
| `n_disabled_transformers` | Transformers among the disabled connections |

Each converged sample also has one **`summary`** row:

| Column | Meaning |
|---|---|
| `n_bus_energised`, `n_bus_islanded` | Bus counts |
| `vm_pu_min`, `vm_pu_max` | Voltage range over energised buses (p.u.) |
| `n_undervoltage`, `n_overvoltage` | Energised buses below `vm_min` / above `vm_max` |
| `n_branch_in_service` | Branches in service |
| `max_loading_pct` | Highest `loading_pct` (`null` when no branch has a rating) |
| `n_thermal_violations`, `n_thermal_violations_emergency` | Branches over their continuous / emergency rating |
| `p_load`, `q_load` | W, VAr — demand drawn by energised buses |
| `q_shunt` | VAr — reactive power generated by capacitor banks (negative for reactors) at the solved voltage |
| `p_loss`, `q_loss` | W, VAr — sum of the branch losses (`q_loss` includes cable charging) |
| `p_gen`, `q_gen` | W, VAr — DER injection (generator convention) |
| `p_curtailed` | W — active power held back by P(V) control |
| `p_slack`, `q_slack` | W, VAr — what the substation supplies: load + losses − shunt generation − DER generation |

Each power column has a `_pu` twin on `S_base`.

Rows written before these columns existed have `scenario`,
`load_scenario_idx` and `energised` set to `null`, and those older runs omit
outaged elements instead of flagging them. Runs from before the summary
table have no `summary` rows.

## Endpoints

### `POST /pf-datagen/mv/{mv_grid_id}/run`

Starts a PF data-generation run for an MV grid: draws `scenario_count`
timesteps from a historical database as bus injections, optionally
perturbs topology/admittances, and solves power flow for each resulting
scenario.

**Path parameters**

| Name | Type | Description |
|---|---|---|
| `mv_grid_id` | string | The MV grid to generate scenarios for. |

**Request body**

```json
{
  "historical_database_id": "hist001",
  "reassignment_period_timesteps": 100,
  "scenario_count": 1000,
  "timeline": "spread",
  "timeline_start": null,
  "load_aggregation_s": null,
  "load_noise_sigma": 0.0,
  "load_level": { "mode": "off", "min": null, "max": null, "sampling": "stratified", "bins": 10 },
  "load_shape": {
    "point_sizing": "auto", "utilisation": 0.4, "reactive": "series", "pf_min": 0.90, "pf_max": 0.99,
    "spatial_sigma": 0.0, "correlation_km": 2.0, "q_sigma": 0.0, "scale_over": "database", "size_by": "mean",
    "feeders_per_point": "one", "feeder_bands": [{ "max_kva": 100, "min": 1, "max": 2 }, "…"]
  },
  "der": {
    "source": "grid", "groups": [],
    "generation": { "generation_start": null, "keep_time_of_day": true, "timezone": "UTC", "reference_year": null,
                    "performance_min": 0.90, "performance_max": 1.00, "rho": 0.85, "sigma": 0.08 },
    "cluster_km": 2.0, "cos_phi_min": 0.95, "cos_phi_max": 1.0, "v_set_min": 0.99, "v_set_max": 1.03,
    "resample_grid_settings": false, "layout": null
  },
  "setpoints": {
    "slack_mode": null, "slack_sampling": "stratified", "slack_bins": 10,
    "slack_v0": 1.00, "slack_v1": 1.04, "slack_noise": 0.01, "per_substation": false,
    "shunt_steps": "fixed", "shunt_band_min": 0.97, "shunt_band_max": 1.03,
    "regulators_resample": false, "reg_v_set_min": 0.98, "reg_v_set_max": 1.03,
    "reg_deadband_min": 0.015, "reg_deadband_max": 0.025, "transformer_taps_resample": false,
    "transformer_tap_steps": null
  },
  "topology_perturbation": { "type": "none", "k": 1, "n_variants": 10 },
  "reconfiguration": { "enabled": false, "n_configs": 10, "switchable": "marked", "allow_meshed": false,
                       "extra_closed": 1 },
  "outages": { "transformers": false, "der_trip_p": 0.0, "shunt_out_p": 0.0, "regulator_bypass_p": 0.0 },
  "variants_per_sample": "all",
  "design": { "mode": "independent" },
  "target_mix": { "min_violation_share": null, "violations": "any", "pilot_share": 0.2 },
  "measurements": { "enabled": false, "cp_coverage": 0.3, "cable_coverage": 0.1, "der_min_mva": 1.0,
                    "profile": "medium", "substation_profile": "easy", "missing_p": 0.02, "pseudo_sigma": 0.3 },
  "admittance_perturbation": { "enabled": false, "sigma": 0.2 },
  "parameters": { "length_sigma": 0.0, "temperature": null, "swap_p": 0.0, "transformer_uk_sigma": 0.0,
                  "transformer_xr_sigma": 0.0, "draw": "per_variant", "catalogue": [] },
  "voltage_limits": { "vm_min": 0.90, "vm_max": 1.10 },
  "slack_voltage_pu": 1.0,
  "solver": "sweep",
  "load_model": { "p": { "z": 0, "i": 0, "p": 1 }, "q": { "z": 0, "i": 0, "p": 1 } },
  "timezone": "Europe/Lisbon",
  "pv_clearness": { "min": 0.3, "max": 1.0 },
  "s_base_mva": 1.0,
  "chunk_commit_size": 100,
  "seed": null
}
```

`historical_database_id` (see [Historical](historical.md)) must have all
series sharing one `grid_level`, which determines how injections are
assigned to MV connection points. `topology_perturbation.type` is one of
`none`, `n_minus_k` (enumerated combinations of up to `k` simultaneous
branch outages, capped by `perturbation.MAX_N_MINUS_K_COMBOS`), or
`random` (`n_variants` distinct random outage sets of 1 to `k` branches).
If `n_variants` exceeds the number of distinct sets the grid allows (e.g. 5
variants with `k = 1` on a 4-branch grid), the run ends `failed` with an
error that says so.

`design` and `target_mix` are described in
[Design, target mix, groups and splits](#design-target-mix-groups-and-splits);
`target_mix` needs a load level.

`measurements`, `timeline`, `timeline_start` and `load_aggregation_s` are
described in [Measurement view and timeline](#measurement-view-and-timeline);
`timeline_start` (an ISO date-time) needs `timeline` `window`, and
`load_aggregation_s` must divide a day.

`reconfiguration`, `outages`, `variants_per_sample` and `parameters` are
described in [Reconfiguration, outages and parameters](#reconfiguration-outages-and-parameters);
`reconfiguration.allow_meshed` needs `solver` `newton`. A run with more than
20 000 topology variants is rejected.

`der` and `setpoints` are described in
[Generated DER and set-points](#generated-der-and-set-points). Mixes take
the listed keys only, with non-negative weights and a positive sum; every
type with a positive weight needs a mode mix. `setpoints.slack_mode`
`uniform` needs `slack_voltage_pu` as `{min, max}`; `fixed` needs one value.

`load_noise_sigma` (0 to < 1), `load_level` and `load_shape` are described
in [Load: shape and level](#load-shape-and-level). Empty `load_level.min` /
`max` take the mode's default range; `min` must not exceed `max` (equal
values fix the level). `load_shape.pf_min` must not exceed `pf_max`.

`voltage_limits` is the band (p.u. of nominal) that sets `v_violation` on
each energised bus. `vm_min` must be in (0, 1], `vm_max` in [1, 2), and
`vm_min < vm_max`.

`s_base_mva` (> 0, up to 1000) is the power base for the
[per-unit columns](#per-unit-columns).

`slack_voltage_pu` is the substation busbar set-point in p.u. of nominal,
between 0.8 and 1.2. It's either a fixed value (default `1.0`) or
`{"min": ..., "max": ...}` to draw it uniformly for each sample, from its own
random stream so the rest of the run's draws don't change. The value used is
recorded per sample in the runtime table's `v_slack_pu`.

`solver` is `sweep` (default) or `newton`; see
[Units and three-phase convention](#units-and-three-phase-convention) for
when each applies.

`load_model` sets how every load in the run depends on voltage, with
separate constant-impedance / constant-current / constant-power fractions
(`z`, `i`, `p`) for P and for Q. Each set must sum to 1. The default is
constant power; a common "static" mix is e.g. `{"z": 0.4, "i": 0.3, "p": 0.3}`.

`timezone` is the IANA time zone the historical timestamps are written in
(local clock time, default `Europe/Lisbon`), and `pv_clearness` the range the
daily clearness factor of the [solar profile](#generators-der) is drawn
from (`min` ≤ `max`, both in [0, 1]; equal values fix it). The clearness only
matters for the grid's own DER with a solar profile. `timezone` also places
the load's timestamps in time when generated DER read a
[generation database](#the-generation-databases).

The New Dataset page (`/pf-datagen/ui/new`) sends:

- from step 2 (Load): the data source, `scenario_count`, `timeline` (and for
  a window an optional `timeline_start`), `load_aggregation_s` (*Load
  resolution*), the reassignment period,
  `load_noise_sigma`, `load_level` and `load_shape`;
- from step 3 (DER): `der` -- the source, the plant groups, the shared
  `generation` settings, the cos φ / voltage set-point ranges and the
  layout (fixed to `per_run` for a `window` timeline). It shows the
  selected grids' own DER
  ([`GET /pf-datagen/devices`](#get-pf-datagendevices)) and a preview of
  the load against the shifted generation
  ([`GET /pf-datagen/generation-preview`](#get-pf-datagengeneration-preview));
- from step 4 (Set-points): `setpoints`, with `slack_voltage_pu` as a range
  when the busbar voltage is drawn; one card per device type, with the
  selected grids' counts;
- from step 5: `topology_perturbation`, `outages`, `variants_per_sample`,
  `reconfiguration`, `solver`, `parameters` (with a cable catalogue file if
  one is chosen) and `admittance_perturbation`;
- from step 6: `design` and `target_mix`; from step 7: `measurements`;
- from step 8: `seed` and `chunk_commit_size`.

With more than one grid selected it starts a group instead
([`POST /pf-datagen/group/run`](#post-pf-datagengrouprun)). It sends none of
the other settings in this paragraph group, so its runs use their defaults
(0.90–1.10 p.u., 1 MVA, constant power, Europe/Lisbon, clearness 0.3–1.0). `seed` is a base RNG seed; omit it for a fresh random
seed (logged in the run's metrics).

**Response** `200`

```json
{
  "run_id": "d4e5f6a7",
  "status": "queued",
  "message": "PF data-generation run started for MV grid 'mv001'."
}
```

**Errors**

| Status | Cause |
|---|---|
| `404` | `mv_grid_id` doesn't reference an existing MV grid. |
| `422` | Request body fails schema validation (e.g. `scenario_count` out of `1`-`200000`). |

---

### `GET /pf-datagen/devices`

For the New Dataset page: what the selected grids have. Query `grids`, a
comma-separated list of MV grid ids; the first 20 are summarised.

```json
{
  "grids": [
    { "mv_grid_id": "SYN-urban-0055",
      "ders": { "count": 0, "mw": 0.0, "types": {} },
      "substations": 1, "capacitor_banks": 0, "regulators": 0, "transformers": 8 }
  ],
  "n_requested": 1, "n_summarised": 1
}
```

`ders` counts the grid's own in-service DER (EV hubs excluded), with their
total rating in MW and a count per type. A grid that can't be read has
`error` instead.

### `GET /pf-datagen/generation-preview`

For the New Dataset page: the load's first days next to each generation
database over the window a run would read
(`der_profiles.py::generation_preview`).

| Query | Default | Meaning |
|---|---|---|
| `database_ids` | required | Comma-separated generation databases |
| `load_database_id` | none | The load database; its first timestamp is the load start without `load_start` |
| `load_start` | none | Start of the load window (local clock), e.g. a `window` timeline's start |
| `generation_start`, `keep_time_of_day`, `timezone` | none, `true`, `UTC` | As in `der.generation` |
| `load_timezone` | `Europe/Lisbon` | The load's clock |
| `days` | 3 | Days shown (at most 14) |
| `load_aggregation_s` | none | Average the load, and finer generation, over intervals of this length, as a run would |

The answer has `load_start`, `shift_days`, `load` (`step_s` -- the
database's own step --, `times`, `values`, `max_w`) and, per database, `generation[]`. Each entry has `data_type`,
`der_type`, `step_s`, `first`, `last`, `window_first`, `window_last`,
`times` (on the load's clock), `values`, `max_w`, `n_producing` and
`ratings_w_total` (what an individual group installs). Values are each
series' total, normalised by its maximum in the window. `422` when there is
no load database or start, or a database or time zone is unknown.

### `GET /pf-datagen/mv/{mv_grid_id}/runs`

Lists PF data-generation runs for one MV grid, most recently created
first.

**Path parameters**

| Name | Type | Description |
|---|---|---|
| `mv_grid_id` | string | The MV grid to list runs for. |

**Response** `200`

```json
[
  {
    "run_id": "d4e5f6a7",
    "mv_grid_id": "mv001",
    "status": "completed",
    "created_at": "2026-08-27T09:00:00+00:00",
    "completed_at": "2026-08-27T09:12:00+00:00",
    "error": null,
    "config": { "historical_database_id": "hist001", "scenario_count": 1000 },
    "total_scenarios": 1000,
    "completed_scenarios": 1000,
    "failed_scenarios": 0,
    "n_topology_variants": 1,
    "seed_used": 42,
    "metrics": null
  }
]
```

Returns an empty list, not `404`, for an `mv_grid_id` that doesn't exist.

---

### `GET /pf-datagen/runs`

Lists PF data-generation runs across every MV grid, most recently created
first — backs the "PF Data Generation" tab on the MV grids home page.

**Response** `200`

Same array shape as [`GET /pf-datagen/mv/{mv_grid_id}/runs`](#get-pf-datagenmvmv_grid_idruns),
unfiltered.

---

### `GET /pf-datagen/runs/{run_id}`

Returns one run's status. For a `completed` run, also includes a
`summary` computed from the stored result rows, across all samples:

- `converged` out of `total_output_groups`;
- `thermal_violations` and `voltage_violations`: branch and bus row counts;
- `thermal_violations_emergency`: branch rows over their emergency rating;
- `islanded_bus_rows`: bus rows with `energised = false`;
- `vm_pu_min` / `vm_pu_max`: voltage range over energised buses;
- `max_loading_pct`: the highest branch loading;
- `failure_reasons`: non-converged samples counted by failure category
  (e.g. `{"max_iterations": 3}`);
- `p_loss_mean` / `p_loss_max`: mean and largest per-sample active loss (W),
  and `loss_pct_mean`: the mean loss as a share of `p_slack` (%), from the
  `summary` table.

The voltage range and loading are `null`, and `failure_reasons` is empty,
for runs written before those columns existed; the losses are `null` for
runs from before the summary table.

**Path parameters**

| Name | Type | Description |
|---|---|---|
| `run_id` | string | The run to fetch. |

**Response** `200`

```json
{
  "run_id": "d4e5f6a7",
  "mv_grid_id": "mv001",
  "status": "completed",
  "created_at": "2026-08-27T09:00:00+00:00",
  "completed_at": "2026-08-27T09:12:00+00:00",
  "error": null,
  "config": { "historical_database_id": "hist001", "scenario_count": 1000 },
  "total_scenarios": 1000,
  "completed_scenarios": 1000,
  "failed_scenarios": 0,
  "n_topology_variants": 1,
  "seed_used": 42,
  "metrics": null,
  "summary": {
    "total_output_groups": 1000,
    "converged": 987,
    "thermal_violations": 12,
    "thermal_violations_emergency": 2,
    "voltage_violations": 41,
    "islanded_bus_rows": 0,
    "vm_pu_min": 0.887,
    "vm_pu_max": 1.0,
    "max_loading_pct": 104.2,
    "failure_reasons": { "max_iterations": 13 },
    "p_loss_mean": 41250.0,
    "p_loss_max": 96800.0,
    "loss_pct_mean": 1.9
  }
}
```

**Errors**

| Status | Cause |
|---|---|
| `404` | No run with this ID exists. |

---

### `GET /pf-datagen/runs/{run_id}/report`

The run's **diversity report**, accumulated while the run solves and stored
as `report.json` in its run directory. It describes the whole run, not a
preview. The run's page (`/pf-datagen/ui/runs/{run_id}`) draws it.

| Key | Content |
|---|---|
| `version` | Report format version (5) |
| `outcomes`, `failure_reasons` | Total / converged / failed samples; failures per category |
| `voltage_pu` | Histogram of energised bus voltages: `edges` (0.80–1.20 p.u., 0.005 steps), `counts`, plus `below` / `above` the edges |
| `loading_pct` | Histogram of in-service branch loading (0–200 %, 5 % steps), same shape |
| `per_sample` | Percentiles (p0, p5, p25, p50, p75, p95, p100), mean and count over converged samples: `vm_pu_min`, `vm_pu_max`, `max_loading_pct`, `p_load_w`, `q_load_var`, `loss_pct` |
| `load` | Same statistics over load samples (before the topology variants): `gross_demand_w`, `load_pf` (load-weighted power factor), `spatial_cv` (coefficient of variation of the load magnitude across load buses) |
| `load_level` | With a load level only: the band `edges`, `samples` and `converged` per band, `undervoltage_rate` / `overvoltage_rate` / `thermal_rate` (share of converged samples in the band with that violation, `null` for an empty band), `unscaled_samples` |
| `loadability` | When P_max was searched (`loadability` level or generated DER): `p_max_w`, `p_max_scale` (P_max relative to the average pattern), `p_max_limit`, `p_max_solves`; otherwise `null` |
| `der_penetration` | With generated DER only: band `edges`, `samples` and `converged` for the **installed** penetration (all in-service DER capacity ÷ P_max — with `add`, the grid's own DER count). The groups fix it, so there is one band, at its value. Also `reverse_flow_rate` (power flowing back through the substation), `overvoltage_rate`, `thermal_rate` |
| `der_load_ratio` | Percentiles of DER generation ÷ gross load (positive demand of energised buses), per converged sample |
| `setpoints` | Percentiles of the busbar set-point (`v_slack_pu`, mean over substations) and of the capacitor steps switched in (`shunt_steps_in`); `regulator_taps`: per regulator, how many samples ended at each tap |
| `topology` | Shown in the table below |
| `outages` | With per-sample outages only: percentiles of `ders_tripped`, `shunts_out` and `regulators_bypassed` per sample |
| `parameters` | With temperature or swaps only: percentiles of `conductor_temp_c` and `cable_swaps` per sample |
| `load.near_duplicate_share` | Share of load samples whose injection vector matches an earlier one's to about 1 % in direction and magnitude (approximate: quantised and hashed) |
| `design` | With a drawn axis: `mode`, `axes` (name, range, bins), `max_rank_correlation` between two axes, `samples_per_stratum`, `n_strata`, `strata_filled` |
| `timeline` | `mode`, the number `n` of timestamps, `first` and `last`, the typical step `step_s` (seconds) and `n_gaps`. Under `window`, `per_sample_axes` lists the settings that still draw per sample |
| `measurements` | With the measurement view only: `meters` per reason, `readings`, `pseudo_readings`, `dead_readings`, the realised `missing_rate` and `outlier_rate` (of meter readings), `relative_error` per profile (p10 / p50 / p90 of the relative error of live readings, outliers excluded, voltage `v` apart from P / Q / current `pqi`) and `pseudo_relative_error` |
| `target_mix` | With a target mix: the settings, `achieved_share` of violating converged samples, `rounds` (tilt, expected share and whether the target was reachable, per round), and the last round's `rate` and `outputs` per stratum, `weights` and `last_round_samples_per_stratum` |

The `topology` key holds:

| Field | Content |
|---|---|
| `n_variants`, `n_configurations`, `radial_configurations` | How many variants and configurations the run has, and how many radial configurations the grid allows (`null` without reconfiguration) |
| `variants_by_kind` | Number of variants of each kind |
| `distinct_topologies` | Distinct energised networks among the variants |
| `nodes_per_substation` | Percentiles of the nodes each substation feeds, over variants and substations |
| `outcomes_by_kind` | Per variant kind: `outputs`, `converged`, and the under-voltage, over-voltage and thermal rates |
| `islanded_load_share` | Percentiles of the share of gross load each sample's variant islands |
| `outputs_with_islanded_load` | Samples whose variant islands some load |

The run's `metrics` (`GET /pf-datagen/runs/{run_id}`) also carry
`load_level_mode` and, in `loadability` mode, the same `p_max_*` values.

**Errors**

| Status | Cause |
|---|---|
| `404` | No such run, or no report yet: the run is still going, failed, or ran before reports existed. |

---

### `GET /pf-datagen/runs/{run_id}/splits`

A leakage-safe train / validation / test split of the run's samples (see
[Design, target mix, groups and splits](#design-target-mix-groups-and-splits)).
It is computed from the runtime table on request; nothing is stored.

**Query parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `by` | `"time"` \| `"topology"` | `time` | Whole days, or whole configurations / topology variants |
| `train`, `val` | float | 0.7, 0.15 | Shares of samples; test gets the rest |
| `seed` | int | 0 | Shuffles the units |
| `format` | `"json"` \| `"csv"` | `json` | CSV gives one row per sample |

**Response** `200` (JSON)

```json
{
  "by": "time", "unit": "day", "ratios": { "train": 0.7, "val": 0.15, "test": 0.15 }, "seed": 0,
  "counts": { "train": 252, "val": 54, "test": 54 },
  "units": { "train": ["2023-03-31", "..."], "val": ["..."], "test": ["..."] },
  "note": null,
  "samples": [ { "scenario": 0, "load_scenario_idx": 0, "topology_variant_idx": 0,
                 "scenario_timestamp": "2023-03-31 10:00:00", "split": "train" } ]
}
```

`note` warns when there are fewer than three units.

**Errors**

| Status | Cause |
|---|---|
| `404` | No run with this ID exists. |
| `422` | Bad ratios, or the run has no samples. |

---

### `POST /pf-datagen/group/run`

Starts one run per grid, sharing a `group_id`.

**Request body**

```json
{
  "mv_grid_ids": ["53_MVFeeder0641", "93_MVFeeder1430"],
  "request": { "historical_database_id": "feeder_measurements", "scenario_count": 1000, "seed": 5 },
  "min_nodes": 10,
  "min_connection_points": 3
}
```

`mv_grid_ids` may be `"all"`. `request` is the
[run request](#post-pf-datagenmvmv_grid_idrun); each grid's seed is derived
from its `seed` (or a fresh one) and the grid's position in the group.

**Response** `200`

```json
{
  "group_id": "1280f7e4",
  "runs": [ { "mv_grid_id": "53_MVFeeder0641", "run_id": "f5a283e8" } ],
  "skipped": [ { "mv_grid_id": "MV_TINY", "reason": "5 nodes (fewer than 10)" } ],
  "message": "Started 2 run(s) as group '1280f7e4'."
}
```

**Errors**

| Status | Cause |
|---|---|
| `404` | A listed grid doesn't exist. |
| `422` | No grid passes the size filter (`detail.skipped` says why), or the request fails validation. |

---

### `GET /pf-datagen/groups`

Lists groups, newest first: `group_id`, `created_at`, `n_runs`, `n_skipped`.

---

### `GET /pf-datagen/groups/{group_id}`

The group's runs, each with its grid, size, seed, status and progress, plus
the skipped grids and the overall `progress` (`solved` / `total` samples).
Returns `404` for an unknown group.

---

### `GET /pf-datagen/groups/{group_id}/report`

The group's diversity report:

| Key | Content |
|---|---|
| `status_counts`, `complete`, `skipped` | Runs per status, whether every run completed, grids left out |
| `outcomes`, `failure_reasons` | Summed over the completed runs |
| `voltage_pu`, `loading_pct` | The runs' histograms, summed |
| `per_sample` | Percentiles over every completed run's summary rows: `vm_pu_min`, `vm_pu_max`, `max_loading_pct`, `p_load_w`, `q_load_var`, `loss_pct` |
| `grids` | Per grid: run, status, nodes, connection points, outputs, converged samples, under-voltage / over-voltage / thermal rates, P_max |

Returns `404` for an unknown group.

---

### `GET /pf-datagen/groups/{group_id}/splits`

Grids to train / validation / test (`train`, `val`, `seed` as for a run).
Grids are sorted by node count, cut into blocks of about
1 / min(val, test) grids, and each block is split by the ratios, so every
split gets small and large grids. Returns `grids` (grid → split), `counts`
(grids and samples per split), `node_counts` and `runs` (grid → run).

---

### `GET /pf-datagen/runs/{run_id}/results`

Returns a page of raw result rows from one of the run's output tables.

**Path parameters**

| Name | Type | Description |
|---|---|---|
| `run_id` | string | The run to fetch results for. |

**Query parameters**

| Name | Type | Required | Default | Description |
|---|---|---|---|---|
| `table` | `"bus"` \| `"branch"` \| `"ybus"` \| `"runtime"` \| `"summary"` \| `"der"` \| `"variants"` \| `"measurements"` \| `"layout"` | yes | — | Which output table to read. |
| `limit` | integer | no | `500` | Max rows to return (1-5000). |
| `offset` | integer | no | `0` | Row offset for pagination. |

**Response** `200`

```json
{
  "run_id": "d4e5f6a7",
  "table": "bus",
  "columns": ["scenario", "load_scenario_idx", "output_group_id", "scenario_timestamp", "topology_variant_idx",
              "node_id", "energised", "pd", "qd", "vm", "va", "bus_type", "pd_pu", "qd_pu", "vm_pu",
              "vm_min", "vm_max", "v_violation", "vm_lin", "vm_lin_pu"],
  "rows": [
    {
      "scenario": 0,
      "load_scenario_idx": 0,
      "output_group_id": "d4e5f6a7-2024-08-20 14:00:00-0",
      "scenario_timestamp": "2024-08-20 14:00:00",
      "topology_variant_idx": 0,
      "node_id": "B1",
      "energised": true,
      "pd": 120000.0, "qd": 30000.0, "vm": 14850.0, "va": -0.4, "bus_type": "PQ",
      "pd_pu": 0.12, "qd_pu": 0.03, "vm_pu": 0.99,
      "vm_min": 0.9, "vm_max": 1.1, "v_violation": false,
      "vm_lin": 14855.0, "vm_lin_pu": 0.9903
    }
  ]
}
```

`rows` are objects keyed by `columns`, which mirror the underlying
`MVPFScenario*` table for the requested `table` (`measurements` and `layout`:
`MVPFMeasurement` and `MVPFMeasurementLayout`, see
[Measurement view and timeline](#measurement-view-and-timeline); see also
[Units](#units-and-three-phase-convention) and
[Data Model › PF Data Generation](../database-schema.md#pf-data-generation)).

**Errors**

| Status | Cause |
|---|---|
| `404` | No run with this ID exists. |
| `400` | The results query failed (response `detail` is generic; specifics are logged server-side). |

---

### `GET /pf-datagen/runs/{run_id}/export.csv`

Streams every row of one output table for a run as a CSV file download.
Unlike `/results`, this is not paginated — it returns the full table for
the run.

**Path parameters**

| Name | Type | Description |
|---|---|---|
| `run_id` | string | The run to export results for. |

**Query parameters**

| Name | Type | Required | Default | Description |
|---|---|---|---|---|
| `table` | `"bus"` \| `"branch"` \| `"ybus"` \| `"runtime"` \| `"summary"` \| `"der"` \| `"variants"` \| `"measurements"` \| `"layout"` | yes | — | Which output table to export. List columns (`disabled_connections`, `islanded_nodes`) are joined with `;` in the CSV. |

**Response** `200`

`text/csv` body with `Content-Disposition: attachment; filename="pf_datagen_<run_id>_<table>.csv"`.

**Errors**

| Status | Cause |
|---|---|
| `404` | No run with this ID exists. |
| `400` | The export query failed (response `detail` is generic; specifics are logged server-side). |

---

### `POST /pf-datagen/mv/{mv_grid_id}/package`

Builds a **standalone package** of a run, to generate the dataset outside
GridArena. It does not start a run. It does the part of a run that needs the
database (the grid, the cable catalogue, and the load scenarios built from
the historical series) and returns a zip with:
- those inputs,
- a copy of the solver code (no web app, no database code),
- `run.py`,
- `how_to_run.md`.

Unzipped anywhere with Python 3.10+ and `requirements.txt` installed,
`python run.py` writes one file per table. The columns are those of
[`export.csv`](#get-pf-datagenrunsrun_idexportcsv). The rows are the ones a
run of the same request would store, up to floating-point rounding on
another platform. `--format parquet` writes Parquet, with column types from
the database, and needs pyarrow.

The New Dataset page's **Download standalone package** button sends the
reviewed request here (to
[`/group/package`](#post-pf-datagengrouppackage) for several grids).

**Request body**: the [run request](#post-pf-datagenmvmv_grid_idrun). An
omitted `seed` is drawn and fixed in the package.

**Response** `200`: `application/zip`, with
`Content-Disposition: attachment; filename="gridarena_dataset_<package_id>.zip"`.

```
gridarena_dataset_<package_id>/
    how_to_run.md, run.py, requirements.txt, package.json, column_types.json
    gridarena/                      solver code (pf_datagen + full_pf + the noise module)
    runs/<grid>/run_config.json     the request, seed fixed
    runs/<grid>/inputs/             grid.json, catalogue.json, scenarios.npz, manifest.json
```

`run.py` puts each grid's results in `results/<grid>/`: the nine tables,
`report.json`, and `run_config.json` with status and metrics. Options:
`--format csv|parquet`, `--grid <mv_grid_id>` (repeatable) and `--out <dir>`.
It exits with code 1 if any grid failed.

Most settings in `runs/<grid>/run_config.json` can be edited before
running. The exceptions are the fields the stored scenarios were built
from, and `run.py` refuses to run if any of them changed:
- `historical_database_id`, `scenario_count`, `load_noise_sigma`,
- `reassignment_period_timesteps`, `load_shape`,
- `timeline`, `timeline_start`, `load_aggregation_s`, `seed`.

**Errors**

| Status | Cause |
|---|---|
| `404` | No MV grid with this ID exists. |
| `400` | The inputs can't be built; `detail` says why (e.g. the historical database has no usable series, or the grid has no connected LV grids for an `LV` source). |
| `422` | The request fails validation. |

---

### `POST /pf-datagen/group/package`

The standalone package of a [group](#post-pf-datagengrouprun): the same
body and size filter, with one `runs/<grid>/` folder per grid that passes
it. Each grid's seed is derived from the group seed and its position, as in
a group run. `package.json` lists the runs, the group seed and any grids
the filter skipped.

**Response** `200`: `application/zip`, as above.

**Errors**: as for [`/group/run`](#post-pf-datagengrouprun), plus `400` when
a grid's inputs can't be built.
