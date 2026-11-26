# MV Grid

Endpoints for registering, validating, reading, and deleting MV (medium-voltage)
grids, and for bridging a registered LV grid to an MV connection point via a
transformer.

**Base path:** `/mv_grid`

Also serves a browser UI at `/mv_grid/ui` and `/mv_grid/ui/{mv_grid_id}`.

## Endpoints

### `POST /mv_grid/validate`

Validates an uploaded MV grid JSON file without saving it: parses it against
the `MVGrid` schema, then checks internal references (connections pointing
at nodes/cables that don't exist, connection points pointing at nodes that
don't exist or duplicated, transformers pointing at connection points that
don't exist or whose `primary_voltage_kv` doesn't match the grid's
`nominal_voltage_kv`). Like `/grid/validate`, reference errors come back as
a `200` with `valid: false`, not an HTTP error.

**Request**

| Name | Type | Description |
|---|---|---|
| `file` (multipart) | file, required | An MV grid JSON file matching the `MVGrid` schema (see below). |

Uploaded file content:

```json
{
  "mv_grid_id": "mv001",
  "name": "Feeder 12",
  "nominal_voltage_kv": 20.0,
  "region": "North",
  "nodes": [
    { "node_id": "MVN1", "coord_lat": 41.15, "coord_lon": -8.61, "node_type": "substation" }
  ],
  "cables": [
    { "cable_id": "MVCBL1", "imp_real": 0.12, "imp_imag": 0.09, "nom_curr": 400.0,
      "cap_uf_per_km": 0.3, "emerg_curr": 480.0 }
  ],
  "connections": [
    { "connection_id": "MVC1", "from_node_id": "MVN1", "to_node_id": "MVN2", "cable_id": "MVCBL1", "length": 500.0 },
    { "connection_id": "TIE1", "from_node_id": "MVN2", "to_node_id": "MVN7", "cable_id": "MVCBL1", "length": 300.0,
      "is_switch": true, "normally_open": true }
  ],
  "connection_points": [
    { "connection_point_id": "CP1", "node_id": "MVN1", "name": "Substation A" }
  ],
  "transformers": [
    {
      "transformer_id": "TR1", "connection_point_id": "CP1", "lv_grid_id": "grid001",
      "rated_power_kva": 400.0, "primary_voltage_kv": 20.0, "secondary_voltage_v": 230.0,
      "imp_real": 0.01, "imp_imag": 0.04, "tap_pos": 1, "tap_step_pct": 2.5, "tap_min": -2, "tap_max": 2
    }
  ],
  "regulators": [
    { "regulator_id": "REG1", "connection_id": "MVC1", "v_set_pu": 1.0, "deadband_pu": 0.02,
      "tap_step_pct": 0.625, "tap_min": -16, "tap_max": 16, "tap_pos": 0 }
  ],
  "shunts": [
    { "shunt_id": "CB1", "node_id": "MVN2", "q_step_kvar": 300.0, "n_steps": 2, "step": 1 }
  ],
  "ders": [
    { "der_id": "PV1", "node_id": "MVN2", "type": "pv", "s_rated_kva": 2000.0, "control_mode": "q_v" }
  ]
}
```

Cable fields, all per phase:

| Field | Unit | Required | Meaning |
|---|---|---|---|
| `imp_real`, `imp_imag` | Ω/km | yes | Series resistance and reactance |
| `nom_curr` | A | yes | Continuous current rating |
| `cap_uf_per_km` | µF/km | no | Shunt (charging) capacitance at 50 Hz, must be ≥ 0. Omit it to model the cable without charging. [PF data generation](pf-datagen.md) uses it as a pi model. |
| `emerg_curr` | A | no | Short-term emergency rating, must be > 0. Omit it to skip the emergency-rating check in PF data generation. |

`length` on a connection is in metres. The direction of a connection
(`from_node_id` / `to_node_id`) doesn't matter for power flow, and neither
does the order connections are listed in. The one exception is a connection
with a regulator, which must point away from the substation (see below).

Connections also take three optional fields:

| Field | Default | Meaning |
|---|---|---|
| `status` | `"active"` | Any other value takes the connection out of service. |
| `is_switch` | `false` | The connection can be opened and closed. |
| `normally_open` | `false` | Open in the base topology; needs `is_switch`. A tie line between two feeders is a normally-open switch. |

A grid can have **several substations** (nodes with `node_type`
`"substation"`); each is a source in [PF data generation](pf-datagen.md).
The MV grid page draws open connections dashed.

Transformer fields (MV/LV transformer at a connection point):

| Field | Unit | Required | Meaning |
|---|---|---|---|
| `rated_power_kva` | kVA | yes | Rating |
| `primary_voltage_kv` | kV | yes | MV side, line-to-line; must equal the grid's `nominal_voltage_kv` |
| `secondary_voltage_v` | V | no (230) | LV side, **phase-to-neutral** |
| `imp_real`, `imp_imag` | p.u. | no | Series resistance and reactance, in p.u. on the transformer's own rating. Each missing value takes a typical default: short-circuit impedance 4 % up to 630 kVA or 6 % above, with X/R = 5. |
| `status` | — | no (`"active"`) | Any other value means the transformer is out of service. |
| `tap_pos` | — | no (0) | Fixed off-load tap position. A **positive position raises the LV-side voltage**. Must lie within [`tap_min`, `tap_max`]. |
| `tap_step_pct` | % | no (2.5) | Voltage change per tap step, > 0 and ≤ 10 |
| `tap_min`, `tap_max` | — | no (−2, +2) | Tap range |

Regulator fields (optional `regulators` list: a step voltage regulator on a
connection, at most one per connection):

| Field | Unit | Default | Meaning |
|---|---|---|---|
| `regulator_id`, `connection_id` | — | required | The regulator and the connection it sits on. `connection_id` must exist. |
| `v_set_pu` | p.u. | 1.0 | Target voltage at the connection's to node, in (0.8, 1.2) |
| `deadband_pu` | p.u. | 0.02 | Width of the band around the set-point, in which the tap doesn't move |
| `tap_step_pct` | % | 0.625 | Voltage change per step; positive positions raise the to-side voltage |
| `tap_min`, `tap_max` | — | −16, +16 | Tap range (the defaults give ±10 % in 32 steps) |
| `tap_pos` | — | 0 | Starting position, within the range |

The regulator sits at the connection's **to** end, so the connection must
point away from the substation. In [PF data generation](pf-datagen.md#transformers-and-voltage-regulators),
each sample moves its tap until the to-node voltage is inside the band, or
the tap reaches a limit.

Shunt fields (optional `shunts` list: capacitor banks and shunt reactors at
MV nodes):

| Field | Unit | Default | Meaning |
|---|---|---|---|
| `shunt_id`, `node_id` | — | required | The element and its MV node. `node_id` must exist; `shunt_id` must be unique. |
| `q_step_kvar` | kvar | required | Three-phase reactive power per step at nominal voltage. **Positive = capacitor bank** (generates), negative = reactor; non-zero. |
| `n_steps` | — | 1 | Number of switchable steps, ≥ 1 |
| `step` | — | 1 | Steps switched in, between 0 and `n_steps` |
| `status` | — | `"active"` | Any other value means the shunt is out of service. |
| `name` | — | — | Optional label |

[PF data generation](pf-datagen.md) models a shunt as a constant admittance
generating `step × q_step_kvar` at 1 p.u., so its reactive power scales with
V². The step is fixed for every sample.

DER fields (optional `ders` list: generators, batteries and EV hubs at MV
nodes; generator convention, positive = injected):

| Field | Unit | Default | Meaning |
|---|---|---|---|
| `der_id`, `node_id` | — | required | The unit and its MV node. `node_id` must exist and not be the substation; `der_id` must be unique. |
| `type` | — | required | `pv`, `wind`, `chp`, `battery`, `ev_hub` or `other` |
| `s_rated_kva` | kVA | required | Rating, > 0 |
| `p_max_kw` | kW | `s_rated_kva` | Maximum active power the profile scales (e.g. a PV array's peak) |
| `profile` | — | `solar` for `pv`, else `constant` | `solar`: clear-sky PV output at the node's coordinates (the node needs `coord_lat`/`coord_lon`) times a daily clearness factor; `constant`: `profile_value × p_max_kw` |
| `profile_value` | — | 1.0 | Share of `p_max_kw` for a constant profile, in [−1, 1]; negative consumes |
| `control_mode` | — | `fixed_pf` | `fixed_pf`, `q_v` (Q(V) droop), `p_v` (P(V) curtailment) or `voltage` (holds the node voltage) |
| `cos_phi`, `excitation` | —, — | 1.0, `under` | Power factor for `fixed_pf` and `p_v`; `under` absorbs reactive power, `over` injects it |
| `q_limit_pu` | p.u. of `s_rated_kva` | 0.33 | Reactive-power limit outside `fixed_pf` (0.33 ≈ cos φ 0.95 at full output) |
| `qv_curve` | p.u. | [0.95, 0.99, 1.01, 1.05] | Q(V) breakpoints v1 < v2 ≤ v3 < v4 (VDE-AR-N 4110 style) |
| `pv_v_start`, `pv_v_end` | p.u. | 1.06, 1.09 | P(V): full output up to the start, zero from the end |
| `v_set_pu` | p.u. | 1.02 | Set-point for `voltage` mode, in [0.9, 1.1] |
| `status` | — | `"active"` | Any other value means the unit is out of service. |

See [PF data generation › Generators](pf-datagen.md#generators-der) for how
profiles and controls are solved.

**Response** `200`

```json
{
  "valid": true,
  "mv_grid": {
    "mv_grid_id": "mv001",
    "name": "Feeder 12",
    "description": null,
    "nominal_voltage_kv": 20.0,
    "region": "North",
    "nodes": [{ "node_id": "MVN1", "coord_lat": 41.15, "coord_lon": -8.61, "node_type": "substation" }],
    "cables": [{ "cable_id": "MVCBL1", "imp_real": 0.12, "imp_imag": 0.09, "nom_curr": 400.0, "cap_uf_per_km": 0.3, "emerg_curr": 480.0 }],
    "connections": [{ "connection_id": "MVC1", "from_node_id": "MVN1", "to_node_id": "MVN2", "cable_id": "MVCBL1", "length": 500.0 }],
    "connection_points": [{ "connection_point_id": "CP1", "node_id": "MVN1", "name": "Substation A" }],
    "transformers": [{ "transformer_id": "TR1", "connection_point_id": "CP1", "lv_grid_id": "grid001", "rated_power_kva": 400.0, "primary_voltage_kv": 20.0, "secondary_voltage_v": 230.0, "imp_real": null, "imp_imag": null, "name": null, "status": "active", "tap_pos": null, "tap_step_pct": null, "tap_min": null, "tap_max": null }],
    "regulators": null,
    "shunts": null,
    "ders": null
  },
  "graph": {
    "nodes": [{ "id": "MVN1", "x": -8.61, "y": 41.15, "type": "connection_point", "connection_point_id": "CP1", "connected_lv": ["grid001"] }],
    "edges": [{ "from": "MVN1", "to": "MVN2" }]
  }
}
```

**Errors**

| Status | Cause |
|---|---|
| `400` | Uploaded file is not valid JSON. |
| `422` | File content doesn't conform to the `MVGrid` schema. |

---

### `POST /mv_grid/`

Registers a new MV grid: uploads an MV grid JSON file (same shape as
`/mv_grid/validate`), validates schema and internal references, and
persists the grid, its nodes, cables, connections, connection points,
transformers, regulators, shunts and DER.

**Request**

| Name | Type | Description |
|---|---|---|
| `file` (multipart) | file, required | An MV grid JSON file matching the `MVGrid` schema. |

The upload is subject to the shared [200 MB upload cap](index.md#resource-safety-limits).

**Response** `200`

```json
{ "message": "MV Grid 'mv001' registered successfully." }
```

**Errors**

| Status | Cause |
|---|---|
| `413` | Uploaded file exceeds the [200 MB upload cap](index.md#resource-safety-limits). |
| `400` | Uploaded file is not valid JSON. |
| `422` | Schema validation failed, or reference validation failed (response `detail` is `{"validation_errors": [...]}`). |
| `500` | Database error while inserting the MV grid (e.g. duplicate `mv_grid_id`). |

---

### `GET /mv_grid/list`

Lists all registered MV grids with summary fields.

**Response** `200`

```json
[
  { "mv_grid_id": "mv001", "name": "Feeder 12", "nominal_voltage_kv": 20.0, "region": "North" }
]
```

**Errors**

| Status | Cause |
|---|---|
| `500` | Database error while querying. |

---

### `GET /mv_grid/data/{mv_grid_id}`

Retrieves the full MV grid: header fields plus nodes, cables, connections,
connection points, transformers, regulators, shunts, and DER.

**Path parameters**

| Name | Type | Description |
|---|---|---|
| `mv_grid_id` | string | Unique identifier of the MV grid. |

**Response** `200`

```json
{
  "mv_grid_id": "mv001",
  "name": "Feeder 12",
  "description": null,
  "nominal_voltage_kv": 20.0,
  "region": "North",
  "nodes": [{ "node_id": "MVN1", "coord_lat": 41.15, "coord_lon": -8.61, "node_type": "substation" }],
  "cables": [{ "cable_id": "MVCBL1", "imp_real": 0.12, "imp_imag": 0.09, "nom_curr": 400.0, "cap_uf_per_km": 0.3, "emerg_curr": 480.0 }],
  "connections": [{ "connection_id": "MVC1", "from_node_id": "MVN1", "to_node_id": "MVN2", "cable_id": "MVCBL1", "length": 500.0 }],
  "connection_points": [{ "connection_point_id": "CP1", "node_id": "MVN1", "name": "Substation A" }],
  "transformers": [{ "transformer_id": "TR1", "connection_point_id": "CP1", "lv_grid_id": "grid001", "rated_power_kva": 400.0, "primary_voltage_kv": 20.0, "secondary_voltage_v": 230.0, "imp_real": null, "imp_imag": null, "name": null, "status": "active", "tap_pos": null, "tap_step_pct": null, "tap_min": null, "tap_max": null }],
  "regulators": [{ "regulator_id": "REG1", "connection_id": "MVC1", "v_set_pu": 1.0, "deadband_pu": 0.02, "tap_step_pct": 0.625, "tap_min": -16, "tap_max": 16, "tap_pos": 0 }],
  "shunts": [{ "shunt_id": "CB1", "node_id": "MVN2", "q_step_kvar": 300.0, "n_steps": 2, "step": 1, "status": "active", "name": null }],
  "ders": [{ "der_id": "PV1", "node_id": "MVN2", "type": "pv", "s_rated_kva": 2000.0, "p_max_kw": 2000.0, "profile": "solar", "profile_value": 1.0, "control_mode": "q_v", "cos_phi": 1.0, "excitation": "under", "q_limit_pu": 0.33, "qv_curve": [0.95, 0.99, 1.01, 1.05], "pv_v_start": 1.06, "pv_v_end": 1.09, "v_set_pu": 1.02, "status": "active", "name": null }]
}
```

Cables written before `cap_uf_per_km` / `emerg_curr` existed come back with
those fields `null`.
Likewise, transformer tap fields are `null` when they weren't given (meaning
the defaults: position 0, 2.5 % steps, −2..+2).

**Errors**

| Status | Cause |
|---|---|
| `404` | No MV grid with this `mv_grid_id` exists. |
| `500` | Database error while querying. |

---

### `DELETE /mv_grid/{mv_grid_id}`

Deletes an MV grid and its associated nodes, cables, connections,
connection points, and transformers.

!!! warning "Destructive and irreversible"
    Note that deleting an MV grid does **not** delete any LV grid connected
    to it — per [Data Model › MV grids](../database-schema.md#mv-grids),
    `MVTransformer.lv_grid_id` is `ON DELETE SET NULL` in the other
    direction (deleting the *LV* grid unlinks the transformer), and deleting
    the MV grid itself simply removes its own rows including that
    transformer.

**Path parameters**

| Name | Type | Description |
|---|---|---|
| `mv_grid_id` | string | Unique identifier of the MV grid to delete. |

**Response** `200`

```json
{ "message": "MV Grid 'mv001' and all associated data deleted." }
```

**Errors**

| Status | Cause |
|---|---|
| `404` | No MV grid with this `mv_grid_id` exists. |
| `500` | Database error during deletion. |

---

### `PATCH /mv_grid/{mv_grid_id}/configuration`

Changes the operating settings of a grid's components without re-uploading
it. This is what the **Current configuration** box on the grid's page sends
when you press **Update**. Each edit names one element by its ID and lists
only the fields that change; omitted fields keep their stored value. All
lists are optional.

| List | ID field | Editable fields |
|---|---|---|
| `connections` | `connection_id` | `status` (`active` / `inactive`), `normally_open` (switches only) |
| `transformers` | `transformer_id` | `status`, `tap_pos` |
| `regulators` | `regulator_id` | `v_set_pu`, `deadband_pu`, `tap_pos` (starting position) |
| `shunts` | `shunt_id` | `step`, `status` |
| `ders` | `der_id` | `status`, `control_mode`, `cos_phi`, `excitation`, `profile_value`, `v_set_pu`, `q_limit_pu` |

Every edited element is merged with its stored values and checked against
the same rules as an upload: tap within its range, `step` ≤ `n_steps`,
`normally_open` only on a switch, DER limits, and so on. If any edit is
invalid, nothing is written. Valid edits are written in one transaction.
Edits that match the stored value are skipped.

**Request body**

```json
{
  "connections": [{"connection_id": "11_TIE01", "normally_open": false}],
  "shunts": [{"shunt_id": "CB01", "step": 2}],
  "ders": [{"der_id": "WT01", "control_mode": "q_v"}]
}
```

**Response** `200`

```json
{ "message": "Updated 3 element(s) of MV grid '11_MVFeeder0725_P1'.", "updated": 3 }
```

**Errors**

| Status | Cause |
|---|---|
| `404` | No MV grid with this `mv_grid_id` exists. |
| `422` | Unknown field or invalid value in the body, an element ID not in the grid, or an edit that breaks the element's rules (`detail.validation_errors` lists them). |
| `500` | Database error while updating. |

---

### `GET /mv_grid/{mv_grid_id}/tie-suggestions`

Proposes normally-open **tie lines** for the grid, giving
[reconfiguration](pf-datagen.md#reconfiguration-outages-and-parameters)
something to switch. Nothing is stored: `POST` the ties to keep to
[`/mv_grid/{mv_grid_id}/ties`](#post-mv_gridmv_grid_idties). The grid page's
**Suggest tie lines** button calls it, draws the proposals dashed on the graph
and adds the ones you tick.

**Which pairs qualify:**

- **Nodes:** MV nodes with coordinates, not substations, and not already
  connected.
- **Distance:** less than `max_distance_m` apart in a straight line, and
  either more than `min_cable_distance_m` apart along the cables or on
  different feeders. A feeder is the part of the base topology below one
  cable leaving a substation.
- **Existing ties:** cable distances run over every active connection, open
  ties included, so a pair that an existing tie already shortcuts isn't
  proposed.

**How ties are picked:**

- **Ranking:** by cable distance ÷ straight-line distance.
- **Spread:** a candidate is skipped if either end lies within
  `max_distance_m` of an end of a tie already picked.
- **Tie values:** each tie gets the grid's most common cable type and a
  length of 1.2 × the straight-line distance.

**Query parameters**

| Name | Type | Description |
|---|---|---|
| `max_distance_m` | float | Largest straight-line distance between the ends (default 500, at most 5000) |
| `min_cable_distance_m` | float | Smallest distance along the cables, unless the ends are on different feeders (default 2000) |
| `max_ties` | int | At most this many ties (default 5, 1–50) |

**Response** `200`

```json
{
  "suggestions": [
    {
      "connection_id": "TIE01", "from_node_id": "53_MVLV38757", "to_node_id": "53_MVLV76572",
      "distance_m": 339.8, "cable_distance_m": 5462.0, "reason": "distant_along_cables",
      "cable_id": "53_101203", "length": 407.8
    }
  ],
  "n_candidates": 13,
  "cable_id": "53_101203"
}
```

`reason` is `distant_along_cables` or `different_feeders`.
`cable_distance_m` is `null` when the two nodes aren't joined by cables at
all.

**Errors**

| Status | Cause |
|---|---|
| `404` | The MV grid doesn't exist. |

---

### `POST /mv_grid/{mv_grid_id}/ties`

Stores tie lines as active, normally-open switches. They then appear in the
grid page's configuration box like any switch. All ties are validated first,
and nothing is written if any is invalid.

**Request body**

```json
{ "ties": [ { "connection_id": "TIE01", "from_node_id": "N12", "to_node_id": "N40",
              "cable_id": "CBL_240AL", "length": 407.8 } ] }
```

**Response** `200`

```json
{ "message": "Added 1 tie line(s) to MV grid 'mv001'.", "added": 1 }
```

**Errors**

| Status | Cause |
|---|---|
| `404` | The MV grid doesn't exist. |
| `422` | `validation_errors` lists every problem found: an unknown node or cable, both ends on the same node, a connection id already used, or two nodes already connected. |

---

### `POST /mv_grid/synthetic`

Generates **synthetic MV grids** and stores them as ordinary grids
(`gridarena/synthetic/`). They go where the stored grids don't reach:
other sizes, several feeders, a second substation, ties.

**Calibration.** On every request the generator is calibrated on the stored
grids (synthetic ones excluded), grouped by archetype. The archetype is the
overhead share of cable length: `urban` below 10 %, `rural` above 50 %,
`mixed` between. A cable is overhead if it has less than 0.05 µF/km of
charging capacitance. What it fits per archetype:

- **Node count:** log-normal, so sizes up to 500 nodes are possible.
- **Section length:** a typical length per grid, drawn from the stored
  grids' medians, with each section varying around it.
- **Overhead share** and **transformer ratings:** drawn from the stored
  grids.
- **Connection-point share:** the archetype's median.
- **Growth:** two probabilities, found by simulation so that the share of
  branching nodes and the depth relative to size match the stored grids.

**Shape.**

- **Growth:**
  - each feeder grows from its substation as a trunk;
  - a new node either extends the branch being built or starts a lateral off
    a node that doesn't branch yet;
  - a lateral returns to extending the trunk with the calibrated
    probability.
- **Geometry:**
  - sections keep their parent's heading within a few degrees, and laterals
    leave at 60–90°;
  - each cable is 1.2 × the straight-line distance;
  - positions are placed around the reference location.
- **Equipment:**
  - every branch end, and a calibrated share of the other nodes, is a
    connection point with one MV/LV transformer;
  - whole branches are overhead until their length reaches the grid's
    overhead share;
  - each feeder's first section and the section into each tie end are
    closed switches.
- **Cable types:** per family, the smaller cable is the stored type used by
  the most grids and the larger one the highest-rated stored type. They are
  copied as `SYN-UG-S` / `-L` and `SYN-OH-S` / `-L`.

**Planning check.** At design load (every transformer at 60 % of its rating,
power factor 0.95 lagging), every section must stay within its rating and
every node within 5 % of the busbar.

1. Overloaded sections move to their family's larger cable.
2. While a node is too low, the smaller-cable section carrying the most
   current on the path to it moves up.
3. A grid that still fails is grown again at the same size. After 5 failed
   shapes the size is drawn again, since it may be too much load for one
   feeder. After 20 attempts the request fails.

**Storage.** Grids are named `SYN-<archetype>-NNNN`, numbered after the
existing ones, with region `synthetic`. Their description holds the
generator version, seed, position in the batch and options; grid k of a
batch draws from the seed and k, so a batch reproduces. No LV grid is
attached, so use them with MV or Feeder historical databases. The
**Synthetic Grids** page of the Synthetic Data section
(`/pf-datagen/ui/synthetic-grids`) runs this endpoint and the delete.

**Request body**

```json
{
  "archetype": "rural", "count": 10, "seed": 11,
  "feeders_per_substation": 1, "second_substation": false, "ties_between_feeders": 0,
  "n_nodes_min": null, "n_nodes_max": null,
  "reference_lat": 48.85, "reference_lon": 2.2
}
```

| Field | Description |
|---|---|
| `archetype` | `urban`, `mixed`, `rural`, or `all` (archetypes in their stored proportions) |
| `count` | 1–100 grids |
| `seed` | Omit for a fresh one (returned) |
| `feeders_per_substation` | 1–4; nodes are split evenly |
| `second_substation` | Adds a substation with its own feeders, joined to the first by a normally-open tie |
| `ties_between_feeders` | 0–3 more normally-open ties, each between the closest nodes of two different feeders |
| `n_nodes_min`, `n_nodes_max` | Fix the node range (both or neither, 5–500) instead of the fit |

**Response** `200`

```json
{
  "created": ["SYN-rural-0001", "SYN-rural-0002"],
  "seed": 11,
  "message": "Generated 2 synthetic grid(s).",
  "report": {
    "generator_version": 1, "seed": 11, "count": 2, "sizes_redrawn": 0,
    "archetypes": { "rural": { "generated": 2, "metrics": {
      "n_nodes": { "generated": { "p10": 58, "p50": 99, "p90": 153 }, "stored": { "p10": 69, "p50": 133, "p90": 200 } }
    } } },
    "grids": [ { "mv_grid_id": "SYN-rural-0001", "archetype": "rural", "attempts": 1, "sizes_redrawn": 0,
                 "upgrades": 2, "max_loading": 0.71, "min_voltage_pu": 0.962, "n_ties": 0, "n_nodes": 104 } ]
  }
}
```

The report compares generated and stored grids per archetype, on node count,
section length, depth (sections), farthest node (km), share of branching
nodes, connection-point share and overhead share.

**Errors**

| Status | Cause |
|---|---|
| `422` | No stored grid of that archetype (or none at all) to calibrate on; a grid failed the planning check 20 times; or the request fails validation. |

---

### `GET /mv_grid/synthetic/calibration`

The generator's calibration: per archetype, the number of stored grids, the
fitted parameters (`log_nodes`, `log_section`, `section_spread`,
`cp_share`, `branching`, `depth_ratio`, `ratings`, `p_lateral`, `p_return`)
and the stored grids' p10 / p50 / p90 for each statistic.

---

### `DELETE /mv_grid/synthetic`

Deletes every grid whose id starts with `SYN-`, with everything that cascades
from it. PF data-generation results are kept, as for any deleted grid.

**Response** `200`: `{"deleted": 33, "message": "Deleted 33 synthetic grid(s)."}`

---

### `GET /mv_grid/lv-grids`

Lists the `grid_id` of every registered LV grid, for populating a
"connect an LV grid" picker.

**Response** `200`

```json
[{ "grid_id": "grid001" }, { "grid_id": "grid002" }]
```

**Errors**

| Status | Cause |
|---|---|
| `500` | Database error while querying. |

---

### `POST /mv_grid/{mv_grid_id}/connect/{connection_point_id}`

Connects an existing LV grid to an MV connection point by creating a new
`MVTransformer` row bridging the two.

**Path parameters**

| Name | Type | Description |
|---|---|---|
| `mv_grid_id` | string | The MV grid owning the connection point. |
| `connection_point_id` | string | The connection point to attach the LV grid to. |

**Request body**

```json
{
  "lv_grid_id": "grid001",
  "transformer_id": null,
  "rated_power_kva": 400.0,
  "primary_voltage_kv": 20.0,
  "secondary_voltage_v": 230.0,
  "imp_real": null,
  "imp_imag": null,
  "name": null
}
```

Only `lv_grid_id` is required. Defaults applied server-side when omitted:
`transformer_id` → a generated `TR-<8 hex chars>`, `rated_power_kva` →
`400.0`, `primary_voltage_kv` → `20.0`, `secondary_voltage_v` → `230.0`.

**Response** `200`

```json
{ "message": "LV grid 'grid001' connected to point 'CP1' via transformer 'TR-a1b2c3d4'." }
```

**Errors**

| Status | Cause |
|---|---|
| `400` | `connection_point_id` not found in this MV grid; `lv_grid_id` not found; or the LV grid is already connected to this connection point. |
| `422` | Request body doesn't conform to `ConnectLVRequest` (e.g. missing `lv_grid_id`). |

!!! note
    All failure modes from the underlying connect operation surface as
    `400`, including a not-found MV connection point or LV grid — there is
    no `404` from this endpoint.

---

### `DELETE /mv_grid/{mv_grid_id}/disconnect/{connection_point_id}/{lv_grid_id}`

Removes the transformer bridge between an MV connection point and an LV
grid.

**Path parameters**

| Name | Type | Description |
|---|---|---|
| `mv_grid_id` | string | The MV grid owning the connection point. |
| `connection_point_id` | string | The connection point the LV grid is attached to. |
| `lv_grid_id` | string | The LV grid to disconnect. |

**Response** `200`

```json
{ "message": "LV grid 'grid001' disconnected from point 'CP1'." }
```

**Errors**

| Status | Cause |
|---|---|
| `400` | No matching transformer connects this connection point and LV grid. |

---

### `GET /mv_grid/{mv_grid_id}/pv/{lv_grid_id}`

Returns aggregated active-power/voltage pairs for a connected LV grid — the
average `power_active`, `power_reactive`, and `voltage_magnitude` per
timestamp across all of the LV grid's nodes and phases. Represents a
per-meter load profile at the MV/LV boundary (not a summed total).

**Path parameters**

| Name | Type | Description |
|---|---|---|
| `mv_grid_id` | string | The MV grid the LV grid is connected to. |
| `lv_grid_id` | string | The connected LV grid to aggregate measurements for. |

**Query parameters**

| Name | Type | Required | Default | Description |
|---|---|---|---|---|
| `limit` | integer (1-10000) | no | `500` | Maximum number of timestamps to return. |

**Response** `200`

```json
{
  "mv_grid_id": "mv001",
  "lv_grid_id": "grid001",
  "n_nodes": 42,
  "aggregation": "avg_per_timestamp_across_all_nodes",
  "count": 500,
  "pv_pairs": [
    { "datetime": "2024-08-20 14:00:00", "avg_power_active": 3210.0, "avg_power_reactive": 550.0, "avg_voltage_magnitude": 229.8 }
  ]
}
```

**Errors**

| Status | Cause |
|---|---|
| `404` | `lv_grid_id` is not connected to `mv_grid_id` (no matching `MVTransformer` row), or no measurement data exists for the LV grid. |
| `500` | Database error while querying. |

---
