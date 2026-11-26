# Historical

Endpoints for uploading, reading, and deleting **historical databases** —
named collections of standalone power/voltage series ("historicals") that
all share one declared timeline. Unlike [Measurements](measurements.md),
none of this is tied to a specific grid or node; each series just carries a
`series_id` (unique within its database) and a `grid_level` tag (`MV`,
`LV`, or `Feeder`). Each database also has a `data_type`: `"load"`
(consumption, the default) or a kind of generation injected into the grid —
`"pv"` (solar PV), `"chp"` (cogeneration), `"wind"`, `"hydro"`,
`"other_generation"` (other technologies) or `"injection_total"` (all
generation together). Every type has exactly the same structure; the type
only marks what the power values are. See
[Data Model › Measurements and historical data](../database-schema.md#measurements-and-historical-data).

**Base path:** `/historical`

Also serves a browser UI at `/historical/ui`, `/historical/ui/{database_id}`,
and `/historical/ui/{database_id}/{series_id}`.

## Endpoints

### `POST /historical/`

Uploads and inserts a historical database: a shared timeline plus one or
more series, each carrying values that must fall on that timeline.

**Request**

| Name | Type | Description |
|---|---|---|
| `file` (multipart) | file, required | A JSON file matching the `HistoricalDatabaseUpload` schema (see below). |
| `data_type` (multipart) | `"load"` \| `"pv"` \| `"chp"` \| `"wind"` \| `"hydro"` \| `"other_generation"` \| `"injection_total"`, optional | The database type, as chosen on the Historical page. Used when the file has no `data_type`; if the file has one, the two must match. |

Uploaded file content:

```json
{
  "database_id": "db001",
  "name": "Feeder measurements 2024",
  "description": "Hourly feeder readings",
  "data_type": "load",
  "timestamps": ["2024-08-20T14:00:00", "2024-08-20T15:00:00"],
  "historicals": [
    {
      "series_id": "series1",
      "grid_level": "LV",
      "name": "Feeder A",
      "description": null,
      "values": [
        {
          "datetime": "2024-08-20T14:00:00",
          "phase": "R",
          "power_active": 3200.0,
          "power_reactive": 600.0,
          "voltage_magnitude": 229.5,
          "voltage_angle": -0.4
        }
      ]
    }
  ]
}
```

Validated at the schema level (`HistoricalDatabaseUpload`) before any
database access: at least one timestamp and one series are required,
`series_id` must be unique within the database, every value's `datetime`
must be one of the declared `timestamps`, and each value must include
`power_active` and/or `voltage_magnitude`. Power is in W / VAr (load
convention) and voltage in V. LV series may carry one row per phase. MV
and Feeder series hold three-phase totals, either as one row with `phase`
omitted or as per-phase rows that consumers sum — see
[Units](index.md#units). The upload is subject to the
shared [200 MB upload cap](index.md#resource-safety-limits); bulk loads that
legitimately exceed it go through a direct script against the database
instead of this endpoint.

`data_type` defaults to `"load"`. Every other type holds generation (for
example `"pv"`, solar PV production): `power_active` is the power generated
(positive = produced, the opposite sign of the load convention), and series
still carry a `grid_level` for where the generation is connected. Small
negative values, such as inverter standby draw at night, are accepted.
Re-uploading to an existing `database_id` must use the same `data_type` as
the database (`409` otherwise), so generation and load series are never
merged into one database.
[PF Data Generation](pf-datagen.md) uses only `"load"` databases as its load
source; generation databases drive its generated DER, one per plant group
(`der.groups`).

**Response** `200`

```json
{ "message": "Historical database 'db001' inserted successfully (1 series)." }
```

**Errors**

| Status | Cause |
|---|---|
| `413` | Uploaded file exceeds the [200 MB upload cap](index.md#resource-safety-limits). |
| `400` | Uploaded file is not valid JSON. |
| `409` | `database_id` already exists with a different `data_type`. |
| `422` | File content doesn't conform to the `HistoricalDatabaseUpload` schema — including a value whose `datetime` isn't in the declared `timestamps`, a duplicate `series_id`, a value missing both `power_active` and `voltage_magnitude`, or an unknown `data_type` — or the `data_type` field contradicts the file's. |
| `500` | Database error during insertion. |

---

### `DELETE /historical/{database_id}`

Deletes an entire historical database, including its timeline and every
series/record within it (via `ON DELETE CASCADE`).

!!! warning "Destructive and irreversible"
    This removes the database's timeline and all series/records in one call.

**Path parameters**

| Name | Type | Description |
|---|---|---|
| `database_id` | string | The database to delete. |

**Response** `200`

```json
{ "message": "1 database(s) deleted for 'db001'." }
```

!!! note
    Deleting a nonexistent `database_id` still returns `200` with
    `"0 database(s) deleted..."` rather than a `404` — the delete is
    unconditional (`rowcount` just happens to be 0).

**Errors**

| Status | Cause |
|---|---|
| `500` | Database error during deletion. |

---

### `DELETE /historical/{database_id}/{series_id}`

Deletes records for one series within a database, optionally narrowed by
phase and/or datetime range. The series and database registrations
themselves are left in place even if this empties the series' records.

**Path parameters**

| Name | Type | Description |
|---|---|---|
| `database_id` | string | The database the series belongs to. |
| `series_id` | string | The series to delete records from. |

**Query parameters**

| Name | Type | Required | Default | Description |
|---|---|---|---|---|
| `start` | string (ISO datetime) | no | — | Delete only records at or after this datetime (inclusive). |
| `end` | string (ISO datetime) | no | — | Delete only records before this datetime (exclusive). |
| `phase` | `"R"` \| `"S"` \| `"T"` | no | — | Restrict deletion to one phase; omit to delete aggregated and all-phase rows. |

**Response** `200`

```json
{ "message": "12 record(s) deleted for series 'series1' in database 'db001'." }
```

**Errors**

| Status | Cause |
|---|---|
| `400` | `start` or `end` is not a valid ISO datetime. |
| `500` | Database error during deletion. |

---

### `GET /historical/data/{database_id}/{series_id}`

Retrieves records for one series within a database, optionally filtered by
datetime range and phase.

**Path parameters**

| Name | Type | Description |
|---|---|---|
| `database_id` | string | The database the series belongs to. |
| `series_id` | string | The series to retrieve records for. |

**Query parameters**

| Name | Type | Required | Default | Description |
|---|---|---|---|---|
| `start` | string (ISO datetime) | no | — | Lower bound (inclusive). |
| `end` | string (ISO datetime) | no | — | Upper bound (exclusive). |
| `phase` | `"R"` \| `"S"` \| `"T"` | no | — | Filter to a single phase. |

**Response** `200`

```json
{
  "database_id": "db001",
  "data_type": "load",
  "series_id": "series1",
  "grid_level": "LV",
  "name": "Feeder A",
  "description": null,
  "datetime_range": { "start": null, "end": null },
  "records": [
    {
      "datetime": "2024-08-20T14:00:00",
      "phase": "R",
      "power_active": 3200.0,
      "power_reactive": 600.0,
      "voltage_magnitude": 229.5,
      "voltage_angle": -0.4
    }
  ],
  "count": 1,
  "truncated": false
}
```

Results are capped at 50,000 rows — see the shared
[resource limits](index.md#resource-safety-limits) (the underlying
`HistoricalRecords` table can hold tens of millions of rows). When the cap
is hit, `truncated` is `true` and a `truncation_hint` field is added:

```json
{ "truncation_hint": "Result capped at 50000 records; narrow with start/end." }
```

**Errors**

| Status | Cause |
|---|---|
| `400` | Invalid `start`/`end` datetime format. |
| `404` | `series_id` does not exist in `database_id`. |
| `500` | Database error while querying. |

---

### `GET /historical/databases`

Lists every historical database with its `data_type` (`"load"`, `"pv"`, ...),
a detected `grid_level`
(`"MV"`/`"LV"`/`"Feeder"`/`"Mixed"`/`"Empty"` — `"Mixed"` if its series span
more than one level, `"Empty"` if it has none), series counts per level, its
record count, and its shared timeline (number of timestamps, first and
last). Used to populate data-source pickers elsewhere in the app (e.g.
[PF Data Generation](pf-datagen.md), whose New Dataset page loads it at its
data-source step and offers only `"load"` databases).

!!! note "Stored record counts"
    Record counts aren't counted on request: counting `HistoricalRecords`
    means scanning the whole table (about 90 s for a 40-million-row
    database). Each series stores its own count instead, kept exact by
    triggers in Postgres on every insert, update, delete and truncate,
    including bulk imports made directly against Postgres. See
    [Data Model › Historical databases](../database-schema.md). The response
    reads only small tables, so it answers in well under a second and is
    never stale.

**Response** `200`

```json
[
  {
    "database_id": "db001",
    "name": "Feeder measurements 2024",
    "data_type": "load",
    "n_series": 3,
    "n_mv": 0,
    "n_lv": 3,
    "n_feeder": 0,
    "n_records": 8931,
    "n_timestamps": 2977,
    "first_ts": "2024-08-20 14:00",
    "last_ts": "2024-09-20 14:00",
    "grid_level": "LV"
  }
]
```

**Errors**

None — a database error while querying is swallowed and the endpoint
returns an empty list (`[]`) rather than a `5xx`.

---
