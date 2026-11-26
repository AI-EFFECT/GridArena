# How to run this synthetic-data package

GridArena package `$package_id`, created $created_at.

This folder holds everything needed to generate an MV power-flow dataset on
your own machine: the grid(s), the load scenarios already built from the
historical database `$historical_database_id`, the generation settings and
the solver code. No database, server or GridArena account is needed, only
Python.

Running it gives the rows GridArena would have stored for the same settings:
- the same samples and random draws;
- the same flags, counts and categories;
- the same values, up to floating-point rounding. A different numpy build
  or operating system can change the last digits, around 1e-8 W.

## Contents

| MV grid | Folder | Load scenarios | Buses | Seed |
|---|---|---|---|---|
$grid_table
$skipped
```
how_to_run.md        this file
run.py               the entry point
requirements.txt     Python packages to install
package.json         what is in the package
gridarena/           the solver code (a copy of GridArena's pf_datagen)
runs/<grid>/
    run_config.json  the generation settings (you can edit some of them, see below)
    inputs/          grid.json, catalogue.json, scenarios.npz, manifest.json
```

## 1. Install

You need Python 3.10 or newer. In this folder:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Run

```bash
python run.py
```

This solves every grid in the package, one after another, and prints
progress as it goes. Options:

| Option | What it does |
|---|---|
| `--format parquet` | write Parquet files instead of CSV (run `pip install pyarrow` first) |
| `--grid <mv_grid_id>` | solve only this grid; repeat it for several, e.g. `--grid $first_grid` |
| `--out <dir>` | write the results to `<dir>` instead of `results/` |

To use several CPU cores with a multi-grid package, start one process per
grid, each with its own `--grid`.

`run.py` exits with code 0 when every grid completed and 1 when any failed.
A failed grid writes its error to `results/<grid>/run_config.json`.

## 3. Results

Each grid gets a folder, for example `$first_results/`:

| File | One row per | Contents |
|---|---|---|
| `runtime.csv` | sample | converged, iterations, failure reason, solve time, the drawn load level, DER penetration, set-points and outages |
| `bus.csv` | sample × bus | voltage magnitude/angle (SI and p.u.), demand, generation, shunts, violation flags |
| `branch.csv` | sample × branch | flows at both ends, losses, current, loading, ratings, taps, Yff..Ytt |
| `ybus.csv` | sample × non-zero Y-bus entry | G and B (SI and p.u.) |
| `summary.csv` | converged sample | totals and extremes: load, losses, slack power, min/max voltage, violation counts |
| `der.csv` | sample × generator | available / produced P, Q, curtailment, Q limit |
| `variants.csv` | topology variant | disabled connections, islanded nodes, open switches, closed ties |
| `measurements.csv` | sample × meter reading | noisy measurements (only with the measurement view on) |
| `layout.csv` | meter | where the meters are (only with the measurement view on) |
| `report.json` | run | the diversity report shown on GridArena's run page |
| `run_config.json` | run | the settings, plus status, counts and metrics |

Units: power in W / VAr, voltage in V (line-to-line base `vn_kv` per bus),
current in A, impedance in Ω, admittance in S; `*_pu` columns are per unit on
the bases in `run_config.json` → `metrics.per_unit_bases`. A sample is
identified by `scenario` (unique integer), `load_scenario_idx` (the load
scenario) and `topology_variant_idx`; `output_group_id` joins the tables.
Non-converged samples appear only in `runtime.csv`, with their
`failure_reason`.

The columns are the same as GridArena's CSV export of a run.

## Changing the settings

Every grid's settings are in `runs/<grid>/run_config.json`. You can edit any
setting applied while solving and run again, for example:
- the DER options (`der`),
- outages and reconfiguration (`outages`, `reconfiguration`, `topology_perturbation`),
- set-points (`setpoints`, `slack_voltage_pu`),
- the load level (`load_level`),
- the solver (`solver`: `sweep` or `newton`),
- the measurement view (`measurements`),
- branch parameters (`parameters`),
- voltage limits (`voltage_limits`).

These cannot be changed here, because the load scenarios in `inputs/` were
built with them: $locked. `run.py` stops with a message if one of them was
edited. To change them, create a new package in GridArena.

## Troubleshooting

- **`ModuleNotFoundError`**: activate the virtual environment and run
  `pip install -r requirements.txt` again.
- **Slow run**: install `threadpoolctl` (in `requirements.txt`). Without it,
  numpy's multi-threaded BLAS can make the many small solves several times
  slower.
- **`ZoneInfoNotFoundError`**: install `tzdata`.
- **Parquet conversion runs out of memory**: each table is converted in
  memory at the end. Use the CSV output for very large runs and convert
  table by table.
