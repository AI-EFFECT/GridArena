"""Solve this GridArena synthetic-data package and write its results.

    python run.py                       every grid, CSV files in results/
    python run.py --format parquet      Parquet files instead (needs pyarrow)
    python run.py --grid <mv_grid_id>   one grid only (repeat for several)
    python run.py --out <dir>           write somewhere else than results/

See how_to_run.md. Each grid's results go to <out>/<grid folder>/: one file
per table, report.json (the diversity report) and run_config.json (the
request plus the status, counts and metrics of the run).
"""

import argparse
import contextlib
import json
import logging
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from gridarena.pf_datagen import job  # noqa: E402
from gridarena.pf_datagen.inputs import LOCKED_FIELDS, load_inputs, load_manifest, locked_values  # noqa: E402
from gridarena.pf_datagen.sinks import FileSink  # noqa: E402

try:
    from threadpoolctl import threadpool_limits
except ImportError:
    threadpool_limits = None


def _check_locked(run_data: dict, request, manifest: dict) -> None:
    """The stored scenarios were built from these fields: refuse an edited one."""
    changed = [f for f in LOCKED_FIELDS if locked_values(request)[f] != manifest["locked"].get(f)]
    if changed:
        raise SystemExit(
            f"{run_data['mv_grid_id']}: run_config.json changes {', '.join(changed)}, but the load scenarios in "
            "inputs/ were built with the original values. Put them back, or create a new package in GridArena."
        )


def _progress(name):
    start = time.time()

    def report(completed, failed, total):
        done = completed + failed
        rate = done / max(time.time() - start, 1e-9)
        eta = (total - done) / rate if rate > 0 else 0
        print(f"  {name}: {done}/{total} solved ({failed} not converged), ~{eta:.0f} s left", flush=True)

    return report


def run_one(folder: Path, out: Path, fmt: str) -> dict:
    run_data = json.loads((folder / "run_config.json").read_text())
    manifest = load_manifest(folder / "inputs")
    request, _ = job.request_and_seed(run_data)       # validates the (possibly edited) request
    _check_locked(run_data, request, manifest)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(folder / "run_config.json", out / "run_config.json")
    run_data = job.start_run(str(out))
    try:
        inputs = load_inputs(folder / "inputs")
        types_path = HERE / "column_types.json"
        column_types = json.loads(types_path.read_text()) if types_path.exists() else None
        sink = FileSink(out, run_data["run_id"], fmt, on_progress=_progress(run_data["mv_grid_id"]),
                        column_types=column_types)
        # Many tiny matrix solves: a multi-threaded BLAS only slows them down.
        limit = threadpool_limits(limits=1) if threadpool_limits is not None else contextlib.nullcontext()
        with limit:
            return job.execute(str(out), run_data, request, inputs, sink)
    except Exception as e:
        return job.record_failure(str(out), e)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Solve a GridArena synthetic-data package.")
    parser.add_argument("--out", default=str(HERE / "results"), help="output directory (default: results/)")
    parser.add_argument("--format", choices=("csv", "parquet"), default="csv", help="output file format")
    parser.add_argument("--grid", action="append", help="only this MV grid id (repeatable)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    package = json.loads((HERE / "package.json").read_text())
    runs = package["runs"]
    if args.grid:
        unknown = set(args.grid) - {r["mv_grid_id"] for r in runs}
        if unknown:
            raise SystemExit(f"Not in this package: {', '.join(sorted(unknown))}.")
        runs = [r for r in runs if r["mv_grid_id"] in args.grid]

    failures = 0
    for k, r in enumerate(runs, 1):
        folder = HERE / r["folder"]
        out = Path(args.out) / Path(r["folder"]).name
        print(f"[{k}/{len(runs)}] {r['mv_grid_id']}: {r['n_scenarios']} load scenarios -> {out}", flush=True)
        t0 = time.time()
        result = run_one(folder, out, args.format)
        if result["status"] == "completed":
            m = result["metrics"]
            print(f"  done in {time.time() - t0:.0f} s: {m['converged']} converged, "
                  f"{m['non_converged']} not converged", flush=True)
        else:
            failures += 1
            print(f"  FAILED: {result['error']}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
