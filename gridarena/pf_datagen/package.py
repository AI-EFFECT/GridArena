"""Standalone dataset package: a synthetic-data run to solve outside GridArena.

GridArena does the part of a run that needs its database -- the grid, the
cable catalogue and the load scenarios built from the historical series
(pf_datagen/inputs.py) -- and zips it with a copy of the solver code, a
`run.py` and `how_to_run.md`. Anyone can then unzip it and run it on their
own machine with plain Python; the results are one file per table, with the
columns of GridArena's CSV export, and are the rows GridArena would have
stored for the same request and seed (pf_datagen/job.py::execute runs in
both places).

Zip layout (one folder per grid; a single-grid dataset has one):

    gridarena_dataset_<package_id>/
        how_to_run.md, requirements.txt, run.py, package.json
        column_types.json               each output column's type (for the Parquet files)
        gridarena/...                   the solver code (no web app, no database)
        runs/<grid>/run_config.json     the request, with its seed fixed
        runs/<grid>/inputs/             grid.json, catalogue.json, scenarios.npz, manifest.json
"""

import ast
import json
import re
import shutil
import tempfile
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from string import Template
from typing import List, Tuple

import numpy as np

from gridarena.pf_datagen.inputs import LOCKED_FIELDS, prepare_inputs, save_inputs

PACKAGE_FORMAT = 1
_GRIDARENA_DIR = Path(__file__).resolve().parent.parent       # .../gridarena
_FILES_DIR = Path(__file__).resolve().parent / "package_files"
# Entry points of what run.py imports; the rest is found from their imports.
_ROOTS = ("gridarena.pf_datagen.job", "gridarena.pf_datagen.inputs", "gridarena.pf_datagen.sinks",
          "gridarena.pf_datagen.schemas")
# Only the GridArena side touches these (prepare_inputs / DatabaseSink).
_EXCLUDED = ("gridarena.database",)


def new_package_id() -> str:
    return str(uuid.uuid4())[:8]


def grid_seeds(seed: int, n: int) -> List[int]:
    """Each grid's seed in a group, from the group seed and its position (as
    pf_datagen/run_manager.py::start_group does)."""
    return [int(np.random.SeedSequence([seed, k]).generate_state(1)[0]) for k in range(n)]


def folder_name(mv_grid_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", mv_grid_id)


# ── The code that goes in ──

def _module_path(module: str) -> Path:
    rel = Path(*module.split(".")[1:])
    if (_GRIDARENA_DIR / rel).is_dir():
        return _GRIDARENA_DIR / rel / "__init__.py"
    return _GRIDARENA_DIR / rel.with_suffix(".py")


def _gridarena_imports(path: Path) -> set:
    """Every gridarena module a file imports, at module level or inside functions."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    package = ".".join(("gridarena",) + path.relative_to(_GRIDARENA_DIR).parent.parts)
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                base = ".".join(package.split(".")[: len(package.split(".")) - node.level + 1] + ([base] if base else []))
            names = [base] + [f"{base}.{a.name}" for a in node.names]
        else:
            continue
        for name in names:
            if name.startswith("gridarena.") and not name.startswith(_EXCLUDED):
                if _module_path(name).exists():
                    found.add(name)
    return found


def code_modules() -> List[str]:
    """The gridarena modules the standalone run needs, sorted."""
    todo, seen = list(_ROOTS), set()
    while todo:
        m = todo.pop()
        if m in seen:
            continue
        seen.add(m)
        todo.extend(_gridarena_imports(_module_path(m)) - seen)
    return sorted(seen)


def _write_code(root: Path) -> List[str]:
    """Copy the modules under root/gridarena; every package __init__ is left
    empty, so importing one module doesn't pull in the web app."""
    written = []
    for m in code_modules():
        src = _module_path(m)
        if src.name == "__init__.py":
            continue
        dst = root / src.relative_to(_GRIDARENA_DIR.parent)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        written.append(str(dst.relative_to(root)).replace("\\", "/"))
    for d in sorted({p.parent for p in (root / "gridarena").rglob("*.py")} | {root / "gridarena"}):
        init = d / "__init__.py"
        if not init.exists():
            init.write_text("")
    return written


# ── Building it ──

def build_package(zip_path, grids: List[Tuple[str, object]], db, package_id: str, group_seed=None,
                  skipped=()) -> dict:
    """Prepare every (mv_grid_id, request) of `grids` -- each request with its
    seed already fixed -- and write the zip to `zip_path`. Returns the
    package.json written into it."""
    created = datetime.now(timezone.utc).isoformat()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / f"gridarena_dataset_{package_id}"
        runs = []
        for mv_grid_id, request in grids:
            run_id = new_package_id()
            folder = root / "runs" / folder_name(mv_grid_id)
            inputs = prepare_inputs(mv_grid_id, request, request.seed, db)
            save_inputs(inputs, request, folder / "inputs")
            config = request.model_dump(mode="json")
            config.update({"run_id": run_id, "mv_grid_id": mv_grid_id, "status": "packaged", "created_at": created,
                           "config": request.model_dump(mode="json"), "group_id": package_id if len(grids) > 1 else None,
                           "package_id": package_id})
            (folder / "run_config.json").write_text(json.dumps(config, indent=2))
            runs.append({"mv_grid_id": mv_grid_id, "folder": f"runs/{folder.name}", "run_id": run_id,
                         "seed": request.seed, "n_scenarios": len(inputs.scenarios),
                         "n_nodes": len(inputs.topology.node_id_to_index)})

        files = _write_code(root)
        conn, cursor = db.get_db_connection()
        try:
            column_types = db.get_pf_datagen_column_types(cursor)
        finally:
            conn.close()
        (root / "column_types.json").write_text(json.dumps(column_types, indent=1))
        manifest = {"format": PACKAGE_FORMAT, "package_id": package_id, "created_at": created,
                    "group_seed": group_seed, "runs": runs, "skipped": list(skipped), "code": files}
        (root / "package.json").write_text(json.dumps(manifest, indent=2))
        shutil.copyfile(_FILES_DIR / "run.py", root / "run.py")
        shutil.copyfile(_FILES_DIR / "requirements.txt", root / "requirements.txt")
        (root / "how_to_run.md").write_text(_how_to_run(manifest, grids), encoding="utf-8")

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for p in sorted(root.rglob("*")):
                if p.is_file():
                    z.write(p, p.relative_to(root.parent).as_posix())
    return manifest


def _how_to_run(manifest: dict, grids) -> str:
    request = grids[0][1]
    lines = [f"| `{r['mv_grid_id']}` | `{r['folder']}` | {r['n_scenarios']} | {r['n_nodes']} | {r['seed']} |"
             for r in manifest["runs"]]
    skipped = "".join(f"\n- `{s['mv_grid_id']}`: {s['reason']}" for s in manifest["skipped"])
    return Template((_FILES_DIR / "how_to_run.md").read_text(encoding="utf-8")).substitute(
        package_id=manifest["package_id"],
        created_at=manifest["created_at"][:19].replace("T", " ") + " UTC",
        n_grids=len(manifest["runs"]),
        historical_database_id=request.historical_database_id,
        grid_table="\n".join(lines),
        skipped=f"\nSkipped by the size filter:{skipped}\n" if skipped else "",
        locked=", ".join(f"`{f}`" for f in LOCKED_FIELDS),
        first_folder=manifest["runs"][0]["folder"],
        first_grid=manifest["runs"][0]["mv_grid_id"],
        first_results=f"results/{manifest['runs'][0]['folder'].split('/', 1)[1]}",
    )
