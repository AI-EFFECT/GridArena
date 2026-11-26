"""Runs a user's benchmark submission against prepared inputs.

Executed as a standalone script inside the sandbox (``python harness.py <workdir>``),
so it only uses the standard library and never imports gridarena: the sandbox may be
a container that has the scientific stack but not the app, and the user's code must
not reach the app's database helpers through it.

Layout of <workdir> (written by run_manager.py before the run):

    manifest.json         {"entrypoint": "submission.py", "function": "predict",
                           "inputs": ["0000", "0001", ...]}
    code/                 the user's files, entrypoint at its root
    inputs/<name>.json    {"args": [...]}: one call of the user's function each

Written by this script:

    status.json           {"loaded": true} or {"loaded": false, "error": "<traceback>"}
    outputs/<name>.json   {"ok": true, "result": ..., "seconds": 1.2}
                          or {"ok": false, "error": "<traceback>", "seconds": 1.2}

Exit code 0 when every input was attempted (some calls may still have failed),
2 when the submission could not be loaded at all.
"""

import importlib.util
import json
import os
import sys
import time
import traceback
from pathlib import Path


def _jsonable(obj):
    """Convert numpy/torch values (anything with .tolist()) to plain JSON types."""
    if isinstance(obj, dict):
        return {(k if isinstance(k, str) else str(_jsonable(k))): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in obj]
    if not isinstance(obj, (str, bytes)) and hasattr(obj, "tolist"):
        return _jsonable(obj.tolist())
    return obj


def _write_json(path: Path, data) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def _load_function(code_dir: Path, entrypoint: str, function: str):
    # The user's own helper modules (shipped in a .zip) must be importable, and
    # relative paths (e.g. "weights.pt") must resolve next to the entrypoint.
    sys.path.insert(0, str(code_dir))
    os.chdir(code_dir)
    spec = importlib.util.spec_from_file_location("submission", code_dir / entrypoint)
    module = importlib.util.module_from_spec(spec)
    sys.modules["submission"] = module
    spec.loader.exec_module(module)
    fn = getattr(module, function, None)
    if not callable(fn):
        raise TypeError(f"{entrypoint} does not define a callable '{function}'.")
    return fn


def main(workdir: Path) -> int:
    manifest = json.loads((workdir / "manifest.json").read_text(encoding="utf-8"))
    outputs_dir = workdir / "outputs"
    outputs_dir.mkdir(exist_ok=True)

    try:
        fn = _load_function(workdir / "code", manifest["entrypoint"], manifest["function"])
    except BaseException:
        _write_json(workdir / "status.json", {"loaded": False, "error": traceback.format_exc()})
        traceback.print_exc()
        return 2
    _write_json(workdir / "status.json", {"loaded": True})

    for name in manifest["inputs"]:
        with open(workdir / "inputs" / f"{name}.json", encoding="utf-8") as f:
            args = json.load(f)["args"]
        started = time.perf_counter()
        try:
            result = _jsonable(fn(*args))
            record = {"ok": True, "result": result}
            json.dumps(record)  # fail here, inside the try, if the result isn't serialisable
        except Exception:
            traceback.print_exc()
            record = {"ok": False, "error": traceback.format_exc()}
        record["seconds"] = round(time.perf_counter() - started, 3)
        _write_json(outputs_dir / f"{name}.json", record)
        print(f"[harness] input {name}: {'ok' if record['ok'] else 'error'} "
              f"in {record['seconds']} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]).resolve()))
