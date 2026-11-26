import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, get_args

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates

import gridarena.benchmarks.voltage_control as vc
import gridarena.database as db
from gridarena.benchmarks.voltage_control.evaluation import VOLTAGE_CONTROL_TASK
from gridarena.routers import _submissions as subs
from gridarena.routers._upload_guard import ensure_upload_size_ok

logger = logging.getLogger(__name__)

_TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=_TEMPLATES_DIR)

_SUBMISSION_TEMPLATE = Path(__file__).resolve().parent.parent / "benchmarks" / "voltage_control" / "submission_template.py"
_TEMPLATE_DOWNLOAD_NAME = "voltage_control_submission.py"

router = APIRouter()

Difficulty = Literal["clean", "easy", "medium", "hard"]
ScenarioType = Literal["Overvoltages", "Undervoltages"]


def _to_plain_dict(x: Any) -> Any:
    """Recursively convert defaultdicts and nested containers to plain serialisable objects."""
    if isinstance(x, defaultdict):
        x = dict(x)
    if isinstance(x, dict):
        return {k: _to_plain_dict(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_to_plain_dict(v) for v in x]
    return x


@router.get("/training-scenarios")
async def get_training_voltage_control_data(
    voltage_limit: float,
    Scenario: ScenarioType = Query(...),
    difficulty: Difficulty = Query("clean"),
):
    """
    Retrieve training snapshots + reference solutions.
    Difficulty filters which snapshots you return (scenario hardness), not just data corruption.

    Returns:
      {
        "difficulty": "...",
        "difficulty_profile": {...},
        "grids": {
          grid_id: {
            datetime: {
              "measurements": [...],
              "solutions": [...],
              "metrics": {...}
            }
          }
        }
      }
    """
    conn, cursor = db.get_db_connection()
    try:
        cursor.execute("""SELECT grid_id FROM GridUsage WHERE voltage_control_train = 1""")
        grids_for_training = set(row[0] for row in cursor.fetchall())

        if Scenario == "Overvoltages":
            measurements = vc.get_all_measurements_over(
                cursor, voltage_limit, grids_for_training, phase_type="train"
            )
        else:
            measurements = vc.get_all_measurements_under(
                cursor, voltage_limit, grids_for_training, phase_type="train"
            )

        # Group into snapshots: (grid_id, dt) -> list[measurement_dict]
        snapshots: Dict[Tuple[str, Any], List[Dict[str, Any]]] = {}
        tmp = defaultdict(lambda: defaultdict(list))

        for grid_id, node_id, phase, dt, p_act, p_rea, v_mag, v_ang in measurements:
            tmp[grid_id][dt].append(
                {
                    "node_id": node_id,
                    "phase": phase,
                    "datetime": dt,
                    "power_active": p_act,
                    "power_reactive": p_rea,
                    "voltage_magnitude": v_mag,
                    "voltage_angle": v_ang,
                }
            )

        for grid_id, dt_dict in tmp.items():
            for dt, meas in dt_dict.items():
                snapshots[(grid_id, dt)] = meas

        sol_query = """
            SELECT node_id, phase, corrected_voltage, adjusted_power_active, adjusted_power_reactive, created_at
            FROM "VoltageControlSolutions"
            WHERE grid_id = %s AND datetime = %s
        """

        scenario_payload: Dict[Tuple[str, Any], Dict[str, Any]] = {}
        hardness_by_key: Dict[Tuple[str, Any], float] = {}
        has_solutions_by_key: Dict[Tuple[str, Any], bool] = {}

        grid_complexity_cache: Dict[str, Dict[str, Any]] = {}

        for (grid_id, dt), meas in snapshots.items():
            if grid_id not in grid_complexity_cache:
                grid_complexity_cache[grid_id] = vc.get_grid_complexity_metrics(cursor, grid_id)
            gcp = grid_complexity_cache[grid_id]
            grid_cnorm = float(gcp["grid_complexity_norm"])

            cursor.execute(sol_query, (grid_id, dt))
            sol_rows = cursor.fetchall()

            solutions = [
                {
                    "node_id": r[0],
                    "phase": r[1],
                    "corrected_voltage": r[2],
                    "adjusted_power_active": r[3],
                    "adjusted_power_reactive": r[4],
                    "created_at": r[5],
                }
                for r in sol_rows
            ]

            has_solutions = len(solutions) > 0
            has_solutions_by_key[(grid_id, dt)] = has_solutions

            metrics = vc.compute_snapshot_metrics(
                measurements=meas,
                solutions=solutions if solutions else None,
                nominal_voltage=vc.NOMINAL_VOLTAGE,
                band=vc.VOLTAGE_BAND,
                grid_complexity_norm=grid_cnorm,
                grid_complexity_profile=gcp,
            )
            hardness = float(metrics["hardness_score"])

            scenario_payload[(grid_id, dt)] = {
                "measurements": meas,
                "solutions": solutions,
                "metrics": metrics,
            }
            hardness_by_key[(grid_id, dt)] = hardness

        # Prefer snapshots that have solutions
        keys_all = list(snapshots.keys())
        keys_with_solutions = [k for k in keys_all if has_solutions_by_key.get(k, False)]

        if keys_with_solutions:
            preferred_keys = keys_with_solutions
            hardness_for_filter = {k: hardness_by_key[k] for k in preferred_keys}
            prefer_mode = "solutions_only"
        else:
            preferred_keys = keys_all
            hardness_for_filter = hardness_by_key
            prefer_mode = "no_solutions_found_fallback_all"

        selected_keys, diff_profile = vc.difficulty_filter_keys(hardness_for_filter, difficulty)
        diff_profile = {
            **diff_profile,
            "preference": prefer_mode,
            "preferred_population": len(preferred_keys),
        }

        all_data = defaultdict(lambda: defaultdict(dict))
        ordered_selected = [k for k in preferred_keys if k in selected_keys]
        for grid_id, dt in ordered_selected:
            all_data[grid_id][dt] = scenario_payload[(grid_id, dt)]

        conn.commit()

        return _to_plain_dict(
            {
                "difficulty": difficulty,
                "difficulty_profile": diff_profile,
                "scenario": Scenario,
                "voltage_limit": voltage_limit,
                "grids": all_data,
            }
        )

    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    finally:
        conn.close()


@router.get("/submission-template", response_class=FileResponse)
async def download_submission_template():
    """
    Downloads the blank submission script. Fill in its ``predict`` function
    with your controller and upload it via ``POST /submissions``.
    """
    return subs.template_file(_SUBMISSION_TEMPLATE, _TEMPLATE_DOWNLOAD_NAME)


async def get_submission_template_source():
    """The blank submission script as text (for in-process callers such as the chat assistant)."""
    return subs.template_source(_SUBMISSION_TEMPLATE, _TEMPLATE_DOWNLOAD_NAME)


async def submit_voltage_control_script(
    source: bytes, filename: str, submission_name: str, voltage_limit: float, scenario: str, difficulty: str,
):
    """Validate a submission, queue its evaluation and return its record.

    Shared by the upload endpoint and in-process callers (the chat assistant)."""
    subs.check_choice("scenario", scenario, get_args(ScenarioType))
    subs.check_choice("difficulty", difficulty, get_args(Difficulty))
    try:
        voltage_limit = float(voltage_limit)
    except (TypeError, ValueError):
        voltage_limit = float("nan")
    if not (math.isfinite(voltage_limit) and voltage_limit > 0):
        raise HTTPException(status_code=422, detail="voltage_limit must be a positive number of volts.")
    return subs.submit_script(
        VOLTAGE_CONTROL_TASK, source=source, filename=filename, submission_name=submission_name,
        settings={"voltage_limit": voltage_limit, "scenario": scenario, "difficulty": difficulty},
    )


@router.post("/submissions", status_code=202)
async def create_voltage_control_submission(
    file: UploadFile = File(..., description="A .py file, or a .zip with submission.py at its root."),
    submission_name: str = Form(..., description="Label for this attempt, e.g. alice_v1."),
    voltage_limit: float = Form(..., description="Voltage threshold (V) that selects the violation snapshots."),
    Scenario: ScenarioType = Form("Overvoltages"),
    difficulty: Difficulty = Form("clean"),
):
    """
    Submits a voltage control script for evaluation on the hidden test snapshots.

    The script must define ``predict(snapshot) -> [{"node_id", "phase",
    "corrected_voltage", "adjusted_power_active", "adjusted_power_reactive"}, ...]``
    (see ``GET /submission-template``). It is checked for syntax and for the
    function, queued, and then run in a sandbox once per test snapshot; poll
    ``GET /submissions/{submission_id}`` for its status and score.
    """
    ensure_upload_size_ok(file)
    source = await file.read()
    return await submit_voltage_control_script(
        source=source, filename=file.filename or "", submission_name=submission_name,
        voltage_limit=voltage_limit, scenario=Scenario, difficulty=difficulty,
    )


@router.get("/submissions")
async def list_voltage_control_submissions(submission_name: Optional[str] = Query(None)):
    """
    Lists voltage control submissions, newest first, optionally only those
    with the given ``submission_name``.
    """
    return subs.list_submissions(VOLTAGE_CONTROL_TASK, submission_name)


@router.get("/submissions/{submission_id}")
async def get_voltage_control_submission(submission_id: str):
    """
    Returns a submission's status (``queued``, ``running``, ``completed`` or
    ``failed``), its score once completed, its error if it failed, and the
    tail of its log.

    The score is the node-weighted mean of the snapshot scores (0-1, higher is better)::

          {
              "score": 0.71,
              "snapshots": 24, "nodes": 960, "failed_snapshots": 0, "grids": 2,
              "runtime_seconds": 0.3,
              "per_grid": [{"grid_id": ..., "snapshots": ..., "nodes": ..., "score": ...,
                            "failed_snapshots": ..., "seconds": ..., "error": null}, ...],
              "per_snapshot": [{"grid_id": ..., "datetime": ..., "nodes": ..., "changed_nodes": ...,
                                "invalid": ..., "mode": ..., "timestamp_score": ...,
                                "effort_norm": ..., "reference_voltage_mae": ..., "seconds": ...,
                                "error": null}, ...]
          }
    """
    return subs.get_submission(VOLTAGE_CONTROL_TASK, submission_id)


@router.get("/ui", response_class=HTMLResponse, include_in_schema=False)
def voltage_control_ui(request: Request):
    """UI for Voltage Control Benchmark."""
    return templates.TemplateResponse(request, "benchmarks/voltage_control.html", {
        "request": request,
        "title": "Voltage Control",
        "active": "voltage_control",
    })
