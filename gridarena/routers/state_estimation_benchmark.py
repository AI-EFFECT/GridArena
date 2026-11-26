"""State Estimation Router with endpoints to retrieve training data, download the
submission template, submit an estimation script and get its score."""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Literal, Optional, get_args

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates

import gridarena.benchmarks.state_estimation as se
import gridarena.database as db
from gridarena.benchmarks.state_estimation.evaluation import STATE_ESTIMATION_TASK
from gridarena.routers import _submissions as subs
from gridarena.routers._upload_guard import ensure_upload_size_ok

logger = logging.getLogger(__name__)

_TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=_TEMPLATES_DIR)

_SUBMISSION_TEMPLATE = Path(__file__).resolve().parent.parent / "benchmarks" / "state_estimation" / "submission_template.py"
_TEMPLATE_DOWNLOAD_NAME = "state_estimation_submission.py"

Difficulty = Literal["clean", "easy", "medium", "hard"]

router = APIRouter()


@router.get("/training-state-data")
async def get_training_state_data(
    noise_difficulty: Difficulty = Query("clean"),
    observability: Difficulty = Query("medium"),
):
    """
    Retrieves historical measurement data for all nodes and timestamps in the grids
    that allow information to be retrieved for state estimation training,
    structured by masked grid and node identifiers. This endpoint ensures that
    no real identifiers are exposed.

    The function ensures that grid and node masks are generated beforehand,
    and then returns the data grouped hierarchically as follows:

    Returns:
        dict: Dictionary with training data, eg.::

        {
            "grids": {
                masked_grid_id: {
                    masked_node_id: [
                        {
                            "datetime": str,
                            "phase": str,
                            "power_active": float,
                            "power_reactive": float,
                            "voltage_magnitude": float,
                            "voltage_angle": float
                        },
                        ...
                    ],
                    ...
                },
                ...
            }
        }
    """
    conn, cursor = db.get_db_connection()
    try:
        # Get grids marked for training state estimation
        cursor.execute(
            """
            SELECT grid_id FROM GridUsage WHERE state_estimation_train = 1
        """
        )
        grids_for_training = set(row[0] for row in cursor.fetchall())

        se.ensure_grid_masks(cursor)
        se.ensure_all_node_masks(cursor)

        data = {"grids": {}}

        # Use psycopg3's support for arrays when passing grid IDs
        cursor.execute(
            """
            SELECT m.grid_id, gm.masked_id, m.node_id, nm.masked_node_id,
                   m.datetime, m.phase, m.power_active, m.power_reactive,
                   m.voltage_magnitude, m.voltage_angle
            FROM "Measurements" m
            JOIN GridMask gm ON gm.grid_id = m.grid_id
            JOIN NodeMask nm ON nm.grid_id = m.grid_id AND nm.node_id = m.node_id AND nm.phase = m.phase
            WHERE m.grid_id = ANY(%s)
            ORDER BY m.grid_id, m.node_id, m.datetime
            """,
            (list(grids_for_training),),  # Pass grids as a list
        )

        grids = defaultdict(lambda: defaultdict(list))

        BATCH_SIZE = 5000
        while True:
            batch = cursor.fetchmany(BATCH_SIZE)
            if not batch:
                break
            for (
                grid_id,
                masked_gid,
                node_id,
                masked_nid,
                dt,
                phase,
                p_act,
                p_rea,
                v_mag,
                v_ang,
            ) in batch:
                grids[masked_gid][masked_nid].append(
                    {
                        "datetime": dt,
                        "phase": phase,
                        "power_active": p_act,
                        "power_reactive": p_rea,
                        "voltage_magnitude": v_mag,
                        "voltage_angle": v_ang,
                    }
                )

        data["grids"] = grids
        noise_profile = se.get_noise_profile(noise_difficulty)
        corrupted = se.corrupt_training_dataset(grids, noise_profile, difficulty=noise_difficulty)

        conn.commit()
        return {
            "noise_difficulty": noise_difficulty,
            "noise_profile": noise_profile,
            "grids": corrupted,
        }
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    finally:
        conn.close()


@router.get("/submission-template", response_class=FileResponse)
async def download_submission_template():
    """
    Downloads the blank submission script. Fill in its ``predict`` function
    with your estimator and upload it via ``POST /submissions``.
    """
    return subs.template_file(_SUBMISSION_TEMPLATE, _TEMPLATE_DOWNLOAD_NAME)


async def get_submission_template_source():
    """The blank submission script as text (for in-process callers such as the chat assistant)."""
    return subs.template_source(_SUBMISSION_TEMPLATE, _TEMPLATE_DOWNLOAD_NAME)


async def submit_state_estimation_script(
    source: bytes, filename: str, submission_name: str, noise_difficulty: str, observability: str,
):
    """Validate a submission, queue its evaluation and return its record.

    Shared by the upload endpoint and in-process callers (the chat assistant)."""
    subs.check_choice("noise_difficulty", noise_difficulty, get_args(Difficulty))
    subs.check_choice("observability", observability, get_args(Difficulty))
    return subs.submit_script(
        STATE_ESTIMATION_TASK, source=source, filename=filename, submission_name=submission_name,
        settings={"noise_difficulty": noise_difficulty, "observability": observability},
    )


@router.post("/submissions", status_code=202)
async def create_state_estimation_submission(
    file: UploadFile = File(..., description="A .py file, or a .zip with submission.py at its root."),
    submission_name: str = Form(..., description="Label for this attempt, e.g. alice_v1."),
    noise_difficulty: Difficulty = Form("clean"),
    observability: Difficulty = Form("medium"),
):
    """
    Submits a state estimation script for evaluation on hidden estimation tasks.

    The script must define ``predict(task) -> {series_key: {"voltage_magnitude",
    "voltage_angle"}}`` (see ``GET /submission-template``). It is checked for
    syntax and for the function, queued, and then run in a sandbox once per
    task (a few timestamps per test grid); poll ``GET /submissions/{submission_id}``
    for its status and score.
    """
    ensure_upload_size_ok(file)
    source = await file.read()
    return await submit_state_estimation_script(
        source=source, filename=file.filename or "", submission_name=submission_name,
        noise_difficulty=noise_difficulty, observability=observability,
    )


@router.get("/submissions")
async def list_state_estimation_submissions(submission_name: Optional[str] = Query(None)):
    """
    Lists state estimation submissions, newest first, optionally only those
    with the given ``submission_name``.
    """
    return subs.list_submissions(STATE_ESTIMATION_TASK, submission_name)


@router.get("/submissions/{submission_id}")
async def get_state_estimation_submission(submission_id: str):
    """
    Returns a submission's status (``queued``, ``running``, ``completed`` or
    ``failed``), its score once completed, its error if it failed, and the
    tail of its log.

    The score is over every unknown series of every task (lower is better)::

          {
              "score": 0.42,                    # mean (dV)^2 + mean |sin(dθ)|
              "rmse_voltage_magnitude": 0.61,   # over usable estimates
              "mae_voltage_angle": 0.35,        # over usable estimates
              "estimates": 120, "missing": 0, "invalid": 0,
              "tasks": 9, "failed_tasks": 0, "grids": 3,
              "runtime_seconds": 0.8,
              "per_grid": [{"grid_id": ..., "tasks": ..., "score": ..., "rmse_voltage_magnitude": ...,
                            "mae_voltage_angle": ..., "estimates": ..., "missing": ..., "invalid": ...,
                            "failed_tasks": ..., "seconds": ..., "error": null}, ...]
          }
    """
    return subs.get_submission(STATE_ESTIMATION_TASK, submission_id)


@router.get("/ui", response_class=HTMLResponse, include_in_schema=False)
def state_estimation_ui(request: Request):
    """Main UI page for State Estimation."""
    return templates.TemplateResponse(request, "benchmarks/state_estimation.html", {
        "request": request,
        "title": "State Estimation",
        "active": "state_estimation",
    })
