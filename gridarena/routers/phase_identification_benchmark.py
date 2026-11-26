import hashlib
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Literal, Optional, get_args

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates

import gridarena.benchmarks.phase_identification as pi
import gridarena.database as db
from gridarena.benchmarks.phase_identification.evaluation import PHASE_TASK
from gridarena.routers import _submissions as subs
from gridarena.routers._upload_guard import ensure_upload_size_ok

logger = logging.getLogger(__name__)

_TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=_TEMPLATES_DIR)

_SUBMISSION_TEMPLATE = Path(pi.__file__).resolve().parent / "submission_template.py"
_TEMPLATE_DOWNLOAD_NAME = "phase_identification_submission.py"

Difficulty = Literal["clean", "easy", "medium", "hard"]

router = APIRouter()


@router.get("/training-phase-data")
async def get_training_phase_identification_data(
    difficulty: Difficulty = Query("clean"),
):
    """
    Retrieves historical measurement data for all grids for training
    a phase detection algorithm

    Returns:
        dict: {
            grid_id1: {
                node1_id: [ {measurement1}, ... ],
                node2_id: [ {measurement2}, ... ],
                ...
            },
            grid_id2: {
                ...
            }
        }

    Raises:
        HTTPException: If no measurements are found or a database error occurs.
    """
    conn, cursor = db.get_db_connection()
    try:
        # Get measurements filtered for phase detection training
        measurements = pi.get_all_measurements(
            cursor, task_type="phase_detection", phase_type="train"
        )

        # Difficulty profile (platform-controlled)
        profile = pi.get_corruption_profile(difficulty=difficulty)

        # Seeded RNG per (grid_id, node_id, phase, difficulty) => deterministic across calls
        rngs: Dict[tuple, random.Random] = {}

        all_data = defaultdict(lambda: defaultdict(list))

        for grid_id, node_id, phase, dt, p_act, p_rea, v_mag, v_ang in measurements:
            rng_key = (grid_id, node_id, phase)
            if rng_key not in rngs:
                seed = hashlib.sha256(f"{grid_id}:{node_id}:{phase}:{difficulty}".encode()).hexdigest()
                rngs[rng_key] = random.Random(seed)

            record = {
                "datetime": dt,
                "power_active": p_act,
                "power_reactive": p_rea,
                "voltage_magnitude": v_mag,
                "voltage_angle": v_ang,
                "phase": phase,
            }

            # Corrupt only numeric measurement fields; datetime/phase unchanged.
            record = pi.corrupt_measurement_record(record, rngs[rng_key], profile)

            all_data[grid_id][node_id].append(record)

        conn.commit()
        return {
            "difficulty": difficulty,
            "profile": profile,
            "grids": all_data,
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
    with your model and upload it via ``POST /submissions``.
    """
    return subs.template_file(_SUBMISSION_TEMPLATE, _TEMPLATE_DOWNLOAD_NAME)


async def get_submission_template_source():
    """The blank submission script as text (for in-process callers such as the chat assistant)."""
    return subs.template_source(_SUBMISSION_TEMPLATE, _TEMPLATE_DOWNLOAD_NAME)


async def submit_phase_script(source: bytes, filename: str, submission_name: str, difficulty: str):
    """Validate a submission, queue its evaluation and return its record.

    Shared by the upload endpoint and in-process callers (the chat assistant)."""
    subs.check_choice("difficulty", difficulty, get_args(Difficulty))
    return subs.submit_script(
        PHASE_TASK, source=source, filename=filename, submission_name=submission_name,
        settings={"difficulty": difficulty},
    )


@router.post("/submissions", status_code=202)
async def create_phase_submission(
    file: UploadFile = File(..., description="A .py file, or a .zip with submission.py at its root."),
    submission_name: str = Form(..., description="Label for this attempt, e.g. alice_v1."),
    difficulty: Difficulty = Form("clean"),
):
    """
    Submits a phase identification script for evaluation on the hidden test grids.

    The script must define ``predict(measurements) -> {series_key: "R" | "S" | "T"}``
    (see ``GET /submission-template``). It is checked for syntax and for the
    function, queued, and then run in a sandbox once per test grid; poll
    ``GET /submissions/{submission_id}`` for its status and score.

    Returns:
        dict: The submission record, eg.::

          {
              "submission_id": "3f9c1a2b7d4e",
              "submission_name": "alice_v1",
              "settings": {"difficulty": "clean"},
              "status": "queued",
              ...
          }
    """
    ensure_upload_size_ok(file)
    source = await file.read()
    return await submit_phase_script(
        source=source, filename=file.filename or "", submission_name=submission_name, difficulty=difficulty,
    )


@router.get("/submissions")
async def list_phase_submissions(submission_name: Optional[str] = Query(None)):
    """
    Lists phase identification submissions, newest first, optionally only
    those with the given ``submission_name``.
    """
    return subs.list_submissions(PHASE_TASK, submission_name)


@router.get("/submissions/{submission_id}")
async def get_phase_submission(submission_id: str):
    """
    Returns a submission's status (``queued``, ``running``, ``completed`` or
    ``failed``), its score once completed, its error if it failed, and the
    tail of its log (anything the script printed, plus tracebacks).

    The score is computed over every series of every test grid::

          {
              "accuracy": 0.83,          # correct / total
              "correct": 166,
              "total": 200,
              "grids": 4,
              "failed_grids": 0,         # grids whose predict() call raised
              "runtime_seconds": 3.2,
              "confusion": {"R": {"R": 60, "S": 5, "T": 2, "other": 0}, ...},
              "per_grid": [{"grid_id": ..., "series": ..., "correct": ..., "accuracy": ...,
                            "missing": ..., "invalid": ..., "unknown_keys": ...,
                            "seconds": ..., "error": null}, ...]
          }
    """
    return subs.get_submission(PHASE_TASK, submission_id)


@router.get("/ui", response_class=HTMLResponse, include_in_schema=False)
def phase_identification_ui(request: Request):
    """Main UI page for Phase Identification."""
    return templates.TemplateResponse(request, "benchmarks/phase_identification.html", {
        "request": request,
        "title": "Phase Identification",
        "active": "phase_mapping",
    })
