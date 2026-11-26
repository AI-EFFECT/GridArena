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
import gridarena.benchmarks.topology_discovery as td
import gridarena.database as db
from gridarena.benchmarks.topology_discovery.evaluation import TOPOLOGY_TASK
from gridarena.routers import _submissions as subs
from gridarena.routers._upload_guard import ensure_upload_size_ok

logger = logging.getLogger(__name__)

_TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=_TEMPLATES_DIR)

_SUBMISSION_TEMPLATE = Path(__file__).resolve().parent.parent / "benchmarks" / "topology_discovery" / "submission_template.py"
_TEMPLATE_DOWNLOAD_NAME = "topology_discovery_submission.py"

Difficulty = Literal["clean", "easy", "medium", "hard"]

router = APIRouter()


@router.get("/training-topology-data")
async def get_topology_identification_data(
    difficulty: Difficulty = Query("clean"),
):
    """
    Retrieves historical measurement and grid data for all grids for training
    a topology discovery algorithm

    Returns:
        dict: Dictionary with the retrieved information, eg.::

        {
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
        # Filter grids marked for topology detection (training)
        cursor.execute(
            """
            SELECT grid_id FROM GridUsage WHERE topology_detection_train = 1
            """
        )
        grids_for_training = set(row[0] for row in cursor.fetchall())

        # Get measurements filtered for topology detection training
        measurements = pi.get_all_measurements(
            cursor, task_type="topology_detection", phase_type="train"
        )

        # Fetch topology data for the same grids
        node_id_to_index_per_grid, connections_per_grid, conn_data_per_grid = (
            td.get_all_grids_topology(cursor, grids_for_training)
        )

        # Difficulty profile (platform-controlled)
        profile = td.get_corruption_profile(difficulty=difficulty)

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
            record = td.corrupt_measurement_record(record, rngs[rng_key], profile)

            all_data[grid_id][node_id].append(record)

        conn.commit()
        return {
            "difficulty": difficulty,
            "profile": profile,
            "grids": all_data,
            "node to corresponding index": node_id_to_index_per_grid,
            "topology (index)": connections_per_grid,
            "conn_data": conn_data_per_grid,
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
    with your method and upload it via ``POST /submissions``.
    """
    return subs.template_file(_SUBMISSION_TEMPLATE, _TEMPLATE_DOWNLOAD_NAME)


async def get_submission_template_source():
    """The blank submission script as text (for in-process callers such as the chat assistant)."""
    return subs.template_source(_SUBMISSION_TEMPLATE, _TEMPLATE_DOWNLOAD_NAME)


async def submit_topology_script(source: bytes, filename: str, submission_name: str, difficulty: str):
    """Validate a submission, queue its evaluation and return its record.

    Shared by the upload endpoint and in-process callers (the chat assistant)."""
    subs.check_choice("difficulty", difficulty, get_args(Difficulty))
    return subs.submit_script(
        TOPOLOGY_TASK, source=source, filename=filename, submission_name=submission_name,
        settings={"difficulty": difficulty},
    )


@router.post("/submissions", status_code=202)
async def create_topology_submission(
    file: UploadFile = File(..., description="A .py file, or a .zip with submission.py at its root."),
    submission_name: str = Form(..., description="Label for this attempt, e.g. alice_v1."),
    difficulty: Difficulty = Form("clean"),
):
    """
    Submits a topology discovery script for evaluation on the hidden test grids.

    The script must define ``predict(grid) -> [[node_key, node_key], ...]``
    (see ``GET /submission-template``). It is checked for syntax and for the
    function, queued, and then run in a sandbox once per test grid; poll
    ``GET /submissions/{submission_id}`` for its status and score.
    """
    ensure_upload_size_ok(file)
    source = await file.read()
    return await submit_topology_script(
        source=source, filename=file.filename or "", submission_name=submission_name, difficulty=difficulty,
    )


@router.get("/submissions")
async def list_topology_submissions(submission_name: Optional[str] = Query(None)):
    """
    Lists topology discovery submissions, newest first, optionally only
    those with the given ``submission_name``.
    """
    return subs.list_submissions(TOPOLOGY_TASK, submission_name)


@router.get("/submissions/{submission_id}")
async def get_topology_submission(submission_id: str):
    """
    Returns a submission's status (``queued``, ``running``, ``completed`` or
    ``failed``), its score once completed, its error if it failed, and the
    tail of its log.

    The score compares the predicted edges (undirected) with the true ones::

          {
              "f1": 0.81,                # of precision and recall, over all grids
              "precision": 0.84,         # correct / predicted edges
              "recall": 0.78,            # correct / true edges
              "correct": 140, "extra": 27, "missed": 39,
              "grids": 3,
              "failed_grids": 0,
              "runtime_seconds": 1.4,
              "per_grid": [{"grid_id": ..., "nodes": ..., "true_edges": ..., "predicted_edges": ...,
                            "correct": ..., "extra": ..., "missed": ..., "invalid": ...,
                            "precision": ..., "recall": ..., "f1": ..., "seconds": ..., "error": null}, ...]
          }
    """
    return subs.get_submission(TOPOLOGY_TASK, submission_id)


@router.get("/ui", response_class=HTMLResponse, include_in_schema=False)
def topology_discovery_ui(request: Request):
    """Main UI page for Topology Discovery."""
    return templates.TemplateResponse(request, "benchmarks/topology_discovery.html", {
        "request": request,
        "title": "Topology Discovery",
        "active": "topology_discovery",
    })
