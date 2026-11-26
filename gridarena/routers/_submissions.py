"""Shared handling of the benchmark code-submission endpoints.

Every benchmark router exposes the same four endpoints (template download,
submit, get one, list) around its own BenchmarkTask; these helpers keep their
behaviour identical. They are plain functions, so the routers' in-process
callers (the chat assistant) go through the same validation as HTTP clients.
"""

import re
from pathlib import Path
from typing import Optional

from fastapi import HTTPException
from fastapi.responses import FileResponse

from gridarena.benchmarks.submissions import BenchmarkTask, SubmissionError, submission_registry

SUBMISSION_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


def template_file(path: Path, download_name: str) -> FileResponse:
    return FileResponse(path, media_type="text/x-python", filename=download_name)


def template_source(path: Path, download_name: str) -> dict:
    return {"filename": download_name, "source": path.read_text(encoding="utf-8")}


def submit_script(task: BenchmarkTask, *, source: bytes, filename: str,
                  submission_name: str, settings: dict) -> dict:
    """Validate a submission, queue its evaluation and return its record."""
    if not SUBMISSION_NAME_RE.match(submission_name or ""):
        raise HTTPException(
            status_code=422,
            detail="submission_name must be 1-64 characters: letters, digits, '_', '-' or '.'.",
        )
    try:
        return submission_registry.submit(
            task, filename=filename, data=source, submission_name=submission_name, settings=settings,
        )
    except SubmissionError as e:
        raise HTTPException(status_code=422, detail=str(e))


def get_submission(task: BenchmarkTask, submission_id: str) -> dict:
    record = submission_registry.get(submission_id, task.name)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Submission '{submission_id}' not found.")
    return record


def list_submissions(task: BenchmarkTask, submission_name: Optional[str]) -> dict:
    return {"submissions": submission_registry.list(task.name, submission_name=submission_name)}


def check_choice(name: str, value, choices) -> None:
    """422 for an invalid setting (in-process callers bypass FastAPI's own validation)."""
    if value not in choices:
        raise HTTPException(status_code=422, detail=f"{name} must be one of {', '.join(map(str, choices))}.")
