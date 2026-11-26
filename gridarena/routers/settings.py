"""Setup (first-run database choice) and Settings (environment variables) pages."""

import hmac
import logging
import os
from pathlib import Path
from typing import Dict, Optional

from fastapi import APIRouter, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from gridarena import settings_store as store

logger = logging.getLogger(__name__)

router = APIRouter()

_TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=_TEMPLATES_DIR)


def _admin_key_ok(provided: Optional[str]) -> bool:
    """Saving is open unless LVGPLAY_API_KEY is set, in which case it must match."""
    expected = os.environ.get("LVGPLAY_API_KEY")
    if not expected:
        return True
    return hmac.compare_digest((provided or "").encode(), expected.encode())


def _in_docker() -> bool:
    return Path("/.dockerenv").exists()


async def _initialize_database():
    from gridarena.app import initialize_database

    await run_in_threadpool(initialize_database)


# ── Setup ────────────────────────────────────────────────────────────────────


def _render_setup(request: Request, error: Optional[str] = None, values: Optional[dict] = None, status: int = 200):
    return templates.TemplateResponse(
        request,
        "setup.html",
        {
            "request": request,
            "title": "Setup",
            "active": "setup",
            "error": error,
            "values": values or {},
            "in_docker": _in_docker(),
            "needs_key": bool(os.environ.get("LVGPLAY_API_KEY")),
            "env_file": str(store.env_file()),
        },
        status_code=status,
    )


@router.get("/setup", include_in_schema=False)
async def setup_page(request: Request):
    if store.db_configured():
        return RedirectResponse("/settings", status_code=303)
    return _render_setup(request)


@router.post("/setup", include_in_schema=False)
async def setup_submit(request: Request):
    if store.db_configured():
        # Once a database is set, it can only be changed from Settings.
        return RedirectResponse("/settings", status_code=303)

    form = await request.form()
    values = {k: str(form.get(k, "")) for k in ("mode", "host", "port", "dbname", "user", "sslmode", "dsn")}
    if not _admin_key_ok(form.get("admin_key")):
        return _render_setup(request, "Wrong access key.", values, status=403)

    if values["mode"] == "dsn":
        dsn = values["dsn"].strip()
        if not dsn:
            return _render_setup(request, "Enter a connection string.", values, status=400)
    else:
        if not (values["host"].strip() and values["dbname"].strip() and values["user"].strip()):
            return _render_setup(request, "Host, database and user are required.", values, status=400)
        dsn = store.build_dsn(values["host"], values["port"], values["dbname"], values["user"],
                              str(form.get("password", "")), values["sslmode"])

    values["create_db"] = "1" if form.get("create_db") else ""
    if values["create_db"]:
        error = await run_in_threadpool(store.create_database, dsn)
        if error:
            return _render_setup(request, f"Could not create the database: {error}", values, status=400)

    error = await run_in_threadpool(store.check_connection, dsn)
    if error and store.is_missing_database(error):
        # The server answered; only the database is missing. Pre-tick the
        # option so creating it is one explicit click away.
        values["create_db"] = "1"
        name = store.database_name(dsn)
        return _render_setup(
            request,
            f'The server is reachable, but the database "{name}" does not exist. Leave '
            f'"Create the database if it doesn\'t exist" ticked and submit again to create an empty one, '
            f"or correct the name.",
            values, status=400,
        )
    if error:
        return _render_setup(request, f"Could not connect: {error}", values, status=400)

    store.save({"GRID_DB_DSN": dsn})
    await _initialize_database()
    logger.info("Database configured from /setup: %s", store.redact_dsn(dsn))
    return RedirectResponse("/", status_code=303)


# ── Settings ─────────────────────────────────────────────────────────────────


def _rows() -> Dict[str, list]:
    file_vals = store.read_env_file()
    settings = list(store.CATALOG) + [
        store.custom_setting(k) for k in sorted(file_vals) if k not in store.CATALOG_BY_KEY
    ]
    groups: Dict[str, list] = {g: [] for g in store.GROUPS + ["Custom"]}
    for s in settings:
        value = os.environ.get(s.key)
        if s.key in file_vals:
            source = "file"
        elif value is not None:
            source = "environment"
        else:
            source = ""
        if value is None:
            shown = ""
        elif s.key in store.DB_KEYS:
            shown = store.redact_dsn(value)
        elif s.secret:
            shown = "••••••"
        else:
            shown = value
        groups[s.group].append({"setting": s, "is_set": value is not None, "shown": shown, "source": source})
    return {g: rows for g, rows in groups.items() if rows or g == "Custom"}


def _render_settings(request: Request, error: Optional[str] = None, status: int = 200):
    q = request.query_params
    return templates.TemplateResponse(
        request,
        "settings.html",
        {
            "request": request,
            "title": "Settings",
            "active": "settings",
            "groups": _rows(),
            "error": error,
            "saved": q.get("saved"),
            "restart": q.get("restart") == "1",
            "db_configured": store.db_configured(),
            "needs_key": bool(os.environ.get("LVGPLAY_API_KEY")),
            "env_file": str(store.env_file()),
            "in_docker": _in_docker(),
        },
        status_code=status,
    )


@router.get("/settings", include_in_schema=False)
async def settings_page(request: Request):
    return _render_settings(request)


@router.post("/settings", include_in_schema=False)
async def settings_submit(request: Request):
    form = await request.form()
    if not _admin_key_ok(form.get("admin_key")):
        return _render_settings(request, "Wrong access key — nothing was saved.", status=403)

    file_vals = store.read_env_file()
    keys = [s.key for s in store.CATALOG] + [k for k in file_vals if k not in store.CATALOG_BY_KEY]
    updates: Dict[str, Optional[str]] = {}

    for key in keys:
        setting = store.CATALOG_BY_KEY.get(key) or store.custom_setting(key)
        current = os.environ.get(key)
        if form.get(f"del__{key}"):
            if current is not None or key in file_vals:
                updates[key] = None
            continue
        raw = form.get(f"val__{key}")
        if raw is None:
            continue
        value = str(raw).strip()
        if setting.secret:
            if value and value != current:  # blank = keep the stored secret
                updates[key] = value
        elif value != (current or ""):
            updates[key] = value or None

    new_key = str(form.get("new_key", "")).strip()
    if new_key:
        updates[new_key] = str(form.get("new_value", "")).strip()

    if not updates:
        return RedirectResponse("/settings?saved=0", status_code=303)

    for key, value in updates.items():
        error = store.validate(key, value or "")
        if error:
            return _render_settings(request, error, status=400)

    db_changed = any(k in store.DB_KEYS or k.startswith("DB_POOL_") for k in updates)
    if any(k in store.DB_KEYS for k in updates):
        def after(k):
            return updates[k] if k in updates else os.environ.get(k)

        new_dsn = after("GRID_DB_DSN") or after("DATABASE_URL")
        if new_dsn:
            error = await run_in_threadpool(store.check_connection, new_dsn)
            if error:
                return _render_settings(request, f"Not saved — could not connect to the new database: {error}",
                                        status=400)

    store.save(updates)
    if db_changed and store.db_configured():
        await _initialize_database()
    logger.info("Settings saved from the UI: %s", ", ".join(sorted(updates)))

    if not store.db_configured():
        return RedirectResponse("/setup", status_code=303)
    restart = any((store.CATALOG_BY_KEY.get(k) and store.CATALOG_BY_KEY[k].applies == "restart") for k in updates)
    return RedirectResponse(f"/settings?saved={len(updates)}&restart={int(restart)}", status_code=303)


@router.post("/settings/test-db", include_in_schema=False)
async def settings_test_db(request: Request):
    """Test a connection string (or the current one when empty) without saving."""
    body = await request.json()
    if not _admin_key_ok(body.get("admin_key")):
        return JSONResponse({"ok": False, "error": "Wrong access key."}, status_code=403)
    dsn = (body.get("dsn") or "").strip() or store.current_dsn()
    if not dsn:
        return JSONResponse({"ok": False, "error": "No connection string to test."}, status_code=400)
    error = await run_in_threadpool(store.check_connection, dsn)
    return {"ok": error is None, "error": error, "target": store.redact_dsn(dsn)}
