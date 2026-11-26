"""Main API file"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

from gridarena import settings_store

# override=True: values saved from the Settings page win over the process
# environment, so an edit made in the UI survives a restart.
load_dotenv(settings_store.env_file(), override=True)

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import gridarena.routers as rt

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent


def initialize_database():
    """(Re)connect to the configured database and prepare its tables in the
    background (gridarena/startup.py); pages show a loading screen until it
    is done. Called at startup and again when the connection is changed from
    the Setup / Settings pages."""
    import gridarena.database as db
    from gridarena.startup import db_startup

    db.close_pool()  # drop connections to a previous database, if any
    db_startup.start()

@asynccontextmanager
async def lifespan(app: FastAPI):
    import gridarena.database as db
    from gridarena.benchmarks.submissions import submission_registry
    from gridarena.diffusion.run_manager import diffusion_registry
    from gridarena.llm import init_rag
    from gridarena.pf_datagen.run_manager import pf_datagen_registry
    from gridarena.rl.run_manager import registry

    from gridarena.digital_twin.runner import dt_runner
    from gridarena.digital_twin.scenario_runner import scenario_runner
    from gridarena.startup import db_startup

    if settings_store.db_configured():
        db_startup.start()  # prepares the tables in the background; pages wait on a loading screen
    else:
        logger.warning("No database configured (GRID_DB_DSN / DATABASE_URL) - open /setup to choose one")
    init_rag()
    registry.startup()
    diffusion_registry.startup()
    pf_datagen_registry.startup()
    yield
    dt_runner.shutdown()
    scenario_runner.shutdown()
    registry.shutdown()
    diffusion_registry.shutdown()
    pf_datagen_registry.shutdown()
    submission_registry.shutdown()
    db.close_pool()


app = FastAPI(
    title="GridArena",
    description="GridArena is a simulation and analytics API for low-voltage electrical \
                distribution grids.It allows users to upload or define grid topologies, execute \
                power flow simulations under various operating conditions, and access historical \
                simulation results for further analysis or forecasting. The API supports grid \
                updates, and enables integration with energy management algorithms.",
    version="0.1",
    lifespan=lifespan,
    contact={
        "name": "INESCTEC - Centro de Produção Energia e Sistemas (CPES)",
        "url": "https://www.inesctec.pt/",
        "email": "david.lima@inesctec.pt, goncalo.cunha@inesctec.pt",
    },
)

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

templates = Jinja2Templates(directory=BASE_DIR / "templates")

app.include_router(rt.grid.router, prefix="/grid", tags=["Grid"])
app.include_router(rt.historical.router, prefix="/historical", tags=["Historical"])
app.include_router(rt.measurements.router, prefix="/measurements", tags=["Measurements"])
app.include_router(rt.powerflow.router, prefix="/powerflow", tags=["Algorithm"])
app.include_router(
    rt.phase_identification_benchmark.router,
    prefix="/phase_identification_benchmark",
    tags=["Phase Identification Benchmark"],
)
app.include_router(
    rt.topology_discovery_benchmark.router,
    prefix="/topology_discovery_benchmark",
    tags=["Topology Discovery Benchmark"],
)
app.include_router(
    rt.state_estimation_benchmark.router,
    prefix="/state_estimation_benchmark",
    tags=["State Estimation Benchmark"],
)
app.include_router(
    rt.voltage_control_benchmark.router,
    prefix="/voltage_control_benchmark",
    tags=["Voltage Control Benchmark"],
)
app.include_router(rt.chat.router, prefix="/chat", tags=["Chat"])
app.include_router(rt.rl.router, prefix="/rl", tags=["Reinforcement Learning"])
app.include_router(rt.diffusion.router, prefix="/diffusion", tags=["Diffusion Models"])
app.include_router(rt.mv_grid.router, prefix="/mv_grid", tags=["MV Grid"])
app.include_router(rt.digital_twin.router, prefix="/digital-twins", tags=["Digital Twin"])
app.include_router(rt.offline_scenario.router, prefix="/offline-scenarios", tags=["Offline Scenarios"])
app.include_router(rt.pf_datagen.router, prefix="/pf-datagen", tags=["PF Data Generation"])
app.include_router(rt.settings.router, tags=["Settings"])

# Reachable while no database is configured: the setup flow itself, settings,
# static assets and the API docs.
_NO_DB_ALLOWED = ("/setup", "/settings", "/static/", "/docs", "/redoc", "/openapi.json", "/favicon.ico")
# Also reachable while the database is being prepared: the loading page's own endpoints.
_STARTUP_ALLOWED = _NO_DB_ALLOWED + ("/startup/",)


def _wants_html(request: Request) -> bool:
    return request.method == "GET" and "text/html" in request.headers.get("accept", "")


@app.middleware("http")
async def require_database(request: Request, call_next):
    from gridarena.startup import db_startup

    path = request.url.path
    if not settings_store.db_configured():
        if path.startswith(_NO_DB_ALLOWED):
            return await call_next(request)
        if _wants_html(request):
            return RedirectResponse("/setup", status_code=303)
        return JSONResponse(
            status_code=503,
            content={"detail": "No database configured. Open /setup to choose one."},
        )
    if not db_startup.blocks_requests() or path.startswith(_STARTUP_ALLOWED):
        return await call_next(request)
    # The database is still being prepared (or preparing it failed).
    if _wants_html(request):
        return templates.TemplateResponse(
            request, "startup.html", {"request": request, "status": db_startup.status()},
            status_code=503, headers={"Retry-After": "2", "Cache-Control": "no-store"},
        )
    return JSONResponse(
        status_code=503, headers={"Retry-After": "2"},
        content={"detail": "GridArena is preparing the database; retry shortly.", "startup": db_startup.status()},
    )


@app.get("/startup/status", include_in_schema=False)
def startup_status():
    """Progress of the background database preparation (polled by the loading page)."""
    from gridarena.startup import db_startup

    return db_startup.status()


@app.post("/startup/retry", include_in_schema=False)
def startup_retry():
    """Run the database preparation again after it failed."""
    from gridarena.startup import db_startup

    if db_startup.state == "failed":
        db_startup.start()
    return db_startup.status()

_cors_env = os.getenv("CORS_ORIGINS")
origins = [o.strip() for o in _cors_env.split(",") if o.strip()] if _cors_env else [
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
def root(request: Request):
    import gridarena.database as db
    from gridarena.diffusion.run_manager import diffusion_registry
    from gridarena.pf_datagen.run_manager import pf_datagen_registry
    from gridarena.rl.run_manager import registry as rl_registry

    stats = {"n_lv_grids": 0, "n_mv_grids": 0, "n_historical_databases": 0, "n_active_runs": 0}
    try:
        conn, cursor = db.get_db_connection()
        try:
            cursor.execute("SELECT COUNT(*) FROM grids")
            stats["n_lv_grids"] = cursor.fetchone()[0] or 0
            cursor.execute('SELECT COUNT(*) FROM "MVGrid"')
            stats["n_mv_grids"] = cursor.fetchone()[0] or 0
            cursor.execute('SELECT COUNT(*) FROM "HistoricalDatabase"')
            stats["n_historical_databases"] = cursor.fetchone()[0] or 0
        finally:
            conn.close()
    except Exception:
        logger.warning("Failed to load home page grid/database stats", exc_info=True)

    try:
        active = {"queued", "running"}
        stats["n_active_runs"] = (
            sum(1 for r in rl_registry.list_runs() if r.status in active)
            + sum(1 for r in diffusion_registry.list_runs() if r.status in active)
            + sum(1 for r in pf_datagen_registry.list_runs() if r.status in active)
        )
    except Exception:
        logger.warning("Failed to load home page active-run stats", exc_info=True)

    return templates.TemplateResponse(
        request,
        "home.html",
        {
            "request": request,
            "title": "Grid",
            "active": "home",
            "stats": stats,
        },
    )
