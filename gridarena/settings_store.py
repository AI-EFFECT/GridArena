"""Runtime configuration stored in a .env file, editable from the web UI.

The file is ``GRIDARENA_ENV_FILE`` if set (the Docker image points it at a
mounted volume), otherwise ``.env`` at the repository root. app.py loads it at
startup with ``override=True``, so values saved from the UI win over the
process environment on the next start; saving also updates ``os.environ`` so
most changes apply immediately.
"""

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import dotenv_values, set_key, unset_key

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DB_KEYS = ("GRID_DB_DSN", "DATABASE_URL")
KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def env_file() -> Path:
    return Path(os.getenv("GRIDARENA_ENV_FILE") or REPO_ROOT / ".env")


@dataclass(frozen=True)
class Setting:
    key: str
    group: str
    description: str
    secret: bool = False
    # "now": read on every use; "restart": read once at startup / import.
    applies: str = "now"
    kind: str = "text"  # "text" | "int" | "bool01" | "truefalse" | "choice"
    choices: tuple = ()
    placeholder: str = ""


CATALOG: List[Setting] = [
    Setting("GRID_DB_DSN", "Database", "PostgreSQL connection string. Takes priority over DATABASE_URL.",
            secret=True, placeholder="postgresql://user:password@host:5432/gridarena"),
    Setting("DATABASE_URL", "Database", "Alternative name for the connection string (used if GRID_DB_DSN is empty).",
            secret=True, placeholder="postgresql://user:password@host:5432/gridarena"),
    Setting("DB_POOL_MIN_SIZE", "Database", "Minimum pooled connections per process.", kind="int", placeholder="2"),
    Setting("DB_POOL_MAX_SIZE", "Database", "Maximum pooled connections per process.", kind="int", placeholder="10"),

    Setting("LVGPLAY_API_KEY", "Access", "Key required to save settings from this page and to open "
            "document-assistant chats. Leave empty for an open instance.", secret=True),
    Setting("CORS_ORIGINS", "Access", "Comma-separated origins allowed to call the API from a browser.",
            applies="restart", placeholder="http://localhost:5173"),

    Setting("LVGPLAY_ANGLE_UNIT", "Benchmarks", "Unit of voltage angles in generated benchmark noise.",
            applies="restart", kind="choice", choices=("deg", "rad"), placeholder="deg"),

    Setting("LLM_API_URL", "Chat assistant", "Base URL of the LLM API.",
            placeholder="https://llm.example.com/api"),
    Setting("LLM_API_KEY", "Chat assistant", "API key for the LLM. The documentation index is built at startup, "
            "so restart after setting it for the first time.", secret=True, applies="restart"),
    Setting("LLM_MODEL", "Chat assistant", "Model name.", placeholder="gemma4-31b"),
    Setting("LLM_VERIFY_SSL", "Chat assistant", "Verify the LLM server's TLS certificate.",
            kind="truefalse", choices=("true", "false"), placeholder="false"),
    Setting("LVGPLAY_LLM_EMBEDDING_MODEL", "Chat assistant", "Embedding model for the documentation index.",
            applies="restart", placeholder="all-MiniLM-L6-v2"),
    Setting("LVGPLAY_LLM_ENABLE_DESTRUCTIVE", "Chat assistant", "Expose delete tools to the model (still need "
            "explicit confirmation).", applies="restart", kind="bool01", choices=("0", "1"), placeholder="0"),
    Setting("LVGPLAY_LLM_MAX_TOOL_RESULT_CHARS", "Chat assistant", "Size cap per tool result (characters).",
            applies="restart", kind="int", placeholder="8000"),
    Setting("LVGPLAY_LLM_HISTORY_TOKEN_BUDGET", "Chat assistant", "Conversation history budget (estimated tokens).",
            applies="restart", kind="int", placeholder="24000"),
    Setting("LVGPLAY_LLM_MAX_TOOL_ROUNDS", "Chat assistant", "Tool-call rounds per message before answering.",
            applies="restart", kind="int", placeholder="4"),
    Setting("LVGPLAY_DATA_DIR", "Chat assistant", "Writable directory for assistant data.",
            applies="restart", placeholder="./data/llm"),

    Setting("LVGPLAY_ALLOW_DOCUMENT_AGENTS", "Document assistants", "Enable document assistants.",
            kind="bool01", choices=("0", "1"), placeholder="0"),
    Setting("LVGPLAY_LLM_MAX_SOURCE_MB", "Document assistants", "Upload cap per document (MB).",
            kind="int", placeholder="5"),
    Setting("LVGPLAY_LLM_MAX_SOURCES_PER_AGENT", "Document assistants", "Documents per assistant.",
            kind="int", placeholder="20"),
    Setting("LVGPLAY_LLM_DOC_CHUNKS_PER_ANSWER", "Document assistants", "Excerpts retrieved per question.",
            kind="int", placeholder="6"),

    Setting("REDIS_URL", "Digital twin", "Message broker for digital twins.", applies="restart",
            placeholder="redis://localhost:6379"),
]
CATALOG_BY_KEY: Dict[str, Setting] = {s.key: s for s in CATALOG}
GROUPS: List[str] = list(dict.fromkeys(s.group for s in CATALOG))


def custom_setting(key: str) -> Setting:
    """Variables outside the catalog, e.g. a digital twin's credential named by
    its secret_ref. Treated as secrets: their values are never shown."""
    return Setting(key, "Custom", "Custom variable (e.g. a digital-twin credential).", secret=True)


# ── Reading ──────────────────────────────────────────────────────────────────


def read_env_file() -> Dict[str, Optional[str]]:
    path = env_file()
    return dict(dotenv_values(path)) if path.exists() else {}


def current_dsn() -> Optional[str]:
    return os.getenv("GRID_DB_DSN") or os.getenv("DATABASE_URL") or None


def db_configured() -> bool:
    return current_dsn() is not None


def redact_dsn(dsn: Optional[str]) -> str:
    """Connection string with the password hidden, for display."""
    if not dsn:
        return ""
    try:
        from psycopg.conninfo import conninfo_to_dict

        parts = conninfo_to_dict(dsn)
        host = parts.get("host", "localhost")
        port = parts.get("port", "5432")
        user = parts.get("user", "")
        dbname = parts.get("dbname", "")
        pw = ":••••" if parts.get("password") else ""
        return f"postgresql://{user}{pw}@{host}:{port}/{dbname}"
    except Exception:
        return "•••• (set)"


def build_dsn(host: str, port: str, dbname: str, user: str, password: str, sslmode: str = "") -> str:
    from psycopg.conninfo import make_conninfo

    params = {"host": host.strip(), "port": (port or "5432").strip(), "dbname": dbname.strip(),
              "user": user.strip()}
    if password:
        params["password"] = password
    if sslmode:
        params["sslmode"] = sslmode
    return make_conninfo("", **{k: v for k, v in params.items() if v})


def check_connection(dsn: str, timeout: int = 5) -> Optional[str]:
    """None if the database answers, otherwise a short error message."""
    import psycopg

    try:
        with psycopg.connect(dsn, connect_timeout=timeout) as conn:
            conn.execute("SELECT 1")
        return None
    except Exception as e:  # psycopg errors carry the useful text in str(e)
        return _short_error(e)


def _short_error(e: Exception) -> str:
    text = str(e).strip()
    return text.splitlines()[0] if text else type(e).__name__


def database_name(dsn: str) -> str:
    from psycopg.conninfo import conninfo_to_dict

    try:
        return conninfo_to_dict(dsn).get("dbname", "") or ""
    except Exception:
        return ""


def is_missing_database(error: Optional[str]) -> bool:
    """True when a check_connection() error means the server is fine but the database doesn't exist."""
    return bool(error) and "database" in error and "does not exist" in error


def create_database(dsn: str, timeout: int = 5) -> Optional[str]:
    """Create the (empty) database named in `dsn` if it doesn't exist yet.

    Connects to the server's maintenance database (postgres, or template1) with
    the same credentials. Returns None on success or when the database already
    exists, otherwise a short error message. GridArena creates its own tables
    on first use, so an empty database is all it needs.
    """
    import psycopg
    from psycopg import sql
    from psycopg.conninfo import make_conninfo

    name = database_name(dsn)
    if not name:
        return "The connection string doesn't name a database."

    conn = None
    error = None
    for maintenance_db in ("postgres", "template1"):
        try:
            conn = psycopg.connect(make_conninfo(dsn, dbname=maintenance_db), connect_timeout=timeout,
                                   autocommit=True)
            break
        except Exception as e:
            error = _short_error(e)
    if conn is None:
        return f"Could not reach the server to create the database: {error}"

    try:
        with conn:
            exists = conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (name,)).fetchone()
            if not exists:
                conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
                logger.info("Created database %r", name)
        return None
    except psycopg.errors.InsufficientPrivilege:
        return (f'This user is not allowed to create databases. Ask the server administrator to create '
                f'"{name}" (or grant CREATEDB), then try again.')
    except Exception as e:
        return _short_error(e)


# ── Writing ──────────────────────────────────────────────────────────────────


def validate(key: str, value: str) -> Optional[str]:
    """An error message, or None when key/value can be stored."""
    if not KEY_RE.match(key):
        return f"{key!r} is not a valid variable name (letters, digits and _; not starting with a digit)."
    if "\n" in value or "\r" in value:
        return f"{key}: the value must be on one line."
    setting = CATALOG_BY_KEY.get(key)
    if setting and value:
        if setting.kind == "int" and not re.fullmatch(r"\d+", value):
            return f"{key} must be a whole number."
        if setting.choices and value not in setting.choices:
            return f"{key} must be one of: {', '.join(setting.choices)}."
    return None


def save(updates: Dict[str, Optional[str]]) -> None:
    """Write to the env file and os.environ. A value of None removes the key."""
    path = env_file()
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# GridArena settings — managed from the Settings page.\n", encoding="utf-8")
    for key, value in updates.items():
        if value is None:
            unset_key(path, key, quote_mode="always")
            os.environ.pop(key, None)
        else:
            set_key(path, key, value, quote_mode="always")
            os.environ[key] = value
