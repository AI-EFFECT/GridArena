"""Setup / Settings pages: no real database, a temporary .env file."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from gridarena import settings_store as store
from gridarena.app import app

GOOD_DSN = "postgresql://alice:s3cret@db.example:5432/gridarena"


@pytest.fixture
def env(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    monkeypatch.setenv("GRIDARENA_ENV_FILE", str(path))
    for key in ("GRID_DB_DSN", "DATABASE_URL", "LVGPLAY_API_KEY", "LLM_MODEL", "LLM_API_KEY", "MY_TOKEN"):
        monkeypatch.delenv(key, raising=False)
    with patch("gridarena.routers.settings.store.check_connection", return_value=None), \
         patch("gridarena.routers.settings._initialize_database", new=AsyncMock()):
        yield path


@pytest.fixture
def client():
    return TestClient(app)


def test_pages_redirect_to_setup_without_database(env, client):
    r = client.get("/", headers={"accept": "text/html"}, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/setup"


def test_api_answers_503_without_database(env, client):
    r = client.get("/rl/runs", headers={"accept": "application/json"})
    assert r.status_code == 503
    assert "setup" in r.json()["detail"].lower()


def test_setup_saves_dsn_after_successful_test(env, client):
    r = client.post("/setup", data={"mode": "dsn", "dsn": GOOD_DSN}, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/"
    assert store.read_env_file()["GRID_DB_DSN"] == GOOD_DSN
    assert store.db_configured()


def test_setup_builds_dsn_from_fields(env, client):
    r = client.post("/setup", data={"mode": "fields", "host": "db", "port": "5433", "dbname": "ga",
                                    "user": "bob", "password": "pw"}, follow_redirects=False)
    assert r.status_code == 303
    dsn = store.read_env_file()["GRID_DB_DSN"]
    assert "host=db" in dsn and "port=5433" in dsn and "user=bob" in dsn


def test_setup_refuses_unreachable_database(env, client):
    with patch("gridarena.routers.settings.store.check_connection", return_value="connection refused"):
        r = client.post("/setup", data={"mode": "dsn", "dsn": GOOD_DSN})
    assert r.status_code == 400 and "connection refused" in r.text
    assert not env.exists() or "GRID_DB_DSN" not in store.read_env_file()


def test_setup_is_closed_once_configured(env, client, monkeypatch):
    monkeypatch.setenv("GRID_DB_DSN", GOOD_DSN)
    r = client.post("/setup", data={"mode": "dsn", "dsn": "postgresql://evil@x/db"}, follow_redirects=False)
    assert r.headers["location"] == "/settings"
    assert "evil" not in (env.read_text() if env.exists() else "")


def test_settings_never_renders_secrets(env, client, monkeypatch):
    monkeypatch.setenv("GRID_DB_DSN", GOOD_DSN)
    store.save({"LLM_API_KEY": "sk-very-secret", "MY_TOKEN": "tok-123"})
    html = client.get("/settings").text
    assert "sk-very-secret" not in html and "tok-123" not in html and "s3cret" not in html
    assert "MY_TOKEN" in html  # custom variable listed


def test_settings_updates_and_keeps_blank_secrets(env, client, monkeypatch):
    monkeypatch.setenv("GRID_DB_DSN", GOOD_DSN)
    store.save({"LLM_API_KEY": "old-key"})
    r = client.post("/settings", data={"val__LLM_MODEL": "llama3", "val__LLM_API_KEY": "",
                                       "new_key": "MY_TOKEN", "new_value": "abc"}, follow_redirects=False)
    assert r.status_code == 303
    saved = store.read_env_file()
    assert saved["LLM_MODEL"] == "llama3" and saved["LLM_API_KEY"] == "old-key" and saved["MY_TOKEN"] == "abc"


def test_settings_rejects_invalid_values(env, client, monkeypatch):
    monkeypatch.setenv("GRID_DB_DSN", GOOD_DSN)
    r = client.post("/settings", data={"val__DB_POOL_MAX_SIZE": "lots"})
    assert r.status_code == 400
    assert "DB_POOL_MAX_SIZE" not in store.read_env_file()


def test_settings_require_access_key_when_set(env, client, monkeypatch):
    monkeypatch.setenv("GRID_DB_DSN", GOOD_DSN)
    monkeypatch.setenv("LVGPLAY_API_KEY", "k")
    assert client.post("/settings", data={"val__LLM_MODEL": "x"}).status_code == 403
    r = client.post("/settings", data={"val__LLM_MODEL": "x", "admin_key": "k"}, follow_redirects=False)
    assert r.status_code == 303


def test_new_database_is_tested_before_saving(env, client, monkeypatch):
    monkeypatch.setenv("GRID_DB_DSN", GOOD_DSN)
    with patch("gridarena.routers.settings.store.check_connection", return_value="no route to host"):
        r = client.post("/settings", data={"val__GRID_DB_DSN": "postgresql://u@other/db"})
    assert r.status_code == 400 and "no route to host" in r.text
    assert "GRID_DB_DSN" not in store.read_env_file()


MISSING = 'connection failed: FATAL:  database "gridarena" does not exist'


def test_setup_offers_to_create_missing_database(env, client):
    with patch("gridarena.routers.settings.store.check_connection", return_value=MISSING):
        r = client.post("/setup", data={"mode": "dsn", "dsn": GOOD_DSN})
    assert r.status_code == 400 and "does not exist" in r.text
    assert 'name="create_db" value="1" checked' in r.text  # pre-ticked for the retry
    assert not env.exists() or "GRID_DB_DSN" not in store.read_env_file()


def test_setup_creates_database_when_asked(env, client):
    with patch("gridarena.routers.settings.store.create_database", return_value=None) as create:
        r = client.post("/setup", data={"mode": "dsn", "dsn": GOOD_DSN, "create_db": "1"},
                        follow_redirects=False)
    create.assert_called_once_with(GOOD_DSN)
    assert r.status_code == 303 and store.read_env_file()["GRID_DB_DSN"] == GOOD_DSN


def test_setup_does_not_create_database_unless_asked(env, client):
    with patch("gridarena.routers.settings.store.create_database") as create:
        client.post("/setup", data={"mode": "dsn", "dsn": GOOD_DSN}, follow_redirects=False)
    create.assert_not_called()


def test_setup_reports_create_failure(env, client):
    with patch("gridarena.routers.settings.store.create_database", return_value="not allowed"):
        r = client.post("/setup", data={"mode": "dsn", "dsn": GOOD_DSN, "create_db": "1"})
    assert r.status_code == 400 and "not allowed" in r.text
    assert not env.exists() or "GRID_DB_DSN" not in store.read_env_file()


class _FakeConn:
    """Minimal psycopg connection: records SQL, knows which databases exist."""

    def __init__(self, existing, log):
        self.existing, self.log = existing, log

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, query, params=None):
        self.log.append((query, params))
        row = (1,) if params and params[0] in self.existing else None
        return type("Cur", (), {"fetchone": lambda _self: row})()


def _fake_connect(existing, log, dbnames):
    def connect(conninfo, **kwargs):
        from psycopg.conninfo import conninfo_to_dict

        dbnames.append((conninfo_to_dict(conninfo)["dbname"], kwargs.get("autocommit")))
        return _FakeConn(existing, log)
    return connect


def test_create_database_runs_create_on_maintenance_db():
    from psycopg import sql

    log, dbnames = [], []
    with patch("psycopg.connect", _fake_connect(set(), log, dbnames)):
        assert store.create_database(GOOD_DSN) is None
    assert dbnames == [("postgres", True)]  # CREATE DATABASE can't run inside a transaction
    create = log[-1][0]
    assert isinstance(create, sql.Composed) and sql.Identifier("gridarena") in list(create)


def test_create_database_is_noop_when_it_exists():
    log, dbnames = [], []
    with patch("psycopg.connect", _fake_connect({"gridarena"}, log, dbnames)):
        assert store.create_database(GOOD_DSN) is None
    assert len(log) == 1  # only the existence check


def test_missing_database_error_detection():
    assert store.is_missing_database(MISSING)
    assert not store.is_missing_database("connection refused")
    assert not store.is_missing_database(None)
