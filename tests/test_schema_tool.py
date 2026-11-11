import sqlite3

import pytest

from app.tools import schema
from app.tools.schema import open_readonly, read_db_schema


@pytest.fixture
def db(tmp_path):
    path = str(tmp_path / "target.db")
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, email TEXT NOT NULL)")
    conn.execute("CREATE TABLE access_log (user_id INTEGER, action TEXT)")
    conn.commit()
    conn.close()
    return path


def test_lists_tables(db):
    res = read_db_schema(db, {})
    assert res.ok and res.data == {"tables": ["access_log", "users"]}


def test_describes_a_table(db):
    res = read_db_schema(db, {"table": "users"})
    assert res.ok
    cols = {c["name"]: c for c in res.data["columns"]}
    assert cols["id"]["pk"] is True
    assert cols["email"]["notnull"] is True and cols["email"]["type"] == "TEXT"


def test_missing_table_error_lists_available_tables(db):
    res = read_db_schema(db, {"table": "usres"})
    assert not res.ok
    assert "users" in res.error and "access_log" in res.error


def test_wrong_argument_type_is_a_result_not_an_exception(db):
    res = read_db_schema(db, {"table": 42})
    assert not res.ok and res.error.startswith("invalid arguments")


@pytest.mark.parametrize(
    "evil", ["users; DROP TABLE users", "users)--", "../etc/passwd", "a b", ""]
)
def test_injection_attempts_are_rejected(db, evil):
    res = read_db_schema(db, {"table": evil})
    assert not res.ok and res.error.startswith("invalid arguments")
    # and the table is still there
    assert read_db_schema(db, {"table": "users"}).ok


def test_unknown_or_bad_json_marker_arguments_are_rejected_not_ignored(db):
    for raw in ({"__bad_json__": "{oops"}, {"tabel": "users"}):
        res = read_db_schema(db, raw)
        assert not res.ok and res.error.startswith("invalid arguments")


def test_open_readonly_rejects_writes(db):
    conn = open_readonly(db)
    with pytest.raises(sqlite3.OperationalError):
        conn.execute("INSERT INTO users (email) VALUES ('x')")
    conn.close()


def test_tool_uses_the_readonly_connection(db, monkeypatch):
    calls = []

    def spy(path):
        calls.append(path)
        return open_readonly(path)

    monkeypatch.setattr(schema, "open_readonly", spy)
    assert schema.read_db_schema(db, {}).ok
    assert calls == [db]


def test_schema_summary_lists_columns_keys_and_foreign_keys(tmp_path):
    path = str(tmp_path / "fk.db")
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, email TEXT NOT NULL)")
    conn.execute("CREATE TABLE logs (user_id INTEGER REFERENCES users(id), action TEXT)")
    conn.execute("CREATE TABLE loose (user_id INTEGER)")
    conn.commit()
    conn.close()
    text = schema.schema_summary(path)
    assert "users(id INTEGER PRIMARY KEY, email TEXT NOT NULL)" in text
    assert "FOREIGN KEY user_id -> users.id" in text
    lines = text.splitlines()
    assert lines[lines.index("loose(user_id INTEGER)") + 1] == "  (no foreign keys)"
