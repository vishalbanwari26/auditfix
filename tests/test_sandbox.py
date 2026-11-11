import hashlib
import os
import sqlite3

import pytest

from app import sandbox
from app.demo_data import make_demo_db
from app.sandbox import run_in_sandbox

needs_drop_column = pytest.mark.skipif(
    sqlite3.sqlite_version_info < (3, 35, 0), reason="DROP COLUMN needs SQLite 3.35+"
)


@pytest.fixture
def target(tmp_path):
    path = str(tmp_path / "target.db")
    make_demo_db(path)
    return path


def digest(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


@needs_drop_column
def test_a_good_migration_passes_and_the_new_schema_is_returned(target):
    res = run_in_sandbox(target, "ALTER TABLE users DROP COLUMN password_plain;")
    assert res.ok and res.error is None
    assert "password_plain" not in res.schema_after and "email" in res.schema_after
    assert res.rows_before == res.rows_after == {"users": 3, "access_log": 4, "backups": 2}


def test_dropping_and_recreating_a_table_loses_rows_and_fails(target):
    sql = "DROP TABLE users; CREATE TABLE users (id INTEGER PRIMARY KEY, email TEXT NOT NULL);"
    res = run_in_sandbox(target, sql)
    assert not res.ok and res.error == "rows lost in users: 3 -> 0"


def test_dropping_a_table_without_recreating_it_fails(target):
    res = run_in_sandbox(target, "DROP TABLE backups;")
    assert not res.ok and res.error == "table backups no longer exists"


def test_adding_rows_and_tables_is_allowed(target):
    res = run_in_sandbox(target, "CREATE TABLE extra (x INTEGER); INSERT INTO backups VALUES (3, '/b', 1);")
    assert res.ok and res.rows_after["backups"] == 3


def test_sql_errors_come_back_as_readable_results(target):
    res = run_in_sandbox(target, "ALTER TABLE nope ADD COLUMN x INTEGER;")
    assert not res.ok and res.error.startswith("SQL error:") and "nope" in res.error


def test_empty_migration_is_rejected(target):
    assert run_in_sandbox(target, "   \n").error == "empty migration"


def test_foreign_key_violations_are_caught(target):
    sql = "CREATE TABLE c (uid INTEGER REFERENCES users(id)); INSERT INTO c VALUES (999);"
    res = run_in_sandbox(target, sql)
    assert not res.ok
    assert res.error == "foreign key violations: 1 row(s), e.g. c rowid 1 references a missing row in users"


def test_attach_is_blocked_and_no_file_is_created(target, tmp_path):
    outside = tmp_path / "stolen.db"
    res = run_in_sandbox(target, f"ATTACH DATABASE '{outside}' AS x; CREATE TABLE x.t (a);")
    assert not res.ok and "not authorized" in res.error
    assert not outside.exists()


def test_other_pragmas_are_blocked(target):
    res = run_in_sandbox(target, "PRAGMA writable_schema = 1;")
    assert not res.ok and "not authorized" in res.error


def test_a_runaway_query_times_out(target):
    sql = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM c) SELECT count(*) FROM c;"
    res = run_in_sandbox(target, sql, timeout_s=0.3)
    assert not res.ok and res.error == "timed out"


def test_the_real_target_is_never_modified(target):
    before = digest(target)
    run_in_sandbox(target, "DROP TABLE users; DELETE FROM backups;")
    run_in_sandbox(target, "INSERT INTO backups VALUES (50, 'x', 1);")
    assert digest(target) == before


def test_the_temp_copy_is_deleted_even_when_the_migration_fails(target, monkeypatch):
    made = []
    real = sandbox.tempfile.mkdtemp

    def spy(**kw):
        path = real(**kw)
        made.append(path)
        return path

    monkeypatch.setattr(sandbox.tempfile, "mkdtemp", spy)
    run_in_sandbox(target, "THIS IS NOT SQL;")
    run_in_sandbox(target, "ALTER TABLE users ADD COLUMN note TEXT;")
    assert len(made) == 2 and not any(os.path.exists(p) for p in made)


def test_a_missing_target_returns_a_failed_result_not_an_exception(tmp_path):
    res = run_in_sandbox(str(tmp_path / "missing.db"), "SELECT 1;")
    assert not res.ok and res.error.startswith("cannot copy target database:")


def test_make_demo_db_refuses_to_overwrite(tmp_path):
    path = str(tmp_path / "t.db")
    make_demo_db(path)
    with pytest.raises(FileExistsError):
        make_demo_db(path)


FK_REBUILD = """
CREATE TABLE access_log_new (user_id INTEGER REFERENCES users(id), action TEXT, ip TEXT, at TEXT);
INSERT INTO access_log_new SELECT * FROM access_log;
DROP TABLE access_log;
ALTER TABLE access_log_new RENAME TO access_log;
"""


def test_default_demo_data_has_an_orphan_so_adding_a_foreign_key_is_refused(target):
    res = run_in_sandbox(target, FK_REBUILD)
    assert not res.ok
    assert res.error == "foreign key violations: 1 row(s), e.g. access_log rowid 4 references a missing row in users"


def test_clean_demo_data_lets_the_same_foreign_key_migration_pass(tmp_path):
    clean = str(tmp_path / "clean.db")
    make_demo_db(clean, with_orphan=False)
    res = run_in_sandbox(clean, FK_REBUILD)
    assert res.ok and res.rows_before == res.rows_after == {"users": 3, "access_log": 3, "backups": 2}
    assert "FOREIGN KEY user_id -> users.id" in res.schema_after
