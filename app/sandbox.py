import os
import shutil
import sqlite3
import tempfile
import time
from dataclasses import dataclass, field

from app.tools.schema import open_readonly, schema_summary


@dataclass
class SandboxResult:
    ok: bool
    error: str | None
    schema_after: str | None = None
    rows_before: dict[str, int] = field(default_factory=dict)
    rows_after: dict[str, int] = field(default_factory=dict)


def _authorizer(action, arg1, arg2, db_name, source):
    if action in (sqlite3.SQLITE_ATTACH, sqlite3.SQLITE_DETACH):
        return sqlite3.SQLITE_DENY  # cannot reach other files
    if action == sqlite3.SQLITE_PRAGMA and (arg1 or "").lower() != "foreign_keys":
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


def copy_database(src: str, dst: str) -> None:
    """Consistent copy through SQLite's backup API; the source is opened read-only."""
    source = open_readonly(src)
    dest = sqlite3.connect(dst)
    try:
        source.backup(dest)
    finally:
        dest.close()
        source.close()


def _row_counts(conn: sqlite3.Connection) -> dict[str, int]:
    names = [
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
    ]
    counts = {}
    for name in names:
        quoted = '"' + name.replace('"', '""') + '"'
        counts[name] = conn.execute(f"SELECT COUNT(*) FROM {quoted}").fetchone()[0]
    return counts


def run_in_sandbox(target_db: str, sql: str, timeout_s: float = 5.0) -> SandboxResult:
    if not sql.strip():
        return SandboxResult(False, "empty migration")
    tmp = tempfile.mkdtemp(prefix="auditfix-sbx-")
    try:
        copy = os.path.join(tmp, "copy.db")
        try:
            copy_database(target_db, copy)
        except sqlite3.Error as exc:
            return SandboxResult(False, f"cannot copy target database: {exc}")
        conn = sqlite3.connect(copy)
        try:
            before = _row_counts(conn)
            deadline = time.monotonic() + timeout_s
            conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
            conn.set_authorizer(_authorizer)
            try:
                conn.executescript(sql)
            except sqlite3.Error as exc:
                msg = "timed out" if "interrupted" in str(exc) else f"SQL error: {exc}"
                return SandboxResult(False, msg, rows_before=before)
            conn.set_authorizer(None)
            conn.set_progress_handler(None, 0)
            after = _row_counts(conn)
            problems = []
            for table, n in before.items():
                if table not in after:
                    problems.append(f"table {table} no longer exists")
                elif after[table] < n:
                    problems.append(f"rows lost in {table}: {n} -> {after[table]}")
            violations = conn.execute("PRAGMA foreign_key_check").fetchall()
            if violations:
                table, rowid, parent, _ = violations[0]
                problems.append(
                    f"foreign key violations: {len(violations)} row(s), e.g. {table} "
                    f"rowid {rowid} references a missing row in {parent}"
                )
            return SandboxResult(
                not problems,
                "; ".join(problems) or None,
                schema_summary(copy),
                before,
                after,
            )
        finally:
            conn.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)  # the copy is always deleted
