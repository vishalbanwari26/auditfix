import json
import sqlite3

from app.jobs import Job, JobStatus

DB_PATH = "auditfix.db"

_JSON_COLUMNS = ("grounding", "migration")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                result TEXT,
                error TEXT,
                grounding TEXT,
                migration TEXT
            )"""
        )
        # databases created before a column existed lack it
        cols = {r[1] for r in conn.execute("PRAGMA table_info(jobs)")}
        for name in _JSON_COLUMNS:
            if name not in cols:
                conn.execute(f"ALTER TABLE jobs ADD COLUMN {name} TEXT")
        conn.execute(
            """CREATE TABLE IF NOT EXISTS trace_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL,
                turn INTEGER,
                kind TEXT NOT NULL,
                name TEXT,
                args TEXT,
                ok INTEGER,
                error TEXT,
                ms REAL,
                prompt_tokens INTEGER,
                completion_tokens INTEGER
            )"""
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_trace_job ON trace_events (job_id)"
        )


def _insert_events(conn: sqlite3.Connection, job_id: str, events: list[dict]) -> None:
    conn.executemany(
        """INSERT INTO trace_events
           (job_id, turn, kind, name, args, ok, error, ms, prompt_tokens, completion_tokens)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            (
                job_id,
                e["turn"],
                e["kind"],
                e.get("name"),
                json.dumps(e["args"]) if "args" in e else None,
                None if e.get("ok") is None else int(e["ok"]),
                e.get("error"),
                e.get("ms"),
                e.get("prompt_tokens"),
                e.get("completion_tokens"),
            )
            for e in events
        ],
    )


def append_trace(job_id: str, events: list[dict]) -> None:
    with _connect() as conn:
        _insert_events(conn, job_id, events)


def save_trace(job_id: str, events: list[dict]) -> None:
    """Replace the job's whole trace."""
    with _connect() as conn:
        conn.execute("DELETE FROM trace_events WHERE job_id = ?", (job_id,))
        _insert_events(conn, job_id, events)


def load_trace(job_id: str) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM trace_events WHERE job_id = ? ORDER BY id", (job_id,)
        ).fetchall()
    return [
        {
            "turn": r["turn"],
            "kind": r["kind"],
            "name": r["name"],
            "args": None if r["args"] is None else json.loads(r["args"]),
            "ok": None if r["ok"] is None else bool(r["ok"]),
            "error": r["error"],
            "ms": r["ms"],
            "prompt_tokens": r["prompt_tokens"],
            "completion_tokens": r["completion_tokens"],
        }
        for r in rows
    ]


def _dump(value: dict | None) -> str | None:
    return None if value is None else json.dumps(value)


def save_job(job: Job) -> None:
    with _connect() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO jobs (id, status, result, error, grounding, migration)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                job.id,
                job.status.value,
                job.result,
                job.error,
                _dump(job.grounding),
                _dump(job.migration),
            ),
        )


def load_job(job_id: str) -> Job | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if row is None:
        return None
    data = dict(row)
    for name in _JSON_COLUMNS:
        data[name] = json.loads(data[name]) if data[name] else None
    return Job(**data)


def fail_interrupted_jobs() -> int:
    """Run at startup: anything still 'running' or 'pending' lost its worker."""
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE jobs SET status = ?, error = ? WHERE status IN (?, ?)",
            (
                JobStatus.FAILED.value,
                "interrupted by restart",
                JobStatus.RUNNING.value,
                JobStatus.PENDING.value,
            ),
        )
    return cur.rowcount
