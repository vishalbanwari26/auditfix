import pytest

from app import store
from app.jobs import Job, JobStatus


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DB_PATH", str(tmp_path / "test.db"))
    store.init_db()


def test_save_then_load_roundtrip():
    job = Job(id="a", status=JobStatus.DONE, result="ok")
    store.save_job(job)
    assert store.load_job("a") == job


def test_load_unknown_returns_none():
    assert store.load_job("missing") is None


def test_fail_interrupted_jobs_only_touches_unfinished():
    store.save_job(Job(id="pending", status=JobStatus.PENDING))
    store.save_job(Job(id="running", status=JobStatus.RUNNING))
    store.save_job(Job(id="done", status=JobStatus.DONE, result="ok"))

    assert store.fail_interrupted_jobs() == 2

    for job_id in ("pending", "running"):
        job = store.load_job(job_id)
        assert job.status == JobStatus.FAILED
        assert job.error == "interrupted by restart"
    assert store.load_job("done").status == JobStatus.DONE
    assert store.load_job("done").error is None


def test_trace_roundtrip_preserves_order_and_args():
    events = [
        {"turn": 1, "kind": "model", "ms": 12.5, "prompt_tokens": 100, "completion_tokens": 20},
        {"turn": 1, "kind": "tool", "name": "read_db_schema", "args": {"table": "users"},
         "ok": False, "error": "no table", "ms": 1.0},
    ]
    store.save_trace("j", events)
    got = store.load_trace("j")
    assert [e["kind"] for e in got] == ["model", "tool"]
    assert got[0]["prompt_tokens"] == 100 and got[0]["args"] is None and got[0]["ok"] is None
    assert got[1]["args"] == {"table": "users"}
    assert got[1]["ok"] is False and got[1]["error"] == "no table"


def test_save_trace_twice_replaces_instead_of_appending():
    store.save_trace("j", [{"turn": 1, "kind": "model", "ms": 1.0}])
    store.save_trace("j", [{"turn": 1, "kind": "model", "ms": 2.0},
                           {"turn": 2, "kind": "model", "ms": 3.0}])
    assert [e["ms"] for e in store.load_trace("j")] == [2.0, 3.0]


def test_traces_are_kept_per_job():
    store.save_trace("a", [{"turn": 1, "kind": "model", "ms": 1.0}])
    store.save_trace("b", [{"turn": 1, "kind": "model", "ms": 9.0}])
    assert [e["ms"] for e in store.load_trace("a")] == [1.0]
    assert store.load_trace("missing") == []


def test_grounding_roundtrips_and_defaults_to_none():
    g = {"ok": False, "cited": ["POL-1.1"], "invented": [], "external": ["GDPR"]}
    store.save_job(Job(id="g", status=JobStatus.NEEDS_REVIEW, result="r", grounding=g))
    store.save_job(Job(id="n", status=JobStatus.DONE))
    assert store.load_job("g").grounding == g
    assert store.load_job("g").status == JobStatus.NEEDS_REVIEW
    assert store.load_job("n").grounding is None


def test_init_db_upgrades_a_database_created_before_grounding_existed(tmp_path, monkeypatch):
    import sqlite3

    old = str(tmp_path / "old.db")
    conn = sqlite3.connect(old)
    conn.execute("CREATE TABLE jobs (id TEXT PRIMARY KEY, status TEXT NOT NULL, result TEXT, error TEXT)")
    conn.execute("INSERT INTO jobs VALUES ('old1', 'done', 'r', NULL)")
    conn.commit()
    conn.close()

    monkeypatch.setattr(store, "DB_PATH", old)
    store.init_db()
    store.init_db()  # idempotent
    job = store.load_job("old1")
    assert job.status == JobStatus.DONE and job.grounding is None
