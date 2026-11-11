import hashlib
import json
import sqlite3

import httpx
import pytest

from app import store
from app.agent.loop import LoopResult
from app.demo_data import make_demo_db
from app.main import app
from app.policy.grounding import CLAUSE
from tests.helpers import wait_for_status


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DB_PATH", str(tmp_path / "api.db"))
    store.init_db()  # ASGITransport does not run lifespan
    target = tmp_path / "target.db"
    conn = sqlite3.connect(target)
    conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, email TEXT)")
    conn.commit()
    conn.close()
    monkeypatch.setenv("AUDITFIX_TARGET_DB", str(target))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def _agent_returning(text: str):
    async def fake(job_id: str):
        return LoopResult("done", text, 1, [])

    return fake


class _StubJudge:
    """Answers both judge prompts: verdicts for the audit, resolutions for the re-check."""

    def __init__(self, payload=None, exc=None, recheck=None):
        self.payload, self.exc, self.recheck, self.calls = payload, exc, recheck, 0

    async def complete_json(self, system, user):
        self.calls += 1
        if self.exc:
            raise self.exc
        if "resolutions" in system:
            if self.recheck is not None:
                return self.recheck
            ids = sorted(set(CLAUSE.findall(user)))
            return json.dumps({"resolutions": [
                {"clause_id": c, "finding": "f", "resolved": True, "reason": "ok"} for c in ids
            ]})
        return self.payload


class _Engineer:
    def __init__(self, *sqls):
        self.sqls, self.calls, self.prompts = list(sqls), 0, []

    async def complete_json(self, system, user):
        self.calls += 1
        self.prompts.append(user)
        item = self.sqls.pop(0)
        return json.dumps(item if isinstance(item, dict) else {"sql": item})


def _verdicts(*items):
    return json.dumps({"verdicts": [
        {"finding": f"finding for {c}", "clause_id": c, "verdict": v, "reason": f"reason {c}"}
        for c, v in items
    ]})


def _digest(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


# ---- basic API --------------------------------------------------------------------


@pytest.mark.anyio
async def test_submit_then_poll_until_done(client, tmp_path):
    target = str(tmp_path / "target.db")
    before = _digest(target)
    resp = await client.post("/jobs")
    assert resp.status_code == 202
    body = resp.json()
    assert body["status_url"] == f"/jobs/{body['job_id']}"

    job = await wait_for_status(client, body["job_id"], "done")
    assert job["result"] == "demo audit: inspected 2 tool results [POL-1.1]"
    g = job["grounding"]
    assert (g["ok"], g["cited"], g["invented"], g["external"]) == (True, ["POL-1.1"], [], [])
    assert g["judge"]["ok"] is True
    assert [v["clause_id"] for v in g["judge"]["verdicts"]] == ["POL-1.1"]
    assert job["error"] is None
    assert _digest(target) == before  # the real target was never modified

    agent = [e for e in job["trace"] if e["kind"] != "node"]
    assert [e["kind"] for e in agent] == ["model", "tool", "model", "tool", "model"]
    assert [e["name"] for e in agent if e["kind"] == "tool"] == ["read_db_schema"] * 2

    nodes = [e for e in job["trace"] if e["kind"] == "node"]
    assert [e["name"] for e in nodes] == ["auditor", "engineer", "sandbox"]
    assert all(e["ok"] is True and e["ms"] >= 0 for e in nodes)

    m = job["migration"]
    assert m["outcome"] == "verified" and m["attempts"] == 1
    assert m["sql"].startswith("CREATE TABLE IF NOT EXISTS")
    assert "audit_demo_note" in m["schema_after"] and m["recheck"]["ok"] is True


@pytest.mark.anyio
async def test_unknown_job_is_404(client):
    resp = await client.get("/jobs/does-not-exist")
    assert resp.status_code == 404


@pytest.mark.anyio
async def test_wait_for_status_times_out_with_clear_message(client):
    job_id = (await client.post("/jobs")).json()["job_id"]
    with pytest.raises(TimeoutError, match="never reached 'failed'"):
        await wait_for_status(client, job_id, "failed", timeout=0.3)


@pytest.mark.anyio
async def test_agent_crash_marks_job_failed(client, monkeypatch):
    async def boom(job_id: str) -> str:
        raise RuntimeError("sandbox exploded")

    monkeypatch.setattr("app.runner.run_agent", boom)

    job_id = (await client.post("/jobs")).json()["job_id"]
    job = await wait_for_status(client, job_id, "failed")
    assert job["error"] == "sandbox exploded"
    assert job["result"] is None


@pytest.mark.anyio
async def test_unknown_llm_mode_fails_the_job_loudly(client, monkeypatch):
    monkeypatch.setenv("AUDITFIX_LLM", "grok")
    job_id = (await client.post("/jobs")).json()["job_id"]
    job = await wait_for_status(client, job_id, "failed")
    assert job["error"] == "unknown AUDITFIX_LLM 'grok'"
    assert job["result"] is None


@pytest.mark.anyio
async def test_trace_is_kept_when_the_agent_hits_its_turn_limit(client, monkeypatch):
    events = [{"turn": 1, "kind": "model", "ms": 5.0, "prompt_tokens": 1, "completion_tokens": 1}]

    async def capped(job_id: str):
        return LoopResult("turn_limit", None, 1, events)

    monkeypatch.setattr("app.runner.run_agent", capped)
    job_id = (await client.post("/jobs")).json()["job_id"]
    job = await wait_for_status(client, job_id, "failed")
    assert job["error"] == "agent stopped: turn_limit"
    assert [e["kind"] for e in job["trace"]] == ["model"]


# ---- grounding: the deterministic citation check ----------------------------------


@pytest.mark.anyio
async def test_outside_standard_sends_the_job_to_review_but_keeps_the_report(client, monkeypatch):
    text = "Plaintext passwords [POL-1.1] violate GDPR Art. 32."
    monkeypatch.setattr("app.runner.run_agent", _agent_returning(text))
    job_id = (await client.post("/jobs")).json()["job_id"]
    job = await wait_for_status(client, job_id, "needs_review")
    assert job["result"] == text and job["error"] is None
    assert job["grounding"]["ok"] is False and job["grounding"]["external"] == ["GDPR"]
    assert job["migration"] is None  # the engineer never ran on an untrusted audit


@pytest.mark.anyio
async def test_invented_clause_sends_the_job_to_review(client, monkeypatch):
    monkeypatch.setattr("app.runner.run_agent", _agent_returning("See [POL-9.9]."))
    job_id = (await client.post("/jobs")).json()["job_id"]
    job = await wait_for_status(client, job_id, "needs_review")
    assert job["grounding"]["invented"] == ["POL-9.9"]


@pytest.mark.anyio
async def test_report_without_citations_goes_to_review(client, monkeypatch):
    monkeypatch.setattr("app.runner.run_agent", _agent_returning("Looks risky."))
    job_id = (await client.post("/jobs")).json()["job_id"]
    job = await wait_for_status(client, job_id, "needs_review")
    assert job["grounding"]["cited"] == []


# ---- grounding: the judge ---------------------------------------------------------


@pytest.mark.anyio
async def test_unsupported_judge_verdict_sends_a_grounded_report_to_review(client, monkeypatch):
    judge = _StubJudge(_verdicts(("POL-1.1", "unsupported")))
    monkeypatch.setattr("app.runner.run_agent", _agent_returning("Bad column [POL-1.1]."))
    monkeypatch.setattr("app.runner.make_judge", lambda: judge)
    job_id = (await client.post("/jobs")).json()["job_id"]
    job = await wait_for_status(client, job_id, "needs_review")
    g = job["grounding"]
    assert g["ok"] is True  # citations were fine; the judge is what rejected it
    assert g["judge"]["ok"] is False
    assert g["judge"]["verdicts"][0]["reason"] == "reason POL-1.1"
    assert job["migration"] is None


@pytest.mark.anyio
async def test_a_crashing_judge_fails_closed_to_review(client, monkeypatch):
    monkeypatch.setattr("app.runner.run_agent", _agent_returning("Bad column [POL-1.1]."))
    monkeypatch.setattr("app.runner.make_judge", lambda: _StubJudge(exc=RuntimeError("down")))
    job_id = (await client.post("/jobs")).json()["job_id"]
    job = await wait_for_status(client, job_id, "needs_review")
    assert job["error"] is None and job["result"] == "Bad column [POL-1.1]."
    assert job["grounding"]["judge"]["error"] == "judge failed: RuntimeError: down"


@pytest.mark.anyio
async def test_judge_skipping_a_cited_clause_goes_to_review(client, monkeypatch):
    judge = _StubJudge(_verdicts(("POL-1.1", "supported")))
    monkeypatch.setattr("app.runner.run_agent", _agent_returning("A [POL-1.1] and B [POL-3.1]."))
    monkeypatch.setattr("app.runner.make_judge", lambda: judge)
    job_id = (await client.post("/jobs")).json()["job_id"]
    job = await wait_for_status(client, job_id, "needs_review")
    assert "POL-3.1" in job["grounding"]["judge"]["error"]


@pytest.mark.anyio
async def test_judge_is_skipped_when_the_deterministic_check_already_failed(client, monkeypatch):
    judge = _StubJudge(_verdicts(("POL-1.1", "supported")))
    monkeypatch.setattr("app.runner.run_agent", _agent_returning("See [POL-9.9]."))
    monkeypatch.setattr("app.runner.make_judge", lambda: judge)
    job_id = (await client.post("/jobs")).json()["job_id"]
    job = await wait_for_status(client, job_id, "needs_review")
    assert judge.calls == 0 and "judge" not in job["grounding"]


# ---- the engineer + sandbox + re-check stage --------------------------------------


@pytest.mark.anyio
async def test_a_trusted_audit_with_a_verified_fix_is_done_and_offers_the_sql(client, monkeypatch):
    judge = _StubJudge(_verdicts(("POL-1.1", "supported")))
    engineer = _Engineer("ALTER TABLE users ADD COLUMN password_hash TEXT;")
    monkeypatch.setattr("app.runner.run_agent", _agent_returning("Bad column [POL-1.1]."))
    monkeypatch.setattr("app.runner.make_judge", lambda: judge)
    monkeypatch.setattr("app.runner.make_engineer", lambda: engineer)
    job_id = (await client.post("/jobs")).json()["job_id"]
    job = await wait_for_status(client, job_id, "done")
    m = job["migration"]
    assert m["outcome"] == "verified" and m["sql"] == "ALTER TABLE users ADD COLUMN password_hash TEXT;"
    assert "password_hash" in m["schema_after"]
    assert m["history"] == [{"attempt": 1, "sql": m["sql"], "error": None, "unfixable": []}]
    assert judge.calls == 2 and engineer.calls == 1  # audit judge + one re-check


@pytest.mark.anyio
async def test_three_failed_attempts_send_a_trusted_report_to_review_with_the_history(client, monkeypatch):
    bad = "ALTER TABLE nope ADD COLUMN x INTEGER;"
    judge = _StubJudge(_verdicts(("POL-1.1", "supported")))
    engineer = _Engineer(bad, bad, bad, "ALTER TABLE users ADD COLUMN y TEXT;")
    monkeypatch.setattr("app.runner.run_agent", _agent_returning("Bad column [POL-1.1]."))
    monkeypatch.setattr("app.runner.make_judge", lambda: judge)
    monkeypatch.setattr("app.runner.make_engineer", lambda: engineer)
    job_id = (await client.post("/jobs")).json()["job_id"]
    job = await wait_for_status(client, job_id, "needs_review")
    assert job["result"] == "Bad column [POL-1.1]." and job["error"] is None
    m = job["migration"]
    assert m["outcome"] == "exhausted" and m["attempts"] == 3 and engineer.calls == 3
    assert m["sql"] is None and m["schema_after"] is None  # nothing verified is offered
    assert [h["sql"] for h in m["history"]] == [bad] * 3
    assert all(h["error"].startswith("SQL error:") for h in m["history"])
    assert judge.calls == 1  # the re-check judge was never asked: every attempt failed the gate


@pytest.mark.anyio
async def test_a_clean_sandbox_but_unresolved_finding_is_retried_and_ends_in_review(client, monkeypatch):
    unresolved = json.dumps({"resolutions": [
        {"clause_id": "POL-1.1", "finding": "f", "resolved": False, "reason": "still plaintext"}]})
    judge = _StubJudge(_verdicts(("POL-1.1", "supported")), recheck=unresolved)
    engineer = _Engineer(*[f"ALTER TABLE users ADD COLUMN c{i} TEXT;" for i in range(4)])
    monkeypatch.setattr("app.runner.run_agent", _agent_returning("Bad column [POL-1.1]."))
    monkeypatch.setattr("app.runner.make_judge", lambda: judge)
    monkeypatch.setattr("app.runner.make_engineer", lambda: engineer)
    job_id = (await client.post("/jobs")).json()["job_id"]
    job = await wait_for_status(client, job_id, "needs_review")
    m = job["migration"]
    assert (m["outcome"], m["attempts"], engineer.calls) == ("exhausted", 3, 3)
    assert m["sql"] is None
    assert [h["error"] for h in m["history"]] == ["[POL-1.1] still unresolved: still plaintext"] * 3
    assert judge.calls == 1 + 3  # the audit judge, then one re-check per clean sandbox run


@pytest.mark.anyio
async def test_node_events_carry_the_attempt_and_error_for_a_retry(client, monkeypatch):
    judge = _StubJudge(_verdicts(("POL-1.1", "supported")))
    engineer = _Engineer("ALTER TABLE nope ADD COLUMN x INTEGER;", "ALTER TABLE users ADD COLUMN y TEXT;")
    monkeypatch.setattr("app.runner.run_agent", _agent_returning("Bad column [POL-1.1]."))
    monkeypatch.setattr("app.runner.make_judge", lambda: judge)
    monkeypatch.setattr("app.runner.make_engineer", lambda: engineer)
    job_id = (await client.post("/jobs")).json()["job_id"]
    job = await wait_for_status(client, job_id, "done")
    nodes = [(e["name"], e["turn"], e["ok"]) for e in job["trace"] if e["kind"] == "node"]
    assert nodes == [("auditor", 0, True), ("engineer", 1, True), ("sandbox", 1, False),
                     ("engineer", 2, True), ("sandbox", 2, True)]
    failed = [e for e in job["trace"] if e["kind"] == "node" and e["ok"] is False][0]
    assert failed["error"].startswith("SQL error:")


@pytest.mark.anyio
async def test_a_destructive_migration_never_touches_the_real_target(tmp_path, monkeypatch):
    target = str(tmp_path / "seeded.db")
    make_demo_db(target)
    monkeypatch.setattr(store, "DB_PATH", str(tmp_path / "api2.db"))
    store.init_db()
    monkeypatch.setenv("AUDITFIX_TARGET_DB", target)
    before = _digest(target)
    engineer = _Engineer("DROP TABLE users; DELETE FROM backups;", "ALTER TABLE users DROP COLUMN password_plain;")
    monkeypatch.setattr("app.runner.run_agent", _agent_returning("Plaintext [POL-1.1]."))
    monkeypatch.setattr("app.runner.make_judge", lambda: _StubJudge(_verdicts(("POL-1.1", "supported"))))
    monkeypatch.setattr("app.runner.make_engineer", lambda: engineer)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        job_id = (await c.post("/jobs")).json()["job_id"]
        job = await wait_for_status(c, job_id, "done")
    assert job["migration"]["history"][0]["error"] == (
        "table users no longer exists; rows lost in backups: 2 -> 0"
    )
    assert job["migration"]["attempts"] == 2
    assert "password_plain" not in job["migration"]["schema_after"]
    assert _digest(target) == before


@pytest.mark.anyio
async def test_rejected_findings_are_listed_but_only_verified_ones_reach_the_engineer(client, monkeypatch):
    judge = _StubJudge(_verdicts(("POL-1.1", "supported"), ("POL-2.1", "unsupported")))
    engineer = _Engineer("ALTER TABLE users ADD COLUMN password_hash TEXT;")
    monkeypatch.setattr("app.runner.run_agent", _agent_returning("A [POL-1.1]. B [POL-2.1]."))
    monkeypatch.setattr("app.runner.make_judge", lambda: judge)
    monkeypatch.setattr("app.runner.make_engineer", lambda: engineer)
    job_id = (await client.post("/jobs")).json()["job_id"]
    job = await wait_for_status(client, job_id, "done")

    j = job["grounding"]["judge"]
    assert j["ok"] is False and j["usable"] is True
    assert [v["clause_id"] for v in j["rejected"]] == ["POL-2.1"]
    prompt = engineer.prompts[0]
    assert "finding for POL-1.1" in prompt and "finding for POL-2.1" not in prompt
    assert job["migration"]["outcome"] == "verified"


@pytest.mark.anyio
async def test_when_every_finding_is_rejected_the_engineer_never_runs(client, monkeypatch):
    judge = _StubJudge(_verdicts(("POL-1.1", "unsupported"), ("POL-2.1", "unsupported")))
    engineer = _Engineer("ALTER TABLE users ADD COLUMN x TEXT;")
    monkeypatch.setattr("app.runner.run_agent", _agent_returning("A [POL-1.1]. B [POL-2.1]."))
    monkeypatch.setattr("app.runner.make_judge", lambda: judge)
    monkeypatch.setattr("app.runner.make_engineer", lambda: engineer)
    job_id = (await client.post("/jobs")).json()["job_id"]
    job = await wait_for_status(client, job_id, "needs_review")
    assert engineer.calls == 0 and job["migration"] is None
    assert len(job["grounding"]["judge"]["rejected"]) == 2


@pytest.mark.anyio
async def test_a_partial_fix_is_done_with_the_declined_finding_listed_for_a_human(client, monkeypatch):
    judge = _StubJudge(_verdicts(("POL-1.1", "supported"), ("POL-2.1", "supported")))
    engineer = _Engineer({"sql": "ALTER TABLE users ADD COLUMN h TEXT;",
                          "unfixable": [{"finding_id": "F2", "reason": "needs app code"}]})
    monkeypatch.setattr("app.runner.run_agent", _agent_returning("A [POL-1.1]. B [POL-2.1]."))
    monkeypatch.setattr("app.runner.make_judge", lambda: judge)
    monkeypatch.setattr("app.runner.make_engineer", lambda: engineer)
    job_id = (await client.post("/jobs")).json()["job_id"]
    job = await wait_for_status(client, job_id, "done")
    m = job["migration"]
    assert m["outcome"] == "verified" and m["sql"].startswith("ALTER TABLE users")
    assert [(u["id"], u["clause_id"], u["reason"]) for u in m["unfixable"]] == [("F2", "POL-2.1", "needs app code")]
    assert "F1 [POL-1.1]" in engineer.prompts[0] and "F2 [POL-2.1]" in engineer.prompts[0]


@pytest.mark.anyio
async def test_declining_everything_sends_the_job_to_review_with_no_sql_offered(client, monkeypatch):
    judge = _StubJudge(_verdicts(("POL-1.1", "supported"), ("POL-2.1", "supported")))
    engineer = _Engineer({"sql": "", "unfixable": [
        {"finding_id": "F1", "reason": "r1"}, {"finding_id": "F2", "reason": "r2"}]})
    monkeypatch.setattr("app.runner.run_agent", _agent_returning("A [POL-1.1]. B [POL-2.1]."))
    monkeypatch.setattr("app.runner.make_judge", lambda: judge)
    monkeypatch.setattr("app.runner.make_engineer", lambda: engineer)
    job_id = (await client.post("/jobs")).json()["job_id"]
    job = await wait_for_status(client, job_id, "needs_review")
    m = job["migration"]
    assert m["outcome"] == "unfixable" and m["sql"] is None and m["schema_after"] is None
    assert [u["reason"] for u in m["unfixable"]] == ["r1", "r2"]
    assert judge.calls == 1  # the audit judge only; no re-check, nothing was attempted
