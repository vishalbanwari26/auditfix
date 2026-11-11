import hashlib
import json

import pytest
from langgraph.errors import GraphRecursionError
from langgraph.graph import END

from app import graph
from app.demo_data import make_demo_db
from app.graph import Deps, after_audit, after_engineer, after_sandbox, run_graph
from app.tools.schema import schema_summary

GOOD_SQL = "ALTER TABLE users DROP COLUMN password_plain;"
LOSSY_SQL = "DROP TABLE users; CREATE TABLE users (id INTEGER PRIMARY KEY, email TEXT NOT NULL);"
FINDINGS = [
    {"id": "F1", "clause_id": "POL-1.1", "finding": "plaintext passwords in users.password_plain"},
    {"id": "F2", "clause_id": "POL-2.1", "finding": "no foreign key on access_log.user_id"},
]
CLAUSES = {
    "POL-1.1": {"title": "Credential storage", "text": "Passwords must be hashed."},
    "POL-2.1": {"title": "Referential integrity", "text": "Logs need a foreign key."},
}


class Sequence:
    """A model stub that plays back payloads (str) or raises exceptions, recording prompts."""

    def __init__(self, *items):
        self.items, self.prompts = list(items), []

    async def complete_json(self, system: str, user: str) -> str:
        self.prompts.append(user)
        item = self.items.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def sql(text: str) -> str:
    return json.dumps({"sql": text})


def resolutions(*pairs) -> str:
    return json.dumps({"resolutions": [
        {"clause_id": c, "finding": "f", "resolved": ok, "reason": why} for c, ok, why in pairs
    ]})


ALL_RESOLVED = resolutions(("POL-1.1", True, "gone"), ("POL-2.1", True, "fk present"))


@pytest.fixture
def target(tmp_path):
    path = str(tmp_path / "target.db")
    make_demo_db(path)
    return path


def make_deps(target, engineer, judge, trusted=True):
    async def audit():
        return {
            "trusted": trusted,
            "findings": FINDINGS if trusted else [],
            "clauses": CLAUSES,
            "schema": schema_summary(target),
        }

    return Deps(audit=audit, engineer_client=engineer, judge_client=judge, target_db=target)


# ---- routers ----------------------------------------------------------------------


def test_after_audit_stops_when_there_are_no_trusted_findings():
    assert after_audit({"findings": []}) == END
    assert after_audit({"findings": FINDINGS}) == "engineer"


def test_after_engineer_routes_by_whether_it_produced_sql_and_the_cap():
    assert after_engineer({"sql": "x", "attempts": 1}) == "sandbox"
    assert after_engineer({"sql": None, "attempts": 2}) == "engineer"
    assert after_engineer({"sql": None, "attempts": 3}) == END


def test_after_sandbox_ends_on_success_or_at_the_cap_otherwise_retries():
    assert after_sandbox({"outcome": "verified", "attempts": 1}) == END
    assert after_sandbox({"outcome": None, "attempts": 2}) == "engineer"
    assert after_sandbox({"outcome": "exhausted", "attempts": 3}) == END


# ---- whole graph, real LangGraph, real sandbox, stub models -----------------------


@pytest.mark.anyio
async def test_happy_path_verifies_on_the_first_attempt(target):
    eng, judge = Sequence(sql(GOOD_SQL)), Sequence(ALL_RESOLVED)
    out = await run_graph(make_deps(target, eng, judge))
    assert out["outcome"] == "verified" and out["attempts"] == 1
    assert out["sql"] == GOOD_SQL and out["sandbox"]["ok"] and out["recheck"]["ok"]
    assert len(eng.prompts) == 1 and len(judge.prompts) == 1
    assert [h["error"] for h in out["history"]] == [None]


@pytest.mark.anyio
async def test_an_untrusted_audit_stops_before_the_engineer_runs(target):
    eng, judge = Sequence(), Sequence()
    out = await run_graph(make_deps(target, eng, judge, trusted=False))
    assert out["outcome"] == "audit_untrusted" and out["attempts"] == 0
    assert eng.prompts == [] and judge.prompts == []


@pytest.mark.anyio
async def test_a_sandbox_failure_is_fed_back_and_the_judge_is_never_asked(target):
    eng = Sequence(sql(LOSSY_SQL), sql(GOOD_SQL))
    judge = Sequence(ALL_RESOLVED)  # exactly one call allowed
    out = await run_graph(make_deps(target, eng, judge))
    assert out["outcome"] == "verified" and out["attempts"] == 2
    second = eng.prompts[1]
    assert "rows lost in users: 3 -> 0" in second and LOSSY_SQL in second
    assert len(judge.prompts) == 1  # attempt 1 failed the hard gate, so no judge call
    assert out["history"][0]["error"] == "rows lost in users: 3 -> 0"


@pytest.mark.anyio
async def test_an_unresolved_finding_goes_back_to_the_engineer_with_the_reason(target):
    eng = Sequence(sql(GOOD_SQL), sql("ALTER TABLE users DROP COLUMN password_plain; CREATE TABLE note (x);"))
    judge = Sequence(
        resolutions(("POL-1.1", True, "gone"), ("POL-2.1", False, "FK still missing")),
        ALL_RESOLVED,
    )
    out = await run_graph(make_deps(target, eng, judge))
    assert out["outcome"] == "verified" and out["attempts"] == 2
    assert "[POL-2.1] still unresolved: FK still missing" in eng.prompts[1]


@pytest.mark.anyio
async def test_the_loop_stops_after_exactly_three_attempts(target):
    eng = Sequence(sql(LOSSY_SQL), sql(LOSSY_SQL), sql(LOSSY_SQL), sql(GOOD_SQL))  # 4th never used
    out = await run_graph(make_deps(target, eng, Sequence()))
    assert out["outcome"] == "exhausted" and out["attempts"] == 3
    assert len(eng.prompts) == 3 and len(out["history"]) == 3
    assert len(eng.items) == 1  # the 4th reply was never requested


@pytest.mark.anyio
async def test_a_malformed_engineer_reply_still_counts_as_an_attempt(target):
    eng = Sequence(sql(LOSSY_SQL), "not json", sql(GOOD_SQL))
    judge = Sequence(ALL_RESOLVED)
    out = await run_graph(make_deps(target, eng, judge))
    assert out["outcome"] == "verified" and out["attempts"] == 3
    assert out["history"][1]["sql"] is None and out["history"][1]["error"].startswith("engineer failed")
    third = eng.prompts[2]
    assert "engineer failed" in third and LOSSY_SQL in third  # still sees the last real SQL


@pytest.mark.anyio
async def test_an_engineer_that_never_produces_valid_output_exhausts_without_the_sandbox(target):
    eng = Sequence("x", "y", "z")
    out = await run_graph(make_deps(target, eng, Sequence()))
    assert out["outcome"] == "exhausted" and out["attempts"] == 3
    assert out.get("sandbox") is None and all(h["sql"] is None for h in out["history"])


@pytest.mark.anyio
async def test_an_engineer_exception_is_an_attempt_not_a_crash(target):
    eng = Sequence(RuntimeError("provider down"), sql(GOOD_SQL))
    out = await run_graph(make_deps(target, eng, Sequence(ALL_RESOLVED)))
    assert out["outcome"] == "verified" and out["attempts"] == 2
    assert out["history"][0]["error"] == "engineer failed: RuntimeError: provider down"


@pytest.mark.anyio
async def test_a_crashing_recheck_judge_fails_closed_and_exhausts(target):
    eng = Sequence(sql(GOOD_SQL), sql(GOOD_SQL), sql(GOOD_SQL))
    judge = Sequence(RuntimeError("a"), RuntimeError("b"), RuntimeError("c"))
    out = await run_graph(make_deps(target, eng, judge))
    assert out["outcome"] == "exhausted" and out["recheck"]["ok"] is False
    assert out["last_error"] == "recheck failed: RuntimeError: c"


@pytest.mark.anyio
async def test_the_real_target_is_untouched_by_a_whole_run(target):
    before = hashlib.sha256(open(target, "rb").read()).hexdigest()
    eng = Sequence(sql(LOSSY_SQL), sql(GOOD_SQL))
    await run_graph(make_deps(target, eng, Sequence(ALL_RESOLVED)))
    assert hashlib.sha256(open(target, "rb").read()).hexdigest() == before


@pytest.mark.anyio
async def test_langgraphs_recursion_limit_is_a_backstop_under_the_attempt_cap(target, monkeypatch):
    monkeypatch.setattr(graph, "MAX_ATTEMPTS", 100)  # pretend the cap is broken
    eng = Sequence(*[sql(LOSSY_SQL)] * 100)
    with pytest.raises(GraphRecursionError):
        await run_graph(make_deps(target, eng, Sequence()))
    assert len(eng.prompts) < 20


# ---- the engineer may decline findings that SQL cannot fix -------------------------


def declined(sql_text, *ids):
    return json.dumps({"sql": sql_text, "unfixable": [{"finding_id": i, "reason": f"{i} needs app code"} for i in ids]})


def test_a_declined_everything_outcome_ends_the_graph_even_on_the_first_attempt():
    assert after_engineer({"outcome": "unfixable", "sql": None, "attempts": 1}) == END
    assert after_engineer({"outcome": None, "sql": None, "attempts": 1}) == "engineer"


@pytest.mark.anyio
async def test_a_partial_fix_is_verified_and_the_declined_finding_is_kept_with_its_reason(target):
    eng = Sequence(declined(GOOD_SQL, "F2"))
    judge = Sequence(resolutions(("POL-1.1", True, "gone")))
    out = await run_graph(make_deps(target, eng, judge))
    assert out["outcome"] == "verified"
    assert [(u["id"], u["clause_id"], u["reason"]) for u in out["unfixable"]] == [("F2", "POL-2.1", "F2 needs app code")]
    assert [f["id"] for f in out["fixable"]] == ["F1"]
    assert "plaintext passwords" in judge.prompts[0]  # the re-check only sees what was attempted
    assert "no foreign key on access_log.user_id" not in judge.prompts[0]
    assert out["history"][0]["unfixable"] == ["F2"]


@pytest.mark.anyio
async def test_declining_every_finding_ends_unfixable_without_touching_the_sandbox_or_judge(target):
    eng = Sequence(declined("", "F1", "F2"))
    out = await run_graph(make_deps(target, eng, Sequence()))  # an empty judge raises if asked
    assert out["outcome"] == "unfixable" and out["attempts"] == 1
    assert out["sql"] is None and out.get("sandbox") is None and out.get("recheck") is None
    assert [u["id"] for u in out["unfixable"]] == ["F1", "F2"]
    assert out["history"][0]["unfixable"] == ["F1", "F2"]


@pytest.mark.anyio
async def test_naming_a_nonexistent_finding_counts_as_a_failed_attempt(target):
    eng = Sequence(declined(GOOD_SQL, "F9"), sql(GOOD_SQL))
    out = await run_graph(make_deps(target, eng, Sequence(ALL_RESOLVED)))
    assert out["outcome"] == "verified" and out["attempts"] == 2
    assert out["history"][0]["error"].startswith("engineer failed: ValueError: unknown finding ids")


@pytest.mark.anyio
async def test_the_sandbox_gate_still_applies_to_the_part_that_was_attempted(target):
    eng = Sequence(declined(LOSSY_SQL, "F2"), declined(GOOD_SQL, "F2"))
    out = await run_graph(make_deps(target, eng, Sequence(resolutions(("POL-1.1", True, "gone")))))
    assert out["outcome"] == "verified" and out["attempts"] == 2
    assert out["history"][0]["error"] == "rows lost in users: 3 -> 0"
