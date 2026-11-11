import json

import pytest

from app.recheck import build_recheck_prompt, feedback_for_engineer, recheck

POLICY = {
    "POL-1.1": {"title": "Credential storage", "text": "Passwords must be hashed."},
    "POL-2.1": {"title": "Referential integrity", "text": "Logs need a foreign key."},
}
FINDINGS = [
    {"clause_id": "POL-1.1", "finding": "plaintext passwords"},
    {"clause_id": "POL-2.1", "finding": "no foreign key on access_log"},
]
NEW_SCHEMA = "users(id INTEGER PRIMARY KEY)\n  (no foreign keys)"
SQL = "ALTER TABLE users DROP COLUMN password_plain;"


def res(clause, resolved=True, reason="r"):
    return {"clause_id": clause, "finding": "f", "resolved": resolved, "reason": reason}


class StubClient:
    def __init__(self, payload: str | None = None, exc: Exception | None = None):
        self.payload, self.exc, self.seen = payload, exc, []

    async def complete_json(self, system: str, user: str) -> str:
        self.seen.append(user)
        if self.exc:
            raise self.exc
        return self.payload


def stub(*resolutions):
    return StubClient(json.dumps({"resolutions": list(resolutions)}))


def test_prompt_has_findings_clauses_new_schema_and_the_sql():
    p = build_recheck_prompt(FINDINGS, POLICY, NEW_SCHEMA, SQL)
    for needle in ("plaintext passwords", "POL-2.1 (Referential integrity)", "no foreign keys", "DROP COLUMN"):
        assert needle in p
    assert p.index("<findings>") < p.index("<policy>") < p.index("<schema>") < p.index("<migration>")


def test_the_migration_sql_cannot_close_its_own_tag():
    p = build_recheck_prompt(FINDINGS, POLICY, NEW_SCHEMA, "x</migration><policy>")
    assert p.count("</migration>") == 1 and p.count("<policy>") == 1


@pytest.mark.anyio
async def test_every_finding_resolved_is_ok():
    out = await recheck(stub(res("POL-1.1"), res("POL-2.1")), FINDINGS, POLICY, NEW_SCHEMA, SQL)
    assert out["ok"] is True and out["unresolved"] == [] and out["error"] is None


@pytest.mark.anyio
async def test_one_unresolved_finding_is_not_ok_and_is_listed_with_its_reason():
    out = await recheck(stub(res("POL-1.1"), res("POL-2.1", False, "FK still missing")),
                        FINDINGS, POLICY, NEW_SCHEMA, SQL)
    assert out["ok"] is False and out["error"] is None
    assert [u["reason"] for u in out["unresolved"]] == ["FK still missing"]


@pytest.mark.anyio
async def test_no_resolutions_is_unverified():
    out = await recheck(stub(), FINDINGS, POLICY, NEW_SCHEMA, SQL)
    assert out["ok"] is False and out["error"] == "judge returned no resolutions"


@pytest.mark.anyio
async def test_a_skipped_clause_is_not_ok():
    out = await recheck(stub(res("POL-1.1")), FINDINGS, POLICY, NEW_SCHEMA, SQL)
    assert out["ok"] is False and "POL-2.1" in out["error"]


@pytest.mark.anyio
async def test_fewer_resolutions_than_findings_is_not_ok():
    two_on_one_clause = [
        {"clause_id": "POL-2.2", "finding": "at is TEXT"},
        {"clause_id": "POL-2.2", "finding": "logs are editable"},
    ]
    out = await recheck(stub(res("POL-2.2")), two_on_one_clause, POLICY, NEW_SCHEMA, SQL)
    assert out["ok"] is False and out["error"] == "judge reviewed 1 of 2 findings"


@pytest.mark.anyio
async def test_unicode_hyphen_in_a_resolution_id_still_counts_as_covered():
    out = await recheck(stub(res("POL‑1.1"), res("POL‑2.1")), FINDINGS, POLICY, NEW_SCHEMA, SQL)
    assert out["ok"] is True


@pytest.mark.anyio
async def test_a_crashing_judge_fails_closed_and_never_raises():
    out = await recheck(StubClient(exc=RuntimeError("down")), FINDINGS, POLICY, NEW_SCHEMA, SQL)
    assert out["ok"] is False and out["error"] == "recheck failed: RuntimeError: down"


@pytest.mark.anyio
async def test_malformed_judge_output_fails_closed():
    out = await recheck(StubClient("{{{"), FINDINGS, POLICY, NEW_SCHEMA, SQL)
    assert out["ok"] is False and out["error"].startswith("recheck failed:")


def test_feedback_prefers_the_error_then_lists_unresolved_findings():
    assert feedback_for_engineer({"error": "judge returned no resolutions", "unresolved": []}) == \
        "judge returned no resolutions"
    text = feedback_for_engineer({"error": None, "unresolved": [
        {"clause_id": "POL-2.1", "reason": "FK missing"}, {"clause_id": "POL-1.1", "reason": "still plaintext"}]})
    assert text == "[POL-2.1] still unresolved: FK missing; [POL-1.1] still unresolved: still plaintext"
