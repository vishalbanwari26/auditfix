import asyncio
import json

from evals.aggrefact import (
    MAX_DOC_CHARS,
    DailyLimitReached,
    judge_row,
    load_store,
    pending,
    row_key,
    run,
    sample_rows,
    save_store,
    score,
    stored_results,
)


def row(ds, label, doc="d", claim="c"):
    return {"dataset": ds, "doc": doc, "claim": claim, "label": label}


class Stub:
    def __init__(self, *replies):
        self.replies = list(replies)

    async def complete_json(self, system, user):
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def verdict(v):
    return json.dumps({"verdict": v, "reason": "x"})


def test_sampling_is_balanced_across_sources_and_repeatable():
    rows = [row("a", 1)] * 50 + [row("b", 0)] * 5
    picked = sample_rows(rows, 6, seed=1)
    assert sorted(r["dataset"] for r in picked) == ["a", "a", "a", "b", "b", "b"]
    assert sample_rows(rows, 6, seed=1) == picked


def test_long_documents_are_skipped_not_truncated():
    rows = [row("a", 1, doc="x" * (MAX_DOC_CHARS + 1)), row("a", 1, doc="short")]
    assert [r["doc"] for r in sample_rows(rows, 5, 0)] == ["short"]


def test_judge_row_reads_the_verdict_and_rejects_anything_else():
    r = row("a", 1)
    assert asyncio.run(judge_row(Stub(verdict("supported")), r)) == "supported"
    assert asyncio.run(judge_row(Stub('{"verdict": "maybe"}'), r)) is None
    assert asyncio.run(judge_row(Stub("not json"), r)) is None
    assert asyncio.run(judge_row(Stub(RuntimeError("boom")), r)) is None


def test_score_counts_false_accepts_false_rejects_and_errors():
    results = [
        (row("a", 1), "supported"),  # right
        (row("a", 1), "unsupported"),  # false reject
        (row("a", 0), "supported"),  # false accept
        (row("a", 0), None),  # error: left out of the rates
    ]
    s = score(results)
    assert s["sampled"] == 4 and s["n"] == 3 and s["judge_errors"] == 1
    assert abs(s["accuracy"] - 1 / 3) < 1e-9
    assert s["false_accept_rate"] == 1.0 and s["false_reject_rate"] == 0.5
    assert s["per_source"]["a"]["n"] == 3


def test_run_judges_every_row():
    rows = [row("a", 1), row("a", 0)]
    out = asyncio.run(run(Stub(verdict("supported"), verdict("unsupported")), rows))
    assert [p for _, p in out] == ["supported", "unsupported"]


def test_a_daily_limit_stops_the_run_and_keeps_the_rows_already_judged():
    rows = [row("a", 1), row("a", 0), row("a", 1)]
    out = asyncio.run(run(Stub(verdict("supported"), RuntimeError("Groq 429: tokens per day (TPD)")), rows))
    assert [p for _, p in out] == ["supported"]  # the failing row and the one after it are dropped


def test_judge_row_raises_on_a_daily_limit_but_not_on_other_errors():
    import pytest

    with pytest.raises(DailyLimitReached):
        asyncio.run(judge_row(Stub(RuntimeError("tokens per day")), row("a", 1)))
    assert asyncio.run(judge_row(Stub(RuntimeError("other")), row("a", 1))) is None


def test_resume_skips_judged_rows_and_retries_errored_ones(tmp_path):
    rows = [row("a", 1, doc="d1"), row("a", 0, doc="d2"), row("a", 1, doc="d3")]
    store, path = {}, tmp_path / "v.json"

    def keep(r, verdict):
        if verdict is not None:
            store[row_key(r)] = {"dataset": r["dataset"], "label": r["label"], "verdict": verdict}
            save_store(path, store)

    stub = Stub(verdict("supported"), "not json", verdict("unsupported"))  # row 2 errors
    asyncio.run(run(stub, pending(rows, store), on_result=keep))
    again = load_store(path)
    assert [r["doc"] for r in pending(rows, again)] == ["d2"]  # only the errored row is left
    assert [p for _, p in stored_results(rows, again)] == ["supported", "unsupported"]


def test_the_store_keeps_no_document_text(tmp_path):
    path = tmp_path / "v.json"
    r = row("a", 1, doc="SECRET DOCUMENT TEXT", claim="SECRET CLAIM")
    save_store(path, {row_key(r): {"dataset": "a", "label": 1, "verdict": "supported"}})
    assert "SECRET" not in path.read_text()
