import asyncio

from app import store
from app.agent.demo import DemoClient
from app.agent.groq_client import MODEL, GroqClient
from app.agent.loop import LoopResult, ModelClient, run_loop
from app.engineer import DemoEngineer
from app.graph import Deps, run_graph
from app.jobs import Job, JobStatus
from app.judge import DemoJudge, GroqJsonClient, GroqJudge, JudgeClient, evaluate
from app.policy.grounding import check_citations, normalise
from app.policy.policy import load_policy
from app.settings import llm_mode, target_db
from app.tools.registry import build_registry
from app.tools.schema import schema_summary

TASK = (
    "Audit the database schema and report compliance risks. "
    "Use read_policy to see the company policy. "
    "Cite only policy clause IDs, written like [POL-1.1]. "
    "Do not cite any outside standards or regulations."
)


def make_client() -> ModelClient:
    mode = llm_mode()
    if mode == "demo":
        return DemoClient()
    if mode == "groq":
        return GroqClient()
    raise ValueError(f"unknown AUDITFIX_LLM {mode!r}")


def make_judge() -> JudgeClient:
    mode = llm_mode()
    if mode == "demo":
        return DemoJudge()
    if mode == "groq":
        return GroqJudge()  # the larger model grades; it never grades its own work
    raise ValueError(f"unknown AUDITFIX_LLM {mode!r}")


def make_engineer() -> JudgeClient:
    mode = llm_mode()
    if mode == "demo":
        return DemoEngineer()
    if mode == "groq":
        # the agent model writes; the judge checks. Full reasoning mode breaks Groq's JSON
        # mode on this model (json_validate_failed, 3 of 3), low effort works (3 of 3).
        return GroqJsonClient(model=MODEL, reasoning_effort="low")
    raise ValueError(f"unknown AUDITFIX_LLM {mode!r}")


async def run_agent(job_id: str) -> LoopResult:
    registry = build_registry(target_db())
    return await run_loop(make_client(), registry, TASK)


async def _save(job: Job) -> None:
    await asyncio.to_thread(store.save_job, job)


async def _ground(text: str) -> tuple[dict, bool]:
    """Deterministic citation check first; the judge only runs if it passes."""
    policy = load_policy()
    report = check_citations(text, set(policy))
    grounding = report.to_dict()
    if not report.ok:
        return grounding, False
    summary = await asyncio.to_thread(schema_summary, target_db())
    grounding["judge"] = await evaluate(make_judge(), text, report.cited, policy, summary)
    # per finding: rejected findings are listed, the verified ones still drive the engineer
    return grounding, grounding["judge"]["usable"]


def _supported_findings(grounding: dict, policy: dict) -> list[dict]:
    findings = []
    for v in grounding.get("judge", {}).get("verdicts", []):
        clause = normalise(v["clause_id"]).strip()
        if v["verdict"] == "supported" and clause in policy:
            findings.append(
                {"id": f"F{len(findings) + 1}", "clause_id": clause, "finding": v["finding"]}
            )
    return findings


async def audit(job: Job) -> dict:
    """The graph's first node: agent loop, citation check, judge. Raises if the agent fails."""
    res = await run_agent(job.id)
    await asyncio.to_thread(store.append_trace, job.id, res.trace)  # kept even on failure
    if res.status != "done":
        raise RuntimeError(f"agent stopped: {res.status}")
    job.result = res.text
    job.grounding, trusted = await _ground(res.text or "")
    policy = load_policy()
    findings = _supported_findings(job.grounding, policy) if trusted else []
    return {
        "trusted": trusted and bool(findings),
        "findings": findings,
        "clauses": {c: policy[c] for c in {f["clause_id"] for f in findings}},
        "schema": await asyncio.to_thread(schema_summary, target_db()),
    }


def _migration_summary(final: dict) -> dict | None:
    if not final.get("history"):
        return None  # the engineer never ran
    verified = final.get("outcome") == "verified"
    sandbox = final.get("sandbox") or {}
    return {
        "outcome": final.get("outcome"),
        "attempts": final["attempts"],
        "sql": final.get("sql") if verified else None,  # only a verified fix is proposed
        "schema_after": sandbox.get("schema_after") if verified else None,
        "recheck": final.get("recheck"),
        "unfixable": final.get("unfixable", []),  # declined by the engineer, for a human
        "history": final["history"],
    }


async def run_job(job: Job) -> None:
    job.status = JobStatus.RUNNING
    await _save(job)
    events: list[dict] = []
    try:
        deps = Deps(
            audit=lambda: audit(job),
            engineer_client=make_engineer(),
            judge_client=make_judge(),
            target_db=target_db(),
        )
        final = await run_graph(deps, on_event=events.append)
        job.migration = _migration_summary(final)
        verified = final.get("outcome") == "verified"
        job.status = JobStatus.DONE if verified else JobStatus.NEEDS_REVIEW
    except Exception as exc:
        job.error = str(exc)
        job.status = JobStatus.FAILED
    finally:
        await asyncio.to_thread(store.append_trace, job.id, events)
    await _save(job)
