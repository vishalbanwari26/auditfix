import asyncio
import time
from dataclasses import asdict, dataclass
from typing import Awaitable, Callable, TypedDict

from langgraph.graph import END, START, StateGraph

from app.engineer import draft_migration
from app.judge import JudgeClient
from app.recheck import feedback_for_engineer, recheck
from app.sandbox import run_in_sandbox

MAX_ATTEMPTS = 3
RECURSION_LIMIT = 15  # LangGraph backstop beneath the attempt cap


class AgentState(TypedDict, total=False):
    findings: list[dict]  # supported findings from the trusted audit, each with an "id"
    clauses: dict[str, dict]  # the policy text for those clauses
    schema: str  # current schema summary
    fixable: list[dict]  # findings the engineer is attempting (all, minus declined)
    unfixable: list[dict]  # findings the engineer declined, each with a "reason"
    sql: str | None  # this attempt's migration (None if the engineer failed)
    attempts: int
    last_error: str | None  # what the engineer is told about the previous attempt
    history: list[dict]  # one entry per attempt: {attempt, sql, error, unfixable}
    sandbox: dict | None
    recheck: dict | None
    outcome: str | None  # "verified" | "exhausted" | "unfixable" | "audit_untrusted"


@dataclass
class Deps:
    # returns {"trusted": bool, "findings": [...], "clauses": {...}, "schema": str}
    audit: Callable[[], Awaitable[dict]]
    engineer_client: JudgeClient
    judge_client: JudgeClient
    target_db: str


# ---- routers: read state only, never write it -------------------------------------


def after_audit(s: AgentState) -> str:
    return "engineer" if s.get("findings") else END  # an untrusted report stops here


def after_engineer(s: AgentState) -> str:
    if s.get("outcome") == "unfixable":  # the engineer declined every finding
        return END
    if s.get("sql") is None:  # the engineer produced nothing usable
        return END if s["attempts"] >= MAX_ATTEMPTS else "engineer"
    return "sandbox"


def after_sandbox(s: AgentState) -> str:
    if s.get("outcome") == "verified" or s["attempts"] >= MAX_ATTEMPTS:
        return END
    return "engineer"


# ---- nodes ------------------------------------------------------------------------


def _with_error(history: list[dict], error: str | None) -> list[dict]:
    """New history list whose last attempt carries the error."""
    return [*history[:-1], {**history[-1], "error": error}]


def build_graph(deps: Deps):
    async def auditor_node(s: AgentState) -> dict:
        audit = await deps.audit()
        trusted = audit["trusted"]
        findings = audit["findings"] if trusted else []
        return {
            "findings": findings,
            "fixable": findings,
            "unfixable": [],
            "clauses": audit["clauses"],
            "schema": audit["schema"],
            "attempts": 0,
            "history": [],
            "sql": None,
            "last_error": None,
            "outcome": None if trusted else "audit_untrusted",
        }

    async def engineer_node(s: AgentState) -> dict:
        attempt = s["attempts"] + 1
        previous = next((h for h in reversed(s["history"]) if h["sql"]), None)
        try:
            draft = await draft_migration(
                deps.engineer_client,
                s["findings"],
                s["schema"],
                previous["sql"] if previous else None,
                s.get("last_error"),
                {f["id"] for f in s["findings"]},
            )
        except Exception as exc:
            reason = " ".join(str(exc).split())[:150]  # one line, bounded
            err = f"engineer failed: {type(exc).__name__}: {reason}"  # still an attempt
            return {
                "attempts": attempt,
                "sql": None,
                "last_error": err,
                "history": [
                    *s["history"],
                    {"attempt": attempt, "sql": None, "error": err, "unfixable": []},
                ],
                "outcome": "exhausted" if attempt >= MAX_ATTEMPTS else None,
            }
        declined = {u.finding_id: u.reason for u in draft.unfixable}
        fixable = [f for f in s["findings"] if f["id"] not in declined]
        unfixable = [{**f, "reason": declined[f["id"]]} for f in s["findings"] if f["id"] in declined]
        sql = draft.sql if fixable else None  # nothing to apply if everything was declined
        return {
            "attempts": attempt,
            "sql": sql,
            "fixable": fixable,
            "unfixable": unfixable,
            "last_error": None,
            "history": [
                *s["history"],
                {"attempt": attempt, "sql": sql, "error": None, "unfixable": [f["id"] for f in unfixable]},
            ],
            "outcome": "unfixable" if not fixable else None,
        }

    def _failed(s: AgentState, sandbox: dict, rc: dict | None, error: str) -> dict:
        return {
            "sandbox": sandbox,
            "recheck": rc,
            "last_error": error,
            "history": _with_error(s["history"], error),
            "outcome": "exhausted" if s["attempts"] >= MAX_ATTEMPTS else None,
        }

    async def sandbox_node(s: AgentState) -> dict:
        res = await asyncio.to_thread(run_in_sandbox, deps.target_db, s["sql"])
        if not res.ok:  # the hard gate: the judge never sees a migration that fails it
            return _failed(s, asdict(res), None, res.error)
        fixable = s["fixable"]  # declined findings are not the judge's concern
        clauses = {c: s["clauses"][c] for c in {f["clause_id"] for f in fixable}}
        rc = await recheck(deps.judge_client, fixable, clauses, res.schema_after, s["sql"])
        if rc["ok"]:
            return {"sandbox": asdict(res), "recheck": rc, "last_error": None, "outcome": "verified"}
        return _failed(s, asdict(res), rc, feedback_for_engineer(rc))

    g = StateGraph(AgentState)
    g.add_node("auditor", auditor_node)
    g.add_node("engineer", engineer_node)
    g.add_node("sandbox", sandbox_node)
    g.add_edge(START, "auditor")
    g.add_conditional_edges("auditor", after_audit, {"engineer": "engineer", END: END})
    g.add_conditional_edges(
        "engineer", after_engineer, {"sandbox": "sandbox", "engineer": "engineer", END: END}
    )
    g.add_conditional_edges("sandbox", after_sandbox, {"engineer": "engineer", END: END})
    return g.compile()


def _node_event(node: str, delta: dict, state: dict, ms: float) -> dict:
    if node == "auditor":
        ok = delta.get("outcome") != "audit_untrusted"
    elif node == "engineer":
        ok = delta.get("sql") is not None or delta.get("outcome") == "unfixable"
    else:  # sandbox
        ok = delta.get("outcome") == "verified"
    return {
        "turn": state.get("attempts", 0),
        "kind": "node",
        "name": node,
        "ok": ok,
        "error": delta.get("last_error"),
        "ms": ms,
    }


async def run_graph(deps: Deps, on_event: Callable[[dict], None] | None = None) -> AgentState:
    """Run the graph; report one event per node (with timing) through on_event."""
    state: dict = {}
    last = time.perf_counter()
    stream = build_graph(deps).astream(
        {}, {"recursion_limit": RECURSION_LIMIT}, stream_mode="updates"
    )
    async for update in stream:
        for node, delta in update.items():
            state.update(delta)  # nodes return whole values, so a dict merge is the state
            now = time.perf_counter()
            if on_event:
                on_event(_node_event(node, delta, state, round((now - last) * 1000, 1)))
            last = now
    return state  # type: ignore[return-value]
