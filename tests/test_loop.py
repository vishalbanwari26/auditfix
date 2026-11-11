import json
import sqlite3

import pytest

from app.agent.loop import ModelReply, ToolCall, run_loop
from app.tools.registry import build_registry


class ScriptedClient:
    """Plays back a fixed list of replies and records what it was shown."""

    def __init__(self, replies: list[ModelReply]):
        self.replies = list(replies)
        self.seen: list[list[dict]] = []

    async def complete(self, messages, tools):
        self.seen.append([dict(m) for m in messages])
        return self.replies.pop(0)


@pytest.fixture
def registry(tmp_path):
    path = str(tmp_path / "t.db")
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, email TEXT)")
    conn.commit()
    conn.close()
    return build_registry(path)


@pytest.mark.anyio
async def test_tool_then_final_answer(registry):
    client = ScriptedClient([
        ModelReply(tool_calls=[ToolCall("c1", "read_db_schema", {"table": "users"})]),
        ModelReply(text="users has 2 columns"),
    ])
    res = await run_loop(client, registry, "audit the schema")
    assert (res.status, res.text, res.turns) == ("done", "users has 2 columns", 2)

    tool_msg = client.seen[1][-1]  # what the model saw on its second turn
    assert tool_msg["role"] == "tool" and tool_msg["tool_call_id"] == "c1"
    assert json.loads(tool_msg["content"])["ok"] is True


@pytest.mark.anyio
async def test_model_sees_tool_error_and_can_retry(registry):
    client = ScriptedClient([
        ModelReply(tool_calls=[ToolCall("c1", "read_db_schema", {"table": "usres"})]),
        ModelReply(tool_calls=[ToolCall("c2", "read_db_schema", {"table": "users"})]),
        ModelReply(text="fixed my typo"),
    ])
    res = await run_loop(client, registry, "audit")
    assert res.status == "done" and res.turns == 3

    first_error = json.loads(client.seen[1][-1]["content"])
    assert first_error["ok"] is False and "users" in first_error["error"]
    assert json.loads(client.seen[2][-1]["content"])["ok"] is True


@pytest.mark.anyio
async def test_turn_cap_stops_a_loop_that_never_answers(registry):
    forever = [
        ModelReply(tool_calls=[ToolCall(f"c{i}", "read_db_schema", {})]) for i in range(10)
    ]
    client = ScriptedClient(forever)
    res = await run_loop(client, registry, "audit", max_turns=3)
    assert (res.status, res.text, res.turns) == ("turn_limit", None, 3)
    assert len(client.seen) == 3  # the model was asked exactly 3 times, no more


@pytest.mark.anyio
async def test_trace_records_model_and_tool_events(registry):
    client = ScriptedClient([
        ModelReply(
            tool_calls=[ToolCall("c1", "read_db_schema", {"table": "usres"})],
            usage={"prompt_tokens": 10, "completion_tokens": 3},
        ),
        ModelReply(text="done"),
    ])
    res = await run_loop(client, registry, "audit")
    assert [(e["turn"], e["kind"]) for e in res.trace] == [(1, "model"), (1, "tool"), (2, "model")]
    model1, tool1, _ = res.trace
    assert (model1["prompt_tokens"], model1["completion_tokens"]) == (10, 3)
    assert tool1["name"] == "read_db_schema" and tool1["args"] == {"table": "usres"}
    assert tool1["ok"] is False and "users" in tool1["error"]
    assert all(e["ms"] >= 0 for e in res.trace)


@pytest.mark.anyio
async def test_trace_is_returned_when_the_turn_cap_is_hit(registry):
    client = ScriptedClient([
        ModelReply(tool_calls=[ToolCall(f"c{i}", "read_db_schema", {})]) for i in range(10)
    ])
    res = await run_loop(client, registry, "audit", max_turns=3)
    assert res.status == "turn_limit"
    assert [e["kind"] for e in res.trace] == ["model", "tool"] * 3
