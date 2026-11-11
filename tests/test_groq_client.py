import json

import httpx
import pytest

from app.agent import groq_client
from app.agent.groq_client import (
    GroqClient,
    parse_reply,
    to_openai_messages,
    to_openai_tools,
)

def ok_body(content="hi", tool_calls=None):
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return {"choices": [{"message": msg}]}


def test_tools_are_translated_to_openai_function_format():
    specs = [{"name": "t", "description": "d", "input_schema": {"type": "object"}}]
    assert to_openai_tools(specs) == [
        {
            "type": "function",
            "function": {"name": "t", "description": "d", "parameters": {"type": "object"}},
        }
    ]


def test_assistant_tool_calls_become_json_string_arguments():
    msgs = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "text": None,
            "tool_calls": [{"id": "c1", "name": "t", "args": {"table": "users"}}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "{}"},
    ]
    out = to_openai_messages(msgs)
    assert out[0] == msgs[0] and out[2] == msgs[2]
    call = out[1]["tool_calls"][0]
    assert call["id"] == "c1" and call["type"] == "function"
    assert json.loads(call["function"]["arguments"]) == {"table": "users"}


def test_parse_reply_text_only():
    reply = parse_reply(ok_body("done"))
    assert reply.text == "done" and reply.tool_calls == []


def test_parse_reply_tool_call():
    body = ok_body(None, [{"id": "c1", "function": {"name": "t", "arguments": '{"a": 1}'}}])
    (call,) = parse_reply(body).tool_calls
    assert (call.id, call.name, call.args) == ("c1", "t", {"a": 1})


def test_parse_reply_marks_invalid_json_arguments():
    body = ok_body(None, [{"id": "c1", "function": {"name": "t", "arguments": "{oops"}}])
    (call,) = parse_reply(body).tool_calls
    assert call.args == {"__bad_json__": "{oops"}


@pytest.mark.anyio
async def test_complete_sends_auth_model_and_tools(http):
    http["responses"] = [httpx.Response(200, json=ok_body("hello"))]
    specs = [{"name": "t", "description": "d", "input_schema": {"type": "object"}}]
    reply = await GroqClient(api_key="k").complete([{"role": "user", "content": "x"}], specs)
    assert reply.text == "hello"
    (req,) = http["requests"]
    assert req.headers["authorization"] == "Bearer k"
    sent = json.loads(req.content)
    assert sent["model"] == groq_client.MODEL
    assert sent["tools"][0]["function"]["name"] == "t"


@pytest.mark.anyio
async def test_rate_limit_is_retried_honoring_retry_after(http):
    http["responses"] = [
        httpx.Response(429, headers={"retry-after": "3"}),
        httpx.Response(200, json=ok_body("ok")),
    ]
    reply = await GroqClient(api_key="k").complete([], [])
    assert reply.text == "ok"
    assert len(http["requests"]) == 2 and http["sleeps"] == [3.0]


@pytest.mark.anyio
async def test_backoff_without_retry_after_doubles(http):
    http["responses"] = [httpx.Response(503)] * 3 + [httpx.Response(200, json=ok_body())]
    await GroqClient(api_key="k").complete([], [])
    assert http["sleeps"] == [1.0, 2.0, 4.0]


@pytest.mark.anyio
async def test_gives_up_after_max_attempts(http):
    http["responses"] = [httpx.Response(429)] * 3
    with pytest.raises(httpx.HTTPStatusError):
        await GroqClient(api_key="k", max_attempts=3).complete([], [])
    assert len(http["requests"]) == 3


@pytest.mark.anyio
async def test_client_errors_are_not_retried(http):
    http["responses"] = [httpx.Response(400)]
    with pytest.raises(httpx.HTTPStatusError):
        await GroqClient(api_key="k").complete([], [])
    assert len(http["requests"]) == 1 and http["sleeps"] == []


def test_parse_reply_reads_token_usage():
    body = ok_body("x")
    body["usage"] = {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9}
    assert parse_reply(body).usage == {"prompt_tokens": 7, "completion_tokens": 2}
    assert parse_reply(ok_body("x")).usage is None


@pytest.mark.anyio
async def test_provider_errors_keep_the_status_and_body_in_the_message(http):
    http["responses"] = [httpx.Response(400, json={"error": {"code": "json_validate_failed", "message": "bad call"}})]
    with pytest.raises(httpx.HTTPStatusError) as info:
        await GroqClient(api_key="k").complete([], [])
    msg = str(info.value)
    assert msg.startswith("Groq 400:") and "json_validate_failed" in msg and "bad call" in msg
    assert info.value.response.status_code == 400


@pytest.mark.anyio
async def test_a_long_error_body_is_truncated_in_the_message(http):
    http["responses"] = [httpx.Response(400, text="x" * 5000)]
    with pytest.raises(httpx.HTTPStatusError) as info:
        await GroqClient(api_key="k").complete([], [])
    assert len(str(info.value)) < 450


def _err400(code: str) -> httpx.Response:
    return httpx.Response(400, json={"error": {"code": code, "message": "m"}})


@pytest.mark.anyio
@pytest.mark.parametrize("code", ["output_parse_failed", "tool_use_failed"])
async def test_transient_generation_failures_are_retried(http, code):
    http["responses"] = [_err400(code), httpx.Response(200, json=ok_body("fine"))]
    reply = await GroqClient(api_key="k").complete([], [])
    assert reply.text == "fine" and len(http["requests"]) == 2


@pytest.mark.anyio
async def test_json_validate_failed_is_not_retried_because_it_repeats(http):
    http["responses"] = [_err400("json_validate_failed")]
    with pytest.raises(httpx.HTTPStatusError):
        await GroqClient(api_key="k").complete([], [])
    assert len(http["requests"]) == 1


@pytest.mark.anyio
async def test_a_persistent_transient_failure_still_gives_up(http):
    http["responses"] = [_err400("output_parse_failed")] * 3
    with pytest.raises(httpx.HTTPStatusError):
        await GroqClient(api_key="k", max_attempts=3).complete([], [])
    assert len(http["requests"]) == 3
