import asyncio
import json
import os

import httpx

from app.agent.loop import ModelReply, ToolCall

URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "openai/gpt-oss-20b"
RETRY_STATUS = {429, 500, 502, 503}


def to_openai_tools(specs: list[dict]) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": s["name"],
                "description": s["description"],
                "parameters": s["input_schema"],
            },
        }
        for s in specs
    ]


def to_openai_messages(messages: list[dict]) -> list[dict]:
    out = []
    for m in messages:
        if m["role"] != "assistant":
            out.append(m)  # user + tool already match
            continue
        msg = {"role": "assistant", "content": m["text"]}
        if m["tool_calls"]:
            msg["tool_calls"] = [
                {
                    "id": c["id"],
                    "type": "function",
                    "function": {"name": c["name"], "arguments": json.dumps(c["args"])},
                }
                for c in m["tool_calls"]
            ]
        out.append(msg)
    return out


def parse_reply(body: dict) -> ModelReply:
    msg = body["choices"][0]["message"]
    calls = []
    for c in msg.get("tool_calls") or []:
        raw = c["function"]["arguments"] or "{}"
        try:
            args = json.loads(raw)
        except json.JSONDecodeError:
            args = {"__bad_json__": raw}
        calls.append(ToolCall(c["id"], c["function"]["name"], args))
    usage = body.get("usage")
    return ModelReply(
        text=msg.get("content"),
        tool_calls=calls,
        usage=None
        if not usage
        else {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
        },
    )


# Groq answers 400 when a model's generation could not be parsed. These are intermittent
# and worth retrying; json_validate_failed is not (it repeated 3 of 3 for the same input).
RETRYABLE_400 = {"output_parse_failed", "tool_use_failed"}


def _is_transient(resp: httpx.Response) -> bool:
    if resp.status_code in RETRY_STATUS:
        return True
    if resp.status_code == 400:
        try:
            return resp.json().get("error", {}).get("code") in RETRYABLE_400
        except Exception:
            return False
    return False


async def post_chat(payload: dict, api_key: str, max_attempts: int = 4) -> dict:
    """POST to Groq, retrying rate limits and 5xx with backoff. Shared by agent and judge."""
    headers = {"Authorization": f"Bearer {api_key}"}
    async with httpx.AsyncClient(timeout=60) as http:
        for attempt in range(max_attempts):
            resp = await http.post(URL, json=payload, headers=headers)
            if _is_transient(resp) and attempt < max_attempts - 1:
                wait = float(resp.headers.get("retry-after", 2**attempt))
                await asyncio.sleep(wait)
                continue
            if resp.is_error:  # keep the provider's explanation, not just the status line
                raise httpx.HTTPStatusError(
                    f"Groq {resp.status_code}: {resp.text[:400]}",
                    request=resp.request,
                    response=resp,
                )
            return resp.json()
    raise AssertionError("unreachable")  # loop always returns or raises


class GroqClient:
    def __init__(self, api_key: str | None = None, max_attempts: int = 4):
        self.api_key = api_key or os.environ["GROQ_API_KEY"]
        self.max_attempts = max_attempts

    async def complete(self, messages: list[dict], tools: list[dict]) -> ModelReply:
        payload = {
            "model": MODEL,
            "messages": to_openai_messages(messages),
            "tools": to_openai_tools(tools),
        }
        body = await post_chat(payload, self.api_key, self.max_attempts)
        return parse_reply(body)
