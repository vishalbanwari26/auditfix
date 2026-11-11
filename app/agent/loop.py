import asyncio
import time
from dataclasses import dataclass, field
from typing import Protocol

from app.tools.registry import Tool, call_tool, tool_specs


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class ModelReply:
    text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict | None = None  # {"prompt_tokens": int, "completion_tokens": int}


class ModelClient(Protocol):
    async def complete(self, messages: list[dict], tools: list[dict]) -> ModelReply: ...


@dataclass
class LoopResult:
    status: str  # "done" | "turn_limit"
    text: str | None
    turns: int
    trace: list[dict]


def _ms(since: float) -> float:
    return round((time.perf_counter() - since) * 1000, 1)


async def run_loop(
    client: ModelClient,
    registry: dict[str, Tool],
    task: str,
    max_turns: int = 8,
) -> LoopResult:
    messages: list[dict] = [{"role": "user", "content": task}]
    specs = tool_specs(registry)
    trace: list[dict] = []
    for turn in range(max_turns):
        n = turn + 1
        t0 = time.perf_counter()
        reply = await client.complete(messages, specs)
        usage = reply.usage or {}
        trace.append(
            {
                "turn": n,
                "kind": "model",
                "ms": _ms(t0),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
            }
        )
        if not reply.tool_calls:
            return LoopResult("done", reply.text, n, trace)
        messages.append(
            {
                "role": "assistant",
                "text": reply.text,
                "tool_calls": [vars(c) for c in reply.tool_calls],
            }
        )
        for call in reply.tool_calls:
            t1 = time.perf_counter()
            result = await asyncio.to_thread(call_tool, registry, call.name, call.args)
            trace.append(
                {
                    "turn": n,
                    "kind": "tool",
                    "name": call.name,
                    "args": call.args,
                    "ok": result.ok,
                    "error": result.error,
                    "ms": _ms(t1),
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": result.model_dump_json(),
                }
            )
    return LoopResult("turn_limit", None, max_turns, trace)
