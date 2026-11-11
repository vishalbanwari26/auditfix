import json

from app.agent.loop import ModelReply, ToolCall


class DemoClient:
    """Scripted model: list tables, describe the first one, then answer."""

    async def complete(self, messages: list[dict], tools: list[dict]) -> ModelReply:
        results = [m for m in messages if m["role"] == "tool"]
        if not results:
            return ModelReply(tool_calls=[ToolCall("d1", "read_db_schema", {})])
        if len(results) == 1:
            tables = json.loads(results[0]["content"])["data"]["tables"]
            return ModelReply(
                tool_calls=[ToolCall("d2", "read_db_schema", {"table": tables[0]})]
            )
        return ModelReply(
            text=f"demo audit: inspected {len(results)} tool results [POL-1.1]"
        )
