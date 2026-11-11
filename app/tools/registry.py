from dataclasses import dataclass
from typing import Callable

from pydantic import BaseModel

from app.tools.policy_tool import ReadPolicyArgs, read_policy
from app.tools.schema import ReadDbSchemaArgs, ToolResult, read_db_schema


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    run: Callable[[dict], ToolResult]


def build_registry(db_path: str) -> dict[str, Tool]:
    return {
        "read_db_schema": Tool(
            name="read_db_schema",
            description="List tables in the target database, or describe one table's columns.",
            args_model=ReadDbSchemaArgs,
            run=lambda args: read_db_schema(db_path, args),
        ),
        "read_policy": Tool(
            name="read_policy",
            description="List the company policy clauses, or read one clause by ID (e.g. POL-1.1). Cite only these IDs.",
            args_model=ReadPolicyArgs,
            run=read_policy,
        ),
    }


def tool_specs(registry: dict[str, Tool]) -> list[dict]:
    """What we send to the LLM API so it knows what it can call."""
    return [
        {
            "name": t.name,
            "description": t.description,
            "input_schema": t.args_model.model_json_schema(),
        }
        for t in registry.values()
    ]


def call_tool(registry: dict[str, Tool], name: str, raw_args: dict) -> ToolResult:
    tool = registry.get(name)
    if tool is None:
        return ToolResult(
            ok=False, error=f"unknown tool {name!r}; available: {sorted(registry)}"
        )
    try:
        return tool.run(raw_args)
    except Exception as exc:
        return ToolResult(ok=False, error=f"tool crashed: {type(exc).__name__}")
