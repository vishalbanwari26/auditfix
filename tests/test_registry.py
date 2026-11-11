import sqlite3

import pytest
from pydantic import BaseModel

from app.tools.registry import Tool, build_registry, call_tool, tool_specs
from app.tools.schema import ToolResult


@pytest.fixture
def registry(tmp_path):
    path = str(tmp_path / "t.db")
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY)")
    conn.commit()
    conn.close()
    return build_registry(path)


def test_specs_have_the_shape_the_llm_api_expects(registry):
    specs = {s["name"]: s for s in tool_specs(registry)}
    assert set(specs) == {"read_db_schema", "read_policy"}
    spec = specs["read_db_schema"]
    assert spec["name"] == "read_db_schema"
    assert spec["description"]
    assert spec["input_schema"]["type"] == "object"
    assert "table" in spec["input_schema"]["properties"]
    assert "db_path" not in spec["input_schema"]["properties"]  # model can't pick it


def test_dispatch_runs_the_real_tool(registry):
    res = call_tool(registry, "read_db_schema", {"table": "users"})
    assert res.ok and res.data["columns"][0]["name"] == "id"


def test_unknown_tool_name_lists_real_ones(registry):
    res = call_tool(registry, "read_schema", {})
    assert not res.ok
    assert "read_schema" in res.error and "read_db_schema" in res.error


def test_crashing_tool_becomes_a_result_without_leaking_details(registry):
    class NoArgs(BaseModel):
        pass

    def boom(args: dict) -> ToolResult:
        raise RuntimeError("secret path /etc/shadow")

    registry["boom"] = Tool("boom", "always crashes", NoArgs, boom)
    res = call_tool(registry, "boom", {})
    assert not res.ok
    assert res.error == "tool crashed: RuntimeError"
    assert "shadow" not in res.error
