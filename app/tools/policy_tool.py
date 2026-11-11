import re

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from app.policy.policy import load_policy
from app.tools.schema import ToolResult

CLAUSE_ID = re.compile(r"^POL-\d+\.\d+$")


class ReadPolicyArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clause_id: str | None = Field(
        default=None,
        description="Clause ID like POL-1.1. Omit to list all clauses.",
    )

    @field_validator("clause_id")
    @classmethod
    def must_look_like_a_clause(cls, v: str | None) -> str | None:
        if v is not None and not CLAUSE_ID.match(v):
            raise ValueError("clause_id must look like POL-1.1")
        return v


def read_policy(raw_args: dict) -> ToolResult:
    try:
        args = ReadPolicyArgs(**raw_args)
    except ValidationError as exc:
        return ToolResult(ok=False, error=f"invalid arguments: {exc.errors()[0]['msg']}")

    policy = load_policy()
    if args.clause_id is None:
        clauses = [{"id": cid, "title": c["title"]} for cid, c in sorted(policy.items())]
        return ToolResult(ok=True, data={"clauses": clauses})
    if args.clause_id not in policy:
        return ToolResult(
            ok=False, error=f"no clause {args.clause_id!r}; available: {sorted(policy)}"
        )
    clause = policy[args.clause_id]
    return ToolResult(
        ok=True,
        data={"id": args.clause_id, "title": clause["title"], "text": clause["text"]},
    )
