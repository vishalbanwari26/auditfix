import re
import sqlite3

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")


class ToolResult(BaseModel):
    ok: bool
    data: dict | None = None
    error: str | None = None


class ReadDbSchemaArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    table: str | None = Field(
        default=None, description="Table name. Omit to list all tables."
    )

    @field_validator("table")
    @classmethod
    def must_be_identifier(cls, v: str | None) -> str | None:
        if v is not None and not IDENT.match(v):
            raise ValueError("table must be a plain identifier (letters, digits, _)")
        return v


def open_readonly(db_path: str) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)


def schema_summary(db_path: str) -> str:
    """Plain-text schema incl. foreign keys, so a judge can verify claims about them."""
    conn = open_readonly(db_path)
    try:
        names = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
        ]
        lines = []
        for name in names:
            quoted = '"' + name.replace('"', '""') + '"'
            cols = ", ".join(
                f"{r[1]} {r[2]}"
                + (" PRIMARY KEY" if r[5] else "")
                + (" NOT NULL" if r[3] else "")
                for r in conn.execute(f"PRAGMA table_info({quoted})")
            )
            lines.append(f"{name}({cols})")
            fks = [
                f"  FOREIGN KEY {r[3]} -> {r[2]}.{r[4]}"
                for r in conn.execute(f"PRAGMA foreign_key_list({quoted})")
            ]
            lines.extend(fks or ["  (no foreign keys)"])
        return "\n".join(lines)
    finally:
        conn.close()


def read_db_schema(db_path: str, raw_args: dict) -> ToolResult:
    try:
        args = ReadDbSchemaArgs(**raw_args)
    except ValidationError as exc:
        return ToolResult(ok=False, error=f"invalid arguments: {exc.errors()[0]['msg']}")

    conn = open_readonly(db_path)
    try:
        tables = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
        ]
        if args.table is None:
            return ToolResult(ok=True, data={"tables": tables})
        if args.table not in tables:
            return ToolResult(
                ok=False, error=f"no table {args.table!r}; available: {tables}"
            )
        cols = [
            {"name": r[1], "type": r[2], "notnull": bool(r[3]), "pk": bool(r[5])}
            for r in conn.execute(f"PRAGMA table_info({args.table})")
        ]
        return ToolResult(ok=True, data={"table": args.table, "columns": cols})
    finally:
        conn.close()
