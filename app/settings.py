import os


def target_db() -> str:
    return os.environ.get("AUDITFIX_TARGET_DB", "target.db")


def llm_mode() -> str:
    return os.environ.get("AUDITFIX_LLM", "demo")  # "demo" now, "groq" next
