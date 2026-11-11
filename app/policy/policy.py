import json
from pathlib import Path

POLICY_PATH = Path(__file__).with_name("policy.json")


def load_policy() -> dict[str, dict]:
    return json.loads(POLICY_PATH.read_text())
