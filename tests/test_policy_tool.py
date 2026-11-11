import pytest

from app.policy.policy import load_policy
from app.tools.policy_tool import read_policy


def test_lists_all_clauses_with_titles():
    res = read_policy({})
    assert res.ok
    assert {c["id"] for c in res.data["clauses"]} == set(load_policy())
    assert all(c["title"] for c in res.data["clauses"])


def test_reads_one_clause():
    res = read_policy({"clause_id": "POL-1.1"})
    assert res.ok and res.data["id"] == "POL-1.1"
    assert "salted hashes" in res.data["text"]


def test_unknown_clause_error_lists_available_ids():
    res = read_policy({"clause_id": "POL-9.9"})
    assert not res.ok and "POL-1.1" in res.error


@pytest.mark.parametrize("bad", ["PCI-DSS 10.13.2", "POL-1", "pol-1.1", "POL-1.1; x", ""])
def test_malformed_clause_ids_are_rejected(bad):
    res = read_policy({"clause_id": bad})
    assert not res.ok and res.error.startswith("invalid arguments")


def test_unknown_arguments_are_rejected_not_ignored():
    for raw in ({"__bad_json__": "{oops"}, {"clause": "POL-1.1"}):
        res = read_policy(raw)
        assert not res.ok and res.error.startswith("invalid arguments")
