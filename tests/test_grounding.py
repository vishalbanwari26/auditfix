import re

from app.policy.grounding import CLAUSE, check_citations
from app.policy.policy import load_policy

IDS = {"POL-1.1", "POL-2.1", "POL-3.1"}


def test_only_valid_clauses_passes():
    rep = check_citations("Plaintext passwords [POL-1.1]. No FK on logs [POL-2.1].", IDS)
    assert rep.ok
    assert rep.cited == ["POL-1.1", "POL-2.1"] and rep.invented == [] and rep.external == []


def test_invented_clause_id_fails():
    rep = check_citations("See [POL-1.1] and [POL-9.9].", IDS)
    assert not rep.ok and rep.invented == ["POL-9.9"]


def test_valid_clause_plus_outside_standard_still_fails():
    rep = check_citations("Violates [POL-1.1] and GDPR Art. 32.", IDS)
    assert not rep.ok and rep.external == ["GDPR"] and rep.invented == []


def test_the_made_up_pci_citation_from_the_live_run_is_caught():
    rep = check_citations("Orphaned logs violate PCI-DSS 10.13.2 [POL-2.1].", IDS)
    assert not rep.ok and rep.external == ["PCI-DSS"]


def test_report_with_no_citations_is_not_grounded():
    rep = check_citations("The schema looks risky.", IDS)
    assert not rep.ok and rep.cited == []


def test_outside_standard_spellings_are_normalised_and_deduplicated():
    rep = check_citations("PCI DSS, pci-dss and Pci-Dss all mean one thing [POL-1.1].", IDS)
    assert rep.external == ["PCI-DSS"]


def test_repeated_citations_count_once():
    assert check_citations("[POL-1.1] [POL-1.1] [POL-1.1]", IDS).cited == ["POL-1.1"]


def test_shipped_policy_is_well_formed():
    policy = load_policy()
    assert policy
    for cid, clause in policy.items():
        assert re.fullmatch(CLAUSE.pattern, cid), cid
        assert clause["title"] and clause["text"]


def test_unicode_hyphens_in_clause_ids_still_count_as_citations():
    # the live model wrote U+2011 (non-breaking hyphen), which looks identical to "-"
    rep = check_citations("Plaintext passwords [POL‑1.1] and [POL–2.1].", IDS)
    assert rep.cited == ["POL-1.1", "POL-2.1"] and rep.ok


def test_unicode_hyphen_cannot_hide_an_outside_standard():
    rep = check_citations("Violates PCI‑DSS [POL-1.1].", IDS)
    assert not rep.ok and rep.external == ["PCI-DSS"]


def test_non_breaking_space_cannot_hide_an_outside_standard():
    rep = check_citations("Violates PCI DSS [POL-1.1].", IDS)
    assert not rep.ok and rep.external == ["PCI-DSS"]


def test_unicode_hyphen_in_an_invented_id_is_still_reported_as_invented():
    rep = check_citations("See [POL‑9.9].", IDS)
    assert rep.invented == ["POL-9.9"]
