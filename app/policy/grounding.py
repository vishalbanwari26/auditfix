import re
from dataclasses import dataclass

CLAUSE = re.compile(r"\bPOL-\d+\.\d+\b")
EXTERNAL = re.compile(r"\b(PCI[- ]DSS|GDPR|HIPAA|SOX|ISO[- ]?27001|NIST)\b", re.I)


# Models often emit look-alike characters. Map them to ASCII before matching so a
# real citation is not missed and an outside standard cannot hide behind one.
_LOOKALIKES = {
    **dict.fromkeys(map(ord, "‐‑‒–—―−"), "-"),
    **dict.fromkeys(map(ord, "    "), " "),
}


def normalise(text: str) -> str:
    return text.translate(_LOOKALIKES)


@dataclass
class GroundingReport:
    cited: list[str]  # every POL-x.y the text mentions
    invented: list[str]  # cited IDs that are not in the policy
    external: list[str]  # references to standards outside the policy

    @property
    def ok(self) -> bool:
        return bool(self.cited) and not self.invented and not self.external

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "cited": self.cited,
            "invented": self.invented,
            "external": self.external,
        }


def check_citations(text: str, policy_ids: set[str]) -> GroundingReport:
    text = normalise(text)
    cited = sorted(set(CLAUSE.findall(text)))
    return GroundingReport(
        cited=cited,
        invented=[c for c in cited if c not in policy_ids],
        external=sorted({m.upper().replace(" ", "-") for m in EXTERNAL.findall(text)}),
    )
