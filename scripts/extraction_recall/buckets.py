"""Classify one case's extractor output against its planted fact.

Registry buckets say whether the fact reached the registry and with which
anchor. Claim buckets say whether a wrong claim fired drift and, when it did
not, which condition of the drift rule stopped it. The diagnosis mirrors
`detect_drift` in known_values.rs; the fired/not-fired verdict itself always
comes from the binary.
"""

from __future__ import annotations

from .facts import ATTRIBUTE_KINDS, Fact, expected_triples

REGISTRY_BUCKETS = ("anchored", "unanchored", "misanchored", "missing")
CLAIM_BUCKETS = (
    "fired",
    "fired_misattributed",
    "claim_not_recognized",
    "fact_missing",
    "suppressed_known_elsewhere",
    "ambiguous",
    "anchor_mismatch",
    "unexplained",
)
FOUND_BUCKETS = ("anchored", "unanchored", "misanchored")
BENIGN_BUCKETS = ("silent", "false_positive")


def registry_bucket(fact: Fact, registry: list[dict]) -> str:
    """The entity's own name always counts as anchored: the extractor may anchor
    a plain-word entity through a name-value pattern (`nginx 1.2.3`) even though
    the generic anchor rule would leave it empty."""
    same = [e for e in registry if e["kind"] == fact.kind and e["value"] == fact.value]
    if not same:
        return "missing"
    if any(e["anchor"] in (fact.anchor, fact.entity.lower()) for e in same):
        return "anchored"
    if any(e["anchor"] == "" for e in same):
        return "unanchored"
    return "misanchored"


def spurious_entries(fact: Fact, others: tuple[Fact, ...], registry: list[dict]) -> list[dict]:
    """Attribute-kind registry entries that no planted fact accounts for. Only
    attribute kinds matter: they are the ones a claim can be checked against."""
    expected = {(k, v) for f in (fact, *others) for (k, _a, v) in expected_triples(f)}
    return [e for e in registry if e["kind"] in ATTRIBUTE_KINDS and (e["kind"], e["value"]) not in expected]


def claim_bucket(fact: Fact, wrong: str, registry: list[dict], claims: list[dict], drift: list[dict]) -> str:
    fired = [d for d in drift if d["kind"] == fact.kind and d["claimed"] == wrong]
    if fired:
        return "fired" if any(d["known"] == fact.value for d in fired) else "fired_misattributed"
    recognized = [c for c in claims if c["kind"] == fact.kind and c["value"] == wrong]
    if not recognized:
        return "claim_not_recognized"
    if not any(e["kind"] == fact.kind and e["value"] == fact.value for e in registry):
        return "fact_missing"
    if any(e["kind"] == fact.kind and e["value"] == wrong for e in registry):
        return "suppressed_known_elsewhere"
    for c in recognized:
        same_anchor = [e for e in registry if e["kind"] == fact.kind and e["anchor"] == c["anchor"]]
        if len(same_anchor) > 1:
            return "ambiguous"
        if len(same_anchor) == 1:
            return "unexplained"  # one entry under the claim's anchor: the rule should have fired
    return "anchor_mismatch"


def benign_bucket(fact: Fact, wrong: str, drift: list[dict]) -> str:
    """A benign reply mentioned `wrong` without contradicting the fact."""
    fired = any(d["kind"] == fact.kind and d["claimed"] == wrong for d in drift)
    return "false_positive" if fired else "silent"
