"""Aggregate bucket counts and render the report as Markdown and JSON."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from .buckets import (
    BENIGN_BUCKETS,
    CLAIM_BUCKETS,
    FOUND_BUCKETS,
    REGISTRY_BUCKETS,
    benign_bucket,
    claim_bucket,
    registry_bucket,
    spurious_entries,
)
from .corpus import Case, Corpus


@dataclass
class Miss:
    tier: str
    template_id: str
    kind: str
    bucket: str
    text: str
    expected: str
    claim_template_id: str | None = None


@dataclass
class Report:
    seeds: list[int]
    registry: dict[tuple[str, str], Counter] = field(default_factory=lambda: defaultdict(Counter))
    claims: dict[tuple[str, str], Counter] = field(default_factory=lambda: defaultdict(Counter))
    benign: dict[tuple[str, str], Counter] = field(default_factory=lambda: defaultdict(Counter))
    by_statement_template: dict[tuple[str, str, str], Counter] = field(default_factory=lambda: defaultdict(Counter))
    by_claim_template: dict[tuple[str, str], Counter] = field(default_factory=lambda: defaultdict(Counter))
    misses: list[Miss] = field(default_factory=list)
    spurious: list[Miss] = field(default_factory=list)

    # ---- aggregation --------------------------------------------------------------

    @staticmethod
    def build(seeds: list[int], corpus: Corpus, results: dict[str, dict]) -> "Report":
        rep = Report(seeds=seeds)
        for case in corpus.statements():
            rep._add_statement(case, results[case.id])
        for case in corpus.claims():
            rep._add_claim(case, results[case.id])
        for case in corpus.benign():
            rep._add_benign(case, results[case.id])
        return rep

    def _add_benign(self, case: Case, result: dict) -> None:
        assert case.wrong_value is not None and case.claim_template_id is not None
        bucket = benign_bucket(case.fact, case.wrong_value, result["drift"])
        key = (case.fact.kind, case.claim_template_id)
        self.benign[key][bucket] += 1
        self.benign[key]["n"] += 1
        if bucket == "false_positive":
            self.misses.append(
                Miss(
                    case.tier,
                    case.template_id,
                    case.fact.kind,
                    bucket,
                    f"{case.statement!r} then {case.claim!r}",
                    f"known {case.fact.value}, mentioned {case.wrong_value}",
                    claim_template_id=case.claim_template_id,
                )
            )

    def _add_statement(self, case: Case, result: dict) -> None:
        bucket = registry_bucket(case.fact, result["registry"])
        key = (case.fact.kind, case.tier)
        self.registry[key][bucket] += 1
        self.registry[key]["n"] += 1
        self.by_statement_template[(case.tier, case.template_id, case.fact.kind)][bucket] += 1
        self.by_statement_template[(case.tier, case.template_id, case.fact.kind)]["n"] += 1
        expected = f"{case.fact.kind} {case.fact.anchor or '-'} {case.fact.value}"
        if bucket != "anchored":
            got = [e for e in result["registry"] if e["kind"] == case.fact.kind and e["value"] == case.fact.value]
            detail = f"{expected}; got anchor {got[0]['anchor'] or '-'}" if got else expected
            self.misses.append(Miss(case.tier, case.template_id, case.fact.kind, bucket, case.statement, detail))
        for e in spurious_entries(case.fact, case.others, result["registry"]):
            self.registry[key]["spurious"] += 1
            self.spurious.append(
                Miss(case.tier, case.template_id, case.fact.kind, "spurious", case.statement, f"{e['kind']} {e['anchor'] or '-'} {e['value']}")
            )

    def _add_claim(self, case: Case, result: dict) -> None:
        assert case.wrong_value is not None and case.claim_template_id is not None
        bucket = claim_bucket(case.fact, case.wrong_value, result["registry"], result["claims"], result["drift"])
        key = (case.fact.kind, case.tier)
        self.claims[key][bucket] += 1
        self.claims[key]["n"] += 1
        self.by_claim_template[(case.fact.kind, case.claim_template_id)][bucket] += 1
        self.by_claim_template[(case.fact.kind, case.claim_template_id)]["n"] += 1
        if bucket != "fired":
            self.misses.append(
                Miss(
                    case.tier,
                    case.template_id,
                    case.fact.kind,
                    bucket,
                    f"{case.statement!r} then {case.claim!r}",
                    f"known {case.fact.value}, claimed {case.wrong_value}",
                    claim_template_id=case.claim_template_id,
                )
            )

    # ---- headline numbers ---------------------------------------------------------

    def registry_recall(self) -> dict[str, float]:
        """Fraction of statements whose fact reached the registry at all, per kind and overall."""
        return _ratio_by_kind(self.registry, FOUND_BUCKETS)

    def drift_recall(self) -> dict[str, float]:
        """Fraction of wrong claims that fired drift against the right fact, per kind and overall."""
        return _ratio_by_kind(self.claims, ("fired",))

    def benign_fp_rate(self) -> dict[str, float]:
        """Fraction of benign replies that fired drift, per kind and overall. Lower is better."""
        return _ratio_by_kind(self.benign, ("false_positive",))

    # ---- rendering ----------------------------------------------------------------

    def to_markdown(self, limit: int) -> str:
        out: list[str] = []
        n_stmt = sum(c["n"] for c in self.registry.values())
        n_claim = sum(c["n"] for c in self.claims.values())
        out.append(f"# Extraction recall (seeds {', '.join(map(str, self.seeds))}; {n_stmt} statements, {n_claim} claims)\n")

        out.append("## Headline\n")
        out.append("| kind | registry recall | drift-eligible recall | benign false positives |")
        out.append("|------|-----------------|-----------------------|------------------------|")
        rr, dr, fp = self.registry_recall(), self.drift_recall(), self.benign_fp_rate()
        for kind in sorted(set(rr) | set(dr) | set(fp), key=lambda k: (k != "all", k)):
            out.append(f"| {kind} | {_pct(rr.get(kind))} | {_pct(dr.get(kind))} | {_pct(fp.get(kind))} |")
        out.append("")

        out.append("## Registry (statement cases)\n")
        out.append(_table(self.registry, ("kind", "tier"), REGISTRY_BUCKETS + ("spurious",)))
        out.append("## Drift eligibility (claim cases)\n")
        out.append(_table(self.claims, ("kind", "tier"), CLAIM_BUCKETS))
        out.append("## Benign replies (a fired drift is a false positive)\n")
        out.append(_table(self.benign, ("kind", "benign template"), BENIGN_BUCKETS))

        out.append("## Statement templates\n")
        out.append("| tier | template | kind | n | anchored | unanchored | misanchored | missing |")
        out.append("|------|----------|------|---|----------|------------|-------------|---------|")
        for (tier, tid, kind), c in sorted(self.by_statement_template.items()):
            out.append(f"| {tier} | {tid} | {kind} | {c['n']} | " + " | ".join(str(c[b]) for b in REGISTRY_BUCKETS) + " |")
        out.append("")

        out.append("## Claim templates\n")
        out.append("| kind | claim template | n | fired | not recognized | anchor mismatch | ambiguous | other |")
        out.append("|------|----------------|---|-------|----------------|-----------------|-----------|-------|")
        for (kind, tid), c in sorted(self.by_claim_template.items()):
            other = c["n"] - c["fired"] - c["claim_not_recognized"] - c["anchor_mismatch"] - c["ambiguous"]
            out.append(
                f"| {kind} | {tid} | {c['n']} | {c['fired']} | {c['claim_not_recognized']} | {c['anchor_mismatch']} | {c['ambiguous']} | {other} |"
            )
        out.append("")

        out.append(f"## Misses (up to {limit} per bucket, first seed only where duplicated)\n")
        seen: set[tuple] = set()
        per_bucket: Counter = Counter()
        for m in self.misses:
            sig = (m.tier, m.template_id, m.kind, m.bucket, m.claim_template_id)
            if sig in seen or per_bucket[m.bucket] >= limit:
                continue
            seen.add(sig)
            per_bucket[m.bucket] += 1
            where = f"{m.tier}/{m.template_id}" + (f" → {m.claim_template_id}" if m.claim_template_id else "")
            out.append(f"- **{m.bucket}** [{m.kind}] {where}: {_one_line(m.text)}  \n  expected {m.expected}")
        out.append("")

        if self.spurious:
            out.append(f"## Spurious attribute values (up to {limit})\n")
            for m in self.spurious[:limit]:
                out.append(f"- [{m.kind}] {m.tier}/{m.template_id}: {_one_line(m.text)}  \n  registered {m.expected}")
            out.append("")
        return "\n".join(out)

    def to_json(self) -> str:
        return json.dumps(
            {
                "seeds": self.seeds,
                "registry_recall": self.registry_recall(),
                "drift_recall": self.drift_recall(),
                "benign_fp_rate": self.benign_fp_rate(),
                "benign": [{"kind": k, "template": t, **c} for (k, t), c in sorted(self.benign.items())],
                "registry": [{"kind": k, "tier": t, **c} for (k, t), c in sorted(self.registry.items())],
                "claims": [{"kind": k, "tier": t, **c} for (k, t), c in sorted(self.claims.items())],
                "statement_templates": [
                    {"tier": t, "template": tid, "kind": k, **c} for (t, tid, k), c in sorted(self.by_statement_template.items())
                ],
                "claim_templates": [{"kind": k, "template": tid, **c} for (k, tid), c in sorted(self.by_claim_template.items())],
                "misses": [m.__dict__ for m in self.misses],
                "spurious": [m.__dict__ for m in self.spurious],
            },
            indent=2,
            ensure_ascii=False,
        )


def _ratio_by_kind(table: dict[tuple[str, str], Counter], hit_buckets: tuple[str, ...]) -> dict[str, float]:
    hits: Counter = Counter()
    total: Counter = Counter()
    for (kind, _tier), c in table.items():
        h = sum(c[b] for b in hit_buckets)
        hits[kind] += h
        total[kind] += c["n"]
        hits["all"] += h
        total["all"] += c["n"]
    return {k: hits[k] / total[k] for k in total if total[k]}


def _table(table: dict[tuple[str, str], Counter], keys: tuple[str, str], buckets: tuple[str, ...]) -> str:
    head = "| " + " | ".join(keys) + " | n | " + " | ".join(buckets) + " |"
    sep = "|" + "---|" * (len(keys) + 1 + len(buckets))
    rows = [head, sep]
    for (a, b), c in sorted(table.items()):
        rows.append(f"| {a} | {b} | {c['n']} | " + " | ".join(str(c[x]) for x in buckets) + " |")
    rows.append("")
    return "\n".join(rows)


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{100 * x:.0f}%"


def _one_line(text: str) -> str:
    return " ".join(text.split())[:160]
