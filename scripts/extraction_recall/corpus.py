"""Turn fact sheets and templates into cases the extractor binary can run.

A *statement case* has one user or tool message and measures what the registry
records. A *claim case* adds an assistant message carrying a wrong value and
measures whether drift fires. Attribute kinds get claim cases; entity kinds
(path, url, hostname, container) cannot drift and get statement cases only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .facts import ATTRIBUTE_KINDS, Fact, FactSheet
from .templates import BENIGN, CLAIMS, MULTI, STATEMENTS, TOOL, Template

PARAPHRASE_TIER = "paraphrase"
BENIGN_TIER = "benign"


@dataclass(frozen=True)
class Case:
    id: str
    tier: str
    template_id: str
    fact: Fact
    statement: str
    statement_role: str
    # Facts stated alongside `fact` in the same message (multi-fact tier).
    others: tuple[Fact, ...] = ()
    claim_template_id: str | None = None
    claim: str | None = None
    wrong_value: str | None = None
    # A benign claim mentions the wrong value without contradicting the fact;
    # drift firing on it is a false positive.
    benign: bool = False

    @property
    def is_claim(self) -> bool:
        return self.claim is not None

    @property
    def statement_key(self) -> str:
        """Identifies the statement independent of any claim variant."""
        return f"{self.tier}:{self.template_id}:{self.fact.kind}:{self.fact.value}"

    def messages(self) -> list[dict]:
        msgs = [{"role": self.statement_role, "text": self.statement}]
        if self.claim is not None:
            msgs.append({"role": "assistant", "text": self.claim})
        return msgs

    def to_request(self) -> dict:
        return {"id": self.id, "messages": self.messages()}


def render(template: Template, fact: Fact, second: Fact | None = None) -> str:
    fields = {"entity": fact.entity, "value": fact.value}
    if second is not None:
        fields.update(entity2=second.entity, value2=second.value)
    return template.text.format(**fields)


def _with_claims(base: Case, sheet: FactSheet) -> Iterable[Case]:
    yield base
    if base.fact.kind not in ATTRIBUTE_KINDS:
        return
    wrong = sheet.wrong_value(base.fact)
    for ct in CLAIMS[base.fact.kind]:
        yield Case(
            id=f"{base.id}:c:{ct.id}",
            tier=base.tier,
            template_id=base.template_id,
            fact=base.fact,
            statement=base.statement,
            statement_role=base.statement_role,
            others=base.others,
            claim_template_id=ct.id,
            claim=render(ct, Fact(base.fact.kind, base.fact.entity, wrong, base.fact.anchor)),
            wrong_value=wrong,
        )


def template_cases(sheet: FactSheet) -> Iterable[Case]:
    for kind, templates in STATEMENTS.items():
        for t in templates:
            for fact in sheet.by_kind(kind):
                base = Case(
                    id=f"template:{t.id}:{kind}:{fact.value}",
                    tier="template",
                    template_id=t.id,
                    fact=fact,
                    statement=render(t, fact),
                    statement_role=t.role,
                )
                yield from _with_claims(base, sheet)


def multi_cases(sheet: FactSheet) -> Iterable[Case]:
    for kind, templates in MULTI.items():
        facts = sheet.by_kind(kind)
        if len(facts) < 2:
            continue
        first, second = facts[0], facts[1]
        for t in templates:
            base = Case(
                id=f"multi:{t.id}:{kind}:{first.value}",
                tier="multi",
                template_id=t.id,
                fact=first,
                statement=render(t, first, second),
                statement_role=t.role,
                others=(second,),
            )
            yield from _with_claims(base, sheet)


def tool_cases(sheet: FactSheet) -> Iterable[Case]:
    for t in TOOL:
        for fact in sheet.by_kind(t.kind):
            base = Case(
                id=f"tool:{t.id}:{t.kind}:{fact.value}",
                tier="tool",
                template_id=t.id,
                fact=fact,
                statement=render(t, fact),
                statement_role=t.role,
            )
            yield from _with_claims(base, sheet)


def benign_cases(sheet: FactSheet) -> Iterable[Case]:
    """A fact in its two most common phrasings, then a reply that mentions a
    different value benignly. Every drift here is a false positive."""
    for kind, templates in BENIGN.items():
        for st in STATEMENTS[kind][:2]:
            for fact in sheet.by_kind(kind):
                wrong = sheet.wrong_value(fact)
                for bt in templates:
                    yield Case(
                        id=f"{BENIGN_TIER}:{st.id}:{kind}:{fact.value}:b:{bt.id}",
                        tier=BENIGN_TIER,
                        template_id=st.id,
                        fact=fact,
                        statement=render(st, fact),
                        statement_role=st.role,
                        claim_template_id=bt.id,
                        claim=render(bt, Fact(kind, fact.entity, wrong, fact.anchor)),
                        wrong_value=wrong,
                        benign=True,
                    )


def paraphrase_cases(path: Path, sheet: FactSheet) -> Iterable[Case]:
    """Cases from a paraphrase fixture written by `paraphrase.py`. Each line
    holds the fact it was generated from, so ground truth travels with the text."""
    if not path.exists():
        return
    with path.open() as fh:
        for n, line in enumerate(fh):
            if not line.strip():
                continue
            rec = json.loads(line)
            fact = Fact.from_dict(rec["fact"])
            base = Case(
                id=f"{PARAPHRASE_TIER}:{rec['template_id']}:{fact.kind}:{fact.value}:{n}",
                tier=PARAPHRASE_TIER,
                template_id=rec["template_id"],
                fact=fact,
                statement=rec["text"],
                statement_role=rec.get("role", "user"),
            )
            yield from _with_claims(base, sheet)


@dataclass
class Corpus:
    cases: list[Case] = field(default_factory=list)

    @staticmethod
    def build(sheets: Iterable[FactSheet], tiers: Iterable[str], paraphrase_path: Path | None) -> "Corpus":
        tiers = set(tiers)
        corpus = Corpus()
        sheets = list(sheets)
        for sheet in sheets:
            if "template" in tiers:
                corpus.cases.extend(template_cases(sheet))
            if "multi" in tiers:
                corpus.cases.extend(multi_cases(sheet))
            if "tool" in tiers:
                corpus.cases.extend(tool_cases(sheet))
            if BENIGN_TIER in tiers:
                corpus.cases.extend(benign_cases(sheet))
        # The fixture carries its own facts, so it is independent of the sheets;
        # the first sheet only supplies wrong values for its claim cases.
        if PARAPHRASE_TIER in tiers and paraphrase_path is not None and sheets:
            corpus.cases.extend(paraphrase_cases(paraphrase_path, sheets[0]))
        return corpus

    def statements(self) -> list[Case]:
        return [c for c in self.cases if not c.is_claim]

    def claims(self) -> list[Case]:
        return [c for c in self.cases if c.is_claim and not c.benign]

    def benign(self) -> list[Case]:
        return [c for c in self.cases if c.benign]
