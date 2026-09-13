"""Generate paraphrases of the statement templates with a local model through
LiteLLM. The model only varies the phrasing; a rewrite is kept only if every
planted token (the value and the entity) survives verbatim, so ground truth is
preserved by construction. The result is a JSONL fixture that later runs read
without a model, keeping the report reproducible.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Iterable

from .corpus import render
from .facts import FactSheet
from .templates import STATEMENTS

LIST_MARK_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")
THINK_RE = re.compile(r"<think>.*?</think>", re.S)

PROMPT = (
    "Rewrite the sentence below in {n} different ways a developer might type it in a chat "
    "with an assistant. Keep every one of these tokens exactly as written, character for "
    "character: {tokens}. Vary word order and phrasing, mix casual and formal styles, and "
    "sometimes put the value before the name. Output only the rewrites, one per line, with "
    "no numbering and no commentary.\n\nSentence: {sentence}"
)


class LiteLLMClient:
    def __init__(self, url: str, key: str, model: str, timeout: float = 180.0, no_think: bool = True):
        self.url = url.rstrip("/") + "/v1/chat/completions"
        self.key = key
        self.model = model
        self.timeout = timeout
        self.no_think = no_think

    def complete(self, prompt: str) -> str:
        if self.no_think:
            prompt += " /no_think"
        body = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.9,
                "max_tokens": 800,
                "stream": False,
            }
        ).encode()
        req = urllib.request.Request(
            self.url,
            data=body,
            headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.load(resp)
        return data["choices"][0]["message"].get("content") or ""


def parse_rewrites(text: str) -> list[str]:
    text = THINK_RE.sub("", text).replace("/no_think", "")
    lines = [LIST_MARK_RE.sub("", ln).strip().strip('"') for ln in text.splitlines()]
    # A line with an unbalanced quote glued a quote onto a planted token.
    return [ln for ln in lines if ln and ln.count('"') % 2 == 0]


def keep(rewrite: str, original: str, tokens: Iterable[str]) -> bool:
    """A rewrite is usable only if it differs from the original and still
    carries every planted token verbatim."""
    if rewrite == original or not 3 < len(rewrite) < 400:
        return False
    return all(tok in rewrite for tok in tokens)


def generate(
    sheet: FactSheet,
    complete: Callable[[str], str],
    n: int,
    facts_per_template: int,
    kinds: Iterable[str] | None = None,
    log: Callable[[str], None] = lambda _msg: None,
) -> list[dict]:
    records: list[dict] = []
    seen: set[str] = set()
    for kind, templates in STATEMENTS.items():
        if kinds is not None and kind not in kinds:
            continue
        for t in templates:
            for fact in sheet.by_kind(kind)[:facts_per_template]:
                original = render(t, fact)
                tokens = [fact.value] + ([fact.entity] if fact.entity else [])
                prompt = PROMPT.format(n=n, tokens=", ".join(f'"{x}"' for x in tokens), sentence=original)
                try:
                    reply = complete(prompt)
                except (urllib.error.URLError, TimeoutError, KeyError) as e:
                    log(f"{kind}/{t.id}: request failed: {e}")
                    continue
                kept = [r for r in parse_rewrites(reply) if keep(r, original, tokens) and r not in seen]
                for r in kept:
                    seen.add(r)
                    records.append({"template_id": t.id, "role": t.role, "fact": fact.to_dict(), "text": r})
                log(f"{kind}/{t.id}: kept {len(kept)}")
    return records


def write_fixture(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
