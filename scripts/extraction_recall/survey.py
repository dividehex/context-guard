"""Survey real conversations for values the strict extractor missed.

There is no fact sheet for real text, so a deliberately loose sweep proposes
candidates (anything port-, version-, path- or setting-shaped) and the strict
extractor is run on the same text; only candidates the strict pass did not
register are kept. They are written as a CSV for drill-down and summarized by
*shape* (the key, marker or preceding token) so a reviewer scans a few dozen
phrasing patterns rather than every row. Sources are Claude Code transcripts on
disk and Context Guard's own SQLite file. The output stays local: it contains
conversation text.
"""

from __future__ import annotations

import csv
import json
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

SECRET_NAME_RE = re.compile(r"(?i)(key|token|secret|pass|cred|auth)")
# Upper-case prose labels that look like environment variables but are not.
LABEL_WORDS = frozenset(
    "NOTE TODO FIXME XXX WARNING WARN ERROR INFO DEBUG TRACE REMINDER PROGRESS IMPORTANT RESULT RESULTS "
    "OUTPUT STATUS SUMMARY STEP STEPS GET POST PUT DELETE PATCH HEAD OK NULL TRUE FALSE FILE FILES NAME "
    "TYPE VALUE DATE TIME TOTAL COUNT SIZE LINE LINES USER HOST PID URL API HTTP HTTPS JSON YAML TOML".split()
)

# Each loose pattern names the kind it guesses; group `value` is the candidate
# and group `shape`, when present, is what the summary groups by.
LOOSE: dict[str, re.Pattern] = {
    "port": re.compile(
        r"(?i)(?<![\w.:/-])(?P<shape>->|\blisten(?:s|ing)?(?:\s+on)?\s+|\bbind(?:s|ing)?(?:\s+to)?\s+|\bport\b\W{0,3}"
        r"|(?:localhost|\d{1,3}(?:\.\d{1,3}){3}):)(?P<value>\d{2,5})(?![\w.])"
    ),
    "version": re.compile(r"(?<![\w./])(?P<shape>[A-Za-z][\w.-]*[-:=\"'(\s]\s?v?)?(?P<value>\d+\.\d+\.\d+)(?![\w.])"),
    "path": re.compile(r"(?<![\w/#>])(?P<value>(?:~|\.\.?)?/[\w.-]+(?:/[\w.-]+)*\.[A-Za-z0-9]{1,5}|(?:~|\.\.?)/[\w.-]+(?:/[\w.-]+)+)"),
    "env_var": re.compile(r"(?<![\w.])(?P<shape>[A-Z][A-Z0-9_]{2,})\s*[:=]\s*[\"']?(?P<value>[\w./:-]+)"),
    "numeric_cfg": re.compile(r"(?<![\w./:\\-])(?P<shape>[a-z][a-z0-9]{1,30})\s*[:=]\s*(?P<value>\d+)(?![\w.])"),
}


@dataclass(frozen=True)
class Source:
    origin: str
    line: int
    role: str
    text: str


@dataclass(frozen=True)
class Candidate:
    origin: str
    line: int
    role: str
    kind_guess: str
    shape: str
    candidate: str
    context: str


# ---- sources --------------------------------------------------------------------------


def _block_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
    return ""


def _is_injected(text: str) -> bool:
    """Claude Code wraps its own injected user-role text in tags (`<command-name>`,
    `<system-reminder>`); none of it is the user stating a fact."""
    return text.lstrip().startswith("<")


def iter_transcripts(root: Path) -> Iterable[Source]:
    """User text and tool results from Claude Code transcripts under `root`."""
    for path in sorted(root.rglob("*.jsonl")):
        with path.open(errors="replace") as fh:
            for n, line in enumerate(fh, start=1):
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("type") != "user":
                    continue
                content = rec.get("message", {}).get("content")
                if isinstance(content, str):
                    if not _is_injected(content):
                        yield Source(str(path), n, "user", content)
                    continue
                if not isinstance(content, list):
                    continue
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "tool_result":
                        yield Source(str(path), n, "tool", _block_text(block.get("content")))
                    elif block.get("type") == "text" and not _is_injected(block.get("text", "")):
                        yield Source(str(path), n, "user", block.get("text", ""))


def iter_database(db: Path) -> Iterable[Source]:
    """User and tool messages from Context Guard's conversation_events table."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = con.execute("SELECT id, new_messages_json FROM conversation_events WHERE new_messages_json IS NOT NULL")
        for event_id, blob in rows:
            try:
                messages = json.loads(blob)
            except json.JSONDecodeError:
                continue
            for i, m in enumerate(messages):
                role = str(m.get("role", "")).lower()
                if role in ("user", "tool"):
                    yield Source(f"db:{event_id}", i, role, _block_text(m.get("content")))
    finally:
        con.close()


# ---- sweep ----------------------------------------------------------------------------


def loose_candidates(text: str) -> Iterable[tuple[str, str, str, int]]:
    """`(kind_guess, shape, candidate, position)` for every loose match in `text`."""
    for kind, pattern in LOOSE.items():
        for m in pattern.finditer(text):
            shape = (m.group("shape") or "").strip() if "shape" in pattern.groupindex else ""
            value = m.group("value")
            if kind == "env_var" and (SECRET_NAME_RE.search(shape) or shape in LABEL_WORDS):
                continue
            if kind == "path":
                shape = value.split("/")[0] + "/" if not value.startswith("/") else "/" + value.split("/")[1]
            yield kind, shape.lower(), value, m.start("value")


def missed_candidates(source: Source, registry: list[dict]) -> Iterable[Candidate]:
    registered = {e["value"] for e in registry}
    seen: set[tuple[str, str]] = set()
    for kind, shape, value, pos in loose_candidates(source.text):
        if value in registered or (kind, value) in seen:
            continue
        if kind == "path" and any(value in v for v in registered):
            continue  # a prefix of a registered path
        seen.add((kind, value))
        lo, hi = max(0, pos - 60), min(len(source.text), pos + len(value) + 60)
        context = " ".join(source.text[lo:hi].split())
        yield Candidate(source.origin, source.line, source.role, kind, shape, value, context)


# ---- output ---------------------------------------------------------------------------


def write_csv(path: Path, candidates: Iterable[Candidate]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["origin", "line", "role", "kind_guess", "shape", "candidate", "context"])
        for c in candidates:
            w.writerow([c.origin, c.line, c.role, c.kind_guess, c.shape, c.candidate, c.context])
            n += 1
    return n


def shape_summary(candidates: list[Candidate], top: int) -> str:
    """Markdown: per kind, the most frequent shapes with counts, distinct
    sources, and one example each. This is the part a human reviews."""
    out: list[str] = ["# Missed-value shapes\n"]
    for kind in LOOSE:
        subset = [c for c in candidates if c.kind_guess == kind]
        if not subset:
            continue
        counts: Counter = Counter(c.shape for c in subset)
        sources: dict[str, set[str]] = {}
        example: dict[str, Candidate] = {}
        for c in subset:
            sources.setdefault(c.shape, set()).add(c.origin)
            example.setdefault(c.shape, c)
        out.append(f"## {kind} ({len(subset)} candidates, {len(counts)} shapes)\n")
        out.append("| shape | n | sources | example |")
        out.append("|-------|---|---------|---------|")
        for shape, n in counts.most_common(top):
            ex = example[shape]
            ctx = ex.context.replace("|", "\\|")[:120]
            out.append(f"| `{shape or '^'}` | {n} | {len(sources[shape])} | {ex.role}: {ctx} |")
        out.append("")
    return "\n".join(out)
