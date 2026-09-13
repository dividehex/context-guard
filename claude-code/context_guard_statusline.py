#!/usr/bin/env python3
"""Claude Code status line that prints the Context Guard score for the session.

Configure it as the ``statusLine`` command (see ``settings.example.json``).
Claude Code passes its status JSON on stdin; this script reads ``session_id``,
fetches ``GET /api/v1/conversations/{session_id}/health`` and prints the
``summary`` line, the same one Open WebUI shows under a reply:

    🟡 Context Guard 74 · watch · 🟡 context 78% (156,240/200,000) · 1 drift

The whole line is a terminal hyperlink (OSC 8) to the explanation page for
the session, ``{CONTEXT_GUARD_URL}/ui/conversations/{session_id}``, which
renders ``GET /api/v1/conversations/{id}/explain``; terminals such as kitty,
iTerm2 and WezTerm open it on click (usually with Ctrl or Cmd held). Set
``CONTEXT_GUARD_LINK`` to point the line at your own front end instead, with
``{id}`` standing for the session id.

It also records the context window Claude Code reports (``context_window``)
so the hook can pass it to Context Guard as the model's limit. When the
service is unreachable it prints the last line it printed for this session,
and nothing if there is none. One request, half a second, never more.

Environment:
    CONTEXT_GUARD_URL        default http://127.0.0.1:7432
    CONTEXT_GUARD_LINK       default {CONTEXT_GUARD_URL}/ui/conversations/{id}
    CONTEXT_GUARD_STATE_DIR  default $XDG_STATE_HOME/context-guard/claude-code
Standard library only.
"""

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import quote

from context_guard_hook import state_dir

TIMEOUT_SECONDS = 0.5


def context_window_total(event: dict):
    window = event.get("context_window")
    if not isinstance(window, dict):
        return None
    for key in ("total", "context_window_size"):
        value = window.get(key)
        if isinstance(value, int) and value > 0:
            return value
    return None


def remember(path: Path, text: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    except OSError:
        pass


def recall(path: Path) -> str:
    try:
        return path.read_text().strip()
    except OSError:
        return ""


def fetch_summary(url: str, session_id: str) -> str:
    """The summary for the session, or "" when it is not scored yet."""
    target = f"{url.rstrip('/')}/api/v1/conversations/{quote(session_id, safe='')}/health"
    try:
        with urllib.request.urlopen(target, timeout=TIMEOUT_SECONDS) as res:
            result = json.load(res)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return ""
        raise
    summary = result.get("summary")
    return summary if isinstance(summary, str) else f"Context Guard {result.get('score')}"


def run(event: dict, url: str) -> str:
    session_id = event.get("session_id")
    if not session_id:
        return ""
    total = context_window_total(event)
    if total:
        remember(state_dir() / f"{session_id}.window", str(total))
    cache = state_dir() / f"{session_id}.status"
    try:
        summary = fetch_summary(url, session_id)
    except Exception:  # noqa: BLE001 - unreachable service: show what we last knew
        return recall(cache)
    if summary:
        remember(cache, summary)
    return summary


def page_url(url: str, session_id: str, template: "str | None" = None) -> str:
    template = template or f"{url.rstrip('/')}/ui/conversations/{{id}}"
    return template.replace("{id}", quote(session_id, safe=""))


def linkify(text: str, href: str) -> str:
    """OSC 8 hyperlink: terminals without support show the text unchanged."""
    return f"\033]8;;{href}\033\\{text}\033]8;;\033\\"


def main() -> int:
    try:
        event = json.load(sys.stdin)
        url = os.environ.get("CONTEXT_GUARD_URL", "http://127.0.0.1:7432")
        line = run(event, url)
        if line:
            line = linkify(line, page_url(url, str(event.get("session_id")), os.environ.get("CONTEXT_GUARD_LINK")))
    except Exception:  # noqa: BLE001 - a status line must never fail the session
        line = ""
    if line:
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
