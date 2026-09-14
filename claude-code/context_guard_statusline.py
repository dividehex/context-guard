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

A turn's score is produced out of band: the ``Stop`` hook ships the reply and
Context Guard scores it a moment after the reply is already on screen, but
Claude Code only re-runs this command on its own events (a new reply,
``/compact``, ...) and never while the session is idle. Without help the fresh
score would not show until the next reply, one prompt late. Set
``"refreshInterval": 2`` on the ``statusLine`` entry (see
``settings.example.json``) so Claude Code re-runs this command every couple of
seconds; the just-scored turn then appears without waiting for the next prompt.

Environment (see ``agent-hooks/context_guard_shipper.py``):
    CONTEXT_GUARD_URL        default http://127.0.0.1:7432
    CONTEXT_GUARD_LINK       default {CONTEXT_GUARD_URL}/ui/conversations/{id}
    CONTEXT_GUARD_STATE_DIR  default $XDG_STATE_HOME/context-guard/claude-code
Standard library only.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "agent-hooks"))

import context_guard_shipper as shipper  # noqa: E402
from context_guard_hook import state_dir  # noqa: E402

page_url = shipper.page_url


def context_window_total(event: dict):
    window = event.get("context_window")
    if not isinstance(window, dict):
        return None
    for key in ("total", "context_window_size"):
        value = window.get(key)
        if isinstance(value, int) and value > 0:
            return value
    return None


def fetch_summary(url: str, session_id: str) -> str:
    """The summary for the session, or "" when it is not scored yet."""
    result = shipper.fetch_health(url, session_id)
    return shipper.summary_line(result) if result else ""


def run(event: dict, url: str) -> str:
    session_id = event.get("session_id")
    if not session_id:
        return ""
    total = context_window_total(event)
    if total:
        shipper.remember(state_dir() / f"{session_id}.window", str(total))
    cache = state_dir() / f"{session_id}.status"
    try:
        summary = fetch_summary(url, session_id)
    except Exception:  # noqa: BLE001 - unreachable service: show what we last knew
        return shipper.recall(cache)
    if summary:
        shipper.remember(cache, summary)
    return summary


def linkify(text: str, href: str) -> str:
    """OSC 8 hyperlink: terminals without support show the text unchanged."""
    return f"\033]8;;{href}\033\\{text}\033]8;;\033\\"


def render(event: dict) -> str:
    """The linked status line for the event, or "" for nothing."""
    url = shipper.service_url()
    line = run(event, url)
    if not line:
        return ""
    return linkify(line, page_url(url, str(event.get("session_id")), os.environ.get("CONTEXT_GUARD_LINK")))


def main() -> int:
    return shipper.run_hook(render)


if __name__ == "__main__":
    sys.exit(main())
