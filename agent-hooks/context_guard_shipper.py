"""Plumbing shared by the Context Guard agent hooks (``claude-code/``, ``codex/``).

Each hook script adds this directory to ``sys.path`` and imports this module;
nothing here knows a transcript format. What lives here is what every hook
needs: where per-session state goes, how a slice is posted, how the score is
fetched, and a ``main`` that can never fail the agent's session.

Environment (read by every hook):
    CONTEXT_GUARD_URL        default http://127.0.0.1:7432
    CONTEXT_GUARD_STATE_DIR  default $XDG_STATE_HOME/context-guard/<source>
                             (~/.local/state/context-guard/<source>)
    CONTEXT_GUARD_LINK       explanation page template, ``{id}`` is the session id
    CONTEXT_GUARD_HOOK_LOG   optional file; one line per run for troubleshooting
Standard library only.
"""

import json
import os
import sys
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import quote

DEFAULT_URL = "http://127.0.0.1:7432"
SHIP_TIMEOUT_SECONDS = 5
FETCH_TIMEOUT_SECONDS = 0.5


def service_url() -> str:
    return os.environ.get("CONTEXT_GUARD_URL", DEFAULT_URL)


def state_dir(source: str) -> Path:
    override = os.environ.get("CONTEXT_GUARD_STATE_DIR")
    if override:
        return Path(override)
    base = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
    return Path(base) / "context-guard" / source


def read_int(path: Path) -> int:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return 0


def remember(path: Path, text: str) -> None:
    """Best-effort write of a cache file; a failure only loses the fallback."""
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


def log(message: str) -> None:
    path = os.environ.get("CONTEXT_GUARD_HOOK_LOG")
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {message}\n")
    except OSError:
        pass


def ship(url: str, source: str, body: dict) -> None:
    """POST one slice to ``/api/v1/ingest/<source>``; anything but 202 raises."""
    req = urllib.request.Request(
        f"{url.rstrip('/')}/api/v1/ingest/{source}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=SHIP_TIMEOUT_SECONDS) as res:
        if res.status != 202:
            raise OSError(f"ingest answered {res.status}")


def fetch_health(
    url: str, conversation_id: str, message_id: "str | None" = None, timeout: float = FETCH_TIMEOUT_SECONDS
) -> "dict | None":
    """The health document for the conversation (or one completion of it), or
    ``None`` when it is not scored yet. Other failures raise."""
    target = f"{url.rstrip('/')}/api/v1/conversations/{quote(conversation_id, safe='')}/health"
    if message_id:
        target += f"?message_id={quote(message_id, safe='')}"
    try:
        with urllib.request.urlopen(target, timeout=timeout) as res:
            return json.load(res)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def summary_line(result: dict) -> str:
    """The one-line status the service composed, e.g.
    ``🟡 Context Guard 74 · watch · 🟡 context 78% (156,240/200,000) · 1 drift``."""
    summary = result.get("summary")
    return summary if isinstance(summary, str) else f"Context Guard {result.get('score')}"


def page_url(url: str, conversation_id: str, template: "str | None" = None) -> str:
    template = template or f"{url.rstrip('/')}/ui/conversations/{{id}}"
    return template.replace("{id}", quote(conversation_id, safe=""))


def run_hook(run) -> int:
    """Read the hook event from stdin, run it, print what it returns (if
    anything) and exit 0 no matter what: a hook must never fail the session."""
    try:
        output = run(json.load(sys.stdin))
        if output:
            print(output)
    except Exception:  # noqa: BLE001
        log(traceback.format_exc().strip().splitlines()[-1])
    return 0
