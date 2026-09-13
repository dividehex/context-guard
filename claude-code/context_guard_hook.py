#!/usr/bin/env python3
"""Claude Code hook that ships a session transcript to Context Guard.

Configure it as an async ``Stop`` and ``PostToolUse`` hook (see
``settings.example.json``). Claude Code passes the hook event as JSON on stdin;
this script reads ``transcript_path`` and ``session_id`` from it, sends the
transcript records written since the last ship to
``POST /api/v1/ingest/claude-code``, and exits 0 without printing anything.
Nothing it does can reach the model or delay the session.

Only ``user``, ``assistant`` and ``system`` records leave the machine: the
transcript's attachment and bookkeeping records carry environment details and
account information that Context Guard has no use for.

Cursor rule: the state file keeps the byte offset of the first record of the
last assistant group (one API call) that was shipped. Every ship starts there,
so that group is resent and deduplicated by the server, and any user or tool
records written after it are never stranded between two ships.

A ``PostToolUse`` event can fire while the same API response is still
streaming further tool calls, so on that event the trailing assistant group is
held back and shipped by the next event. ``Stop`` and ``SessionEnd`` mark the
end of a turn or of the process: everything is shipped. (A headless
``claude -p`` run exits without reliably waiting for either, so its final
reply may only arrive with a later ship.)

Environment:
    CONTEXT_GUARD_URL        default http://127.0.0.1:7432
    CONTEXT_GUARD_STATE_DIR  default $XDG_STATE_HOME/context-guard/claude-code
                             (~/.local/state/context-guard/claude-code)
    CONTEXT_GUARD_HOOK_LOG   optional file; one line per run (event, records
                             shipped, cursor, or the error) for troubleshooting
Standard library only.
"""

import json
import os
import sys
import time
import traceback
import urllib.request
from pathlib import Path

SHIPPED_TYPES = {"user", "assistant", "system"}
TIMEOUT_SECONDS = 5


def state_dir() -> Path:
    override = os.environ.get("CONTEXT_GUARD_STATE_DIR")
    if override:
        return Path(override)
    base = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
    return Path(base) / "context-guard" / "claude-code"


def read_int(path: Path) -> int:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return 0


def read_context_limit(session_id: str) -> "int | None":
    """The status line stores the context window Claude Code reports to it."""
    value = read_int(state_dir() / f"{session_id}.window")
    return value or None


def slice_transcript(path: Path, start: int, hold_last_group: bool = False):
    """Return (records to ship, offset of the last shipped assistant group's first record).

    Only whole lines are shipped; a partially written last line is left for the
    next ship. Lines that are not JSON objects are skipped. With
    ``hold_last_group`` the trailing assistant group and everything after it
    are kept back, because that reply may still be in progress.
    """
    with path.open("rb") as f:
        f.seek(start)
        data = f.read()
    end = data.rfind(b"\n")
    if end < 0:
        return [], start
    records = []
    offset = start
    group_starts = []  # (byte offset, index into records) of each assistant group
    group_id = None
    for line in data[: end + 1].splitlines(keepends=True):
        line_start = offset
        offset += len(line)
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if not isinstance(record, dict) or record.get("type") not in SHIPPED_TYPES:
            continue
        if record.get("type") == "assistant":
            request_id = record.get("requestId") or record.get("uuid")
            if request_id != group_id:
                group_id = request_id
                group_starts.append((line_start, len(records)))
        records.append(record)
    if hold_last_group and group_starts:
        _, index = group_starts.pop()
        records = records[:index]
    cursor = group_starts[-1][0] if group_starts else start
    return records, cursor


def ship(url: str, records, context_limit) -> None:
    body = json.dumps({"records": records, "context_limit": context_limit}).encode()
    req = urllib.request.Request(
        f"{url.rstrip('/')}/api/v1/ingest/claude-code",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as res:
        if res.status != 202:
            raise OSError(f"ingest answered {res.status}")


def log(message: str) -> None:
    path = os.environ.get("CONTEXT_GUARD_HOOK_LOG")
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {message}\n")
    except OSError:
        pass


def run(event: dict, url: str) -> None:
    transcript = event.get("transcript_path")
    session_id = event.get("session_id")
    name = event.get("hook_event_name")
    if not transcript or not session_id:
        log(f"{name}: no transcript_path or session_id in the event")
        return
    cursor = state_dir() / f"{session_id}.cursor"
    in_progress = name == "PostToolUse"
    records, next_start = slice_transcript(Path(transcript), read_int(cursor), in_progress)
    if not records:
        log(f"{name} {session_id}: nothing new")
        return
    ship(url, records, read_context_limit(session_id))
    cursor.parent.mkdir(parents=True, exist_ok=True)
    cursor.write_text(str(next_start))
    log(f"{name} {session_id}: shipped {len(records)} records, cursor {next_start}")


def main() -> int:
    try:
        event = json.load(sys.stdin)
        run(event, os.environ.get("CONTEXT_GUARD_URL", "http://127.0.0.1:7432"))
    except Exception:  # noqa: BLE001 - a hook must never fail the session
        log(traceback.format_exc().strip().splitlines()[-1])
    return 0


if __name__ == "__main__":
    sys.exit(main())
