#!/usr/bin/env python3
"""Codex CLI hook that ships a session rollout to Context Guard and shows the score.

Configure it as a synchronous ``Stop`` hook, an async ``PostToolUse`` hook and
a ``SessionEnd`` hook (see ``hooks.example.json``). Codex passes the hook
event as JSON on stdin; this script reads ``transcript_path``, ``session_id``
and ``model`` from it, sends the rollout records written since the last ship
to ``POST /api/v1/ingest/codex``, and exits 0. On ``Stop`` it then prints
``{"systemMessage": "<score line>"}``, which Codex shows in the TUI as a dim
``↳ Hook · …`` line under the reply and never sends to the model. On every
other event it prints nothing. Nothing it does can reach the model.

What leaves the machine: ``response_item`` records except ``reasoning``
(encrypted, and large), ``turn_context`` (the model), ``token_usage_record``
and the ``task_started``, ``task_complete``, ``turn_aborted`` and
``token_count`` events, and a ``compacted`` record reduced to its summary
text. ``session_meta`` (the base instructions), ``world_state`` and
everything else stay.

One API response is the run of model output items (assistant messages, tool
calls) closed by the record carrying its token usage: ``token_usage_record``
in paginated history, the ``token_count`` event in legacy history. Cursor
rule: the state file keeps the byte offset of the first record of the last
closed response that was shipped. Every ship starts there, so that response is
resent and deduplicated by the server, and no tool output written after it is
ever stranded between two ships. A response still open when the slice ends has
no usage yet: the server leaves it for the next ship.

``PostToolUse`` fires while the response that called the tool may still be
streaming, so on that event the open response and everything after it are
held back. ``Stop`` waits briefly for the final response's usage record (the
rollout writer runs behind the hook), ships everything, then polls the health
of the response it just shipped so the line shows this turn's score, not the
previous one; a turn that ended in an error is polled by the id the server
gives the failure, so an overflow shows as the red line it is. The model from
the last ``turn_context`` is remembered per session for ships whose slice
starts after it. ``SessionEnd`` is a last chance at exit. Hooks also run for
Codex's internal threads (memory consolidation); a rollout whose
``session_meta.thread_source`` is not ``user`` is ignored.

Environment (see ``agent-hooks/context_guard_shipper.py``):
    CONTEXT_GUARD_URL           default http://127.0.0.1:7432
    CONTEXT_GUARD_STATE_DIR     default $XDG_STATE_HOME/context-guard/codex
    CONTEXT_GUARD_LINK          default {CONTEXT_GUARD_URL}/ui/conversations/{id}
    CONTEXT_GUARD_WAIT_SECONDS  default 2; the Stop hook's whole budget for
                                waiting on the rollout and on the score
    CONTEXT_GUARD_HOOK_LOG      optional troubleshooting log
Codex hands hooks the session's environment snapshot, not the live shell, so
set these in the login environment or inline in the hook ``command``.
Standard library only.
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "agent-hooks"))

import context_guard_shipper as shipper  # noqa: E402

SOURCE = "codex"
SHIPPED_EVENTS = {"task_started", "task_complete", "turn_aborted", "token_count"}
TOOL_CALL_TYPES = {"function_call", "custom_tool_call", "local_shell_call", "tool_search_call", "web_search_call"}
DEFAULT_WAIT_SECONDS = 2.0
SETTLE_INTERVAL = 0.1
POLL_INTERVAL = 0.2
MIN_FETCH_SECONDS = 0.1  # a fetch never blocks the TUI for more than this past the budget

read_int = shipper.read_int
log = shipper.log


def state_dir() -> Path:
    return shipper.state_dir(SOURCE)


def wait_seconds() -> float:
    try:
        return max(0.0, float(os.environ.get("CONTEXT_GUARD_WAIT_SECONDS", DEFAULT_WAIT_SECONDS)))
    except ValueError:
        return DEFAULT_WAIT_SECONDS


def shippable(record: dict) -> "dict | None":
    """The record as it leaves the machine, or None when it stays."""
    kind = record.get("type")
    payload = record.get("payload")
    if not isinstance(payload, dict):
        return None
    if kind == "response_item":
        return None if payload.get("type") == "reasoning" else record
    if kind in ("token_usage_record", "turn_context"):
        return record
    if kind == "event_msg":
        return record if payload.get("type") in SHIPPED_EVENTS else None
    if kind == "compacted":
        return {"type": kind, "timestamp": record.get("timestamp"), "payload": {"message": payload.get("message")}}
    return None


def is_model_output(record: dict) -> bool:
    payload = record["payload"]
    if record["type"] != "response_item":
        return False
    kind = payload.get("type")
    return kind in TOOL_CALL_TYPES or (kind == "message" and payload.get("role") == "assistant")


def usage_record(record: dict) -> "str | None":
    """Which usage record this is (``usage`` or ``token_count``), or None. A
    ``token_usage_record`` closes the response even when its usage cannot be
    read, as it does on the server."""
    payload = record["payload"]
    if record["type"] == "token_usage_record":
        return "usage"
    if record["type"] == "event_msg" and payload.get("type") == "token_count":
        info = payload.get("info")
        if isinstance(info, dict) and isinstance(info.get("last_token_usage"), dict):
            return "token_count"
    return None


def failure_id(record: dict) -> "str | None":
    """The id the server gives a turn that ended in an error, else None."""
    payload = record["payload"]
    if record["type"] == "event_msg" and payload.get("type") == "task_complete" and isinstance(payload.get("error"), dict):
        turn_id = payload.get("turn_id")
        if isinstance(turn_id, str) and turn_id:
            return f"{turn_id}:failure"
    return None


def ends_turn(record: dict) -> bool:
    """What closes an open response without a usage record, exactly as the
    server does: a new turn, an interrupt, a failed turn, or the user's next
    prompt."""
    kind = record["type"]
    payload = record["payload"]
    if kind == "turn_context":
        return True
    if kind == "event_msg":
        return payload.get("type") == "turn_aborted" or failure_id(record) is not None
    return kind == "response_item" and payload.get("type") == "message" and payload.get("role") == "user"


class Slice:
    def __init__(self, records, cursor, open_group, last_closed_id, window, model=None):
        self.records = records
        self.cursor = cursor
        self.open_group = open_group
        self.last_closed_id = last_closed_id
        self.window = window
        self.model = model


def slice_rollout(path: Path, start: int, hold_open_group: bool = False) -> Slice:
    """Read the rollout from ``start`` and group model output into responses.

    Only whole lines are read; a partially written last line waits for the
    next ship. A cursor beyond the end of the file means the rollout was
    rewritten or replaced, so it starts over (the server deduplicates). With
    ``hold_open_group`` a response without its usage record yet, and
    everything after it, is kept back.
    """
    if start > path.stat().st_size:
        start = 0
    with path.open("rb") as f:
        f.seek(start)
        data = f.read()
    end = data.rfind(b"\n")
    if end < 0:
        return Slice([], start, False, None, None)
    records = []
    closed = []  # (byte offset, first item id) of each response closed so far
    open_group = None  # (byte offset, index into records, first item id)
    saw_usage_record = False
    window = None
    model = None
    offset = start
    for line in data[: end + 1].splitlines(keepends=True):
        line_start = offset
        offset += len(line)
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if not isinstance(record, dict):
            continue
        shipped = shippable(record)
        if shipped is None:
            continue
        payload = shipped["payload"]
        if payload.get("type") == "task_started" and isinstance(payload.get("model_context_window"), int):
            window = payload["model_context_window"] or window
        if shipped["type"] == "turn_context" and isinstance(payload.get("model"), str) and payload["model"]:
            model = payload["model"]
        if is_model_output(shipped):
            if open_group is None:
                open_group = (line_start, len(records), payload.get("id") or payload.get("call_id"))
        else:
            usage = usage_record(shipped)
            saw_usage_record = saw_usage_record or usage == "usage"
            closes = usage == "usage" or (usage == "token_count" and not saw_usage_record) or ends_turn(shipped)
            failed = failure_id(shipped)
            if closes and open_group is not None:
                closed.append((open_group[0], failed or open_group[2]))
                open_group = None
            elif failed:
                # A failed turn with nothing open is still the completion the
                # Stop hook must show; resending from here is harmless.
                closed.append((line_start, failed))
        records.append(shipped)
    if hold_open_group and open_group is not None:
        records = records[: open_group[1]]
    cursor = closed[-1][0] if closed else start
    return Slice(records, cursor, open_group is not None, closed[-1][1] if closed else None, window, model)


def user_thread(path: Path) -> bool:
    """False for Codex's internal threads, whose rollouts also run hooks."""
    try:
        with path.open("rb") as f:
            first = json.loads(f.readline())
    except (OSError, ValueError):
        return True
    if not isinstance(first, dict) or first.get("type") != "session_meta":
        return True
    source = (first.get("payload") or {}).get("thread_source")
    return source is None or source == "user"


def score_line(url: str, session_id: str, message_id: "str | None", deadline: float) -> "str | None":
    """The Stop hook's output: this turn's score if it arrives in time, else
    the latest score, else the cached line; None when nothing is known yet."""
    cache = state_dir() / f"{session_id}.status"

    def timeout() -> float:
        """Never block past the deadline by more than MIN_FETCH_SECONDS."""
        return max(MIN_FETCH_SECONDS, min(shipper.FETCH_TIMEOUT_SECONDS, deadline - time.monotonic()))

    try:
        result = None
        while message_id and result is None:
            result = shipper.fetch_health(url, session_id, message_id, timeout=timeout())
            if result is None and time.monotonic() >= deadline:
                break
            if result is None:
                time.sleep(POLL_INTERVAL)
        if result is None:
            result = shipper.fetch_health(url, session_id, timeout=timeout())
    except Exception:  # noqa: BLE001 - unreachable service: show what we last knew
        line = shipper.recall(cache)
        return json.dumps({"systemMessage": line}) if line else None
    if result is None:
        return None
    line = shipper.summary_line(result)
    shipper.remember(cache, line)
    page = shipper.page_url(url, session_id, os.environ.get("CONTEXT_GUARD_LINK"))
    return json.dumps({"systemMessage": f"{line} · {page}"})


def run(event: dict, url: str) -> "str | None":
    transcript = event.get("transcript_path")
    session_id = event.get("session_id")
    name = event.get("hook_event_name")
    if not transcript or not session_id:
        log(f"{name}: no transcript_path or session_id in the event")
        return None
    path = Path(transcript)
    if not user_thread(path):
        log(f"{name} {session_id}: internal thread, ignored")
        return None
    stop = name == "Stop"
    deadline = time.monotonic() + (wait_seconds() if stop else 0)
    cursor = state_dir() / f"{session_id}.cursor"
    window_file = state_dir() / f"{session_id}.window"
    model_file = state_dir() / f"{session_id}.model"
    start = read_int(cursor)
    size = path.stat().st_size
    sl = slice_rollout(path, start, hold_open_group=name == "PostToolUse")
    while stop and sl.open_group and time.monotonic() < deadline:
        time.sleep(SETTLE_INTERVAL)
        grown = path.stat().st_size
        if grown != size:  # only re-read the rollout when the writer has added to it
            size = grown
            sl = slice_rollout(path, start)
    if sl.window:
        shipper.remember(window_file, str(sl.window))
    if sl.model:
        shipper.remember(model_file, sl.model)
    if sl.records:
        shipper.ship(
            url,
            SOURCE,
            {
                "session_id": session_id,
                # A re-ship starts after its turn's turn_context, so the model
                # seen last is kept per session.
                "model": event.get("model") or sl.model or shipper.recall(model_file) or None,
                "context_limit": sl.window or read_int(window_file) or None,
                "records": sl.records,
            },
        )
        cursor.parent.mkdir(parents=True, exist_ok=True)
        cursor.write_text(str(sl.cursor))
        log(f"{name} {session_id}: shipped {len(sl.records)} records, cursor {sl.cursor}")
    else:
        log(f"{name} {session_id}: nothing new")
    if not stop:
        return None
    return score_line(url, session_id, sl.last_closed_id, deadline)


def main() -> int:
    return shipper.run_hook(lambda event: run(event, shipper.service_url()))


if __name__ == "__main__":
    sys.exit(main())
