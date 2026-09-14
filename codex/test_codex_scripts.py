"""Tests for the Codex CLI hook. Run: pytest codex/ (needs pytest only)."""

import json
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "agent-hooks"))

import context_guard_codex_hook as hook  # noqa: E402
from stub_service import Stub  # noqa: E402

SESSION = "01a08cc2-557b-7220-9231-ec71906fb7c6"
TS = "2026-09-10T19:19:45.278Z"
HEALTH = f"/api/v1/conversations/{SESSION}/health"


@pytest.fixture
def stub():
    s = Stub()
    yield s
    s.close()


@pytest.fixture
def state(tmp_path, monkeypatch):
    monkeypatch.setenv("CONTEXT_GUARD_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("CONTEXT_GUARD_WAIT_SECONDS", "1")
    return tmp_path / "state"


def line(kind, payload):
    return json.dumps({"timestamp": TS, "type": kind, "payload": payload}) + "\n"


def meta(thread_source="user"):
    return line("session_meta", {"id": SESSION, "cwd": "/w", "thread_source": thread_source, "base_instructions": {"text": "secret"}})


def turn_context():
    return line("turn_context", {"turn_id": "turn_1", "model": "gpt-6-astra", "cwd": "/w"})


def task_started(window=258400):
    return line("event_msg", {"type": "task_started", "turn_id": "turn_1", "model_context_window": window})


def user(text):
    return line("response_item", {"type": "message", "id": f"msg_{text[:4]}", "role": "user", "content": [{"type": "input_text", "text": text}]})


def reply(item_id, text):
    return line("response_item", {"type": "message", "id": item_id, "role": "assistant", "content": [{"type": "output_text", "text": text}]})


def call(item_id, call_id):
    return line("response_item", {"type": "custom_tool_call", "id": item_id, "call_id": call_id, "name": "exec", "input": "ls"})


def output(call_id, text):
    return line("response_item", {"type": "custom_tool_call_output", "id": "ctco_1", "call_id": call_id, "output": [{"type": "input_text", "text": text}]})


def usage(response_id):
    return line("token_usage_record", {"response_id": response_id, "usage": {"input_tokens": 10, "output_tokens": 1}})


def token_count():
    return line("event_msg", {"type": "token_count", "info": {"last_token_usage": {"input_tokens": 10, "output_tokens": 1}}})


def reasoning():
    return line("response_item", {"type": "reasoning", "id": "rs_1", "encrypted_content": "gAAA"})


def write_rollout(path, lines):
    path.write_text("".join(lines))


def event(transcript, name="Stop"):
    return {"session_id": SESSION, "transcript_path": str(transcript), "hook_event_name": name, "model": "gpt-6-astra"}


def shipped_types(post):
    return [(r["type"], r["payload"].get("type")) for r in post[1]["records"]]


def offset_of(path, index):
    """Byte offset of the index-th line."""
    return sum(len(l.encode()) for l in path.read_text().splitlines(keepends=True)[:index])


def test_hook_ships_conversation_records_only_and_strips_compaction_history(stub, state, tmp_path):
    rollout = tmp_path / "rollout.jsonl"
    compacted = line("compacted", {"message": "Summary: port 8080.", "replacement_history": [{"role": "user"}]})
    write_rollout(rollout, [meta(), line("world_state", {"full": True}), turn_context(), task_started(), user("hi"), reasoning(), reply("msg_1", "hello"), usage("resp_1"), token_count(), compacted, line("event_msg", {"type": "task_complete", "turn_id": "turn_1"})])
    hook.run(event(rollout, "SessionEnd"), stub.url)
    path, body = stub.posts[0]
    assert path == "/api/v1/ingest/codex"
    assert body["session_id"] == SESSION
    assert body["model"] == "gpt-6-astra"
    assert body["context_limit"] == 258400
    assert shipped_types(stub.posts[0]) == [
        ("turn_context", None), ("event_msg", "task_started"), ("response_item", "message"),
        ("response_item", "message"), ("token_usage_record", None), ("event_msg", "token_count"),
        ("compacted", None), ("event_msg", "task_complete"),
    ]
    assert [r for r in body["records"] if r["type"] == "compacted"][0]["payload"] == {"message": "Summary: port 8080."}
    assert hook.read_int(state / f"{SESSION}.window") == 258400


def test_paginated_cursor_resends_the_last_closed_response(stub, state, tmp_path):
    rollout = tmp_path / "rollout.jsonl"
    write_rollout(rollout, [meta(), turn_context(), user("hi"), call("ctc_1", "call_1"), usage("resp_1"), output("call_1", "a b"), reply("msg_2", "done"), usage("resp_2")])
    hook.run(event(rollout), stub.url)
    assert len(stub.posts[0][1]["records"]) == 7
    assert hook.read_int(state / f"{SESSION}.cursor") == offset_of(rollout, 6)  # reply msg_2
    with rollout.open("a") as f:
        f.write(user("more") + reply("msg_3", "sure") + usage("resp_3"))
    hook.run(event(rollout), stub.url)
    assert [r["payload"].get("id") or r["payload"].get("response_id") for r in stub.posts[1][1]["records"]] == [
        "msg_2", "resp_2", "msg_more", "msg_3", "resp_3",
    ]
    assert hook.read_int(state / f"{SESSION}.cursor") == offset_of(rollout, 9)


def test_legacy_history_closes_responses_on_token_count(stub, state, tmp_path):
    rollout = tmp_path / "rollout.jsonl"
    write_rollout(rollout, [meta(), turn_context(), user("hi"), call("ctc_1", "call_1"), output("call_1", "a"), token_count(), reply("msg_2", "done"), token_count()])
    sl = hook.slice_rollout(rollout, 0)
    assert not sl.open_group
    assert sl.cursor == offset_of(rollout, 6)
    assert sl.last_closed_id == "msg_2"


def test_post_tool_use_holds_back_the_open_response(stub, state, tmp_path):
    rollout = tmp_path / "rollout.jsonl"
    write_rollout(rollout, [meta(), turn_context(), user("hi"), call("ctc_1", "call_1")])
    assert hook.run(event(rollout, "PostToolUse"), stub.url) is None
    assert shipped_types(stub.posts[0]) == [("turn_context", None), ("response_item", "message")]
    assert hook.read_int(state / f"{SESSION}.cursor") == 0
    with rollout.open("a") as f:
        f.write(output("call_1", "a") + token_count())
    hook.run(event(rollout, "PostToolUse"), stub.url)
    assert len(stub.posts[1][1]["records"]) == 5
    assert hook.read_int(state / f"{SESSION}.cursor") == offset_of(rollout, 3)


def test_post_tool_use_ships_a_response_already_closed(stub, state, tmp_path):
    rollout = tmp_path / "rollout.jsonl"
    write_rollout(rollout, [meta(), turn_context(), user("hi"), call("ctc_1", "call_1"), usage("resp_1")])
    hook.run(event(rollout, "PostToolUse"), stub.url)
    assert len(stub.posts[0][1]["records"]) == 4
    assert hook.read_int(state / f"{SESSION}.cursor") == offset_of(rollout, 3)


def test_stop_waits_for_the_usage_record_then_prints_this_turns_score(stub, state, tmp_path):
    rollout = tmp_path / "rollout.jsonl"
    write_rollout(rollout, [meta(), turn_context(), user("hi"), reply("msg_1", "hello")])
    stub.gets[f"{HEALTH}?message_id=msg_1"] = (200, {"score": 95, "summary": "🟢 Context Guard 95 · healthy"})
    stub.gets[HEALTH] = (200, {"score": 80, "summary": "stale"})

    def finish():
        time.sleep(0.2)
        with rollout.open("a") as f:
            f.write(usage("resp_1"))

    threading.Thread(target=finish).start()
    out = hook.run(event(rollout), stub.url)
    assert json.loads(out) == {"systemMessage": f"🟢 Context Guard 95 · healthy · {stub.url}/ui/conversations/{SESSION}"}
    assert shipped_types(stub.posts[0])[-1] == ("token_usage_record", None)
    assert (state / f"{SESSION}.status").read_text() == "🟢 Context Guard 95 · healthy"


def test_stop_falls_back_to_the_latest_score_then_the_cache(stub, state, tmp_path, monkeypatch):
    rollout = tmp_path / "rollout.jsonl"
    write_rollout(rollout, [meta(), turn_context(), user("hi"), reply("msg_1", "hello"), usage("resp_1")])
    monkeypatch.setenv("CONTEXT_GUARD_WAIT_SECONDS", "0.3")
    monkeypatch.setenv("CONTEXT_GUARD_LINK", "https://ui.example/{id}")
    assert hook.run(event(rollout), stub.url) is None, "nothing scored yet: silent"
    stub.gets[HEALTH] = (200, {"score": 80, "summary": "🟡 Context Guard 80 · watch"})
    out = hook.run(event(rollout), stub.url)
    assert json.loads(out) == {"systemMessage": f"🟡 Context Guard 80 · watch · https://ui.example/{SESSION}"}
    out = hook.score_line("http://127.0.0.1:9", SESSION, "msg_1", time.monotonic())
    assert json.loads(out) == {"systemMessage": "🟡 Context Guard 80 · watch"}
    assert hook.score_line("http://127.0.0.1:9", "other", None, time.monotonic()) is None


def test_hook_ignores_internal_threads(stub, state, tmp_path):
    rollout = tmp_path / "rollout.jsonl"
    write_rollout(rollout, [meta("subagent"), turn_context(), user("hi"), reply("msg_1", "x"), usage("resp_1")])
    assert hook.run(event(rollout), stub.url) is None
    assert stub.posts == []


def test_hook_starts_over_when_the_cursor_is_beyond_the_file(stub, state, tmp_path):
    rollout = tmp_path / "rollout.jsonl"
    write_rollout(rollout, [meta(), turn_context(), user("hi"), reply("msg_1", "x"), usage("resp_1")])
    state.mkdir(parents=True)
    (state / f"{SESSION}.cursor").write_text("100000")
    hook.run(event(rollout, "SessionEnd"), stub.url)
    assert len(stub.posts[0][1]["records"]) == 4
    assert hook.read_int(state / f"{SESSION}.cursor") == offset_of(rollout, 3)


def test_hook_leaves_a_partial_last_line_and_skips_bad_lines(stub, state, tmp_path):
    rollout = tmp_path / "rollout.jsonl"
    write_rollout(rollout, [meta(), "not json\n", turn_context(), user("hi"), reply("msg_1", "x")[:-10]])
    hook.run(event(rollout, "SessionEnd"), stub.url)
    assert shipped_types(stub.posts[0]) == [("turn_context", None), ("response_item", "message")]


def test_hook_ships_nothing_without_new_records(stub, state, tmp_path):
    rollout = tmp_path / "rollout.jsonl"
    write_rollout(rollout, [meta(), line("world_state", {})])
    hook.run(event(rollout, "SessionEnd"), stub.url)
    assert stub.posts == []
    assert not (state / f"{SESSION}.cursor").exists()


def test_hook_does_not_advance_the_cursor_when_the_service_is_down(state, tmp_path):
    rollout = tmp_path / "rollout.jsonl"
    write_rollout(rollout, [meta(), turn_context(), user("hi"), reply("msg_1", "x"), usage("resp_1")])
    with pytest.raises(Exception):
        hook.run(event(rollout, "SessionEnd"), "http://127.0.0.1:9")
    assert not (state / f"{SESSION}.cursor").exists()


def test_main_prints_only_the_stop_line_and_never_fails(stub, state, tmp_path, monkeypatch, capsys):
    rollout = tmp_path / "rollout.jsonl"
    write_rollout(rollout, [meta(), turn_context(), user("hi"), reply("msg_1", "x"), usage("resp_1")])
    stub.gets[f"{HEALTH}?message_id=msg_1"] = (200, {"score": 100, "summary": "🟢 Context Guard 100 · healthy"})
    monkeypatch.setenv("CONTEXT_GUARD_URL", stub.url)
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO(json.dumps(event(rollout))))
    assert hook.main() == 0
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1 and json.loads(out[0])["systemMessage"].startswith("🟢 Context Guard 100")
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO("{not json"))
    assert hook.main() == 0
    assert capsys.readouterr().out == ""


def task_failed(turn_id="turn_1"):
    return line("event_msg", {"type": "task_complete", "turn_id": turn_id, "last_agent_message": None,
                              "error": {"message": "Codex ran out of room in the model's context window.",
                                        "codex_error_info": "context_window_exceeded"}})


def test_a_failed_turn_is_polled_by_its_failure_id(stub, state, tmp_path):
    rollout = tmp_path / "rollout.jsonl"
    write_rollout(rollout, [meta(), turn_context(), user("Summarize everything"), reply("msg_1", "Here is"), task_failed()])
    stub.gets[f"{HEALTH}?message_id=msg_1"] = (200, {"score": 100, "summary": "stale healthy line"})
    stub.gets[f"{HEALTH}?message_id=turn_1%3Afailure"] = (200, {"score": 80, "summary": "🟢 Context Guard 80 · good · 🔴 context overflow"})
    out = hook.run(event(rollout), stub.url)
    assert json.loads(out)["systemMessage"].startswith("🟢 Context Guard 80 · good · 🔴 context overflow")
    assert hook.read_int(state / f"{SESSION}.cursor") == offset_of(rollout, 3), "the open reply was closed by the failure"
    # A failure with nothing open is still the completion to show.
    write_rollout(rollout, [meta(), turn_context(), user("Summarize everything"), task_failed()])
    sl = hook.slice_rollout(rollout, 0)
    assert (sl.open_group, sl.last_closed_id, sl.cursor) == (False, "turn_1:failure", offset_of(rollout, 3))


def test_a_user_prompt_closes_the_open_response_as_the_server_does(stub, state, tmp_path):
    rollout = tmp_path / "rollout.jsonl"
    write_rollout(rollout, [meta(), turn_context(), user("hi"), reply("msg_1", "hello"), user("more"), reply("msg_2", "sure"), usage("resp_2")])
    sl = hook.slice_rollout(rollout, 0)
    assert not sl.open_group
    assert sl.last_closed_id == "msg_2"
    assert sl.cursor == offset_of(rollout, 5)
    # msg_1 is closed, so PostToolUse ships it instead of holding it back.
    sl = hook.slice_rollout(rollout, 0, hold_open_group=True)
    assert len(sl.records) == 6


def test_a_usage_record_without_usage_closes_the_response(stub, state, tmp_path):
    rollout = tmp_path / "rollout.jsonl"
    write_rollout(rollout, [meta(), turn_context(), user("hi"), reply("msg_1", "hello"),
                            line("token_usage_record", {"response_id": "resp_1", "usage": None})])
    sl = hook.slice_rollout(rollout, 0)
    assert (sl.open_group, sl.last_closed_id, sl.cursor) == (False, "msg_1", offset_of(rollout, 3))


def test_the_model_is_remembered_per_session(stub, state, tmp_path):
    rollout = tmp_path / "rollout.jsonl"
    write_rollout(rollout, [meta(), turn_context(), user("hi"), reply("msg_1", "hello"), usage("resp_1")])
    bare = {"session_id": SESSION, "transcript_path": str(rollout), "hook_event_name": "SessionEnd"}
    hook.run(bare, stub.url)
    assert stub.posts[0][1]["model"] == "gpt-6-astra"
    assert (state / f"{SESSION}.model").read_text() == "gpt-6-astra"
    with rollout.open("a") as f:
        f.write(user("more") + reply("msg_2", "sure") + usage("resp_2"))
    hook.run(bare, stub.url)  # this slice starts at msg_1, after the turn_context
    assert [r["type"] for r in stub.posts[1][1]["records"]][0] == "response_item"
    assert stub.posts[1][1]["model"] == "gpt-6-astra"


def test_hook_runs_standalone_by_path(stub, state):
    import os
    import subprocess

    done = subprocess.run(
        [sys.executable, str(Path(__file__).resolve().parent / "context_guard_codex_hook.py")],
        input="{}",
        capture_output=True,
        text=True,
        env={**os.environ, "CONTEXT_GUARD_URL": stub.url, "CONTEXT_GUARD_STATE_DIR": str(state)},
        timeout=10,
    )
    assert done.returncode == 0 and done.stdout == "" and done.stderr == "", done.stderr
