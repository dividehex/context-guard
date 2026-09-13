"""Tests for the Claude Code hook and status line. Run: pytest claude-code/ (needs pytest only)."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import context_guard_hook as hook
import context_guard_statusline as statusline

SESSION = "880138cf-78cd-4d41-9940-a4aa38c2aaec"


class _Stub:
    """Tiny HTTP server that records POST bodies and scripts GET responses per path."""

    def __init__(self):
        self.posts = []
        self.gets = {}
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                stub.posts.append((self.path, json.loads(self.rfile.read(length))))
                self._reply(202, {"accepted": 1, "dropped": 0})

            def do_GET(self):
                status, body = stub.gets.get(self.path, (404, {"error": {"code": "not_found"}}))
                self._reply(status, body)

            def _reply(self, status, body):
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *_):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def close(self):
        self.server.shutdown()


@pytest.fixture
def stub():
    s = _Stub()
    yield s
    s.close()


@pytest.fixture
def state(tmp_path, monkeypatch):
    monkeypatch.setenv("CONTEXT_GUARD_STATE_DIR", str(tmp_path / "state"))
    return tmp_path / "state"


def record(kind, **fields):
    return json.dumps({"type": kind, "sessionId": SESSION, **fields}) + "\n"


def write_transcript(path, lines):
    path.write_text("".join(lines))


def hook_event(transcript):
    return {"session_id": SESSION, "transcript_path": str(transcript), "hook_event_name": "Stop"}


def test_hook_ships_only_message_records_and_the_context_window(tmp_path, stub, state):
    transcript = tmp_path / "t.jsonl"
    write_transcript(
        transcript,
        [
            record("attachment", attachment={"type": "environment", "secret": "not shipped"}),
            record("user", uuid="u1", message={"role": "user", "content": "hi"}),
            record("assistant", uuid="a1", requestId="req_A", message={"content": [{"type": "text", "text": "hello"}]}),
            record("mode", mode="default"),
        ],
    )
    (state).mkdir(parents=True)
    (state / f"{SESSION}.window").write_text("200000")

    hook.run(hook_event(transcript), stub.url)

    assert len(stub.posts) == 1
    path, body = stub.posts[0]
    assert path == "/api/v1/ingest/claude-code"
    assert [r["type"] for r in body["records"]] == ["user", "assistant"]
    assert body["context_limit"] == 200000


def test_hook_cursor_resends_the_last_group_and_never_strands_trailing_records(tmp_path, stub, state):
    transcript = tmp_path / "t.jsonl"
    lines = [
        record("user", uuid="u1", message={"role": "user", "content": "hi"}),
        record("assistant", uuid="a1", requestId="req_A", message={"content": [{"type": "thinking"}]}),
        record("assistant", uuid="a2", requestId="req_A", message={"content": [{"type": "tool_use", "id": "t1", "name": "Read", "input": {}}]}),
        record("user", uuid="u2", message={"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "x"}]}),
    ]
    write_transcript(transcript, lines)
    hook.run(hook_event(transcript), stub.url)
    first = stub.posts[0][1]["records"]
    assert [r.get("uuid") for r in first] == ["u1", "a1", "a2", "u2"]

    # The cursor points at the first record of req_A, so the next ship starts there.
    cursor = int((state / f"{SESSION}.cursor").read_text())
    assert cursor == len(lines[0].encode())

    lines.append(record("assistant", uuid="a3", requestId="req_B", message={"content": [{"type": "text", "text": "done"}]}))
    write_transcript(transcript, lines)
    hook.run(hook_event(transcript), stub.url)
    second = stub.posts[1][1]["records"]
    assert [r.get("uuid") for r in second] == ["a1", "a2", "u2", "a3"]
    assert int((state / f"{SESSION}.cursor").read_text()) == sum(len(l.encode()) for l in lines[:4])


def test_post_tool_use_holds_back_the_reply_still_in_progress(tmp_path, stub, state):
    transcript = tmp_path / "t.jsonl"
    lines = [
        record("user", uuid="u1", message={"role": "user", "content": "hi"}),
        record("assistant", uuid="a1", requestId="req_A", message={"content": [{"type": "tool_use", "id": "t1", "name": "Read", "input": {}}]}),
        record("user", uuid="u2", message={"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "x"}]}),
    ]
    write_transcript(transcript, lines)
    event = {**hook_event(transcript), "hook_event_name": "PostToolUse"}
    hook.run(event, stub.url)
    # req_A may still be streaming more tool calls: only the prompt goes now.
    assert [r["uuid"] for r in stub.posts[0][1]["records"]] == ["u1"]
    assert hook.read_int(state / f"{SESSION}.cursor") == 0

    # A second tool call of the same response lands, then the turn ends.
    lines.append(record("assistant", uuid="a2", requestId="req_A", message={"content": [{"type": "tool_use", "id": "t2", "name": "Bash", "input": {}}]}))
    lines.append(record("user", uuid="u3", message={"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t2", "content": "y"}]}))
    lines.append(record("assistant", uuid="a3", requestId="req_B", message={"content": [{"type": "text", "text": "done"}]}))
    write_transcript(transcript, lines)
    hook.run(hook_event(transcript), stub.url)  # Stop
    assert [r["uuid"] for r in stub.posts[1][1]["records"]] == ["u1", "a1", "u2", "a2", "u3", "a3"]
    assert int((state / f"{SESSION}.cursor").read_text()) == sum(len(l.encode()) for l in lines[:5])


def test_hook_leaves_a_partial_last_line_for_next_time(tmp_path, stub, state):
    transcript = tmp_path / "t.jsonl"
    complete = record("user", uuid="u1", message={"role": "user", "content": "hi"})
    transcript.write_text(complete + '{"type": "assistant", "uuid": "a1"')
    hook.run(hook_event(transcript), stub.url)
    assert [r["uuid"] for r in stub.posts[0][1]["records"]] == ["u1"]


def test_hook_ships_nothing_without_new_records_and_skips_bad_lines(tmp_path, stub, state):
    transcript = tmp_path / "t.jsonl"
    transcript.write_text("not json\n" + record("cost-state") + "\n")
    hook.run(hook_event(transcript), stub.url)
    assert stub.posts == []
    assert not (state / f"{SESSION}.cursor").exists()


def test_hook_main_never_fails_the_session(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO("{not json"))
    assert hook.main() == 0
    assert capsys.readouterr().out == ""


def test_hook_does_not_advance_the_cursor_when_the_service_is_down(tmp_path, state):
    transcript = tmp_path / "t.jsonl"
    write_transcript(transcript, [record("user", uuid="u1", message={"role": "user", "content": "hi"})])
    with pytest.raises(Exception):
        hook.run(hook_event(transcript), "http://127.0.0.1:9")
    assert not (state / f"{SESSION}.cursor").exists()


def test_statusline_prints_the_summary_and_records_the_context_window(stub, state):
    summary = "🟡 Context Guard 74 · watch · 🟡 context 78% (156,240/200,000) · 1 drift"
    stub.gets[f"/api/v1/conversations/{SESSION}/health"] = (200, {"score": 74, "summary": summary})
    line = statusline.run({"session_id": SESSION, "context_window": {"used": 1000, "total": 200000}}, stub.url)
    assert line == summary
    assert (state / f"{SESSION}.window").read_text() == "200000"
    assert (state / f"{SESSION}.status").read_text() == summary


def test_statusline_is_silent_before_the_first_score(stub, state):
    assert statusline.run({"session_id": SESSION}, stub.url) == ""
    assert not (state / f"{SESSION}.status").exists()


def test_statusline_falls_back_to_the_cached_line_when_unreachable(state):
    state.mkdir(parents=True)
    (state / f"{SESSION}.status").write_text("🟢 Context Guard 100 · healthy\n")
    assert statusline.run({"session_id": SESSION}, "http://127.0.0.1:9") == "🟢 Context Guard 100 · healthy"
    assert statusline.run({"session_id": "other"}, "http://127.0.0.1:9") == ""


def test_statusline_prints_the_whole_line_as_a_link_to_the_page(stub, state, monkeypatch, capsys):
    summary = "🟢 Context Guard 100 · healthy"
    stub.gets[f"/api/v1/conversations/{SESSION}/health"] = (200, {"score": 100, "summary": summary})
    monkeypatch.setenv("CONTEXT_GUARD_URL", stub.url)
    monkeypatch.delenv("CONTEXT_GUARD_LINK", raising=False)
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO(json.dumps({"session_id": SESSION})))
    assert statusline.main() == 0
    out = capsys.readouterr().out
    assert out == f"\033]8;;{stub.url}/ui/conversations/{SESSION}\033\\{summary}\033]8;;\033\\\n"


def test_statusline_link_template_points_at_another_front_end():
    assert statusline.page_url("http://cg:7432/", "a b", "https://ui.example/x/{id}") == "https://ui.example/x/a%20b"
    assert statusline.page_url("http://cg:7432/", "s1") == "http://cg:7432/ui/conversations/s1"


def test_statusline_main_never_fails_the_session(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO("{not json"))
    assert statusline.main() == 0
    assert capsys.readouterr().out == ""
