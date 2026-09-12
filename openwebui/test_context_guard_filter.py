"""Tests for the Open WebUI filter. Run: pytest openwebui/ (needs aiohttp, pydantic, pytest)."""

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from context_guard_filter import Filter


class _Stub:
    """Tiny HTTP server whose responses are scripted per URL path+query."""

    def __init__(self):
        self.responses = {}
        self.requests = []
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                stub.requests.append(self.path)
                queue = stub.responses.get(self.path, [])
                status, body = queue.pop(0) if len(queue) > 1 else (queue[0] if queue else (500, {}))
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


def make_filter(url, **valves):
    f = Filter()
    valves = {"settle_seconds": 0.0, **valves}
    f.valves = Filter.Valves(context_guard_url=url, wait_seconds=1.0, poll_interval=0.05, connect_timeout=0.5, **valves)
    return f


class Emitter:
    def __init__(self):
        self.events = []

    async def __call__(self, event):
        self.events.append(event)


BODY = {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello", "timestamp": 1757600000}]}
META = {"chat_id": "chat-1", "message_id": "msg-1"}
HEALTH = {"score": 74, "status": "watch", "summary": "Context Guard 74 · watch · context 78% · 1 known-value drift"}


def run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def test_shows_status_and_returns_same_body(stub):
    stub.responses["/api/v1/conversations/chat-1/health?message_id=msg-1"] = [(200, HEALTH)]
    f, em = make_filter(stub.url), Emitter()
    body = dict(BODY)
    out = run(f.outlet(body, em, META))
    assert out is body
    assert em.events == [{"type": "status", "data": {"description": HEALTH["summary"], "done": True}}]


def test_polls_until_scored(stub):
    not_yet = (404, {"error": {"code": "not_scored_yet", "message": ""}})
    stub.responses["/api/v1/conversations/chat-1/health?message_id=msg-1"] = [not_yet, not_yet, (200, HEALTH)]
    f, em = make_filter(stub.url), Emitter()
    run(f.outlet(dict(BODY), em, META))
    assert len(stub.requests) == 3
    assert em.events[0]["type"] == "status"


def test_new_chat_unknown_then_scored(stub):
    unknown = (404, {"error": {"code": "unknown_conversation", "message": ""}})
    stub.responses["/api/v1/conversations/chat-1/health?message_id=msg-1"] = [unknown, unknown, (200, HEALTH)]
    f, em = make_filter(stub.url), Emitter()
    run(f.outlet(dict(BODY), em, META))
    assert len(stub.requests) == 3
    assert em.events and em.events[0]["type"] == "status"


def test_settles_on_the_latest_iteration(stub):
    first = {"turn": 3, "score": 100, "status": "healthy", "summary": "first iteration"}
    second = {"turn": 4, "score": 95, "status": "healthy", "summary": "second iteration"}
    stub.responses["/api/v1/conversations/chat-1/health?message_id=msg-1"] = [(200, first), (200, second), (200, second)]
    f, em = make_filter(stub.url, settle_seconds=0.05), Emitter()
    run(f.outlet(dict(BODY), em, META))
    assert em.events[0]["data"]["description"] == "second iteration"
    assert len(stub.requests) == 3  # initial, settle (changed), settle (unchanged)


def test_timestamp_fallback_after_deadline(stub):
    not_yet = (404, {"error": {"code": "not_scored_yet", "message": ""}})
    stub.responses["/api/v1/conversations/chat-1/health?message_id=msg-1"] = [not_yet]
    stub.responses["/api/v1/conversations/chat-1/health?after=1757599999.000"] = [(200, HEALTH)]
    f, em = make_filter(stub.url), Emitter()
    f.valves.wait_seconds = 0.2
    run(f.outlet(dict(BODY), em, META))
    assert any("after=" in r for r in stub.requests)
    assert em.events and em.events[0]["type"] == "status"


def test_connection_refused_is_silent_and_fast(stub):
    stub.close()
    f, em = make_filter(stub.url), Emitter()
    import time

    t = time.monotonic()
    out = run(f.outlet(dict(BODY), em, META))
    assert out == BODY
    assert em.events == []
    assert time.monotonic() - t < 2.0


def test_missing_ids_make_no_request(stub):
    f, em = make_filter(stub.url), Emitter()
    run(f.outlet(dict(BODY), em, {"chat_id": "chat-1"}))
    assert stub.requests == []
    assert em.events == []


def test_show_minimum_and_notification(stub):
    low = {"score": 30, "status": "reset_recommended", "summary": "Context Guard 30 · reset recommended · context 95%"}
    stub.responses["/api/v1/conversations/chat-1/health?message_id=msg-1"] = [(200, low)]
    f, em = make_filter(stub.url, show_minimum="watch", notify_below=40), Emitter()
    run(f.outlet(dict(BODY), em, META))
    assert [e["type"] for e in em.events] == ["status", "notification"]

    stub.responses["/api/v1/conversations/chat-1/health?message_id=msg-1"] = [(200, HEALTH)]
    f, em = make_filter(stub.url, show_minimum="degraded", notify_below=0), Emitter()
    run(f.outlet(dict(BODY), em, META))
    assert em.events == []


def test_never_raises_on_garbage_emitter(stub):
    stub.responses["/api/v1/conversations/chat-1/health?message_id=msg-1"] = [(200, HEALTH)]

    async def broken(_):
        raise RuntimeError("socket gone")

    f = make_filter(stub.url)
    out = run(f.outlet(dict(BODY), broken, META))
    assert out == BODY


def test_no_emitter_makes_no_request(stub):
    f = make_filter(stub.url)
    out = run(f.outlet(dict(BODY), None, META))
    assert out == BODY
    assert stub.requests == []


def test_non_numeric_score_shows_status_without_notification(stub):
    odd = {"score": "n/a", "status": "weird", "summary": "Context Guard ?"}
    stub.responses["/api/v1/conversations/chat-1/health?message_id=msg-1"] = [(200, odd)]
    f, em = make_filter(stub.url, notify_below=40), Emitter()
    run(f.outlet(dict(BODY), em, META))
    assert [e["type"] for e in em.events] == ["status"]


def test_unknown_conversation_after_deadline_gives_up_quietly(stub):
    unknown = (404, {"error": {"code": "unknown_conversation", "message": ""}})
    stub.responses["/api/v1/conversations/chat-1/health?message_id=msg-1"] = [unknown]
    stub.responses["/api/v1/conversations/chat-1/health?after=1757599999.000"] = [unknown]
    f, em = make_filter(stub.url), Emitter()
    f.valves.wait_seconds = 0.15
    run(f.outlet(dict(BODY), em, META))
    assert em.events == []


def test_server_error_stops_polling_immediately(stub):
    stub.responses["/api/v1/conversations/chat-1/health?message_id=msg-1"] = [(500, {})]
    f, em = make_filter(stub.url), Emitter()
    run(f.outlet(dict(BODY), em, META))
    assert len(stub.requests) == 1
    assert em.events == []


def test_settle_stops_on_error_and_keeps_first_result(stub):
    first = {"turn": 1, "score": 100, "status": "healthy", "summary": "first"}
    stub.responses["/api/v1/conversations/chat-1/health?message_id=msg-1"] = [(200, first), (503, {})]
    f, em = make_filter(stub.url, settle_seconds=0.05), Emitter()
    run(f.outlet(dict(BODY), em, META))
    assert em.events[0]["data"]["description"] == "first"


def test_reply_timestamp_helpers():
    assert Filter._reply_timestamp({"messages": [{"role": "assistant", "content": "x"}]}) is None
    assert Filter._reply_timestamp({"messages": []}) is None
    assert Filter._reply_timestamp({"messages": [{"role": "assistant", "timestamp": 5}]}) == 5.0
    f = Filter()
    f.valves.show_minimum = "bogus"
    assert f._should_show("healthy")
    f.valves.show_minimum = "watch"
    assert f._should_show("unknown-status")
