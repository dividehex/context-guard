"""A stand-in Context Guard service for the hook tests (pytest only)."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer


class Stub:
    """Tiny HTTP server that records POST bodies and scripts GET responses per
    path (query string included, so ``.../health?message_id=x`` is its own key)."""

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
