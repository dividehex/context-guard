#!/usr/bin/env python3
"""
End-to-end degradation test for Context Guard.

Drives one scripted chat through LiteLLM (with the same headers Open WebUI
sends) against a real model and checks, turn by turn, that Context Guard
records the expected signal and score.

Scripted assistant wording comes from an "echo" system prompt sent with a
minimal request history (system + "Go."), so the model has nothing to correct
and repeats the text verbatim. Context Guard ignores system prompts when it
builds its fact registry and remembers facts per conversation rather than per
request (the same property that makes it robust to Open WebUI's context
compaction), so the echoed text is judged against what the user said in
earlier turns. Tool-call stages use the model for real.

Stages (all in one chat, so anomalies accumulate inside the scoring window):

  1  baseline: the user states facts                      -> 100 healthy
  2  assistant contradicts a user-stated port             -> known_value_drift      -15
  3  assistant names a near-duplicate model identifier     -> suspicious_identifier   -5
  4-6 assistant repeats the same reply three times         -> response_loop           -5
  7  history carries a tool result nobody asked for        -> tool_result_without_call -20
  8-10 assistant calls the same tool with identical args   -> repeated_tool_call      -5
  11 assistant cites a tool-call id that never existed     -> tool_call_id_reference_unknown -25

Optionally (--context-model) a second short chat pads the prompt to ~75 %
of a small model's context limit and expects the context_70 penalty.

Usage:
  scripts/e2e_degradation.py --litellm-key "$LITELLM_MASTER_KEY" [--model qwen3-30b-a3b] [--context-model qwen3-vl-4b]

Exit status 0 when every stage passed.
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
import uuid

LOOP_TEXT = (
    "I have checked the configuration file and the service is running on the expected port "
    "with no errors reported in the logs at this time, so nothing needs to change right now."
)


class Client:
    def __init__(self, litellm_url, litellm_key, guard_url, model, chat_id, user_id):
        self.litellm_url = litellm_url.rstrip("/")
        self.litellm_key = litellm_key
        self.guard_url = guard_url.rstrip("/")
        self.model = model
        self.chat_id = chat_id
        self.user_id = user_id
        self.messages = []
        self.turn = 0

    # ---- LiteLLM -------------------------------------------------------------------------

    def complete(self, system, user_text=None, tools=None, tool_choice=None, max_tokens=160, with_history=True):
        """Send the running history (plus a new user message) as a streaming request,
        exactly the way Open WebUI does, and append the assistant reply to the history.
        With `with_history=False` only the system prompt and the new user message are
        sent (an echo turn); the reply is still appended to the running history."""
        if user_text is not None:
            self.messages.append({"role": "user", "content": user_text})
        self.turn += 1
        message_id = f"e2e-msg-{self.turn}-{uuid.uuid4().hex[:8]}"
        sent = self.messages if with_history else self.messages[-1:]
        body = {
            "model": self.model,
            "stream": True,
            "max_tokens": max_tokens,
            "temperature": 0,
            "messages": [{"role": "system", "content": system}] + sent,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = tool_choice or "auto"
        req = urllib.request.Request(
            f"{self.litellm_url}/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={
                "Authorization": f"Bearer {self.litellm_key}",
                "Content-Type": "application/json",
                "X-OpenWebUI-Chat-Id": self.chat_id,
                "X-OpenWebUI-User-Id": self.user_id,
                "X-OpenWebUI-Message-Id": message_id,
            },
        )
        content, tool_calls = self._read_stream(urllib.request.urlopen(req, timeout=600))
        assistant = {"role": "assistant", "content": content or None}
        if tool_calls:
            assistant["tool_calls"] = tool_calls
        self.messages.append(assistant)
        return message_id, content, tool_calls

    @staticmethod
    def _read_stream(resp):
        content = ""
        calls = {}
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            for choice in chunk.get("choices", []):
                delta = choice.get("delta") or {}
                content += delta.get("content") or ""
                for tc in delta.get("tool_calls") or []:
                    slot = calls.setdefault(tc.get("index", 0), {"id": None, "type": "function", "function": {"name": "", "arguments": ""}})
                    if tc.get("id"):
                        slot["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["function"]["name"] = fn["name"]
                    slot["function"]["arguments"] += fn.get("arguments") or ""
        tool_calls = [calls[i] for i in sorted(calls)]
        for i, tc in enumerate(tool_calls):
            tc["id"] = tc["id"] or f"call_e2e_{i}"
        return content.strip(), tool_calls

    def add_tool_result(self, call_id, text):
        self.messages.append({"role": "tool", "tool_call_id": call_id, "content": text})

    # ---- Context Guard -------------------------------------------------------------------

    def health(self, message_id, wait=12.0):
        url = f"{self.guard_url}/api/v1/conversations/{self.chat_id}/health?message_id={message_id}"
        deadline = time.monotonic() + wait
        last = None
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=3) as r:
                    return json.load(r)
            except urllib.error.HTTPError as e:
                last = e.read().decode("utf-8", "replace")
                time.sleep(0.4)
        raise RuntimeError(f"no health result for {message_id} within {wait}s: {last}")


class Report:
    def __init__(self):
        self.failures = 0

    def check(self, stage, ok, detail):
        mark = "PASS" if ok else "FAIL"
        if not ok:
            self.failures += 1
        print(f"[{mark}] {stage}: {detail}")


def signals_of(result):
    return [r["signal"] for r in result.get("reasons", [])]


def run_degradation(c: Client, rep: Report):
    plain = "You are a terse assistant. Answer in one short sentence."

    def echo(text):
        """An echo turn: minimal history, verbatim scripted reply."""
        system = (
            "You are an echo service used for automated testing. Output exactly the text below, "
            f"verbatim, with nothing before or after it:\n{text}"
        )
        return c.complete(system, "Go. /no_think", with_history=False)

    # 1 baseline
    mid, text, _ = c.complete(
        plain,
        "For this session: llama.cpp is running on port 8080 on host ai-llama-swap, its config is "
        "/etc/llama-swap/config.yaml, and the model is called qwen3-general. Acknowledge briefly. /no_think",
    )
    r = c.health(mid)
    rep.check("1 baseline", r["score"] == 100 and r["status"] == "healthy", r["summary"])

    # 2 known-value drift
    mid, text, _ = echo("Your llama.cpp server on port 8000 looks healthy.")
    r = c.health(mid)
    rep.check("2 known-value drift", "known_value_drift" in signals_of(r) and r["score"] == 85, f"{r['summary']} | reply: {text!r}")

    # 3 suspicious identifier
    mid, text, _ = echo("I switched the model to qwen3-general-v2 as requested.")
    r = c.health(mid)
    rep.check("3 suspicious identifier", "suspicious_identifier" in signals_of(r) and r["score"] == 80, f"{r['summary']} | reply: {text!r}")

    # 4-6 response loop
    for _ in range(3):
        mid, text, _ = echo(LOOP_TEXT)
        r = c.health(mid)
    rep.check("6 response loop", "response_loop" in signals_of(r) and r["score"] == 75, f"{r['summary']} | reply: {text!r}")

    # 7 orphan tool result in history
    c.add_tool_result("call_orphan_0001", "restart complete")
    mid, text, _ = c.complete(plain, "Did the restart finish? Answer in one sentence. /no_think")
    r = c.health(mid)
    rep.check("7 orphan tool result", "tool_result_without_call" in signals_of(r) and r["score"] == 55, f"{r['summary']} | reply: {text!r}")

    # 8-10 repeated tool call
    tools = [
        {
            "type": "function",
            "function": {
                "name": "restart_service",
                "description": "Restart a service by name.",
                "parameters": {"type": "object", "properties": {"service": {"type": "string"}}, "required": ["service"]},
            },
        }
    ]
    tool_system = (
        "You are an operations assistant. When asked to restart, call restart_service with "
        'exactly {"service": "llama-swap"} and no other arguments. Do not write text.'
    )
    real_ids = []
    for i in range(3):
        mid, text, calls = c.complete(tool_system, f"Restart it again (attempt {i + 1}). /no_think", tools=tools, tool_choice="required", max_tokens=80)
        for tc in calls:
            real_ids.append(tc["id"])
            c.add_tool_result(tc["id"], "ok")
        r = c.health(mid)
    args = [tc for m in c.messages if m.get("tool_calls") for tc in m["tool_calls"]]
    rep.check(
        "10 repeated tool call",
        "repeated_tool_call" in signals_of(r) and r["score"] == 50,
        f"{r['summary']} | calls: {[a['function']['arguments'] for a in args]}",
    )

    # 11 unknown tool-call id reference
    # Forge an id with the same shape as the real ones (llama.cpp uses opaque
    # random strings; other backends use a `call_` prefix): keep the first half,
    # replace the rest.
    if real_ids:
        real = real_ids[0]
        tail = ("zZ9x7q4m" * 8)[: len(real) - len(real) // 2]
        fake = real[: len(real) // 2] + tail
    else:
        fake = "call_zz9x7q4m"
    mid, text, _ = echo(f"Tool call {fake} returned an error, so I could not finish.")
    r = c.health(mid)
    rep.check(
        "11 unknown tool-call reference",
        "tool_call_id_reference_unknown" in signals_of(r) and r["score"] == 25 and r["status"] == "reset_recommended",
        f"{r['summary']} | reply: {text!r}",
    )
    return r


def run_context_pressure(c: Client, rep: Report, limit_hint):
    """Pad the prompt to ~77 % of the model's limit and expect the context_70 penalty.
    Two probes calibrate the model's characters-per-token ratio first."""
    sentence = "The quick brown fox jumps over the lazy dog near the river bank at dawn. "
    system = "Answer in one word."

    bare_mid, _, _ = c.complete(system, "Reply with the single word OK. /no_think", max_tokens=10, with_history=False)
    bare = c.health(bare_mid)
    limit = bare["context"].get("limit") or limit_hint
    if not limit:
        rep.check("context pressure", False, "model limit unknown to Context Guard; set CONTEXT_GUARD_MODEL_LIMITS or --context-limit")
        return
    sample = sentence * 60
    sample_mid, _, _ = c.complete(system, sample + "\nReply with the single word OK. /no_think", max_tokens=10, with_history=False)
    sampled = c.health(sample_mid)
    base_tokens = bare["context"]["prompt_tokens"]
    sample_tokens = sampled["context"]["prompt_tokens"] - base_tokens
    chars_per_token = len(sample) / max(1, sample_tokens)

    target_tokens = int(limit * 0.77)
    filler = sentence * int((target_tokens - base_tokens) * chars_per_token / len(sentence))
    mid, _, _ = c.complete(system, filler + "\nReply with the single word OK. /no_think", max_tokens=10, with_history=False)
    r = c.health(mid, wait=60)
    pct = r["context"].get("percent")
    ok = pct is not None and 70 <= pct <= 90 and any(s.startswith("context_") for s in signals_of(r))
    rep.check("context pressure", ok, r["summary"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--litellm-url", default="http://127.0.0.1:4000")
    ap.add_argument("--litellm-key", required=True)
    ap.add_argument("--guard-url", default="http://127.0.0.1:7432")
    ap.add_argument("--model", default="qwen3-30b-a3b")
    ap.add_argument("--context-model", default=None, help="small-context model for the context-pressure stage (e.g. qwen3-vl-4b)")
    ap.add_argument("--context-limit", type=int, default=None, help="limit to assume if Context Guard does not know it")
    args = ap.parse_args()

    rep = Report()
    run_id = uuid.uuid4().hex[:8]
    chat = Client(args.litellm_url, args.litellm_key, args.guard_url, args.model, f"e2e-degrade-{run_id}", "e2e-user")
    print(f"chat {chat.chat_id} on {args.model}")
    final = run_degradation(chat, rep)
    print(f"final: {final['summary']}")

    if args.context_model:
        ctx = Client(args.litellm_url, args.litellm_key, args.guard_url, args.context_model, f"e2e-context-{run_id}", "e2e-user")
        print(f"chat {ctx.chat_id} on {args.context_model}")
        run_context_pressure(ctx, rep, args.context_limit)

    print("ALL STAGES PASSED" if rep.failures == 0 else f"{rep.failures} STAGE(S) FAILED")
    sys.exit(1 if rep.failures else 0)


if __name__ == "__main__":
    main()
