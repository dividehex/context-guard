# Context Guard

A deterministic, out-of-band health monitor for LLM conversations. Context
Guard watches the completions that flow through [LiteLLM](https://github.com/BerriAI/litellm),
keeps a per-chat record of what was said, and after every reply computes a
health score from 0 to 100 with an explicit list of reasons.
[Open WebUI](https://github.com/open-webui/open-webui) shows the score under
each reply as a UI-only status line:

```text
🟢 Context Guard 100 · healthy · 🟢 context 4% (508/12,288)
🟡 Context Guard 74 · watch · 🟡 context 78% (25,624/32,768) · 1 drift · 1 repeated call
🟢 Context Guard 90 · healthy · 🔴 context overflow (16,456/12,288)
```

> Context Guard does not determine the truth of arbitrary natural-language
> statements. It detects deterministic indicators of conversation degradation,
> inconsistency, context pressure, repetition, identifier drift, and tool
> anomalies.

> Context Guard does not add any messages, prompts, canaries, or health
> information to the model's context.

![Three Open WebUI replies with Context Guard status lines: 100 healthy after the user states facts, 85 with one drift after the assistant names the wrong port, 80 with a drift and a suspicious id after it names a near-duplicate model](docs/images/openwebui-status-lines.png)

Three replies from the walkthrough in `docs/ui-demo.md`: the user states the
facts (100, healthy), the assistant contradicts the port (85, one drift), then
names a near-duplicate model (80, drift plus a suspicious id). The second light
tracks context pressure separately, here 0% of the model's 122,880-token limit.

It is never in the inference path. If it crashes, hangs, loses its database, or
is removed, LiteLLM logs one line per flush and inference continues unchanged.

Verified against LiteLLM v1.94.1 and Open WebUI v0.11.3 with llama.cpp
backends. One Rust binary, one container, SQLite, no other services.

## Quick install

You need a compose stack with LiteLLM and Open WebUI on a shared Docker
network, and Open WebUI already sending its user-info headers
(`ENABLE_FORWARD_USER_INFO_HEADERS=true`, which is how it forwards the chat id).

**1. Add the container.** Clone this repository next to your compose file and
merge `docker-compose.example.yml` into it: one `context-guard` service on the
same network as LiteLLM, with a `/data` volume.

```sh
git clone https://github.com/dividehex/context-guard
mkdir -p data/context-guard
docker compose up -d --build context-guard
curl -s http://127.0.0.1:7432/healthz        # {"status":"ok","database":"ok",...}
```

**2. Point LiteLLM at it.** Two environment variables on the `litellm`
service and one block in the LiteLLM config file, then recreate LiteLLM.

```yaml
# compose: litellm service
    environment:
      GENERIC_LOGGER_ENDPOINT: http://context-guard:7432/api/v1/ingest/litellm
      DEFAULT_FLUSH_INTERVAL_SECONDS: "1"
```

```yaml
# litellm config.yaml
litellm_settings:
  callbacks: ["generic_api"]
  extra_spend_tag_headers:
    - x-openwebui-chat-id
    - x-openwebui-user-id
    - x-openwebui-message-id
    - x-openwebui-task
```

```sh
docker compose up -d litellm
```

**3. Two headers on the Open WebUI connection.** Admin Panel → Settings →
Connections → your LiteLLM connection → *Headers*, paste:

```json
{"X-OpenWebUI-Message-Id": "{{MESSAGE_ID}}", "X-OpenWebUI-Task": "{{TASK}}"}
```

Keep the double braces; Open WebUI fills them per request. The first lets the
status line match its own reply exactly; the second marks background calls
(title, tags, follow-ups) so they are not scored as turns.

**4. Install the filter.** Admin Panel → Functions → *+* → paste
`openwebui/context_guard_filter.py` → Save → enable it → toggle **Global**.
(Or `openwebui/install-filter.sh` with an admin API key.)

**5. Send a message.** The status line appears under the reply within a
second or two. `http://127.0.0.1:7432/api/v1/conversations` lists chats and
scores; `docs/ui-demo.md` walks a chat through every signal so you can watch
the score fall.

## What it does, and does not, claim

It **does**:

* measure how full the model's context window is, and flag a request the
  backend rejected for exceeding it
* notice when the assistant repeats the same tool call or the same reply
* notice when the assistant states a different value (port, IP, path, version,
  setting, …) for something the user established unambiguously
* notice when the assistant mentions an identifier that closely resembles one
  the conversation already uses (`qwen3-general` → `qwen3-general-v2`)
* notice tool results without a matching call, and references to tool-call ids
  that were never issued

It does **not**:

* judge whether prose is true
* use another model, embeddings, or any non-deterministic method
* see anything LiteLLM does not log
* touch the inference path in any way

## Architecture

```text
Open WebUI ──► LiteLLM ──► llama.cpp / other backends
   │             │
   │             │  generic_api batch logger: async, after the response, fire-and-forget
   │             │  POST http://context-guard:7432/api/v1/ingest/litellm
   │             ▼
   │      Context Guard (one Rust binary, one container, SQLite in /data)
   │             ▲
   │  Filter outlet: GET /api/v1/conversations/{chat_id}/health?message_id=…
   └─────────────┘  then a `status` event under the reply (never in messages[])
```

### Zero-context-overhead design

The model receives exactly the request it would receive without Context Guard:

* LiteLLM's logger runs after the response has been delivered and only copies
  the request/response to Context Guard.
* Context Guard has no outbound connections and never modifies anything.
* The Open WebUI filter runs in the outlet after the reply is persisted, reads
  from Context Guard, and emits a `status` event. Open WebUI stores that in the
  message's `statusHistory`; when it builds the next request it uses only the
  message id, role, content, info, timestamp and sources, so the score never
  reaches the model.

### Failure isolation

* No inference proxying, ever, and nothing `depends_on` Context Guard.
* LiteLLM's `generic_api` logger makes zero retries, swallows every error and
  clears its queue; an unreachable Context Guard costs one log line per flush.
* Ingest validates JSON, enqueues, and returns; a single worker processes
  batches in `startTime` order. A full queue drops and counts.
* Malformed payloads are rejected with a counter; a batch with one bad item
  still processes the good ones; oversized or deeply nested bodies are refused
  before parsing does any work.
* Database write failures are logged and counted; the event is dropped and the
  worker continues. If the database cannot be opened at startup the process
  exits non-zero so the container restarts.
* Events are deduplicated by LiteLLM's payload id, so redelivery is harmless.
* On SIGTERM the API stops accepting requests and the worker gets up to five
  seconds to finish the batches it already holds, so a turn is not cut off
  between writes.
* The filter only reads, returns the body untouched, and gives up silently
  after one refused connection (about one second).

## LiteLLM integration in detail

LiteLLM's built-in `generic_api` logger is a batch logger: every completion's
`StandardLoggingPayload` is queued after the response is sent and flushed as a
JSON array every `DEFAULT_FLUSH_INTERVAL_SECONDS` (default 5; set to 1 so the
score is ready when Open WebUI asks) or at 512 events.

`extra_spend_tag_headers` turns the listed request headers into
`request_tags` entries such as `"x-openwebui-chat-id: <uuid>"`; that is how
Context Guard learns which chat, user and message a completion belongs to.
Header names are lowercase because LiteLLM stores them that way. (The
payload's `requester_custom_headers` field is always null in v1.94.1, so tags
are the only config-level path.)

Context limits come from `model_info.max_input_tokens` in the LiteLLM model
list, which the payload carries. Set `CONTEXT_GUARD_MODEL_LIMITS` only for
models without it. Note that this is the *input* budget LiteLLM declares, not
the backend's raw window; a request that exceeds the raw window fails and is
scored as an overflow.

Conversation identity, in order of precedence:

1. `x-openwebui-chat-id` request tag (from the header Open WebUI sends).
2. LiteLLM `trace_id`, only with `CONTEXT_GUARD_TRUST_TRACE_ID=true` (a
   client can set it via an `x-*-session-id` header; without one it is a
   per-request uuid, which is why it is off by default).
3. Fallback: `fallback:` + a hash of user id, model and the first user
   message. Fragile by design; a warning is logged.

Open WebUI's context compaction rewrites `messages[]` mid-chat, so Context
Guard never derives turn numbers or facts from the current message list: turns
are counted from events and known values persist per conversation.

## Open WebUI integration in detail

Open WebUI sends `X-OpenWebUI-Chat-Id` and `X-OpenWebUI-User-Id` to LiteLLM on
its own when `ENABLE_FORWARD_USER_INFO_HEADERS=true`. The two headers from the
quick install are connection-level custom headers with Open WebUI's template
variables (`{{MESSAGE_ID}}`, `{{TASK}}`). Without the task header, Context
Guard falls back to a heuristic: non-streaming single-message requests are
treated as background tasks.

The filter (`openwebui/context_guard_filter.py`) has only an `outlet`. Its
valves:

| Valve | Default | Meaning |
|-------|---------|---------|
| `context_guard_url` | `http://context-guard:7432` | reachable from the Open WebUI container |
| `wait_seconds` | 6 | how long to wait for the score |
| `poll_interval` | 0.5 | seconds between polls |
| `connect_timeout` | 1 | per-request timeout |
| `settle_seconds` | 1.5 | re-check once after a result so a reply with tool or code-interpreter iterations shows its last iteration |
| `show_minimum` | always | show for every reply, or only from `good`/`watch`/`degraded`/`reset_recommended` down |
| `notify_below` | 40 | toast when health falls below this; 0 disables |

Open WebUI shows the status on a single line with an ellipsis, so anomalies
use short labels (`drift`, `suspicious id`, `loop`, `repeated call`, `orphan
result`, `unknown call id`) and fold into `+N more` past about 96 characters.
The full reasons are always in the API response.

## How scoring works

Each chat completion is one **turn**. For every turn:

1. **Context utilization** = `prompt_tokens / context_limit`, using the token
   count the backend reported (exact) against LiteLLM's declared input limit.
   `< 70 %` nothing · `70–80 %` −5 · `80–90 %` −10 · `> 90 %` −20.
   Unknown limit ⇒ reported as unknown, no penalty. A request the backend
   rejected for exceeding its window is scored as an overflow: red light, the
   reported request size, −20.
2. **Signals** run over the new messages and the response and record
   **anomalies**, each deduplicated per conversation:

   | Signal | Penalty | Fires when |
   |--------|---------|------------|
   | `repeated_tool_call` | 5 | the same tool with the same canonicalized arguments (key order and whitespace ignored) appears 3 times in the last 5 calls |
   | `response_loop` | 5 | the reply (≥ 20 words) has Jaccard similarity ≥ 0.90 of word 3-shingles with at least 2 of the previous 3 replies |
   | `known_value_drift` | 15 | see below |
   | `tool_result_without_call` | 20 | a `tool` message's `tool_call_id` was not issued by an earlier assistant message in the same request (counted once per id) |
   | `tool_call_id_reference_unknown` | 25 | the reply cites something shaped like the conversation's real tool-call ids (same `call_` prefix, or for opaque ids such as llama.cpp's, the same length with letters and digits) that was never issued |
   | `suspicious_identifier` | 5 | the reply introduces a name that is not known but is within 20 % edit distance of, or extends by prefix (≥ 6 shared chars), a known model, container, host, tool or name; paths and env vars use edit distance only |

3. **Risk** = context penalty + the penalties of every anomaly recorded in the
   last `window_turns` (default 10) turns. **Health** = `100 − risk`, clamped
   to 0..100. The window lets a conversation recover after the behaviour stops.
4. **Status**: `≥ 90` healthy · `≥ 75` good · `≥ 60` watch · `≥ 40` degraded ·
   below that reset recommended. The lights follow the same scale, for health
   and for context pressure.

Every result stores the reasons, so `risk == min(Σ penalties, 100)` always
holds and nothing is a magic number. Weights, thresholds and the window live in
configuration (`config/context-guard.example.toml`), not in code.

### Known-value drift, precisely

User and tool messages are **sources of truth**; assistant messages are
**claims**; system prompts are ignored. From user and tool text, Context Guard
extracts typed values: IPv4/IPv6 addresses, ports (only with an explicit
marker such as `port 8080`, `host:8080`, `--port 8080`), URLs, absolute
paths, `ENV_VAR=value`, versions (`v1.2.3`, `version 1.2`, `litellm 1.94.1`),
hostnames with a real TLD, container names (configurable prefix, default
`ai-`), and `snake_case_key: 123` settings. Each value gets an **anchor**: the
nearest identifier-like token before it (`llama.cpp` in "llama.cpp is running
on port 8080"), or none.

A claim is drift only when **all** of these hold:

1. the registry has **exactly one** value for the same kind and anchor,
2. the claimed value differs from it,
3. the claimed value never appeared in any user or tool message of the chat,
4. the value was recognized with its kind marker (no bare numbers),
5. the kind describes an attribute of something (IP, port, env var, version,
   setting). A path, URL, hostname or container name *is* the thing, so a
   different one is a different thing, not a contradiction; near-duplicates of
   those are the suspicious-identifier signal's job.

So "the API is on port 4000" followed by an assistant "the API on port 4100"
is drift; "open port 3000 for the UI" is not, because no anchored value
conflicts; and if the user themself mentioned 4100 earlier, nothing fires.
`ENV_VAR=value` pairs whose name looks like a key, token or password are never
learned, so an assistant showing `OPENAI_API_KEY=your-key-here` is not drift.
False positives are treated as worse than misses.

## REST API

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/api/v1/ingest/litellm` | LiteLLM telemetry (JSON array, object, or NDJSON). Always answers `202` with `{accepted, dropped}` once parsed; `400` for unparseable bodies, `413` above `CONTEXT_GUARD_MAX_BODY_BYTES`. Never waits for the database. |
| `GET` | `/healthz` | `{status, database, queue_depth, uptime_s, version}`; `200` even when the database is unavailable. |
| `GET` | `/api/v1/conversations?limit=50&status=watch` | Recent conversations with their latest score. |
| `GET` | `/api/v1/conversations/{id}/health` | Latest result. `?message_id=X` returns the result for that Open WebUI message (`404 not_scored_yet` until it exists); `?after=<unix seconds>` the latest result at or after that time. |
| `GET` | `/api/v1/conversations/{id}/history?limit=200` | All health results in turn order plus every anomaly. |
| `GET` | `/metrics` | Prometheus text format. |

Health response:

```json
{
  "conversation_id": "…", "model": "qwen3-30b-a3b", "user_id": "…",
  "score": 74, "risk": 26, "status": "watch", "turns": 39, "turn": 39,
  "message_id": "…", "ts": "2026-09-12T12:52:54.631Z",
  "context": { "prompt_tokens": 25624, "limit": 32768, "percent": 78.2 },
  "signals": { "known_value_drift": 1, "tool_anomalies": 0, "loop_events": 1, "suspicious_identifiers": 0 },
  "reasons": [
    { "signal": "context_70", "penalty": 5, "detail": "context utilization 78.2% of 32768 tokens" },
    { "signal": "known_value_drift", "penalty": 15, "detail": "assistant said port of llama.cpp 8000 but the conversation established 8080" },
    { "signal": "repeated_tool_call", "penalty": 5, "detail": "restart called with identical arguments 3 times" }
  ],
  "summary": "🟡 Context Guard 74 · watch · 🟡 context 78% (25,624/32,768) · 1 drift · 1 repeated call"
}
```

Errors are `{"error": {"code": "…", "message": "…"}}`. There is no
authentication in V1; bind the port to localhost or an internal network. The
router is structured so a bearer-token layer can wrap `/api/v1/*` later.

## Prometheus metrics (optional)

Nothing requires Prometheus. `/metrics` exposes aggregates only, never
conversation ids or text:

```text
context_guard_events_received_total{kind}          chat | task | failure
context_guard_events_dropped_total{reason}         malformed | unsupported | queue_full
context_guard_processing_errors_total{stage}
context_guard_health_score{model}                  latest scored turn per model
context_guard_risk_score{model}
context_guard_context_utilization_ratio{model}
context_guard_conversations_by_status{status}      active in the last 24 h
context_guard_known_value_drift_total{model}
context_guard_tool_anomalies_total{model,signal}
context_guard_loop_events_total{model,signal}
context_guard_suspicious_identifiers_total{model}
context_guard_ingest_batch_size                    histogram
context_guard_queue_depth
```

For a Prometheus on another host, either publish the port on all interfaces
(that also exposes the unauthenticated API) or set
`CONTEXT_GUARD_METRICS_LISTEN=0.0.0.0:7433` and publish only 7433: that
listener serves `/metrics` and `/healthz` and nothing else.

## Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `CONTEXT_GUARD_LISTEN` | `0.0.0.0:7432` | API listener |
| `CONTEXT_GUARD_METRICS_LISTEN` | unset | optional `/metrics`-only listener |
| `CONTEXT_GUARD_DATABASE` | `/data/context-guard.db` | SQLite file (WAL mode) |
| `CONTEXT_GUARD_RETENTION_DAYS` | 30 | hourly purge of conversations not seen for this long (at least 1) |
| `CONTEXT_GUARD_CONFIG` | unset | optional TOML with `[penalties]`, `[thresholds]`, `[scoring]`, `[model_limits]` |
| `CONTEXT_GUARD_MODEL_LIMITS` | unset | `model=tokens,model=tokens`; overrides the payload's limit |
| `CONTEXT_GUARD_QUEUE_SIZE` | 1024 | bounded ingest queue (batches); full ⇒ dropped and counted |
| `CONTEXT_GUARD_MAX_BODY_BYTES` | 33554432 | ingest body limit |
| `CONTEXT_GUARD_STORE_MESSAGES` | false | also keep the full `messages[]` of every event (disk grows quadratically with chat length) |
| `CONTEXT_GUARD_LOG_PAYLOADS` | false | log normalized events at info (opt-in; contains conversation text) |
| `CONTEXT_GUARD_CAPTURE_DIR` | unset | write every raw ingest body to this directory (fixture building; contains conversation text) |
| `CONTEXT_GUARD_LOG_JSON` | false | JSON log lines |
| `CONTEXT_GUARD_TRUST_TRACE_ID` | false | use LiteLLM `trace_id` as the conversation id when no chat tag exists |
| `CONTEXT_GUARD_CONTAINER_PREFIXES` | `ai-` | tokens with these prefixes are container names |
| `RUST_LOG` | `info` | log filter |

Precedence: environment > TOML file > compiled defaults.

## Security and data

**The SQLite database contains copies of conversation text** (user, tool and
assistant messages, one turn's delta per event). Treat `/data` like the Open
WebUI database: keep it on a protected disk, do not share it, and keep the API
port bound to localhost or the internal network. Prometheus output never
includes conversation ids or text. Default logging never includes message or
response text: ids, counts, hashes and anomaly details only, with values that
look like keys, tokens or passwords redacted. Anomaly details do quote the
specific conflicting values (e.g. `8080` vs `8000`), and they are stored,
served and logged only after the same redaction (`NAME=value` with a
secret-looking name, and common API-key, GitHub, Slack, AWS, Google, JWT and
bearer-token shapes become `[redacted]`).

## Build and test

```sh
cargo build --release
cargo test                                   # 71 unit + integration tests, temp SQLite; also spawns the real binary
cargo clippy --all-targets -- -D warnings
cargo audit                                  # RustSec advisories; `cargo install cargo-audit`
docker build -t context-guard .              # multi-stage; runtime is debian-slim, uid 10001
```

Filter tests need Python with `aiohttp`, `pydantic` and `pytest`:

```sh
python -m pytest openwebui/
```

End-to-end against a live stack (`scripts/e2e_degradation.py`): drives one
scripted chat through LiteLLM and a real model and asserts every signal and
score from 100 down to "reset recommended", plus an optional context-pressure
stage on a small-context model.

```sh
scripts/e2e_degradation.py --litellm-key "$LITELLM_MASTER_KEY" --model <model> [--context-model <small-model>]
```

`docs/ui-demo.md` is the interactive version: messages to type into an Open
WebUI chat, each with the status line it produces.

CI (`.github/workflows/ci.yml`) runs rustfmt, clippy, the Rust tests, `cargo
audit`, the filter tests, and a build-and-smoke-test of the Docker image on
every push.

## Current limitations

* Only what LiteLLM logs is visible. Tools executed by Open WebUI appear as
  `tool_calls` in the reply and `tool` messages in the next request; tools that
  never pass through the model are invisible. Tool signals need the model's
  Function Calling set to Native in Open WebUI.
* Known-value drift is deliberately narrow: typed values with markers and an
  unambiguous anchor. Prose contradictions are out of scope.
* Response looping uses exact-ish text similarity; paraphrased loops are missed.
* The health of a chat that started before Context Guard was deployed only
  reflects turns seen since then.
* No authentication; rely on network placement.
* One process, one SQLite file; sized for a personal or small-team stack.

## Repository layout

```text
src/telemetry   LiteLLM payload → ConversationEvent, identity resolution
src/monitor     the signals (pure functions) and the Monitor that runs them
src/database    SQLite connection, migrations, typed queries, retention
src/api         axum handlers
src/metrics.rs  Prometheus registry
src/worker.rs   queue consumer
openwebui/      the Open WebUI filter, its tests, install script
scripts/        end-to-end degradation test
docs/           UI walkthrough
tests/          integration and fault-tolerance tests; real captured fixtures
```

## License

MIT, see `LICENSE`.
