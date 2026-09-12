# Context Guard — implementation plan

Status: plan only, no code yet. Written 2026-09-12 after inspecting the running
`ai` stack (`~/ai/compose.yaml`), LiteLLM v1.94.1 inside `ai-litellm`, and
Open WebUI v0.11.3 inside `ai-openwebui`. Revised the same day: Open WebUI is
the primary consumer (a Filter shows the score under every reply); Prometheus
is optional.

Context Guard is a deterministic, out-of-band health monitor for LLM
conversations. It observes, records, analyzes, scores and reports. It never
sits in the inference path and never adds a single token to any prompt.

---

## 1. What the environment inspection found

### 1.1 Stack facts that shape the design

| Fact | Consequence |
|------|-------------|
| LiteLLM `ghcr.io/berriai/litellm:v1.94.1`, config at `~/ai/config/litellm/config.yaml`, on network `ai-backend` | Context Guard joins `ai-backend`; LiteLLM and Open WebUI reach it as `http://context-guard:7432`. |
| Every model entry already declares `model_info.max_input_tokens` (12288 / 28672 / 122880) | LiteLLM's logging payload carries the context limit itself. Configured overrides stay possible but are not required. |
| Open WebUI has `ENABLE_FORWARD_USER_INFO_HEADERS=true` | Every chat completion to LiteLLM already carries `X-OpenWebUI-Chat-Id` and `X-OpenWebUI-User-Id` (plus name/email/role) headers. |
| Open WebUI `ENABLE_CONTEXT_COMPACTION=true` (threshold 22000 tokens) | `messages[]` can shrink mid-conversation. Known values must be persisted per conversation, never recomputed from the current `messages[]`. |
| Open WebUI uses native function calling (`CHAT_RESPONSE_MAX_TOOL_CALL_ITERATIONS=8`) | Tool calls appear as `tool_calls` in the assistant response; tool results appear as `role: tool` messages in the next request. Both are visible to LiteLLM logging. |
| Open WebUI task calls (title, tags, follow-ups, query generation) also carry the chat id header | Must be excluded from turn counting and scoring or they poison the health of every chat. See 1.4. |
| Open WebUI 0.11.3 runs **outlet filters inline** (`utils/middleware.py: outlet_filter_handler`) after the reply is persisted and the `chat:completion done` event is sent, and before the background title/tag tasks | A filter outlet can wait a few seconds for the score without delaying the visible reply. It receives `__event_emitter__` and `__metadata__` (chat_id, message_id). See 1.5. |
| A `status` event is stored in the message's `statusHistory` (`socket/main.py`), and the frontend builds the next request from `id, role, content, info, timestamp, sources` only | A status line under the reply is UI-only and never enters `messages[]`. This is the display mechanism. |
| Prometheus runs on another host on the LAN; it is not part of this compose project | `/metrics` is provided but optional. Nothing depends on Prometheus being present. |
| No Rust toolchain on the host (`cargo` not found), Docker 29.8 present | Build and tests run in Docker by default. Installing rustup (user-level, no sudo) is recommended for a faster dev loop. |

### 1.2 The LiteLLM telemetry mechanism: built-in `generic_api` callback

LiteLLM v1.94.1 ships `litellm/integrations/generic_api/generic_api_callback.py`
(`GenericAPILogger`). Verified properties from the installed source:

* Enabled purely by config: `litellm_settings.callbacks: ["generic_api"]` and
  env `GENERIC_LOGGER_ENDPOINT`. No Python file, no image change.
* It is a `CustomBatchLogger`: events are appended to an in-memory queue in
  the async success/failure hooks (which run **after** the response has been
  delivered) and flushed every `DEFAULT_FLUSH_INTERVAL_SECONDS` (default 5,
  env-overridable; we set it to 1 so the score is ready when the Open WebUI
  outlet asks for it) or at 512 events (`DEFAULT_BATCH_SIZE`).
* Flush does a single `httpx` POST of a **JSON array** of
  `StandardLoggingPayload` objects. `max_retries` defaults to 0, every
  exception is swallowed with `verbose_logger.exception`, and the queue is
  cleared in `finally`. A dead Context Guard therefore costs LiteLLM nothing
  but a log line per flush.
* Because it is instantiated as `GenericAPILogger()` with no arguments,
  `log_format`, timeout and headers can only come from env
  (`GENERIC_LOGGER_ENDPOINT`, `GENERIC_LOGGER_HEADERS`). The ingest endpoint
  will accept a JSON array (what we get), a single object, and NDJSON, so a
  future LiteLLM upgrade cannot break ingestion.

`StandardLoggingPayload` fields Context Guard will consume:

```text
id, trace_id, litellm_call_id, call_type, stream, status, error_str
model, model_group, api_base
prompt_tokens, completion_tokens, total_tokens
startTime, endTime
messages            full request messages (base64 images truncated by LiteLLM)
response            full ModelResponse dict incl. choices[].message.tool_calls
model_map_information.model_map_value.max_input_tokens   (context limit)
request_tags        list of strings, see 1.3
metadata.user_api_key_end_user_id, end_user
```

Nothing else is invented. Fields absent from the payload are `None` in the
normalized event and reported as "unknown".

### 1.3 Getting the chat id and message id into the payload

`StandardLoggingMetadata.requester_custom_headers` exists in the type but is
hard-coded to `None` in this version (`litellm_logging.py:4630,5469`), so raw
headers do **not** reach the payload. The config-only path that works:

**`extra_spend_tag_headers`.**
`litellm_settings.extra_spend_tag_headers: [x-openwebui-chat-id, x-openwebui-user-id, x-openwebui-message-id, x-openwebui-task]`
makes `_get_extra_header_tags` append `"x-openwebui-chat-id: <uuid>"` etc. to
`request_tags` for every request. Header names are lowercase because Starlette
lowercases them before LiteLLM stores them.

`X-OpenWebUI-Chat-Id` and `X-OpenWebUI-User-Id` are sent automatically today.
`X-OpenWebUI-Message-Id` and `X-OpenWebUI-Task` are **not** sent by 0.11.3's
OpenAI router, but the connection's custom headers support templates
(`utils/headers.py: parse_custom_headers`): `{{CHAT_ID}}`, `{{MESSAGE_ID}}`,
`{{TASK}}`, `{{USER_ID}}`. Two headers are added on the LiteLLM connection in
Admin → Settings → Connections:

```text
X-OpenWebUI-Message-Id   {{MESSAGE_ID}}     lets the filter fetch exactly its own turn
X-OpenWebUI-Task         {{TASK}}           lets Context Guard skip task calls
```

An alternative that needs no LiteLLM tag config: any `x-*-session-id`
header becomes LiteLLM's `trace_id` (`X-OpenWebUI-Session-Id: {{CHAT_ID}}`).
Documented as an alternative only.

Identifier precedence in Context Guard (documented in the README):

```text
1. request_tags "x-openwebui-chat-id: …"      stable per Open WebUI chat
2. trace_id, only with CONTEXT_GUARD_TRUST_TRACE_ID=true (LiteLLM-generated
   per-request uuids are indistinguishable from session-header values)
3. fallback: sha256(user_id | model | first user message) truncated,
   prefixed "fallback:"  — fragile by design, logged at warn on first use
```

User id comes from the `x-openwebui-user-id` tag, else `end_user`.
Message id comes from the `x-openwebui-message-id` tag, else `None`.

### 1.4 Excluding Open WebUI task calls

`routers/tasks.py` sends title/tag/follow-up/query generation through the
same connection with `metadata.task` set and `chat_id` set, so they carry the
chat header too. LiteLLM cannot see `metadata.task` (Open WebUI pops
`metadata` from the outbound body). Two layers:

* **Deterministic:** the `X-OpenWebUI-Task: {{TASK}}` header from 1.3. A
  non-empty task tag ⇒ event is recorded as `kind = task` and skipped for scoring.
* **Fallback heuristic when the tag is absent:** `stream == false` **and**
  `messages` contains exactly one `user` message and no `assistant` message.
  Real Open WebUI chats stream. Labelled as a heuristic in the README.

### 1.5 The Open WebUI display path: a Filter function

Verified in `utils/middleware.py`:

```text
stream finishes
→ message persisted, chat:completion {done: true} emitted   (user sees the finished reply)
→ outlet_filter_handler(ctx)                                 ← our filter runs here
     extra_params: __event_emitter__, __event_call__, __user__, __metadata__, __request__, __model__
     __metadata__ has chat_id, message_id, session_id
→ background_tasks_handler(ctx)                              (title, tags, follow-ups)
```

So an outlet that waits up to a few seconds delays only the title/tag
generation, never the reply. `__event_emitter__({"type": "status", ...})` is
persisted to `statusHistory` on that message and rendered as the small status
line under the reply, exactly like the "Searching the web…" line. It is never
part of `content`, never sent to the model, and survives page reload.

The filter must be unable to break a chat: every network call wrapped in
try/except with short timeouts, and a total wait budget. If Context Guard is
down, the outlet returns the body unchanged after at most one failed
connection attempt (~1 s).

---

## 2. Architecture

```text
Open WebUI ──► LiteLLM ──► llama-swap / fury llama-swap
   │             │
   │             │  generic_api batch logger (async, after response, fire-and-forget)
   │             │  POST http://context-guard:7432/api/v1/ingest/litellm   [JSON array]
   │             ▼
   │      Context Guard (Rust, one container, ai-backend network)
   │      ┌──────────────────────────────────────────────────────┐
   │      │ axum HTTP ──► bounded mpsc (drop on full) ──► worker  │
   │      │                                              │        │
   │      │            SQLite (/data/context-guard.db) ◄─┘        │
   │      │            monitor: context, repetition, known values,│
   │      │                     identifiers, tools → scoring      │
   │      │ REST API  ·  /healthz  ·  /metrics (optional use)     │
   │      └──────────────────────────────────────────────────────┘
   │             ▲
   │  outlet Filter: GET /api/v1/conversations/{chat_id}/health?message_id=…
   └─────────────┘  then emits a UI-only `status` event under the reply
```

Invariants enforced by construction:

* Context Guard has no route to LiteLLM, llama-swap or Open WebUI. It never
  opens an outbound connection in V1 (`reqwest` is not a dependency).
* The ingest handler validates JSON, pushes to a bounded channel and returns
  `202` immediately. It never waits for the database.
* A single worker task processes events sequentially, ordered by
  `startTime` within a batch, so scoring is reproducible for a given
  event sequence.
* Any error in the worker (DB, parse, scoring) increments a counter and is
  logged; the worker never exits and the process never panics on payload
  content (`serde_json::Value` + explicit `Option`s everywhere, no `unwrap`
  on input).
* The filter reads; it never writes to Context Guard, never modifies
  `body["messages"]`, and never raises.

---

## 3. Project layout

```text
~/dev/context-guard/
├── Cargo.toml
├── Cargo.lock
├── Dockerfile                      multi-stage, non-root runtime
├── docker-compose.example.yml      snippet to paste into ~/ai/compose.yaml
├── README.md
├── PLAN.md                         this file
├── migrations/
│   └── 0001_initial.sql
├── config/
│   └── context-guard.example.toml  weights + model limits, all optional
├── openwebui/
│   ├── context_guard_filter.py     the Open WebUI Filter function (single file)
│   └── install-filter.sh           optional: create/update it through the Open WebUI API
├── src/
│   ├── main.rs          wiring: config → db → worker → axum
│   ├── config.rs        env + optional TOML; weights, thresholds, model limits
│   ├── api/
│   │   ├── mod.rs       router, error type, JSON responses
│   │   ├── ingest.rs    POST /api/v1/ingest/litellm
│   │   ├── conversations.rs  GET list / health / history
│   │   └── system.rs    /healthz, /metrics
│   ├── telemetry/
│   │   ├── litellm.rs   StandardLoggingPayload → ConversationEvent
│   │   ├── event.rs     ConversationEvent, Message, ToolCall, ToolResult
│   │   └── identity.rs  conversation id / user id / message id / task detection
│   ├── monitor/
│   │   ├── mod.rs       Monitor: runs signals, persists, scores one event
│   │   ├── context.rs   utilization + penalty
│   │   ├── repetition.rs shingles/Jaccard, tool-call repetition
│   │   ├── known_values.rs  extraction regexes, registry, drift rule
│   │   ├── identifiers.rs   identifier registry, near-duplicate rule
│   │   ├── tools.rs     ledger, call/result consistency
│   │   ├── scoring.rs   penalties → risk → health/status, reasons
│   │   └── text.rs      normalization helpers shared by the above
│   ├── database/
│   │   ├── mod.rs       pool, migrations, WAL, retention job
│   │   └── repo.rs      typed queries (conversations, events, values, tools, anomalies, health)
│   ├── metrics.rs       prometheus registry and counters/gauges
│   └── worker.rs        bounded channel consumer
└── tests/
    ├── fixtures/        real captured LiteLLM payloads (redacted)
    ├── ingest.rs        end-to-end: POST fixtures → query API
    └── fault_tolerance.rs   malformed/oversized/garbage payloads
```

Crates (all stable, widely used): `axum`, `tokio`, `serde`, `serde_json`,
`sqlx` (sqlite, runtime-tokio, migrate), `tracing`, `tracing-subscriber`
(json + env-filter), `prometheus`, `uuid`, `sha2`, `regex`, `chrono`,
`toml`, `thiserror`, `anyhow` (binary only). No `reqwest`.

Modules depend inward: `api` → `worker`/`database`/`metrics`;
`monitor` depends only on `telemetry::event`, `config` and a small repository
trait so the signal code is unit-testable without SQLite.

---

## 4. Event model

```rust
pub struct ConversationEvent {
    pub event_id: String,             // LiteLLM payload id (dedupe key)
    pub conversation_id: String,
    pub conversation_id_source: IdSource, // ChatTag | TraceId | Fallback
    pub user_id: Option<String>,
    pub message_id: Option<String>,   // Open WebUI assistant message id, from tag
    pub model: String,                // model_group if set, else model
    pub request_id: Option<String>,   // litellm_call_id
    pub timestamp: DateTime<Utc>,     // endTime
    pub kind: EventKind,              // Chat | Task | Failure
    pub stream: Option<bool>,
    pub prompt_tokens: Option<u64>,
    pub completion_tokens: Option<u64>,
    pub context_limit: Option<u64>,   // config override > model_map max_input_tokens > None
    pub messages: Vec<Message>,       // role, text content, tool_call_id, tool_calls
    pub response_text: Option<String>,
    pub tool_calls: Vec<ToolCall>,    // from response.choices[0].message.tool_calls
    pub tool_results: Vec<ToolResult>,// role=tool messages in this request
    pub error: Option<String>,
}
```

`Message.content` is flattened to text: string content as-is; array content
keeps `text` parts only. Image/base64 parts are dropped (LiteLLM already
truncates them). Reasoning fields are ignored.

Turn number = number of `Chat` events previously recorded for the
conversation + 1. It is assigned by the worker, not derived from
`messages.len()`, because compaction changes the latter.

Tool-calling iterations: Open WebUI makes one LiteLLM call per iteration
(assistant asks for a tool, results are appended, the model is called again).
Each call is one event and one turn; they share the same `message_id`. The
filter asks for the **latest** result for its message id, so the score
shown reflects the final iteration of that reply.

---

## 5. Signals (deterministic, V1 scope)

All signals return `Vec<Anomaly { signal, penalty_key, severity, detail }>`
plus signal-specific facts. Penalty values are looked up in config by
`penalty_key`; signal code never contains a number.

### 5.1 Context utilization (`context.rs`)

```text
percent = prompt_tokens / context_limit * 100
< 70 → none   70–80 → context_70   80–90 → context_80   > 90 → context_90
```

Unknown limit ⇒ `context.percent = null`, reason "context limit unknown for
model X" with zero penalty. Limit resolution: `[model_limits]` in config /
`CONTEXT_GUARD_MODEL_LIMITS="qwen3-30b-a3b=122880,fury/qwen3-8b=12288"` >
payload `max_input_tokens` > unknown.

### 5.2 Repetition (`repetition.rs`)

* **Tool-call repetition:** key = sha256(tool name + canonical JSON of
  arguments: keys sorted, whitespace stripped, strings trimmed). Same key ≥ 3
  times within the last 5 tool calls of the conversation ⇒ `repeated_tool_call`
  (once per key per window, so it does not stack every turn).
* **Response looping:** normalize (lowercase, collapse whitespace, strip
  punctuation), word 3-shingles, Jaccard similarity vs each of the previous 3
  assistant responses. Jaccard ≥ 0.90 against ≥ 2 of them ⇒ `response_loop`.
  Responses shorter than 20 words are skipped (short acknowledgements repeat
  legitimately).
* **Repeated failed operation:** identical tool-call key whose previous result
  text started with a failure marker (`error`, `Error:`, `failed`, `Traceback`,
  HTTP 4xx/5xx pattern) and is called again ⇒ counts toward the same
  `repeated_tool_call` key; no separate weight in V1.

### 5.3 Known-value drift (`known_values.rs`)

Sources of truth: `user` messages and `tool` results. Assistant messages
produce **claims**, never facts.

Extractors (each yields `kind`, `value`, and an optional `anchor`):

```text
ipv4        \b(?:\d{1,3}\.){3}\d{1,3}\b   (octets ≤ 255; skipped when preceded by "v")
ipv6        bracketed or ::-containing hex groups; validated with std::net::Ipv6Addr
port        (?i)\bport\s*[:=#]?\s*(\d{2,5})\b   or   host:PORT after an ipv4/hostname   or --port=N
url         https?://[^\s<>"')]+           (host and port also registered as their own kinds)
path        (?:^|[\s"'`(=:])(/[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)+)   (≥ 2 segments)
env_var     \b[A-Z][A-Z0-9_]{2,}\b followed by = or preceded by $ / ${
version     \bv?\d+\.\d+(?:\.\d+)?(?:[-+][A-Za-z0-9.]+)?\b  preceded by "version", "v", or a known identifier
hostname    FQDN with ≥ 1 dot and a letter-only TLD, or the host part of a URL; bare words are NOT hostnames
container   words matching a configurable prefix list (default "ai-") or the object of "container"
model       values of the LiteLLM model list observed so far + "model" anchor
numeric_cfg KEY = NUMBER / KEY: NUMBER with snake_case or kebab-case KEY
```

`anchor` = nearest preceding identifier-like token within 6 tokens
(e.g. `llama.cpp` for "llama.cpp is running on port 8080"). Registry key =
`(kind, anchor)`; when no anchor is found, key = `(kind, "")`.

Drift rule (conservative on purpose):

```text
fire known_value_drift when ALL hold:
  1. assistant claim (kind, anchor, value_a)
  2. registry has exactly one value_k for (kind, anchor)      — unambiguous
  3. value_a != value_k
  4. value_a has never appeared in any user/tool message of the conversation
  5. for kind in {port, numeric_cfg, version}: the claim carried an explicit
     kind marker ("port", key name, "version"/"v")
  6. (kind, anchor, value_a) has not already been flagged in this conversation
```

The user restating a different value updates the registry (latest user
value wins, history kept). Values inside fenced code blocks are extracted
like any other text. No prose semantics.

### 5.4 Identifier drift (`identifiers.rs`)

Registry of identifiers seen in user/tool messages and in tool-call names:
model names, container names, hostnames, paths, env vars, tool names. When an
assistant message introduces an identifier of the same kind that is not in
the registry and is *close* to one entry — normalized Levenshtein distance
≤ 0.2 of the longer length, or one is a strict prefix of the other with ≥ 6
shared characters — record `suspicious_identifier` (low severity), once per
new identifier. Example: known `qwen3-general`, assistant says
`qwen3-general-v2`.

### 5.5 Tool anomalies (`tools.rs`)

Ledger row per call: `call_id, turn, tool name, args hash, result hash,
result status (ok / failed / missing), timestamps`. Calls come from the
response; results are matched by `tool_call_id` from `role: tool` messages
in later requests.

Checks that need only what the payload contains:

* `tool_result_without_call`: a `role: tool` message whose `tool_call_id`
  is not preceded, **within the same `messages[]`**, by an assistant message
  containing that id. Self-contained per request, so it is correct even if
  monitoring started mid-conversation.
* `tool_call_id_reference_unknown`: the assistant's text mentions a string
  matching the observed tool-call id format (learned from real ids in the
  ledger, e.g. `call_[A-Za-z0-9]{8,}`) that is not in the ledger.
* `repeated_tool_call` is produced by 5.2, not here.

Nothing is inferred about tools that were executed but not logged.

---

## 6. Scoring (`scoring.rs`)

```text
risk(turn) = context_penalty(this turn)
           + Σ penalty(anomaly) for anomalies recorded in the last W chat turns
             of this conversation (W = window_turns, default 10)
health     = clamp(100 - risk, 0, 100)
status     = 90..=100 healthy | 75..=89 good | 60..=74 watch | 40..=59 degraded | 0..=39 reset_recommended
```

The sliding window lets a conversation recover once the behaviour stops,
without an opaque decay function. Each `HealthResult` stores
`health_score`, `risk_score`, `status`, `context_percent`, and
`reasons: [{signal, penalty, detail}]`, so `Σ penalty == risk` is a testable
property.

Default weights (all overridable, see 8):

```toml
[penalties]
context_70 = 5
context_80 = 10
context_90 = 20
repeated_tool_call = 5
response_loop = 5
known_value_drift = 15
tool_result_without_call = 20
tool_call_id_reference_unknown = 25
suspicious_identifier = 5

[scoring]
window_turns = 10

[thresholds]
good = 75
watch = 60
degraded = 40
# healthy is ≥ 90 by definition; anything below `degraded` is reset_recommended
```

---

## 7. Persistence (SQLite, `sqlx` migrations)

`PRAGMA journal_mode=WAL`, `synchronous=NORMAL`, `busy_timeout=5000`,
`foreign_keys=ON`. Single writer (the worker); the API uses a read pool.

```sql
conversations      (id PK, id_source, user_id, model, first_seen, last_seen,
                    turns, last_health, last_risk, last_status, last_context_percent)
conversation_events(id PK /* LiteLLM id */, conversation_id FK, turn, kind, ts, model,
                    message_id, prompt_tokens, completion_tokens, context_limit,
                    response_text, new_messages_json /* messages not seen in the previous turn */,
                    full_messages_json NULL /* only when CONTEXT_GUARD_STORE_MESSAGES=true */)
health_results     (id PK, conversation_id FK, turn, ts, message_id, health, risk, status,
                    context_percent, reasons_json)
known_values       (id PK, conversation_id FK, kind, anchor, value, source_role,
                    first_turn, last_turn, occurrences, UNIQUE(conversation_id, kind, anchor, value))
identifiers        (id PK, conversation_id FK, kind, value, source_role, first_turn,
                    UNIQUE(conversation_id, kind, value))
tool_calls         (call_id PK, conversation_id FK, turn, name, args_hash, args_json,
                    result_turn, result_hash, result_status, ts)
anomalies          (id PK, conversation_id FK, turn, ts, signal, penalty, severity,
                    detail_json, dedupe_key, UNIQUE(conversation_id, dedupe_key))
```

* `conversation_events.id` unique ⇒ redelivered batches are idempotent.
* Index on `health_results(conversation_id, message_id)` for the filter's lookup.
* Storing only the message delta keeps the DB size roughly linear in
  conversation length instead of quadratic. Full messages are opt-in.
* Retention job: every hour delete conversations (and cascaded rows) with
  `last_seen < now - CONTEXT_GUARD_RETENTION_DAYS`, then `PRAGMA wal_checkpoint(TRUNCATE)`
  when the WAL exceeds a size. Failures are logged, never fatal.
* Write failures: logged, `context_guard_processing_errors_total{stage="db"}`
  incremented, the event is dropped, the worker continues. If the DB file is
  unopenable at startup the process still serves `/healthz` (reporting
  `"database": "unavailable"`) and `/metrics`, and retries opening every 30 s.

The README states plainly that this database holds copies of conversation
text and must be protected like the Open WebUI database.

---

## 8. Configuration

Environment variables (all optional):

```text
CONTEXT_GUARD_LISTEN=0.0.0.0:7432
CONTEXT_GUARD_DATABASE=/data/context-guard.db
CONTEXT_GUARD_RETENTION_DAYS=30
CONTEXT_GUARD_CONFIG=/data/context-guard.toml # optional; weights, thresholds, model limits
CONTEXT_GUARD_MODEL_LIMITS=qwen3-30b-a3b=122880,fury/qwen3-8b=12288   # overrides [model_limits]
CONTEXT_GUARD_QUEUE_SIZE=1024                 # bounded ingest channel
CONTEXT_GUARD_MAX_BODY_BYTES=33554432         # 32 MiB; LiteLLM batches of long chats are large
CONTEXT_GUARD_STORE_MESSAGES=false            # keep full messages[] per event
CONTEXT_GUARD_LOG_PAYLOADS=false              # opt-in verbose payload logging (secrets redacted)
CONTEXT_GUARD_TRUST_TRACE_ID=false            # accept LiteLLM trace_id as conversation id (see 1.3)
CONTEXT_GUARD_CONTAINER_PREFIXES=ai-          # identifier extraction hint
CONTEXT_GUARD_METRICS_LISTEN=                 # optional extra listener serving only /metrics and /healthz (see 9)
RUST_LOG=info
```

Precedence: env > TOML file > compiled defaults. `config.rs` has one
`Config::load()` and one struct; nothing reads `std::env` elsewhere.

Log redaction: values of keys matching `(?i)(api[_-]?key|token|secret|password|authorization)`
and bearer-looking strings are replaced with `[redacted]` before any payload
fragment is logged. Default logging never includes message or response text,
only ids, lengths, token counts and hashes.

---

## 9. HTTP surface

```text
POST /api/v1/ingest/litellm        JSON array | object | NDJSON of StandardLoggingPayload → 202 {accepted, dropped}
GET  /healthz                      {status, database, queue_depth, uptime_s}
GET  /api/v1/conversations         ?limit=50&status=watch  → recent conversations with current score
GET  /api/v1/conversations/{id}/health
        ?message_id=X              → the result for that Open WebUI message (latest iteration),
                                     404 {code: "not_scored_yet"} until it exists
        ?after=<unix seconds>      → latest result whose event time ≥ after, else 404 not_scored_yet
        (no query)                 → latest result for the conversation
GET  /api/v1/conversations/{id}/history      health results (and anomalies) in turn order
GET  /metrics                      Prometheus text format (optional to use)
```

Responses match the shapes in the specification; the health response also
carries `message_id`, `turn` and a one-line `summary` string that the filter
can display verbatim, e.g.

```text
Context Guard 74 · watch · context 78% · 1 known-value drift · 1 repeated operation
```

so the display text is defined in one place (Rust) and the filter stays dumb.

Errors are `{ "error": { "code": "...", "message": "..." } }`. The router is
built so a bearer-token middleware layer can be added around `/api/v1/*`
later without touching handlers; V1 has no authentication.

Network exposure: Open WebUI and LiteLLM reach Context Guard by container
name over `ai-backend`, so the container needs **no published port at all**
for the primary use case. The compose example publishes `127.0.0.1:7432` for
curl/debugging only. For the LAN Prometheus there are two documented options:
bind `0.0.0.0:7432` (exposes the API too, unauthenticated) or set
`CONTEXT_GUARD_METRICS_LISTEN=0.0.0.0:7433` and publish only 7433. The extra
listener is ~25 lines and off by default; Prometheus is never required.

Prometheus metrics (no conversation ids as labels):

```text
context_guard_events_received_total{kind}          chat | task | failure
context_guard_events_dropped_total{reason}         queue_full | malformed | oversize
context_guard_processing_errors_total{stage}
context_guard_health_score{model}                  latest scored turn for that model
context_guard_risk_score{model}
context_guard_context_utilization_ratio{model}
context_guard_conversations_by_status{status}      gauge, active in the last 24 h
context_guard_known_value_drift_total{model}
context_guard_tool_anomalies_total{model,signal}
context_guard_loop_events_total{model,signal}
context_guard_suspicious_identifiers_total{model}
context_guard_ingest_batch_size                    histogram
context_guard_queue_depth
```

---

## 10. The Open WebUI Filter (`openwebui/context_guard_filter.py`)

One file, installed as a **global** Filter (Admin → Functions → import, then
toggle Global) so it applies to every model without per-model setup. No
`inlet`, no `stream` hook; only `outlet`. Valves:

```text
context_guard_url   http://context-guard:7432
wait_seconds        6.0     total budget to wait for the score
poll_interval       0.5
connect_timeout     1.0     per request
show_minimum        always | watch | degraded   (when to show the line; default always)
notify_below        40      toast (`notification` event) when health drops under this; 0 disables
```

Outlet logic:

```text
chat_id, message_id ← __metadata__
if either is missing → return body unchanged
deadline = now + wait_seconds
loop:
    GET {url}/api/v1/conversations/{chat_id}/health?message_id={message_id}
        (fallback when the message-id header is not configured:
         ?after=<timestamp of the last assistant message in body["messages"]>)
    200 → emit status {description: result.summary, done: true}; optional notification; break
    404 not_scored_yet → sleep poll_interval, retry until deadline
    any other error / connection refused / timeout → log at debug, break   (do not retry; Guard may be down)
return body unchanged                                                       (always)
```

Guarantees: `body` is returned by reference untouched; the emitter is only
called for `status`/`notification` events, never `message`/`replace`; a broken
or absent Context Guard costs at most `connect_timeout` per reply and shows
nothing. The status line is stored in `statusHistory`, which is not part of
what the frontend sends as `messages` (verified: it sends
`id, role, content, info, timestamp, sources`), so the score never reaches
the model.

Why `wait_seconds` is enough: with `DEFAULT_FLUSH_INTERVAL_SECONDS=1` on
LiteLLM the payload arrives ≤ 1 s after the reply completes, and scoring one
event is milliseconds. The default 6 s budget covers the default 5 s flush
interval too, for people who do not change LiteLLM's flush setting.

`openwebui/install-filter.sh` (optional) creates or updates the function via
`POST /api/v1/functions/create` / `POST /api/v1/functions/id/{id}/update`
with an admin API key, so the filter can be redeployed from the repo. Manual
import through the UI is the documented primary path.

---

## 11. Docker

* `Dockerfile`: stage 1 `rust:1-bookworm` builds with `cargo build --release`
  using a dependency-caching layer (copy `Cargo.toml`/`Cargo.lock`, build a
  dummy main, then copy `src`). SQLite is statically linked via
  `libsqlite3-sys` bundled feature. Stage 2 `debian:bookworm-slim` with
  `ca-certificates` only, user `context-guard` (uid 10001), `/data` owned by
  it, `EXPOSE 7432`, `HEALTHCHECK` runs `context-guard healthcheck`
  (a subcommand of the same binary that GETs `/healthz`, so no curl/wget in the image).
* `docker-compose.example.yml`:

```yaml
  context-guard:
    build:
      context: /home/jwatkins/dev/context-guard      # source stays in ~/dev, like tailor
    image: ai-context-guard
    container_name: ai-context-guard
    restart: unless-stopped
    user: "10001:10001"
    ports:
      - "127.0.0.1:7432:7432"     # debugging only; Open WebUI and LiteLLM use the container name
    volumes:
      - ./data/context-guard:/data
    environment:
      RUST_LOG: info
      CONTEXT_GUARD_DATABASE: /data/context-guard.db
      CONTEXT_GUARD_RETENTION_DAYS: "30"
    networks:
      - ai-backend
```

LiteLLM side (`~/ai/compose.yaml` and `config/litellm/config.yaml`):

```yaml
  litellm:
    environment:
      GENERIC_LOGGER_ENDPOINT: http://context-guard:7432/api/v1/ingest/litellm
      DEFAULT_FLUSH_INTERVAL_SECONDS: "1"     # affects only batch loggers; generic_api is the only one in use
```

```yaml
litellm_settings:
  callbacks: ["generic_api"]
  extra_spend_tag_headers:
    - x-openwebui-chat-id
    - x-openwebui-user-id
    - x-openwebui-message-id
    - x-openwebui-task
```

Nothing `depends_on` Context Guard. LiteLLM starts and serves whether or not
the container exists. Applying the LiteLLM change requires
`docker compose up -d litellm` (config file is bind-mounted; a restart is
enough, no rebuild).

Open WebUI side, all through the admin UI, no env or compose change:

1. Connections → the LiteLLM connection → Headers: add
   `X-OpenWebUI-Message-Id = {{MESSAGE_ID}}` and `X-OpenWebUI-Task = {{TASK}}`.
2. Functions → import `openwebui/context_guard_filter.py` → enable → Global.
3. Set the `context_guard_url` valve if the container name differs.

---

## 12. Tests

Unit (in-module `#[cfg(test)]`, no SQLite needed thanks to the repository trait):

* `context`: 50 % → 0, 75 % → 5, 85 % → 10, 95 % → 20, unknown limit → none + reason.
* `known_values`: extraction table tests per kind; drift positive
  (`port 8080` known, assistant `port 8000` with anchor) and negatives
  (assistant repeats 8080; two known ports ⇒ ambiguous ⇒ silent; assistant
  value that the user mentioned earlier ⇒ silent; number without "port" marker ⇒ silent).
* `identifiers`: `qwen3-general` vs `qwen3-general-v2` flags; unrelated name does not.
* `repetition`: same normalized tool call 3× in 5 ⇒ one anomaly; arg key
  order and whitespace do not defeat normalization; near-identical responses
  ⇒ loop; short acks ⇒ no loop.
* `tools`: tool result with unknown id in the same request ⇒ anomaly;
  matching id ⇒ none; unknown call-id reference in text.
* `scoring`: same anomaly set ⇒ identical risk on repeated runs; health
  clamped at 0 with an absurd anomaly set; `Σ reasons.penalty == risk`;
  window drops anomalies older than W turns; status boundaries; `summary`
  string is stable for a given result.
* `telemetry::litellm`: real fixture payloads (captured in step 2, redacted)
  map to the expected `ConversationEvent`, including a tool-calling turn,
  a compacted turn, a task call, and a failure.
* `identity`: tag parsing (chat, user, message, task), precedence, fallback hashing is stable.

Integration (`tests/`, real SQLite in a temp dir, axum served on an ephemeral port):

* POST a fixture sequence, then `GET …/health?message_id=` returns 404
  `not_scored_yet` before and 200 after the matching event; `…/history`
  returns the expected scores and reasons; second identical POST changes nothing.
* Fault tolerance: invalid JSON, JSON of the wrong shape, an array with one
  good and one garbage item, 40 MiB body, deeply nested JSON, strings with
  NUL bytes and invalid UTF-8 surrogates ⇒ process keeps serving, counters
  move, `/healthz` stays 200.
* Retention deletes an old conversation and keeps a fresh one.

Filter (`openwebui/test_context_guard_filter.py`, plain `pytest` + a stub
HTTP server on localhost, run in a throwaway `python:3.12-alpine` container):

* 200 on first poll ⇒ one `status` event with the summary, body returned unchanged (same object).
* 404 then 200 ⇒ polls and shows.
* connection refused ⇒ no event, no exception, returns within `connect_timeout`.
* missing `message_id` ⇒ no request made.

`cargo test` runs inside the builder image when no host toolchain exists:
`docker run --rm -v "$PWD":/src -v cargo-cache:/usr/local/cargo/registry -w /src rust:1-bookworm cargo test`.

---

## 13. Delivery order and commits

Each step ends with a signed commit (global `gpgsign` is on).

1. **Repo bootstrap** — this plan, `.gitignore`, `Cargo.toml`,
   `src/main.rs` serving `/healthz`, Dockerfile skeleton. Commit.
2. **Capture real telemetry** — temporarily point `GENERIC_LOGGER_ENDPOINT`
   at a throwaway listener (a `python:3.12-alpine` container on `ai-backend`
   running a 20-line `http.server` that writes bodies to a file), enable
   `callbacks: ["generic_api"]`, the four tags and the two Open WebUI
   connection headers, run one normal chat, one tool-calling chat, and wait
   for a title generation. Confirms the tag names, the message-id header and
   the task heuristic against reality. Redact and store as
   `tests/fixtures/*.json`. Remove the listener. Commit fixtures.
3. **Telemetry ingestion** — `telemetry/`, `worker.rs`, `api/ingest.rs`,
   bounded channel, counters. Fixture-driven unit tests. Commit.
4. **SQLite persistence** — migrations, `database/`, retention job,
   graceful-degradation behaviour, event dedupe. Integration test. Commit.
5. **Context utilization** + config loading (`config.rs`, TOML, model limits). Tests. Commit.
6. **Scoring + health history + REST API** (window, reasons, thresholds,
   summary string, list/health/history with `message_id` and `after`
   lookups). Pulled forward so the end-to-end display path can be exercised
   with context pressure alone. Integration tests. Commit.
7. **Open WebUI filter** — `openwebui/context_guard_filter.py`, its tests,
   install script. Commit.
8. **Docker packaging + first deployment** — finished Dockerfile, compose
   example, healthcheck subcommand; add the service to `~/ai/compose.yaml`
   (backup first, as with every prior change), set the LiteLLM env and
   config, import the filter, `docker compose up -d context-guard && docker compose up -d litellm`,
   and watch a real chat show its score. Fix what reality disagrees with. Commit.
9. **Repetition** (tool-call keys, shingles/Jaccard). Tests. Commit.
10. **Known values + identifiers** (extractors, registries, drift rules). Tests. Commit.
11. **Tool ledger and anomalies.** Tests. Commit.
12. **Prometheus metrics** + optional metrics listener. Commit.
13. **README** — everything in the specification's list, the two verbatim
    disclaimers, limitations, scoring walkthrough, the filter install steps,
    and the Open WebUI header setup.
14. **Full test pass** (`cargo test`, `cargo clippy -D warnings`,
    `cargo fmt --check`, filter pytest), redeploy, fix failures, tag `v0.1.0`.

Steps 9–11 add signals to an already-running display path, so each one can be
verified in the UI as soon as it lands.

---

## 14. Deviations from the specification, with reasons

* **Open WebUI filter is in V1** (10): the specification made UI display an
  optional future item; you have since made it the primary consumer. The
  mechanism found (inline outlet + `status` event) satisfies the zero-context
  invariant, so no departure from that rule is needed.
* **`message_id` tag and lookup** (1.3, 9): not in the specification. Without
  it the filter can only guess which score belongs to which reply by
  timestamp; with it the match is exact. Costs one Open WebUI header and one
  LiteLLM tag entry.
* **`DEFAULT_FLUSH_INTERVAL_SECONDS=1` on LiteLLM** (11): purely to make the
  score available within a second of the reply. Optional; the filter budget
  covers the default 5 s too.
* **Message delta storage** (7): storing every full `messages[]` would make
  the DB grow quadratically with conversation length on 120k-context models.
  The delta plus an opt-in full-copy switch keeps the data and the disk sane.
* **Sliding window instead of a lifetime sum** (6): a lifetime sum would pin
  a long chat at "reset recommended" forever after one bad patch. The window
  is a single configurable integer, so the behaviour stays transparent.
* **Task-call exclusion** (1.4): without it every chat gets 3–4 extra
  "turns" per user message from title/tag/follow-up generation, and those
  turns would trip the loop detector.
* **Optional metrics-only listener** (9): off by default, exists only because
  the LAN Prometheus would otherwise require exposing the unauthenticated
  API. Prometheus is not required for anything.

## 15. Open points (defaults chosen, change if you disagree)

1. Host toolchain: the plan builds and tests in Docker. Installing rustup
   (user-level, no sudo) would make `cargo test` take seconds instead of
   minutes; recommended, not required.
2. Status line format: `Context Guard 74 · watch · context 78% · …`. Shown on
   every reply by default; the `show_minimum` valve can restrict it to
   watch-or-worse.
3. The two Open WebUI connection headers are set by hand in the admin UI
   during step 8 (they live in Open WebUI's config DB, not in env).
4. Retention default 30 days, full message storage off, verbose payload
   logging off.
