# Plan: integrate Context Guard with opencode

Goal: make opencode a fourth telemetry source for Context Guard, exactly like
Claude Code and Codex — an out-of-band ingest source plus a UI-only status
line. The status line goes into the opencode TUI's right-hand sidebar, in the
same slot list as the built-in "Context" panel. Nothing Context Guard produces
ever reaches the model, and the ingest path preserves every invariant in
`CLAUDE.md` (never in the inference path, deterministic, fault isolation,
privacy).

Read `README.md` and `CLAUDE.md` in this repo first; they are the spec. When
you change behaviour, update the matching README table in the same change.

OpenCode source references below are to `github.com/anomalyco/opencode`
(verified against the `dev` branch, commit `e03db9b`). `packages/sdk/js/src/gen/types.gen.ts`
is the generated client types; `packages/opencode/specs/tui-plugins.md` is the
TUI plugin spec; `packages/tui/src/routes/session/sidebar.tsx` and
`packages/tui/src/feature-plugins/sidebar/context.tsx` are the sidebar slot
host and the built-in Context panel.

---

## 1. Verified facts this plan relies on

- The TUI right-hand sidebar (`packages/tui/src/routes/session/sidebar.tsx`) is a
  42-wide column of **plugin slots**. The built-in panels are all plugins that
  register the same `sidebar_content` slot, ordered: context `100`, mcp `200`,
  lsp `300`, todo `400`, files `500`. `sidebar_content` uses the default slot
  mode = every registered plugin renders, stacked top to bottom by `order`.
  So the status line is a `{ order: 150, slots: { sidebar_content(...) } }`
  registration and lands directly under the built-in Context panel.
- `sidebar_title` and `sidebar_footer` are `single_winner` (a plugin can also
  override the footer/title — do NOT; keep the built-ins). Slots receive
  `{ session_id }`; slot context exposes `ctx.theme.current` (tokens such as
  `text`, `textMuted`, `success`, `warning`, `error`).
- The TUI plugin API (`api.*`) is defined in `packages/opencode/specs/tui-plugins.md`:
  `api.event.on(type, handler)`, `api.state.session.messages(sessionID)`,
  `api.state.session.status(sessionID)`, `api.client` (SDK), `api.theme.current`,
  `api.slots.register({ order, slots })`, `api.lifecycle.onDispose(fn)`.
- Messages are `{ info: Message, parts: Part[] }` (SDK `session.messages()`).
  `Message = UserMessage | AssistantMessage`; `AssistantMessage` carries
  `id`, `modelID`, `providerID`, `tokens {input, output, reasoning,
  cache {read, write}}`, `cost`, `finish`, `error?`, `time.created/completed`.
  Parts are `text` (`synthetic?`, `ignored?`), `reasoning`, `file`, `tool`
  (`callID`, `tool`, `state {pending|running|completed|error}` with
  `state.input` = args and `state.output` / `state.error` = result),
  `step-start`, `step-finish`, `snapshot`, `patch`, `agent`, `subtask`,
  `compaction` (`summary`), `retry`.
- Key structural fact: **a tool call and its result are parts of the same
  assistant message** (`state` transitions pending → running → completed). So
  one assistant message is one complete turn including its tool results; the
  orphan-result machinery in `monitor/mod.rs` only ever sees consistent data.
- Error on an assistant message is typed
  (`MessageOutputLengthError`, `MessageAbortedError`, `APIError`,
  `ProviderAuthError`, `UnknownError`).
- Context-limit is available in the TUI from the provider registry:
  `api.state.provider` → `models[modelID].limit.context` (the built-in Context
  panel already reads `model?.limit.context`).
- Events available to the plugin: `message.updated` (fires with `info` for
  both user and assistant messages), `message.part.updated`, `session.status`
  (`idle | busy | retry`), `session.idle`, `session.created`. `session.status`
  → `type === "idle"` reliably marks the end of a turn.

## 2. End-to-end data flow

```
opencode TUI
  plugin (context_guard.tui.ts + context_guard_shipper.mjs)
    on session.status idle / message.updated(assistant completed):
      records = session.messages(sessionID) since cursor   (raw {info, parts})
      POST /api/v1/ingest/opencode   {session_id, context_limit, records}
        -> src/api/ingest.rs handler (validate, 202, bounded queue)
        -> worker::Batch::OpenCode   (single consumer, FIFO)
        -> src/telemetry/opencode.rs normalize() -> ConversationEvent (messages_are_delta)
        -> Monitor::process: dedupe by message id, run signals, score, persist
  display slot (sidebar_content, order 150):
    GET /api/v1/conversations/{session_id}/health -> render one-line summary
    link to /ui/conversations/{session_id} (OSC 8 terminal hyperlink)
```

## 3. Source mapping (opencode -> ConversationEvent)

`ConversationEvent` and `delta_completion`/`delta_failure` already exist and
are shared by Claude Code and Codex; the opencode normalizer is a third caller.
Use `IdSource::Session` (exists) with `conversation_id = session_id`.

| opencode field | ConversationEvent field | rule |
|---|---|---|
| `info.id` (one assistant message) | `event_id`, `message_id` | dedupe key; redelivery harmless |
| `info.role == "assistant"` with `text` or `tool` parts | one `EventKind::Chat` turn | `starts_prompt` = delta has a `Role::User` message |
| `info.role == "assistant"` with `error` | `EventKind::Failure` | error text = `error.name` + `error.data.message`/`responseBody`; an input-too-long message is scored as overflow by the existing `context_overflow_tokens` matcher in `monitor/mod.rs` |
| `info.role == "user"`, `text` parts | `Role::User` message (already seen = source of truth) | drill into the monitor's normal Fact plumbing |
| `text` part with `synthetic: true` | `Role::System` | injected context (`session.prompt noReply`), never a fact source |
| `compaction` part (`summary`) | `Role::System` | model-written summary, like Claude Code isCompactSummary / Codex `compacted` |
| assistant `text` parts (non-`ignored`) | reply text (claims) | concatenate with `\n` |
| `tool` parts | `ToolCall` + `tool` result messages | `callID` = tool call id, `state.input` (JSON) = args, `state.output` = result content, `state.error` = result error; results belong to calls in the **same** message |
| `reasoning`, `file`, `step-start`, `step-finish`, `snapshot`, `patch`, `agent`, `subtask`, `retry` parts | ignored | never shipped content-wise |
| `info.tokens.input + cache.read + cache.write` | `prompt_tokens` | same "what the model held" semantics as Claude Code |
| `info.tokens.output` | `completion_tokens` | |
| `info.modelID` | `model` | bare model id |
| `envelope.context_limit` | `context_limit` | from the plugin (provider registry) when known, else `CONTEXT_GUARD_MODEL_LIMITS`, else unknown |
| `info.time.created` (ms) | `started_at`, `timestamp` | convert to RFC 3339 UTC (`repo::ts` format) |

Delta semantics: `messages_are_delta = true`. Because tool results are bundled
in the same assistant message as their calls, the delta does **not** need to
open with the previous reply (the Codex/Claude Code orphan-result rationale
does not apply); the delta is the records since the last shipped message.
Skip assistant messages with **no** text and **no** tool parts (empty/step-only
records). A turn whose model is knowable is always emitted.

Failure events: emitted via `delta_failure` (same shape as codex/Claude Code);
the prompt delta is copied, not taken, so retries count the prompt once.

## 4. Decisions (settled — change only if a test forces it)

1. **Envelope required**, shaped exactly like Codex: object
   `{"session_id": "...", "context_limit": number|null, "records": [...]}`.
   Session id comes from the envelope (`IdSource::Session`), not from message
   fields, to keep the worker/identity code identical to the Codex path.
2. **The plugin ships raw `{info, parts}` records**, not normalized events.
   OpenCode owns message-shape drift; the Rust normalizer owns the mapping and
   the tests. (Same split of labour as the Claude Code/Codex hooks.)
3. **Cursor = count of fully-complete messages shipped** (per session, in a
   state-dir file, same pattern as `agent-hooks/context_guard_shipper.py` and
   `codex/context_guard_codex_hook.py`). Only complete messages are shipped
   (assistant messages with `time.completed`/`finish`/`error`, user messages
   with at least one text part). A message still streaming is held back and
   ships on the next turn boundary; dedupe by message id makes the reship of
   the tail harmless. Do NOT advance the cursor for a partially-written
   message.
4. **First ship of a session starts at the current cursor (0)**, i.e. the
   plugin watches a session from the moment it loads, including a session
   already in progress. The monitor only scores turns seen since deploy; this
   matches the existing README limitation and keeps the first POST bounded.
5. **Refresh of the status line**: on `session.status` → `idle` and on
   `message.updated` (assistant only), plus a ~2 s poll while the session is
   busy (same lazy-refresh rationale as the Claude Code status line's
   `refreshInterval: 2`). Cache the last-known score per session; show the
   cached line when the service is unreachable. Never block or throw.
6. **Display is `sidebar_content` order 150**, right under the built-in
   Context panel. Compatible with `ctx.theme.current` tokens. No changes to
   `sidebar_title`/`sidebar_footer`.
7. **Zero context overhead and never-fail are hard requirements.** The plugin
   only reads session state via the SDK and renders TUI slots; it never calls
   `session.prompt` (not even `noReply`), never injects parts, never edits a
   message. Ship failures and fetch failures log and give up; nothing retries
   a call that could stall the agent.
8. **Headless is out of scope for v1.** The TUI plugin only runs in the TUI.
   `opencode run`/headless sessions are not monitored (mirrors the note that
   headless `claude -p` can leave the last reply unscored). A server-side
   plugin module to cover headless is a future extension, not part of this
   plan.
9. **Privacy.** Logs from the JS plugin must never include message text
   (session id, record counts, message ids only, `tracing`-style).
   `CONVERSATION_TEXT` stays server-side; the plugin is stateless except the
   cursor. The Rust normalizer applies the existing `redact_secrets` on any
   anomaly details exactly as the other sources do.
10. **No new signals, penalties, config variables or metrics.** The opencode
    source emits ordinary `ConversationEvent`s; `Signal::ALL`, `Penalties`,
    `Thresholds`, the metrics registry and `config.rs` are untouched. This
    keeps the blast radius a pure "one more source" change.

## 5. Work packages

### WP1 — Rust ingest + normalizer

1. `src/telemetry/opencode.rs` — new module, modelled on `codex.rs`:
   - `pub struct Ingest { session_id, context_limit: Option<u64>, records: Vec<Value> }`
   - `parse_body(&[u8]) -> Result<Ingest, ParseError>` — envelope required;
     `session_id` non-empty; `records` a non-empty array; reuse `split_body`
     if the existing pattern needs NDJSON tolerance, otherwise strict
     single-object like Codex. Reject with `ParseError::MissingField`.
   - `normalize(ingest) -> Normalized` — walk records in order, map per §3,
     call `super::delta_completion` / `delta_failure`. Every unknown field is
     optional and ignored; a record that cannot be read counts
     `malformed` and is skipped (monitor continues). The tool-result parts of
     an assistant message are appended to the pending delta as `Role::Tool`
     messages (call id + content), so `Monitor::process` sees calls and their
     results.
   - Unit tests that mirror the fixture minimally: user facts → assistant
     drift; tool call + result pairing; synthetic-as-system; compaction-as-
     system; failure/overflow; empty-message skip; metadata-only messages.
2. `src/telemetry/mod.rs` — `pub mod opencode;` registration.
3. `src/worker.rs` — add `Batch::OpenCode(Ingest)` variant; the worker's
   `process_in_scope`/match arms call `opencode::normalize` and handle
   `Outcome` like the Codex branch.
4. `src/api/ingest.rs` — `opencode` handler mirroring `codex`: parse, count
   records for `ingest_batch_size`, enqueue, answer 202 `{accepted, dropped}`;
   400 unparseable, 413 over `CONTEXT_GUARD_MAX_BODY_BYTES`.
5. `src/lib.rs` — route `POST /api/v1/ingest/opencode` alongside the others.

### WP2 — opencode plugin (ship + display)

Author both files under a new repo directory `opencode/` (do not reuse the
already-claimed `openwebui/` name).

1. `opencode/context_guard_shipper.mjs` — pure, dependency-free (Node ≥18
   `fetch`, `node:test`-friendly), mirrors `agent-hooks/context_guard_shipper.py`:
   - `readCursor(stateDir, sessionID)` / `writeCursor(stateDir, sessionID, n)`
     (JSON cursor files, atomic write via temp + rename).
   - `collect(messages, cursor)` → records to ship; skips incomplete messages;
     returns `{records, nextCursor}`.
   - `buildPayload(sessionID, contextLimit, records)`.
   - `ship(baseUrl, payload, captureDir?)` — POST JSON; when
     `CONTEXT_GUARD_CAPTURE_DIR` is set, write the exact payload to that dir
     instead (fixture capture, same env semantics as the server). Returns
     `{accepted, dropped}`; network errors return `null`, never throw.
   - `fetchHealth(baseUrl, sessionID)` — one `GET
     /api/v1/conversations/{id}/health`, one refused connection then give up.
   - `statusLine(health)` — the one-line summary (format parity with the
     Rust `summary()`: `🟢 Context Guard 100 · healthy · 🟢 context 4% · 1 drift`).
2. `opencode/context_guard.tui.ts` — the TUI plugin module:
   - `default export { id: "context-guard", tui }` (file plugins must export a
     non-empty `id`).
   - `tui(api)`:
     - read `CONTEXT_GUARD_URL` (default `http://127.0.0.1:7432`),
       `CONTEXT_GUARD_STATE_DIR` (default `~/.local/state/context-guard/opencode`),
       `CONTEXT_GUARD_CAPTURE_DIR`, `CONTEXT_GUARD_NOTIFY_BELOW` (default 0,
       off), `CONTEXT_GUARD_LINK` (`{id}` placeholder for the explain URL).
     - `api.event.on("session.status", …)` and `api.event.on("message.updated", …)`:
       on idle / assistant-completed, `flush(sessionID)` using
       `api.client.session.messages({path:{id}})` (prefer the client over
       `api.state` if the state accessor returns partial parts — verify at
       implement time and pin one).
     - resolve `context_limit` once per session from
       `api.state.provider` → `models[info.modelID].limit.context` when present.
     - `api.slots.register({ order: 150, slots: { sidebar_content(ctx, {session_id}) { … } } })`
       → `GuardView` component (Solid, `/** @jsxImportSource @opentui/solid */`).
       Render: `<b>Context Guard</b>` + the `statusLine()`; colour via
       `ctx.theme.current` (`success`/`warning`/`error` for the light);
       wrap the line in an OSC 8 hyperlink to
       `CONTEXT_GUARD_LINK ?? "http://127.0.0.1:7432/ui/conversations/{id}"`.
     - per-session state map for last-known score; a ~2 s poll timer while
       the session is busy so the score appears shortly after the turn (the
       Claude Code status line rationale).
     - toast (`api.attention`/`ui.toast` is not plugin-required; use
       `api.client.tui.showToast` if available at runtime) when the score drops
       below `CONTEXT_GUARD_NOTIFY_BELOW`.
     - register disposers via `api.lifecycle.onDispose` (timers, event unsubs).
   - Import types from `@opencode-ai/plugin/tui`; keep the pure logic in the
     `.mjs` file so tests never need the opencode runtime.
3. `opencode/settings.example.json` or a README snippet showing the install —
   local file plugins are listed in the user's `tui.json` under `plugin`
   (a relative/absolute path to `context_guard.tui.ts`). Provide the JSON to
   merge, mirroring `claude-code/settings.example.json` and
   `codex/hooks.example.json`.

### WP3 — fixtures and tests

1. Capture real fixtures, do **not** hand-edit:
   - `scripts/capture_opencode_fixture.mjs` — a throwaway (or committed)
     helper that uses `@opencode-ai/sdk` `client.session.messages()` on a
     chosen session id and writes `tests/fixtures/opencode-<version>-<session-part>.json`
     as the exact `{session_id, context_limit, records}` ingest body (with
     `records` = `{info, parts}`). Drive one scripted chat in a scratch repo
     that exhibits: user facts → drift, a repeated tool call, a loop, a
     tool result + unknown call id reference, a failure/overflow message, a
     compaction, and a `synthetic` text part. That one session produces the
     fixture after sanitizing (defang ids: sessions/messages are fine as-is;
     strip any real tokens/paths that look like secrets).
   - `CONTEXT_GUARD_CAPTURE_DIR` in the plugin should produce the same body
     so capture is also possible straight from the plugin.
2. Rust integration: `tests/opencode.rs` through `tests/common/mod.rs`
   (in-process router + real worker, temp SQLite). Add an opencode
   `PayloadBuilder` or post the fixture bytes directly; assert: a turn is
   scored and stored; the known-value/drift flow works from `Role::User`
   deltas; tool-call/result pairing feeds `repeated_tool_call`; failures with
   an input-too-long error are scored as overflow; overlapping/resent tail is
   deduplicated (message id); malformed record in the middle is skipped and
   counted; `?message_id=` health lookup.
3. JS tests: `opencode/test/*.test.mjs` with `node:test` + `node:http` stub
   service (mirror `agent-hooks/stub_service.py`): cursor advance, incomplete-
   message hold-back, payload shape, connection-refused never-fail,
   `statusLine` formatting, capture-dir mode.
4. Unit tests in `opencode.rs` for edge parsing per WP1.

### WP4 — docs, versioning, CI

1. README:
   - architecture diagram: add opencode as a source.
   - new section "opencode integration" (install via `tui.json`, env vars,
     what is monitored, sidebar placement, headless caveat).
   - REST API table: `/api/v1/ingest/opencode` row.
   - "Current limitations": headless/open-`run` not monitored; TUI-only.
   - `agent-hooks/` mention stays; add `opencode/` to the repository layout.
2. Version bump to `0.4.0` (new source = feature): `Cargo.toml`, the filter
   docstring `version:` in `openwebui/context_guard_filter.py`, the image tag
   in the README quick start and `docker-compose.example.yml`. Do not tag yet.
3. CI (`.github/workflows/ci.yml`): add the JS plugin tests (a `node` step
   running `node --test "opencode/test/*.test.mjs"`). Everything existing must
   stay green. (Node 22+ does not treat a bare directory arg as a scan root;
   the quoted glob is the reliable form.)
4. `config/context-guard.example.toml` — no change (no new server config).
5. `docs/ui-demo.md` / `scripts/e2e_degradation.py` — no new stage. The TUI is
   not scriptable from the e2e harness; keep expected scores consistent with
   existing stages only.

### WP5 — implementation log (differs from the plan only where noted)

- WP2 done: `opencode/context_guard_shipper.mjs` (pure) + `opencode/context_guard.tui.tsx`
  + `opencode/settings.example.json`. **Verified deviation:** the built-in
  `ui/link.tsx` is internal to the TUI (not exported to plugins) and the
  plugin API has no clickable-link widget in 1.18.31, so the sidebar line is
  plain text — no OSC 8 / URL rendering from the plugin. The explain page URL
  is still reachable directly. `context_guard.tui.tsx` uses
  `api.state.session.messages()` (reactive, flat `SessionMessage[]`) as the
  pinned source, with a one-shot `api.client.session.messages({limit:1000})`
  fallback when the reactive store is empty; both return the same flat shape
  and the shipper converts both to `{info, parts}`.
- WP3 done: `tests/opencode.rs` (4 integration tests) against
  `tests/fixtures/opencode-session.json` (synthetic, verified against the SDK
  types — no live provider here), `opencode/test/shipper.test.mjs` (11 tests),
  `scripts/capture_opencode_fixture.mjs` (real-capture helper for a live
  install, smoke-tested against a stub).
- WP4 done: README + CLAUDE.md updated, version 0.4.0, CI node step added.
  Remaining before the release branch: run the full local gates, then the
  "definition of done" manual TUI check on a real install.

## 6. Definition of done

- Full gates green, exactly the five commands in `CLAUDE.md` (+ the new node
  test step):
  - `export PATH="$HOME/.cargo/bin:$PATH"` then
    `cargo fmt --check && cargo clippy --all-targets -- -D warnings && cargo test --all-targets && cargo audit`
  - `python -m pytest -q openwebui/ claude-code/ codex/ scripts/extraction_recall`
  - `node --test opencode/`
  - `docker build -t context-guard .` (with `ui/` still copied; unchanged)
- A real end-to-end check that cannot be automated here: start the service,
  load the plugin in an opencode TUI session against the README walkthrough
  facts, and confirm (a) the sidebar shows the score under the Context panel
  within ~2 s of a reply, (b) the score drops on a drift and hydrates the
  explain page link, (c) the plugin survives the service being stopped.
- The "Adding or changing a signal" checklist does not apply (no signal
  changes); the "one conversation model everywhere" rule does: nothing
  downstream of `telemetry/mod.rs` knows opencode exists.

## 7. Risks / unknowns to resolve during implementation

- Which exact shape `api.state.session.messages(sessionID)` returns vs
  `api.client.session.messages()` (parts completeness). Pin one at implement
  time; the fixture capture must use the same source.
- `summary()` format parity between the JS status line and the Rust `summary`
  — the JS side should ideally just print the server's `summary` string from
  the health payload rather than re-derive it (the server already returns it).
- Whether `message.updated` fires once per completed assistant message with a
  terminal `finish`/`error` — the flush trigger has a `session.status → idle`
  fallback so a missed event cannot strand a turn.
- Version drift of the opencode SDK shape — every field in the Rust parser is
  optional by design, so a shape change degrades to `malformed` counts, never
  a crash.