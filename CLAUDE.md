# Context Guard — agent briefing

Read `README.md` first: it is the product spec (signals, penalties, config
variables, REST API, metrics) and this file only adds what the README and the
code do not say. When you change behaviour, update the matching README table in
the same change.

## What this is

A deterministic, out-of-band health monitor for LLM conversations. One Rust
binary (axum + tokio + sqlx/SQLite) ingests LiteLLM `generic_api` batches,
scores each chat turn 0..100 with explicit reasons, and serves the result to an
Open WebUI filter (`openwebui/context_guard_filter.py`) that renders a UI-only
status line. Verified against LiteLLM v1.94.1 and Open WebUI v0.11.3.

## Invariants (do not break these)

1. **Never in the inference path.** No proxying, no outbound connections, no
   retries that could stall a caller. Ingest validates, enqueues, returns 202.
2. **Zero context overhead.** Nothing Context Guard produces may reach the
   model. The filter is outlet-only, returns the body untouched, and emits only
   a `status` event.
3. **Deterministic.** No models, embeddings, randomness, or wall-clock
   dependent scoring. The same events always yield the same score.
4. **False positives are worse than misses.** Signals are deliberately narrow
   (typed values with markers, unambiguous anchors). Do not loosen a signal
   without a test showing the exact case it now catches and the cases it still
   ignores.
5. **Every penalty is explained.** `risk == min(Σ reason penalties, 100)` for
   every stored result. No magic numbers in code: weights, thresholds and the
   window live in `config.rs` defaults and `config/context-guard.example.toml`.
6. **Fault isolation.** A bad payload, a full queue, or a DB write failure is
   logged, counted in metrics, and skipped; the worker continues. Only a DB
   that cannot be opened at startup exits the process.
7. **Privacy.** The SQLite file holds conversation text. Logs at default level
   never include message or response text (ids, counts, hashes, anomaly
   details only). `/metrics` never carries conversation ids or text.

## Layout and data flow

```
POST /api/v1/ingest/litellm  (src/api/ingest.rs)
  -> bounded mpsc queue of raw JSON batches (Batch = Vec<Value>)
  -> src/worker.rs (single consumer, sorts by startTime)
  -> src/telemetry/litellm.rs  normalize() -> ConversationEvent
     src/telemetry/identity.rs  conversation id precedence: chat tag > trace_id (opt-in) > fallback hash
  -> src/monitor/mod.rs  Monitor::process(): dedupe by event id, compute the message delta,
     run signals, persist anomalies, score the window, store health, update metrics
  -> src/database/repo.rs  every SQL statement lives here (runtime sqlx queries, FromRow structs)
GET /api/v1/conversations/...  (src/api/conversations.rs) read the stored results
```

- `src/monitor/{context,known_values,identifiers,repetition,tools,text}.rs`
  are **pure functions** with unit tests. Only `monitor/mod.rs` touches the DB.
- `src/monitor/scoring.rs` owns the `Signal` enum, `Status`, and
  `score()`/`summary()`.
- `src/config.rs` is the **only** place that reads the environment.
  Precedence: env > TOML file > compiled defaults. Every variable has a
  parse-and-reject test; keep that.
- `src/metrics.rs` is the Prometheus registry; signal counters are keyed by
  `Signal::family()`.
- `migrations/` is applied by `sqlx::migrate!` at startup. Never edit
  `0001_initial.sql`; add `000N_<name>.sql`. Foreign keys cascade from
  `conversations`, so retention only deletes conversations.
- `openwebui/` filter + pytest; `scripts/e2e_degradation.py` live-stack test;
  `docs/ui-demo.md` the manual walkthrough (its expected scores are asserted
  by the e2e script, keep them consistent).

## How scoring state works (non-obvious)

- A **turn** is one `EventKind::Chat` completion. Task and failure events are
  recorded but not scored, except a failure whose error text is a context
  overflow, which is scored as an overflow (`monitor/mod.rs`
  `context_overflow_tokens`).
- The monitor never trusts `messages[]` for turn counts or facts (Open WebUI
  compaction rewrites it). It hashes the previous request's message list; if
  the new list extends it, only the **delta** is examined for new facts.
- Known values and identifiers are learned only from `user` and `tool`
  messages; `assistant` text is a claim; `system` is ignored.
- Anomalies are deduplicated per conversation by `dedupe_key`
  (`UNIQUE(conversation_id, dedupe_key)`); design the key so a persistent
  condition fires once, not every turn.
- Risk sums the anomalies recorded in the last `window_turns` turns, so a
  chat recovers when the behaviour stops.
- Events are deduplicated by LiteLLM's payload `id`; redelivery is harmless.

## Adding or changing a signal (checklist)

1. Pure detection function + unit tests in the right `monitor/` submodule.
2. `Signal` variant in `scoring.rs`: `as_str`, `parse`, `severity`, `phrase`,
   `short` (fits the one-line Open WebUI status), `family`.
3. `Penalties` field, `Default`, and `for_signal` in `config.rs`.
4. Wire it in `Monitor::process` with a `Finding` and a sensible `dedupe_key`.
5. `config/context-guard.example.toml`, the README signal table, and if the
   family is new, `metrics.rs` and the README metrics list.
6. An integration test in `tests/` through the real pipeline, and a stage in
   `scripts/e2e_degradation.py` / `docs/ui-demo.md` if it can be driven from
   a chat.

## Build, test, CI

```sh
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test --all-targets            # unit + integration; tests/binary.rs spawns the real binary
python -m pytest -q openwebui/      # needs aiohttp, pydantic, pytest
docker build -t context-guard .     # multi-stage, runs as uid 10001, `context-guard healthcheck` subcommand
scripts/e2e_degradation.py --litellm-key "$LITELLM_MASTER_KEY" --model <model>   # live stack only
```

On this machine cargo is installed under `~/.cargo/bin`, which non-interactive
shells do not have on PATH; prefix commands with
`export PATH="$HOME/.cargo/bin:$PATH"` if `cargo` is not found.

CI (`.github/workflows/ci.yml`) runs exactly the first four plus a Docker
build-and-smoke. All of it must be green before a change is done; rustfmt and
clippy with `-D warnings` are hard gates.

Testing conventions:

- `tests/common/mod.rs` is the harness: temp SQLite, real worker, router driven
  in-process via `tower::ServiceExt::oneshot`. Use `PayloadBuilder`, `user()`,
  `assistant()`, `tool_call()` and `wait_for_message` / `wait_for_turns`
  rather than hand-rolling payloads. `harness_without_worker` exists for
  back-pressure tests.
- `tests/fixtures/*.json` are **real captured** LiteLLM v1.94.1 batches. Do not
  hand-edit them; capture new ones with `CONTEXT_GUARD_CAPTURE_DIR`.
- Tests run in parallel; never bind fixed ports or share DB paths.

## Conventions

- Rust 2021, stable toolchain, `anyhow` at the edges, `thiserror` for typed
  errors (`ConfigError`, `NormalizeError`). Enums carry `as_str`/`parse` pairs
  for their DB string form.
- Timestamps are RFC 3339 UTC text in SQLite (`repo::ts`).
- Log with `tracing` structured fields, not formatted strings; anything that
  looks like a key, token or password must be redacted before logging.
- The API has no auth by design (V1); the router is split so a bearer layer
  can wrap `/api/v1/*` later. Do not add auth piecemeal.
- Versions: bump `Cargo.toml`, the filter docstring `version:` in
  `openwebui/context_guard_filter.py`, and tag `vX.Y.Z`.
- Commit messages are short imperative subjects in the style of `git log`.
