//! Parsing Codex CLI session rollouts (`~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl`).
//!
//! The hook in `codex/context_guard_codex_hook.py` ships the rollout's
//! conversation records incrementally, together with what the records do not
//! carry: the session id, the model and the context window it learned from
//! the hook event and the turn's bookkeeping records. One API response is a
//! run of model output items (`message`, `function_call`, `custom_tool_call`,
//! …) closed by the record that carries that response's token usage: a
//! `token_usage_record` (paginated history) or a `token_count` event (legacy
//! history). That run is one completion, hence one turn. The records between
//! two completions are the request delta: the previous reply with its tool
//! calls, the tool outputs, and the user's next prompt.
//!
//! A response still open when the slice ends has no usage record yet and is
//! not emitted: the hook re-ships from the start of the last closed response,
//! and dedupe never extends an event, so an early flush would lose that
//! turn's tokens. A user prompt, a new turn, an interrupt or a failed turn
//! closes a response without usage (the hook counts the same records as
//! closing, so both sides agree on what one ship contains); a usage record
//! arriving after such a close is dropped, so a prompt written before its
//! reply's usage record costs that turn's token count. A response whose model
//! is not known is emitted as `unknown` rather than dropped: dropping it
//! would strand its tool outputs in the next request without their call.
//! Codex documents nothing about this format, so every field is optional and
//! unknown records are ignored rather than rejected. Verified against Codex
//! 0.154.0.

use chrono::{DateTime, Utc};
use serde::Deserialize;
use serde_json::Value;

use super::event::{Message, Role, ToolCall};
use super::litellm::{split_body, ParseError};
pub use super::Normalized;
use super::{
    content_text, delta_completion, delta_failure, parse_rfc3339, DeltaFailure, DeltaReply,
};
use crate::config::Config;

/// One ingest body: rollout records plus the session id (records do not carry
/// it), the model from the hook event, and the context window the shipper
/// last saw in a `task_started` event.
#[derive(Debug, Clone, Default)]
pub struct Ingest {
    pub session_id: String,
    pub model: Option<String>,
    pub context_limit: Option<u64>,
    pub records: Vec<Value>,
}

/// Accepts `{"session_id": "...", "model": "...", "context_limit": N, "records": [...]}`.
/// The envelope is required because rollout records do not name their session.
pub fn parse_body(body: &[u8]) -> Result<Ingest, ParseError> {
    let mut items = split_body(body)?;
    if items.len() != 1 {
        return Err(ParseError::Shape);
    }
    let mut envelope = items.pop().unwrap_or(Value::Null);
    let session_id = envelope
        .get("session_id")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .map(String::from)
        .ok_or(ParseError::MissingField("session_id"))?;
    let records = match envelope.get_mut("records").map(Value::take) {
        Some(Value::Array(records)) => records,
        _ => return Err(ParseError::MissingField("records")),
    };
    Ok(Ingest {
        session_id,
        model: envelope
            .get("model")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .map(String::from),
        context_limit: envelope.get("context_limit").and_then(Value::as_u64),
        records,
    })
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct Record {
    #[serde(rename = "type")]
    kind: String,
    timestamp: Option<String>,
    payload: Value,
}

#[derive(Debug, Default, Deserialize, Clone, Copy)]
#[serde(default)]
struct Usage {
    input_tokens: Option<u64>,
    output_tokens: Option<u64>,
}

/// The usage record that closes a response.
struct Closing {
    usage: Usage,
    response_id: Option<String>,
}

/// The model output items of one API response, accumulated until its usage
/// record arrives.
#[derive(Default)]
struct Group {
    first_item_id: Option<String>,
    started_at: Option<DateTime<Utc>>,
    timestamp: Option<DateTime<Utc>>,
    text: Vec<String>,
    tool_calls: Vec<ToolCall>,
}

impl Group {
    fn absorb(&mut self, rec: &Record) {
        let ts = rec.timestamp.as_deref().and_then(parse_rfc3339);
        if self.first_item_id.is_none() {
            self.first_item_id = item_id(&rec.payload);
            self.started_at = ts;
        }
        self.timestamp = ts.or(self.timestamp);
    }
}

/// A model output item's own id, or its call id for tool calls (`call_id`
/// pairs a call with its output; `id` is the server-side item id).
fn item_id(payload: &Value) -> Option<String> {
    ["id", "call_id"].iter().find_map(|k| {
        payload
            .get(k)
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .map(String::from)
    })
}

struct Builder<'a> {
    config: &'a Config,
    session_id: &'a str,
    model: Option<String>,
    context_limit: Option<u64>,
    out: Normalized,
    /// Messages added to the request since the last completion was emitted.
    pending: Vec<Message>,
    current: Option<Group>,
    /// Tool outputs written while `current` is still open (legacy history
    /// records the output before the usage record; a parallel call's output can
    /// land between two items of one response). They belong to the next
    /// request, after the reply they answer.
    deferred: Vec<Message>,
    /// Paginated history writes both records per response; the `token_count`
    /// event comes second and must not close the next response.
    saw_usage_record: bool,
}

impl Builder<'_> {
    fn record(&mut self, rec: &Record) {
        let payload_type = rec.payload.get("type").and_then(Value::as_str);
        match rec.kind.as_str() {
            "turn_context" => {
                // A new user turn: whatever was still open never got its usage.
                self.flush(None);
                if let Some(model) = rec.payload.get("model").and_then(Value::as_str) {
                    self.model = Some(model.to_string());
                }
            }
            "token_usage_record" => {
                self.saw_usage_record = true;
                let response_id = rec
                    .payload
                    .get("response_id")
                    .and_then(Value::as_str)
                    .map(String::from);
                self.close(rec.payload.get("usage"), response_id);
            }
            "event_msg" => match payload_type {
                Some("task_started") => {
                    if let Some(window) = rec
                        .payload
                        .get("model_context_window")
                        .and_then(Value::as_u64)
                    {
                        self.context_limit = Some(window);
                    }
                }
                Some("token_count") if !self.saw_usage_record => {
                    let usage = rec
                        .payload
                        .get("info")
                        .and_then(|info| info.get("last_token_usage"));
                    self.close(usage, None);
                }
                Some("task_complete") => {
                    if rec.payload.get("error").is_some_and(Value::is_object) {
                        self.failure(rec);
                    }
                }
                Some("turn_aborted") => self.flush(None),
                _ => {}
            },
            "response_item" => match payload_type {
                Some("message") => self.message(rec),
                Some(
                    "function_call" | "custom_tool_call" | "local_shell_call" | "tool_search_call"
                    | "web_search_call",
                ) => self.call(rec),
                Some(
                    "function_call_output"
                    | "custom_tool_call_output"
                    | "local_shell_call_output"
                    | "tool_search_output",
                ) => self.output(rec),
                Some(_) => {} // reasoning and item types added later
                None => self.out.malformed += 1,
            },
            "compacted" => {
                // The summary is model-written: never a source of truth.
                if let Some(text) = rec.payload.get("message").and_then(Value::as_str) {
                    self.push(Role::System, text.to_string());
                }
            }
            _ => {} // session_meta, world_state, bookkeeping
        }
    }

    fn message(&mut self, rec: &Record) {
        let role = rec
            .payload
            .get("role")
            .and_then(Value::as_str)
            .map(Role::parse)
            .unwrap_or(Role::Other);
        let text = content_text(rec.payload.get("content"));
        match role {
            Role::Assistant => {
                let group = self.current.get_or_insert_with(Group::default);
                group.absorb(rec);
                if !text.is_empty() {
                    group.text.push(text);
                }
            }
            Role::User => {
                // A prompt ends the reply in progress. Environment context and
                // compaction summaries are stored as user messages too; the
                // content kinds tell them apart (legacy history has none).
                self.flush(None);
                let kinds = rec
                    .payload
                    .get("internal_chat_message_metadata_passthrough")
                    .and_then(|m| m.get("content_item_kinds"))
                    .and_then(Value::as_array);
                let role = match kinds {
                    Some(kinds) if !kinds.iter().any(|k| k.as_str() == Some("user.text")) => {
                        Role::System
                    }
                    _ => Role::User,
                };
                self.push(role, text);
            }
            Role::System => self.push(Role::System, text),
            Role::Tool | Role::Other => {}
        }
    }

    fn call(&mut self, rec: &Record) {
        let payload = &rec.payload;
        let kind = payload.get("type").and_then(Value::as_str).unwrap_or("");
        let name = payload
            .get("name")
            .and_then(Value::as_str)
            .map(String::from)
            .unwrap_or_else(|| kind.trim_end_matches("_call").to_string());
        let arguments = ["arguments", "input", "action"]
            .iter()
            .find_map(|k| payload.get(k))
            .map(|v| match v {
                Value::String(s) => s.clone(),
                other => other.to_string(),
            })
            .unwrap_or_default();
        let group = self.current.get_or_insert_with(Group::default);
        group.absorb(rec);
        group.tool_calls.push(ToolCall {
            id: payload
                .get("call_id")
                .and_then(Value::as_str)
                .map(String::from),
            name,
            arguments,
        });
    }

    fn output(&mut self, rec: &Record) {
        let message = Message {
            role: Role::Tool,
            content: content_text(rec.payload.get("output")),
            tool_call_id: rec
                .payload
                .get("call_id")
                .and_then(Value::as_str)
                .map(String::from),
            tool_calls: Vec::new(),
        };
        if self.current.is_some() {
            self.deferred.push(message);
        } else {
            self.pending.push(message);
        }
    }

    fn push(&mut self, role: Role, content: String) {
        self.pending.push(Message {
            role,
            content,
            tool_call_id: None,
            tool_calls: Vec::new(),
        });
    }

    /// A usage record closes the open response. One whose usage cannot be
    /// read still closes it (with no token counts), so the next response is
    /// never merged into this one.
    fn close(&mut self, usage: Option<&Value>, response_id: Option<String>) {
        if self.current.is_none() {
            return;
        }
        let usage = usage
            .filter(|u| u.is_object())
            .and_then(|u| serde_json::from_value::<Usage>(u.clone()).ok())
            .unwrap_or_default();
        self.flush(Some(Closing { usage, response_id }));
    }

    fn flush(&mut self, closing: Option<Closing>) {
        let Some(group) = self.current.take() else {
            return;
        };
        let model = self.model.clone().unwrap_or_else(|| "unknown".to_string());
        let (usage, response_id) = match closing {
            Some(c) => (c.usage, c.response_id),
            None => (Usage::default(), None),
        };
        let Some(event_id) = response_id.or_else(|| group.first_item_id.clone()) else {
            self.out.malformed += 1;
            return;
        };
        let event = delta_completion(
            &mut self.pending,
            &mut self.deferred,
            DeltaReply {
                event_id,
                conversation_id: self.session_id.to_string(),
                message_id: group.first_item_id,
                context_limit: self.config.model_limit(&model).or(self.context_limit),
                model,
                started_at: group.started_at,
                timestamp: group.timestamp,
                prompt_tokens: usage.input_tokens,
                completion_tokens: usage.output_tokens,
                text: group.text,
                tool_calls: group.tool_calls,
            },
        );
        self.out.events.push(event);
    }

    /// A turn that ended in an error is a failed completion of the pending
    /// request; the user will retry it, so the pending delta is kept.
    fn failure(&mut self, rec: &Record) {
        self.flush(None);
        let Some(turn_id) = rec
            .payload
            .get("turn_id")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
        else {
            self.out.malformed += 1;
            return;
        };
        let error = rec.payload.get("error");
        let text = ["message", "codex_error_info"]
            .iter()
            .filter_map(|k| error.and_then(|e| e.get(k)).and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join(" ");
        let model = self.model.clone().unwrap_or_else(|| "unknown".to_string());
        let event = delta_failure(
            &self.pending,
            DeltaFailure {
                event_id: format!("{turn_id}:failure"),
                conversation_id: self.session_id.to_string(),
                message_id: Some(format!("{turn_id}:failure")),
                context_limit: self.config.model_limit(&model).or(self.context_limit),
                model,
                timestamp: rec
                    .timestamp
                    .as_deref()
                    .and_then(parse_rfc3339)
                    .unwrap_or_else(Utc::now),
                error: text,
            },
        );
        self.out.events.push(event);
    }
}

/// Normalize one shipped slice of a rollout into events, in file order. A
/// response left open at the end of the slice is not emitted (see the module
/// doc); a record that cannot be read is counted as malformed.
pub fn normalize(ingest: &Ingest, config: &Config) -> Normalized {
    let mut b = Builder {
        config,
        session_id: &ingest.session_id,
        model: ingest.model.clone(),
        context_limit: ingest.context_limit,
        out: Normalized::default(),
        pending: Vec::new(),
        current: None,
        deferred: Vec::new(),
        saw_usage_record: false,
    };
    for value in &ingest.records {
        match serde_json::from_value::<Record>(value.clone()) {
            Ok(rec) => b.record(&rec),
            Err(_) => b.out.malformed += 1,
        }
    }
    b.out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::telemetry::event::{ConversationEvent, EventKind, IdSource};
    use serde_json::json;

    const SESSION: &str = "01a08cc2-557b-7220-9231-ec71906fb7c6";
    const TS: &str = "2026-09-10T19:19:45.278Z";

    fn record(kind: &str, payload: Value) -> Value {
        json!({"timestamp": TS, "type": kind, "payload": payload})
    }

    fn turn_context(model: &str) -> Value {
        record(
            "turn_context",
            json!({"turn_id": "turn_1", "model": model, "cwd": "/w"}),
        )
    }

    fn task_started(window: u64) -> Value {
        record(
            "event_msg",
            json!({"type": "task_started", "turn_id": "turn_1", "model_context_window": window}),
        )
    }

    fn user(text: &str, kinds: Option<Vec<&str>>) -> Value {
        let mut payload = json!({
            "type": "message", "id": format!("msg_u_{}", text.len()), "role": "user",
            "content": [{"type": "input_text", "text": text}]
        });
        if let Some(kinds) = kinds {
            payload["internal_chat_message_metadata_passthrough"] =
                json!({"content_item_kinds": kinds});
        }
        record("response_item", payload)
    }

    fn developer(text: &str) -> Value {
        record(
            "response_item",
            json!({"type": "message", "id": "msg_dev", "role": "developer",
                   "content": [{"type": "input_text", "text": text}]}),
        )
    }

    fn reply(id: &str, text: &str) -> Value {
        record(
            "response_item",
            json!({"type": "message", "id": id, "role": "assistant", "phase": "final_answer",
                   "content": [{"type": "output_text", "text": text}]}),
        )
    }

    fn call(id: &str, call_id: &str, input: &str) -> Value {
        record(
            "response_item",
            json!({"type": "custom_tool_call", "id": id, "call_id": call_id, "name": "exec",
                   "status": "completed", "input": input}),
        )
    }

    fn output(call_id: &str, text: &str) -> Value {
        record(
            "response_item",
            json!({"type": "custom_tool_call_output", "id": "ctco_x", "call_id": call_id,
                   "output": [{"type": "input_text", "text": text}]}),
        )
    }

    fn usage_record(response_id: &str, input: u64, output: u64) -> Value {
        record(
            "token_usage_record",
            json!({"thread_id": SESSION, "turn_id": "turn_1", "response_id": response_id,
                   "usage": {"input_tokens": input, "cached_input_tokens": 0,
                             "output_tokens": output, "total_tokens": input + output}}),
        )
    }

    fn token_count(input: u64, output: u64) -> Value {
        record(
            "event_msg",
            json!({"type": "token_count", "info": {"last_token_usage": {
                "input_tokens": input, "cached_input_tokens": 0, "output_tokens": output},
                "model_context_window": 258400}, "rate_limits": null}),
        )
    }

    fn ingest(records: Vec<Value>) -> Ingest {
        Ingest {
            session_id: SESSION.into(),
            model: Some("gpt-hint".into()),
            context_limit: Some(100_000),
            records,
        }
    }

    fn roles(event: &ConversationEvent) -> Vec<Role> {
        event.messages.iter().map(|m| m.role).collect()
    }

    #[test]
    fn paginated_responses_are_closed_by_usage_records() {
        let n = normalize(
            &ingest(vec![
                turn_context("gpt-6-astra"),
                task_started(258_400),
                user("Read notes.txt", Some(vec!["user.text"])),
                reply("msg_1", "I'll read it."),
                call("ctc_1", "call_1", "cat notes.txt"),
                usage_record("resp_1", 14_735, 84),
                token_count(14_735, 84),
                output("call_1", "port=8080"),
                reply("msg_2", "The port is 8080."),
                usage_record("resp_2", 22_711, 49),
                token_count(22_711, 49),
                record(
                    "event_msg",
                    json!({"type": "task_complete", "turn_id": "turn_1"}),
                ),
            ]),
            &Config::default(),
        );
        assert_eq!(n.malformed, 0);
        assert_eq!(n.events.len(), 2);
        let first = &n.events[0];
        assert_eq!(first.event_id, "resp_1");
        assert_eq!(first.message_id.as_deref(), Some("msg_1"));
        assert_eq!(first.conversation_id, SESSION);
        assert_eq!(first.conversation_id_source, IdSource::Session);
        assert_eq!(first.model, "gpt-6-astra");
        assert_eq!(first.kind, EventKind::Chat);
        assert!(first.messages_are_delta);
        assert!(first.starts_prompt);
        assert_eq!(first.prompt_tokens, Some(14_735));
        assert_eq!(first.completion_tokens, Some(84));
        assert_eq!(first.context_limit, Some(258_400));
        assert_eq!(roles(first), vec![Role::User]);
        assert_eq!(first.response_text.as_deref(), Some("I'll read it."));
        assert_eq!(first.tool_calls.len(), 1);
        assert_eq!(first.tool_calls[0].name, "exec");
        assert_eq!(first.tool_calls[0].id.as_deref(), Some("call_1"));
        assert_eq!(first.tool_calls[0].arguments, "cat notes.txt");
        let second = &n.events[1];
        assert_eq!(second.event_id, "resp_2");
        assert!(!second.starts_prompt);
        assert_eq!(roles(second), vec![Role::Assistant, Role::Tool]);
        assert_eq!(second.messages[0].tool_calls.len(), 1);
        assert_eq!(second.tool_results.len(), 1);
        assert_eq!(
            second.tool_results[0].tool_call_id.as_deref(),
            Some("call_1")
        );
        assert_eq!(second.tool_results[0].content, "port=8080");
        assert_eq!(second.response_text.as_deref(), Some("The port is 8080."));
    }

    #[test]
    fn legacy_history_closes_on_token_count_and_defers_outputs() {
        let n = normalize(
            &ingest(vec![
                turn_context("gpt-6-astra"),
                user("Read notes.txt", None),
                call("ctc_1", "call_1", "cat notes.txt"),
                output("call_1", "port=8080"),
                token_count(17_233, 93),
                reply("msg_2", "The port is 8080."),
                token_count(27_236, 58),
            ]),
            &Config::default(),
        );
        assert_eq!(n.events.len(), 2);
        assert_eq!(n.events[0].event_id, "ctc_1");
        assert_eq!(n.events[0].prompt_tokens, Some(17_233));
        assert_eq!(roles(&n.events[0]), vec![Role::User]);
        assert!(n.events[0].starts_prompt);
        assert_eq!(n.events[1].event_id, "msg_2");
        assert_eq!(roles(&n.events[1]), vec![Role::Assistant, Role::Tool]);
        assert_eq!(n.events[1].tool_results[0].content, "port=8080");
        assert_eq!(n.events[1].completion_tokens, Some(58));
    }

    #[test]
    fn an_open_response_waits_for_the_next_ship_and_keeps_its_ids() {
        let full = vec![
            turn_context("gpt-6-astra"),
            user("hi", None),
            reply("msg_1", "hello"),
            usage_record("resp_1", 10, 1),
            output("call_0", "late"),
            call("ctc_2", "call_2", "ls"),
            usage_record("resp_2", 20, 2),
            output("call_2", "a b"),
            reply("msg_3", "done"),
        ];
        let n = normalize(&ingest(full.clone()), &Config::default());
        assert_eq!(n.events.len(), 2, "msg_3 has no usage record yet");
        assert_eq!(n.malformed, 0);
        // The next ship starts at the last closed response, as the hook's cursor does.
        let again = normalize(&ingest(full[5..].to_vec()), &Config::default());
        assert_eq!(again.events.len(), 1);
        assert_eq!(again.events[0].event_id, n.events[1].event_id);
        assert_eq!(again.events[0].message_id, n.events[1].message_id);
        assert_eq!(again.events[0].message_id.as_deref(), Some("ctc_2"));
    }

    #[test]
    fn token_count_is_ignored_once_usage_records_appear() {
        let n = normalize(
            &ingest(vec![
                turn_context("gpt-6-astra"),
                reply("msg_1", "one"),
                usage_record("resp_1", 10, 1),
                call("ctc_2", "call_2", "ls"),
                token_count(10, 1), // paginated: the event for resp_1, written late
                usage_record("resp_2", 20, 2),
            ]),
            &Config::default(),
        );
        assert_eq!(n.events.len(), 2);
        assert_eq!(n.events[1].event_id, "resp_2");
        assert_eq!(n.events[1].prompt_tokens, Some(20));
    }

    #[test]
    fn model_and_limit_come_from_the_hints_then_the_transcript_then_config() {
        let n = normalize(
            &ingest(vec![reply("msg_1", "hi"), usage_record("resp_1", 1, 1)]),
            &Config::default(),
        );
        assert_eq!(n.events[0].model, "gpt-hint");
        assert_eq!(n.events[0].context_limit, Some(100_000));

        let n = normalize(
            &ingest(vec![
                turn_context("gpt-6-astra"),
                task_started(258_400),
                reply("msg_1", "hi"),
                usage_record("resp_1", 1, 1),
            ]),
            &Config::default(),
        );
        assert_eq!(n.events[0].model, "gpt-6-astra");
        assert_eq!(n.events[0].context_limit, Some(258_400));

        let mut config = Config::default();
        config.model_limits.insert("gpt-6-astra".into(), 1_000_000);
        let n = normalize(
            &ingest(vec![
                turn_context("gpt-6-astra"),
                task_started(258_400),
                reply("msg_1", "hi"),
                usage_record("resp_1", 1, 1),
            ]),
            &config,
        );
        assert_eq!(n.events[0].context_limit, Some(1_000_000));

        let n = normalize(
            &Ingest {
                session_id: SESSION.into(),
                model: None,
                context_limit: None,
                records: vec![
                    turn_context("gpt-6-astra"),
                    reply("msg_1", "hi"),
                    usage_record("resp_1", 1, 1),
                ],
            },
            &Config::default(),
        );
        assert_eq!(n.events[0].context_limit, None);
    }

    #[test]
    fn instructions_environment_and_compaction_are_system_text() {
        let n = normalize(
            &ingest(vec![
                developer("<skills_instructions>…</skills_instructions>"),
                user(
                    "<environment_context>cwd=/w</environment_context>",
                    Some(vec!["environment_context"]),
                ),
                record(
                    "compacted",
                    json!({"message": "Summary: the port was 8080.", "replacement_history": []}),
                ),
                reply("msg_1", "ok"),
                usage_record("resp_1", 1, 1),
                user("Which port?", Some(vec!["user.text"])),
                reply("msg_2", "8080"),
                usage_record("resp_2", 2, 2),
            ]),
            &Config::default(),
        );
        assert_eq!(n.events.len(), 2);
        assert_eq!(
            roles(&n.events[0]),
            vec![Role::System, Role::System, Role::System]
        );
        assert!(!n.events[0].starts_prompt, "no user prompt in the delta");
        assert_eq!(roles(&n.events[1]), vec![Role::Assistant, Role::User]);
        assert!(n.events[1].starts_prompt);
    }

    #[test]
    fn a_failed_turn_is_a_failure_that_keeps_the_pending_delta() {
        let n = normalize(
            &ingest(vec![
                turn_context("gpt-6-astra"),
                user("Summarize everything", None),
                record(
                    "event_msg",
                    json!({"type": "task_complete", "turn_id": "turn_1", "last_agent_message": null,
                           "error": {"message": "Codex ran out of room in the model's context window. Start a new thread or clear earlier history before retrying.",
                                     "codex_error_info": "context_window_exceeded"}}),
                ),
                reply("msg_1", "Here is a summary."),
                usage_record("resp_1", 5, 5),
            ]),
            &Config::default(),
        );
        assert_eq!(n.events.len(), 2);
        let failure = &n.events[0];
        assert_eq!(failure.kind, EventKind::Failure);
        assert_eq!(failure.event_id, "turn_1:failure");
        assert!(!failure.starts_prompt);
        assert_eq!(roles(failure), vec![Role::User]);
        let error = failure.error.as_deref().unwrap();
        assert!(error.contains("context window") && error.contains("context_window_exceeded"));
        assert_eq!(
            roles(&n.events[1]),
            vec![Role::User],
            "the retry still carries the prompt"
        );
        assert!(n.events[1].starts_prompt);
    }

    #[test]
    fn an_interrupted_reply_is_flushed_without_usage() {
        let n = normalize(
            &ingest(vec![
                turn_context("gpt-6-astra"),
                user("go", None),
                call("ctc_1", "call_1", "sleep 100"),
                output("call_1", "interrupted"),
                record(
                    "event_msg",
                    json!({"type": "turn_aborted", "turn_id": "turn_1", "reason": "interrupted"}),
                ),
                turn_context("gpt-6-astra"),
                user("again", None),
                reply("msg_2", "ok"),
                usage_record("resp_2", 3, 3),
            ]),
            &Config::default(),
        );
        assert_eq!(n.events.len(), 2);
        assert_eq!(n.events[0].event_id, "ctc_1");
        assert_eq!(n.events[0].prompt_tokens, None);
        assert_eq!(n.events[0].tool_calls.len(), 1);
        assert_eq!(
            roles(&n.events[1]),
            vec![Role::Assistant, Role::Tool, Role::User]
        );
        assert_eq!(
            n.events[1].tool_results[0].tool_call_id.as_deref(),
            Some("call_1")
        );
    }

    #[test]
    fn other_tool_kinds_are_calls_with_their_arguments() {
        let n = normalize(
            &ingest(vec![
                record(
                    "response_item",
                    json!({"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "wait",
                           "arguments": "{\"cell_id\":\"15\"}"}),
                ),
                record(
                    "response_item",
                    json!({"type": "web_search_call", "id": "ws_1", "status": "completed",
                           "action": {"type": "search", "query": "codex hooks"}}),
                ),
                usage_record("resp_1", 1, 1),
                record(
                    "response_item",
                    json!({"type": "function_call_output", "id": "fco_1", "call_id": "call_1", "output": "{\"total\":1}"}),
                ),
                reply("msg_2", "done"),
                usage_record("resp_2", 2, 2),
            ]),
            &Config::default(),
        );
        let calls = &n.events[0].tool_calls;
        assert_eq!(calls.len(), 2);
        assert_eq!(
            (calls[0].name.as_str(), calls[0].arguments.as_str()),
            ("wait", "{\"cell_id\":\"15\"}")
        );
        assert_eq!(calls[1].name, "web_search");
        assert!(calls[1].arguments.contains("codex hooks"));
        assert_eq!(n.events[1].tool_results[0].content, "{\"total\":1}");
    }

    #[test]
    fn unusable_records_are_counted_not_fatal() {
        let n = normalize(
            &Ingest {
                session_id: SESSION.into(),
                model: None,
                context_limit: None,
                records: vec![
                    json!("not an object"),
                    record("response_item", json!({"id": "no type"})),
                    reply("msg_0", "no model anywhere"),
                    usage_record("resp_0", 1, 1),
                    turn_context("gpt-6-astra"),
                    reply("msg_1", "fine"),
                    usage_record("resp_1", 1, 1),
                    record("world_state", json!({"full": true})),
                    record(
                        "response_item",
                        json!({"type": "reasoning", "encrypted_content": "gAAA"}),
                    ),
                ],
            },
            &Config::default(),
        );
        assert_eq!(n.malformed, 2);
        assert_eq!(n.events.len(), 2);
        assert_eq!(n.events[0].model, "unknown");
        assert_eq!(n.events[1].event_id, "resp_1");
    }

    #[test]
    fn a_response_without_a_known_model_keeps_its_tool_outputs_with_their_call() {
        // The hook re-ships from a closed response, after that turn's
        // turn_context; when the hook event carries no model either, the
        // response must still be emitted or its outputs turn into orphans.
        let n = normalize(
            &Ingest {
                session_id: SESSION.into(),
                model: None,
                context_limit: None,
                records: vec![
                    user("Read notes.txt", None),
                    reply("msg_1", "I'll read it."),
                    call("ctc_1", "call_1", "cat notes.txt"),
                    usage_record("resp_1", 10, 1),
                    output("call_1", "port=8080"),
                    turn_context("gpt-6-astra"),
                    user("thanks", None),
                    reply("msg_2", "sure"),
                    usage_record("resp_2", 20, 2),
                ],
            },
            &Config::default(),
        );
        assert_eq!(n.malformed, 0);
        assert_eq!(n.events.len(), 2);
        assert_eq!(n.events[0].model, "unknown");
        assert_eq!(n.events[1].model, "gpt-6-astra");
        assert_eq!(
            roles(&n.events[1]),
            vec![Role::Assistant, Role::Tool, Role::User]
        );
        assert_eq!(
            n.events[1].messages[0].tool_calls[0].id.as_deref(),
            Some("call_1")
        );
    }

    #[test]
    fn a_usage_record_without_usage_still_closes_the_response() {
        let n = normalize(
            &ingest(vec![
                turn_context("gpt-6-astra"),
                reply("msg_1", "one"),
                record(
                    "token_usage_record",
                    json!({"response_id": "resp_1", "usage": null}),
                ),
                reply("msg_2", "two"),
                usage_record("resp_2", 20, 2),
            ]),
            &Config::default(),
        );
        assert_eq!(n.events.len(), 2);
        assert_eq!(n.events[0].event_id, "resp_1");
        assert_eq!(n.events[0].prompt_tokens, None);
        assert_eq!(n.events[1].event_id, "resp_2");
        assert_eq!(n.events[1].response_text.as_deref(), Some("two"));
    }

    #[test]
    fn an_empty_item_id_falls_back_to_the_call_id() {
        assert_eq!(
            item_id(&json!({"id": "", "call_id": "call_x"})).as_deref(),
            Some("call_x")
        );
        assert_eq!(item_id(&json!({"id": "", "call_id": ""})), None);
        let n = normalize(
            &ingest(vec![
                turn_context("gpt-6-astra"),
                record(
                    "response_item",
                    json!({"type": "function_call", "id": "", "call_id": "call_x", "name": "wait", "arguments": "{}"}),
                ),
                token_count(1, 1),
            ]),
            &Config::default(),
        );
        assert_eq!(n.events[0].message_id.as_deref(), Some("call_x"));
    }

    #[test]
    fn parse_body_requires_the_envelope() {
        let ok = parse_body(
            br#"{"session_id": "s1", "model": "gpt-6-astra", "context_limit": 258400, "records": [{"type": "turn_context", "payload": {}}]}"#,
        )
        .unwrap();
        assert_eq!(ok.session_id, "s1");
        assert_eq!(ok.model.as_deref(), Some("gpt-6-astra"));
        assert_eq!(ok.context_limit, Some(258_400));
        assert_eq!(ok.records.len(), 1);
        let bare = parse_body(br#"{"session_id": "s1", "records": []}"#).unwrap();
        assert_eq!((bare.model, bare.context_limit), (None, None));
        assert!(parse_body(br#"{"records": []}"#).is_err());
        assert!(parse_body(br#"{"session_id": "s1"}"#).is_err());
        assert!(parse_body(br#"[{"type": "turn_context"}]"#).is_err());
        assert!(parse_body(b"nope").is_err());
    }
}
