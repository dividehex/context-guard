//! Parsing opencode session messages (the `{info, parts}` pairs the SDK's
//! `session.messages()` returns).
//!
//! The TUI plugin in `opencode/` ships a session's messages incrementally,
//! together with what the messages do not carry: the session id and the
//! context limit, which the plugin resolves from the model's provider entry
//! (`provider.models[modelID].limit.context`). One opencode message is
//! already one complete turn: the tool parts carry both the call and its
//! result (`state` transitions pending → running → completed/error), so a
//! message needs no further grouping, the way LiteLLM payloads and Codex
//! rollouts do. The records between two completions are the request delta:
//! the previous reply with its tool calls, its tool results, and the user's
//! next prompt.
//!
//! Events are deduplicated by the assistant message id, which is unique per
//! turn. The plugin re-ships from the last shipped message so the previous
//! reply and its results reconstruct in the next request: an event that the
//! monitor later finds to be a duplicate was still normalized first, so the
//! pending delta is never lost. Redelivery is otherwise harmless.
//!
//! opencode documents nothing about this format, so every field is optional
//! and unknown record kinds and parts are ignored rather than rejected.
//! Verified against opencode's own SDK types.

use chrono::{DateTime, TimeZone, Utc};
use serde::Deserialize;
use serde_json::Value;

use super::event::{Message, Role, ToolCall};
use super::litellm::{split_body, ParseError};
pub use super::Normalized;
use super::{delta_completion, delta_failure, DeltaFailure, DeltaReply};
use crate::config::Config;

/// One ingest body: opencode messages plus the session id (records do not
/// carry it) and the context limit the plugin resolved from the provider.
#[derive(Debug, Clone, Default)]
pub struct Ingest {
    pub session_id: String,
    pub context_limit: Option<u64>,
    pub records: Vec<Value>,
}

/// Accepts `{"session_id": "...", "context_limit": N, "records": [...]}`.
/// The envelope is required because opencode messages do not name their
/// session.
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
        context_limit: envelope.get("context_limit").and_then(Value::as_u64),
        records,
    })
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct Time {
    created: Option<i64>,
    completed: Option<i64>,
}

impl Time {
    fn started_at(&self) -> Option<DateTime<Utc>> {
        self.created.and_then(to_utc)
    }

    fn timestamp(&self) -> Option<DateTime<Utc>> {
        self.completed
            .and_then(to_utc)
            .or_else(|| self.started_at())
    }
}

fn to_utc(ms: i64) -> Option<DateTime<Utc>> {
    Utc.timestamp_millis_opt(ms).single()
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct Cache {
    read: Option<u64>,
    write: Option<u64>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct Tokens {
    input: Option<u64>,
    output: Option<u64>,
    cache: Option<Cache>,
}

impl Tokens {
    /// `input + cache.read + cache.write`: what the model actually held, the
    /// same reading as the Claude Code and Codex normalizers.
    fn prompt(&self) -> Option<u64> {
        let mut sum = 0u64;
        let mut any = false;
        if let Some(n) = self.input {
            sum += n;
            any = true;
        }
        if let Some(c) = &self.cache {
            if let Some(n) = c.read {
                sum += n;
                any = true;
            }
            if let Some(n) = c.write {
                sum += n;
                any = true;
            }
        }
        any.then_some(sum)
    }
}

/// The `info` half of an opencode message; every field is optional. The SDK
/// names its fields in camel case (`modelID`, not `model_id`); only the ones
/// spelled with an acronym differ from serde's own camelCase, so they are
/// renamed explicitly. `summary` only means compaction when it is a boolean
/// `true` (a model-written assistant summary); user messages carry a `summary`
/// object (`{title, body, diffs}`) that must not break deserialization.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct Info {
    id: Option<String>,
    role: Option<String>,
    #[serde(rename = "modelID")]
    model_id: Option<String>,
    summary: Option<Value>,
    time: Option<Time>,
    tokens: Option<Tokens>,
    error: Option<Value>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct Record {
    info: Option<Info>,
    parts: Option<Vec<Value>>,
}

impl Info {
    fn model(&self) -> String {
        self.model_id
            .as_deref()
            .filter(|s| !s.is_empty())
            .unwrap_or("unknown")
            .to_string()
    }

    fn message_id(&self) -> Option<String> {
        self.id
            .as_deref()
            .filter(|s| !s.is_empty())
            .map(String::from)
    }
}

struct Builder<'a> {
    config: &'a Config,
    session_id: &'a str,
    context_limit: Option<u64>,
    out: Normalized,
    /// Messages added to the request since the last completion was emitted:
    /// the previous reply with its calls, its tool results, the next prompt.
    pending: Vec<Message>,
}

impl Builder<'_> {
    fn record(&mut self, rec: &Record) {
        let Some(info) = &rec.info else {
            self.out.malformed += 1;
            return;
        };
        let parts = rec.parts.as_deref().unwrap_or(&[]);
        if matches!(info.summary, Some(Value::Bool(true))) {
            // A compaction summary is a model-written assistant message
            // (`summary: true`); it rewrites history and is never a claim.
            return;
        }
        match info.role.as_deref() {
            Some("user") => self.user(parts),
            Some("assistant") => {
                if info.error.is_some() {
                    self.failure(info);
                } else {
                    self.completion(info, parts);
                }
            }
            _ => {} // system and other record kinds are not shipped
        }
    }

    fn user(&mut self, parts: &[Value]) {
        let mut user_text = Vec::new();
        let mut synthetic = Vec::new();
        for part in parts {
            if part.get("type").and_then(Value::as_str) != Some("text") {
                continue;
            }
            let text = part.get("text").and_then(Value::as_str).unwrap_or("");
            if text.is_empty() {
                continue;
            }
            if part.get("ignored").and_then(Value::as_bool) == Some(true) {
                continue;
            }
            if part.get("synthetic").and_then(Value::as_bool) == Some(true) {
                synthetic.push(text);
            } else {
                user_text.push(text);
            }
        }
        if !synthetic.is_empty() {
            self.push(Role::System, synthetic.join("\n"));
        }
        if !user_text.is_empty() {
            self.push(Role::User, user_text.join("\n"));
        }
    }

    fn completion(&mut self, info: &Info, parts: &[Value]) {
        let mut text = Vec::new();
        let mut calls = Vec::new();
        let mut results = Vec::new();
        for part in parts {
            let kind = part.get("type").and_then(Value::as_str).unwrap_or("");
            match kind {
                "text" => {
                    let t = part.get("text").and_then(Value::as_str).unwrap_or("");
                    if !t.is_empty() && part.get("ignored").and_then(Value::as_bool) != Some(true) {
                        text.push(t.to_string());
                    }
                }
                "tool" => {
                    let call_id = part
                        .get("callID")
                        .and_then(Value::as_str)
                        .filter(|s| !s.is_empty());
                    let name = part
                        .get("tool")
                        .and_then(Value::as_str)
                        .unwrap_or("tool")
                        .to_string();
                    let arguments = part
                        .get("state")
                        .and_then(|s| s.get("input"))
                        .map(value_text)
                        .unwrap_or_default();
                    calls.push(ToolCall {
                        id: call_id.map(String::from),
                        name,
                        arguments,
                    });
                    if let Some(state) = part.get("state") {
                        let content = match state.get("status").and_then(Value::as_str) {
                            Some("completed") => state
                                .get("output")
                                .and_then(Value::as_str)
                                .filter(|s| !s.is_empty())
                                .map(str::to_string),
                            Some("error") => state.get("error").and_then(Value::as_str).map(|s| {
                                let s = s.trim();
                                if s.is_empty() {
                                    s.to_string()
                                } else {
                                    format!("error: {s}")
                                }
                            }),
                            _ => None, // pending/running: the plugin does not ship these
                        };
                        if let Some(content) = content {
                            results.push(Message {
                                role: Role::Tool,
                                content,
                                tool_call_id: call_id.map(String::from),
                                tool_calls: Vec::new(),
                            });
                        }
                    }
                }
                _ => {} // reasoning, file, step, snapshot, patch, agent, subtask, retry, compaction
            }
        }
        if text.is_empty() && calls.is_empty() {
            return; // step-only and other empty messages carry no turn
        }
        let Some(event_id) = info.message_id() else {
            self.out.malformed += 1;
            return;
        };
        let model = info.model();
        let time = info.time.as_ref();
        let mut deferred = Vec::new();
        let event = delta_completion(
            &mut self.pending,
            &mut deferred,
            DeltaReply {
                event_id,
                conversation_id: self.session_id.to_string(),
                message_id: info.message_id(),
                context_limit: self.config.model_limit(&model).or(self.context_limit),
                model,
                started_at: time.and_then(Time::started_at),
                timestamp: time.and_then(Time::timestamp),
                prompt_tokens: info.tokens.as_ref().and_then(Tokens::prompt),
                completion_tokens: info.tokens.as_ref().and_then(|t| t.output),
                text,
                tool_calls: calls,
            },
        );
        self.out.events.push(event);
        // The results belong to the next request, after the reply they
        // answer, exactly where the monitor looks for its call.
        self.pending.extend(results);
    }

    /// A turn that ended in an error is a failed completion of the pending
    /// request; the user will retry it, so the pending delta is kept.
    fn failure(&mut self, info: &Info) {
        let Some(message_id) = info.message_id() else {
            self.out.malformed += 1;
            return;
        };
        let model = info.model();
        let error = info.error.as_ref().map(error_text).unwrap_or_default();
        let event = delta_failure(
            &self.pending,
            DeltaFailure {
                event_id: format!("{message_id}:failure"),
                conversation_id: self.session_id.to_string(),
                message_id: Some(format!("{message_id}:failure")),
                context_limit: self.config.model_limit(&model).or(self.context_limit),
                model,
                timestamp: info
                    .time
                    .as_ref()
                    .and_then(Time::timestamp)
                    .unwrap_or_else(Utc::now),
                error,
            },
        );
        self.out.events.push(event);
    }

    fn push(&mut self, role: Role, content: String) {
        self.pending.push(Message {
            role,
            content,
            tool_call_id: None,
            tool_calls: Vec::new(),
        });
    }
}

/// The argument object of a tool call as text, for the monitor's call
/// canonicalization.
fn value_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        other => serde_json::to_string(other).unwrap_or_default(),
    }
}

/// `name` + `data.message` (+ the HTTP status when present) of a message
/// error, so text like "maximum context length" reaches the overflow matcher.
fn error_text(error: &Value) -> String {
    let mut parts: Vec<String> = Vec::new();
    if let Some(name) = error.get("name").and_then(Value::as_str) {
        parts.push(name.to_string());
    }
    let data = error.get("data");
    let message = data.and_then(|d| d.get("message")).and_then(Value::as_str);
    let status = data
        .and_then(|d| d.get("statusCode"))
        .and_then(Value::as_u64);
    if let (Some(message), Some(status)) = (message, status) {
        parts.push(format!("{status} {message}"));
    } else if let Some(message) = message {
        parts.push(message.to_string());
    }
    parts.join(" ")
}

/// Normalize one shipped slice of a session's messages into events, in file
/// order. A record that cannot be read is counted as malformed; the batch
/// continues.
pub fn normalize(ingest: &Ingest, config: &Config) -> Normalized {
    let mut b = Builder {
        config,
        session_id: &ingest.session_id,
        context_limit: ingest.context_limit,
        out: Normalized::default(),
        pending: Vec::new(),
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

    const SESSION: &str = "sess_openai_session_01a1f2";
    const CREATED: i64 = 1_757_600_000_000;
    const COMPLETED: i64 = 1_757_600_000_500;

    fn ts() -> Value {
        json!({"created": CREATED, "completed": COMPLETED})
    }

    fn info(role: &str, id: &str) -> Value {
        json!({
            "id": id, "sessionID": SESSION, "role": role,
            "parentID": "parent", "time": ts(),
        })
    }

    fn user_info(id: &str) -> Value {
        json!({
            "id": id, "sessionID": SESSION, "role": "user",
            "parentID": "parent", "time": {"created": CREATED},
        })
    }

    fn assistant(id: &str) -> Value {
        info("assistant", id)
    }

    fn text_part(text: &str) -> Value {
        json!({"id": "p1", "sessionID": SESSION, "messageID": "m1", "type": "text", "text": text})
    }

    fn synthetic_part(text: &str) -> Value {
        json!({"id": "p_s", "sessionID": SESSION, "messageID": "m1", "type": "text",
               "text": text, "synthetic": true})
    }

    fn tool_part(call_id: &str, tool: &str, input: Value, status: &str, result: Value) -> Value {
        let mut state = json!({"status": status, "input": input});
        if status == "completed" {
            state["output"] = result;
        } else if status == "error" {
            state["error"] = result;
        }
        json!({"id": "t1", "sessionID": SESSION, "messageID": "m1", "type": "tool",
               "callID": call_id, "tool": tool, "state": state})
    }

    fn heading(part: &str) -> Value {
        json!({"id": "p_h", "sessionID": SESSION, "messageID": "m1", "type": part})
    }

    fn msg(info: Value, parts: Vec<Value>) -> Value {
        json!({"info": info, "parts": parts})
    }

    fn toked(
        role: &str,
        id: &str,
        input: u64,
        output: u64,
        cache_read: u64,
        cache_write: u64,
    ) -> Value {
        let mut v = info(role, id);
        v["tokens"] = json!({"input": input, "output": output, "reasoning": 0,
                             "cache": {"read": cache_read, "write": cache_write}});
        v["modelID"] = json!("claude-sonnet-4-5");
        v["provider_id"] = json!("anthropic");
        v
    }

    fn ingest(records: Vec<Value>) -> Ingest {
        Ingest {
            session_id: SESSION.into(),
            context_limit: Some(200_000),
            records,
        }
    }

    fn roles(event: &ConversationEvent) -> Vec<Role> {
        event.messages.iter().map(|m| m.role).collect()
    }

    #[test]
    fn one_message_is_one_turn() {
        let n = normalize(
            &ingest(vec![
                msg(user_info("msg_u"), vec![text_part("Read notes.txt")]),
                msg(
                    toked("assistant", "msg_1", 10_000, 120, 2_000, 0),
                    vec![text_part("The port of llama.cpp is 8080.")],
                ),
            ]),
            &Config::default(),
        );
        assert_eq!(n.malformed, 0);
        assert_eq!(n.events.len(), 1);
        let e = &n.events[0];
        assert_eq!(e.event_id, "msg_1");
        assert_eq!(e.message_id.as_deref(), Some("msg_1"));
        assert_eq!(e.conversation_id, SESSION);
        assert_eq!(e.conversation_id_source, IdSource::Session);
        assert_eq!(e.model, "claude-sonnet-4-5");
        assert_eq!(e.kind, EventKind::Chat);
        assert!(e.messages_are_delta);
        assert!(e.starts_prompt);
        assert_eq!(
            e.prompt_tokens,
            Some(12_000),
            "input + cache.read + cache.write"
        );
        assert_eq!(e.completion_tokens, Some(120));
        assert_eq!(e.context_limit, Some(200_000));
        assert_eq!(roles(e), vec![Role::User]);
        assert_eq!(
            e.response_text.as_deref(),
            Some("The port of llama.cpp is 8080.")
        );
        assert_eq!(e.tool_calls.len(), 0);
    }

    #[test]
    fn a_message_with_facts_is_the_source_of_truth_and_the_reply_a_claim() {
        let n = normalize(
            &ingest(vec![
                msg(
                    user_info("msg_u"),
                    vec![text_part("llama.cpp is at 0.0.0.0:8080.")],
                ),
                msg(assistant("msg_a"), vec![text_part("I'll look.")]),
                msg(
                    user_info("msg_u2"),
                    vec![text_part("What port is llama.cpp on?")],
                ),
                msg(assistant("msg_b"), vec![text_part("port 8000.")]),
            ]),
            &Config::default(),
        );
        assert_eq!(n.events.len(), 2);
        let first = &n.events[0];
        assert_eq!(first.event_id, "msg_a");
        assert!(first.starts_prompt);
        assert_eq!(roles(first), vec![Role::User]);
        let second = &n.events[1];
        assert_eq!(second.event_id, "msg_b");
        assert!(
            second.starts_prompt,
            "the user question is part of this request"
        );
        assert_eq!(
            roles(second),
            vec![Role::Assistant, Role::User],
            "the previous reply opens the next request"
        );
        assert_eq!(second.message_id.as_deref(), Some("msg_b"));
        assert_eq!(second.response_text.as_deref(), Some("port 8000."));
    }

    #[test]
    fn tool_calls_and_results_are_kept_together_after_the_reply() {
        let n = normalize(
            &ingest(vec![
                msg(user_info("msg_u"), vec![text_part("List the dir")]),
                msg(
                    assistant("msg_1"),
                    vec![
                        text_part("On it."),
                        tool_part(
                            "call_1",
                            "bash",
                            json!({"command": "ls"}),
                            "completed",
                            json!("notes.txt"),
                        ),
                    ],
                ),
                msg(user_info("msg_u2"), vec![text_part("thanks")]),
                msg(assistant("msg_2"), vec![text_part("done")]),
            ]),
            &Config::default(),
        );
        assert_eq!(n.malformed, 0);
        assert_eq!(n.events.len(), 2);
        let first = &n.events[0];
        assert_eq!(first.event_id, "msg_1");
        assert_eq!(first.tool_calls.len(), 1);
        assert_eq!(first.tool_calls[0].name, "bash");
        assert_eq!(first.tool_calls[0].id.as_deref(), Some("call_1"));
        assert_eq!(first.tool_calls[0].arguments, "{\"command\":\"ls\"}");
        // The result travels with the reply into the next request: the call
        // it answers is in the same message list, so it is not an orphan.
        let second = &n.events[1];
        assert_eq!(
            roles(second),
            vec![Role::Assistant, Role::Tool, Role::User],
            "{second:?}"
        );
        assert_eq!(second.messages[0].tool_calls.len(), 1);
        assert_eq!(second.tool_results.len(), 1);
        assert_eq!(
            second.tool_results[0].tool_call_id.as_deref(),
            Some("call_1")
        );
        assert_eq!(second.tool_results[0].content, "notes.txt");
    }

    #[test]
    fn a_tool_error_is_a_result_the_monitor_can_match() {
        let n = normalize(
            &ingest(vec![
                msg(
                    assistant("msg_1"),
                    vec![tool_part(
                        "call_1",
                        "start",
                        json!({"app": "web"}),
                        "error",
                        json!("port 8080 is taken"),
                    )],
                ),
                msg(user_info("msg_u2"), vec![text_part("what happened?")]),
                msg(assistant("msg_2"), vec![text_part("port 8080 failed.")]),
            ]),
            &Config::default(),
        );
        assert_eq!(n.events.len(), 2);
        assert_eq!(n.events[0].tool_calls.len(), 1);
        let second = &n.events[1];
        assert_eq!(second.tool_results.len(), 1);
        assert_eq!(second.tool_results[0].content, "error: port 8080 is taken");
    }

    #[test]
    fn synthetic_text_is_system_and_ignored_parts_stay_home() {
        let n = normalize(
            &ingest(vec![
                msg(
                    user_info("msg_u"),
                    vec![
                        synthetic_part("System reminder: work in /w"),
                        text_part("hi"),
                    ],
                ),
                msg(assistant("msg_1"), vec![text_part("hello there")]),
            ]),
            &Config::default(),
        );
        assert_eq!(n.events.len(), 1);
        let e = &n.events[0];
        assert_eq!(
            roles(e),
            vec![Role::System, Role::User],
            "synthetic text never becomes a fact source"
        );
        assert!(e.starts_prompt);
    }

    #[test]
    fn reasoning_step_file_snapshot_patch_and_agent_parts_are_ignored() {
        let n = normalize(
            &ingest(vec![msg(
                assistant("msg_1"),
                vec![
                    heading("step-start"),
                    heading("snapshot"),
                    heading("patch"),
                    heading("agent"),
                    heading("subtask"),
                    heading("retry"),
                    heading("reasoning"),
                    json!({"id": "p_f", "sessionID": SESSION, "messageID": "m1",
                               "type": "file", "mime": "text/plain", "url": "file:///etc/hosts"}),
                    text_part("the answer"),
                ],
            )]),
            &Config::default(),
        );
        assert_eq!(n.malformed, 0);
        assert_eq!(n.events.len(), 1);
        assert_eq!(n.events[0].response_text.as_deref(), Some("the answer"));
        assert_eq!(n.events[0].tool_calls.len(), 0);
    }

    #[test]
    fn an_empty_assistant_message_carries_no_turn() {
        let n = normalize(
            &ingest(vec![
                msg(user_info("msg_u"), vec![text_part("go")]),
                msg(
                    assistant("msg_1"),
                    vec![heading("step-finish"), heading("reasoning")],
                ),
            ]),
            &Config::default(),
        );
        assert_eq!(n.events.len(), 0, "no text and no tool parts: skipped");
    }

    #[test]
    fn an_error_is_a_failure_that_keeps_the_pending_delta() {
        let mut failed = assistant("msg_1");
        failed["error"] = json!({"name": "APIError",
                                 "data": {"message": "This model's maximum context length is 8192 tokens but you requested 20,000 tokens",
                                          "statusCode": 400, "isRetryable": false}});
        let n = normalize(
            &ingest(vec![
                msg(user_info("msg_u"), vec![text_part("carry on")]),
                msg(failed, vec![]),
            ]),
            &Config::default(),
        );
        assert_eq!(n.events.len(), 1);
        let failure = &n.events[0];
        assert_eq!(failure.kind, EventKind::Failure);
        assert_eq!(failure.event_id, "msg_1:failure");
        assert_eq!(failure.message_id.as_deref(), Some("msg_1:failure"));
        assert!(!failure.starts_prompt);
        assert_eq!(roles(failure), vec![Role::User]);
        let error = failure.error.as_deref().unwrap();
        assert!(
            error.contains("APIError")
                && error.contains("400")
                && error.contains("maximum context length"),
            "{error}"
        );
    }

    #[test]
    fn an_error_without_status_is_still_a_failure_text() {
        let mut failed = assistant("msg_1");
        failed["error"] = json!({"name": "UnknownError", "data": {"message": "ran out of room in the model's context window"}});
        let n = normalize(&ingest(vec![msg(failed, vec![])]), &Config::default());
        assert_eq!(n.events.len(), 1);
        let error = n.events[0].error.as_deref().unwrap();
        assert!(
            error.contains("UnknownError") && error.contains("context window"),
            "{error}"
        );
    }

    #[test]
    fn a_compaction_summary_is_never_a_claim_or_a_turn() {
        let mut summary = assistant("msg_c");
        summary["summary"] = json!(true);
        let n = normalize(
            &ingest(vec![
                msg(user_info("msg_u"), vec![text_part("long conversation")]),
                msg(summary, vec![text_part("Summary: the port was 8080.")]),
                msg(user_info("msg_u2"), vec![text_part("thanks")]),
                msg(assistant("msg_2"), vec![text_part("Sure thing.")]),
            ]),
            &Config::default(),
        );
        assert_eq!(n.events.len(), 1);
        assert_eq!(n.events[0].message_id.as_deref(), Some("msg_2"));
        assert!(n.events[0].starts_prompt);
        // The summary text never appeared in the request of any event.
        assert!(
            n.events
                .iter()
                .all(|e| e.messages.iter().all(|m| !m.content.contains("Summary"))),
            "summaries rewrote nothing"
        );
    }

    #[test]
    fn a_real_user_message_with_a_summary_object_is_not_malformed() {
        // The SDK projects `summary: {title?, body?, diffs}` on user messages;
        // only a boolean `true` on an assistant message means compaction.
        let mut user = user_info("msg_u");
        user["summary"] = json!({"diffs": []});
        user["agent"] = json!("build");
        user["model"] = json!({"providerID": "opencode", "modelID": "big-pickle"});
        let n = normalize(
            &ingest(vec![
                msg(user, vec![text_part("llama.cpp is on port 8080.")]),
                msg(
                    toked("assistant", "msg_1", 10_000, 120, 2_000, 0),
                    vec![text_part("noted.")],
                ),
            ]),
            &Config::default(),
        );
        assert_eq!(n.malformed, 0);
        let e = &n.events[0];
        assert_eq!(roles(e), vec![Role::User]);
        assert_eq!(e.messages[0].content, "llama.cpp is on port 8080.");
    }

    #[test]
    fn model_and_limit_come_from_the_message_then_config() {
        let mut config = Config::default();
        config
            .model_limits
            .insert("claude-sonnet-4-5".into(), 1_000_000);
        let n = normalize(
            &ingest(vec![msg(
                toked("assistant", "msg_1", 1, 1, 0, 0),
                vec![text_part("hi")],
            )]),
            &config,
        );
        assert_eq!(n.events[0].context_limit, Some(1_000_000));
        let mut bare = assistant("msg_2");
        bare["modelID"] = json!("x");
        let bare = normalize(
            &Ingest {
                session_id: SESSION.into(),
                context_limit: None,
                records: vec![msg(bare, vec![text_part("hi")])],
            },
            &Config::default(),
        );
        assert_eq!(bare.events[0].model, "x");
        assert_eq!(bare.events[0].context_limit, None);
    }

    #[test]
    fn unknown_and_missing_fields_are_tolerated() {
        let n = normalize(
            &ingest(vec![
                json!("not an object"),
                json!({"info": "not an object too"}),
                msg(
                    json!({"id": "msg_x", "role": "assistant", "time": {"created": CREATED}}),
                    vec![text_part("no model anywhere")],
                ),
                msg(assistant("msg_y"), vec![text_part("fine")]),
            ]),
            &Config::default(),
        );
        assert_eq!(n.malformed, 2);
        assert_eq!(n.events.len(), 2);
        assert_eq!(n.events[0].model, "unknown");
        assert_eq!(n.events[1].message_id.as_deref(), Some("msg_y"));
    }

    #[test]
    fn ms_timestamps_become_rfc3339_in_repo_order() {
        let mut early = assistant("msg_1");
        early["time"] = json!({"created": 0, "completed": 1_000});
        let n = normalize(
            &ingest(vec![msg(early, vec![text_part("hi")])]),
            &Config::default(),
        );
        let e = &n.events[0];
        assert_eq!(e.started_at.timestamp_millis(), 0);
        assert_eq!(e.timestamp.timestamp_millis(), 1_000);
    }

    #[test]
    fn parse_body_requires_the_envelope() {
        let ok = parse_body(
            br#"{"session_id": "s1", "context_limit": 200000, "records": [{"info": {"id": "m", "role": "assistant"}}]}"#,
        )
        .unwrap();
        assert_eq!(ok.session_id, "s1");
        assert_eq!(ok.context_limit, Some(200_000));
        assert_eq!(ok.records.len(), 1);
        let bare = parse_body(br#"{"session_id": "s1", "records": []}"#).unwrap();
        assert_eq!((bare.context_limit, bare.records.len()), (None, 0));
        assert!(parse_body(br#"{"records": []}"#).is_err());
        assert!(parse_body(br#"{"session_id": "s1"}"#).is_err());
        assert!(parse_body(br#"[{"info": {}}]"#).is_err());
        assert!(parse_body(b"nope").is_err());
    }
}
