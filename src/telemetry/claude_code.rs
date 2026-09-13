//! Parsing Claude Code session transcripts (`~/.claude/projects/<cwd>/<session>.jsonl`).
//!
//! The hook in `claude-code/context_guard_hook.py` ships the transcript's
//! `user`, `assistant` and `system` records incrementally. Each API call the
//! CLI made is a run of `assistant` records sharing one `requestId`; that run
//! is one completion, hence one turn. The records between two completions are
//! the request delta: the previous reply (with its tool calls), the tool
//! results, and the user's next prompt.
//!
//! Claude Code documents this format as internal, so everything is optional
//! and unknown records are ignored rather than rejected. Verified against
//! Claude Code 2.1.270.

use chrono::{DateTime, Utc};
use serde::Deserialize;
use serde_json::Value;

use super::content_text;
use super::event::{ConversationEvent, EventKind, IdSource, Message, Role, ToolCall, ToolResult};
use super::litellm::{split_body, ParseError};
use crate::config::Config;

/// One ingest body: transcript records plus what the shipper knows that the
/// records do not carry (the context window Claude Code reports to its status line).
#[derive(Debug, Clone, Default)]
pub struct Ingest {
    pub records: Vec<Value>,
    pub context_limit: Option<u64>,
}

/// The events one ingest body produced, and how many records were unusable.
#[derive(Debug, Default)]
pub struct Normalized {
    pub events: Vec<ConversationEvent>,
    pub malformed: usize,
}

/// Accepts `{"records": [...], "context_limit": N}` or a bare array / NDJSON of records.
pub fn parse_body(body: &[u8]) -> Result<Ingest, ParseError> {
    let mut items = split_body(body)?;
    if items.len() == 1 && items[0].get("records").is_some_and(Value::is_array) {
        let mut envelope = items.pop().unwrap_or(Value::Null);
        let records = match envelope.get_mut("records").map(Value::take) {
            Some(Value::Array(records)) => records,
            _ => Vec::new(),
        };
        let context_limit = envelope.get("context_limit").and_then(Value::as_u64);
        return Ok(Ingest {
            records,
            context_limit,
        });
    }
    Ok(Ingest {
        records: items,
        context_limit: None,
    })
}

#[derive(Debug, Default, Deserialize)]
#[serde(default, rename_all = "camelCase")]
struct Record {
    #[serde(rename = "type")]
    kind: String,
    uuid: Option<String>,
    session_id: Option<String>,
    request_id: Option<String>,
    timestamp: Option<String>,
    is_sidechain: Option<bool>,
    is_compact_summary: Option<bool>,
    is_api_error_message: Option<bool>,
    message: Value,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct Usage {
    input_tokens: Option<u64>,
    cache_read_input_tokens: Option<u64>,
    cache_creation_input_tokens: Option<u64>,
    output_tokens: Option<u64>,
}

/// The assistant records of one API call, accumulated until the next record
/// belongs to a different request.
struct Group {
    request_id: String,
    first_uuid: Option<String>,
    started_at: Option<DateTime<Utc>>,
    timestamp: Option<DateTime<Utc>>,
    session_id: Option<String>,
    model: Option<String>,
    usage: Usage,
    text: Vec<String>,
    tool_calls: Vec<ToolCall>,
}

impl Group {
    fn new(request_id: String) -> Group {
        Group {
            request_id,
            first_uuid: None,
            started_at: None,
            timestamp: None,
            session_id: None,
            model: None,
            usage: Usage::default(),
            text: Vec::new(),
            tool_calls: Vec::new(),
        }
    }

    fn absorb(&mut self, rec: &Record) {
        let ts = rec.timestamp.as_deref().and_then(parse_time);
        if self.first_uuid.is_none() {
            self.first_uuid = rec.uuid.clone();
            self.started_at = ts;
        }
        self.timestamp = ts.or(self.timestamp);
        self.session_id = self.session_id.take().or_else(|| rec.session_id.clone());
        self.model = rec
            .message
            .get("model")
            .and_then(Value::as_str)
            .map(String::from)
            .or_else(|| self.model.take());
        // Every record of a request repeats the same usage; the last one wins.
        if let Some(usage) = rec.message.get("usage") {
            self.usage = serde_json::from_value(usage.clone()).unwrap_or_default();
        }
        for block in rec
            .message
            .get("content")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            match block.get("type").and_then(Value::as_str) {
                Some("text") => {
                    if let Some(t) = block.get("text").and_then(Value::as_str) {
                        self.text.push(t.to_string());
                    }
                }
                Some("tool_use") => {
                    if let Some(name) = block.get("name").and_then(Value::as_str) {
                        self.tool_calls.push(ToolCall {
                            id: block.get("id").and_then(Value::as_str).map(String::from),
                            name: name.to_string(),
                            arguments: block.get("input").map(Value::to_string).unwrap_or_default(),
                        });
                    }
                }
                _ => {} // thinking, redacted_thinking, server tool blocks
            }
        }
    }
}

struct Builder<'a> {
    config: &'a Config,
    context_limit_hint: Option<u64>,
    out: Normalized,
    /// Messages added to the request since the last completion was emitted.
    pending: Vec<Message>,
    current: Option<Group>,
    /// Tool results written while `current` is still open: Claude Code runs a
    /// tool as soon as its `tool_use` block streams in, so a result can land
    /// between two blocks of the same API response. They belong to the next
    /// request, after the reply they answer.
    deferred: Vec<Message>,
}

impl Builder<'_> {
    fn flush(&mut self) {
        let Some(group) = self.current.take() else {
            return;
        };
        let Some(session_id) = group.session_id.clone() else {
            self.out.malformed += 1;
            return;
        };
        let Some(model) = group.model.clone() else {
            self.out.malformed += 1;
            return;
        };
        let messages = std::mem::take(&mut self.pending);
        let starts_prompt = messages.iter().any(|m| m.role == Role::User);
        let response_text = if group.text.is_empty() {
            None
        } else {
            Some(group.text.join("\n"))
        };
        let timestamp = group.timestamp.unwrap_or_else(Utc::now);
        let usage = &group.usage;
        let prompt_tokens = match (
            usage.input_tokens,
            usage.cache_read_input_tokens,
            usage.cache_creation_input_tokens,
        ) {
            (None, None, None) => None,
            (a, b, c) => Some(a.unwrap_or(0) + b.unwrap_or(0) + c.unwrap_or(0)),
        };
        self.pending.push(Message {
            role: Role::Assistant,
            content: response_text.clone().unwrap_or_default(),
            tool_call_id: None,
            tool_calls: group.tool_calls.clone(),
        });
        self.pending.append(&mut self.deferred);
        self.out.events.push(ConversationEvent {
            event_id: group.request_id,
            conversation_id: session_id,
            conversation_id_source: IdSource::Session,
            user_id: None,
            message_id: group.first_uuid,
            request_id: None,
            started_at: group.started_at.unwrap_or(timestamp),
            timestamp,
            kind: EventKind::Chat,
            stream: None,
            prompt_tokens,
            completion_tokens: usage.output_tokens,
            context_limit: self.config.model_limit(&model).or(self.context_limit_hint),
            model,
            tool_results: messages
                .iter()
                .filter(|m| m.role == Role::Tool)
                .map(|m| ToolResult {
                    tool_call_id: m.tool_call_id.clone(),
                    content: m.content.clone(),
                })
                .collect(),
            messages,
            messages_are_delta: true,
            starts_prompt,
            response_text,
            tool_calls: group.tool_calls,
            error: None,
        });
    }

    fn user(&mut self, rec: &Record) {
        let Some(messages) = user_messages(rec) else {
            self.out.malformed += 1;
            return;
        };
        let only_tool_results = messages.iter().all(|m| m.role == Role::Tool);
        if self.current.is_some() && only_tool_results {
            self.deferred.extend(messages);
        } else {
            // A prompt (or a summary) ends the reply in progress.
            self.flush();
            self.pending.extend(messages);
        }
    }

    /// An API error is written as an assistant record with the error text as
    /// its content. It is a failed completion of the pending request, which the
    /// CLI then retries, so the pending delta is kept for the retry.
    fn api_error(&mut self, rec: &Record) {
        self.flush();
        let (Some(session_id), Some(id)) = (
            rec.session_id.clone(),
            rec.request_id.clone().or_else(|| rec.uuid.clone()),
        ) else {
            self.out.malformed += 1;
            return;
        };
        let model = rec
            .message
            .get("model")
            .and_then(Value::as_str)
            .unwrap_or("unknown")
            .to_string();
        let error = content_text(rec.message.get("content"));
        let timestamp = rec
            .timestamp
            .as_deref()
            .and_then(parse_time)
            .unwrap_or_else(Utc::now);
        self.out.events.push(ConversationEvent {
            event_id: id,
            conversation_id: session_id,
            conversation_id_source: IdSource::Session,
            user_id: None,
            message_id: rec.uuid.clone(),
            request_id: None,
            started_at: timestamp,
            timestamp,
            kind: EventKind::Failure,
            stream: None,
            prompt_tokens: None,
            completion_tokens: None,
            context_limit: self.config.model_limit(&model).or(self.context_limit_hint),
            model,
            messages: self.pending.clone(),
            messages_are_delta: true,
            // The retry of this request will carry the same prompt; count it once.
            starts_prompt: false,
            response_text: None,
            tool_calls: Vec::new(),
            tool_results: Vec::new(),
            error: Some(error).filter(|e| !e.is_empty()),
        });
    }

    fn assistant(&mut self, rec: &Record) {
        if rec.is_api_error_message == Some(true) {
            self.api_error(rec);
            return;
        }
        let Some(request_id) = rec
            .request_id
            .clone()
            .or_else(|| rec.uuid.clone())
            .filter(|s| !s.is_empty())
        else {
            self.out.malformed += 1;
            return;
        };
        if self
            .current
            .as_ref()
            .is_some_and(|g| g.request_id != request_id)
        {
            self.flush();
        }
        self.current
            .get_or_insert_with(|| Group::new(request_id))
            .absorb(rec);
    }
}

/// The messages one `user` record contributes: tool results as `tool`
/// messages, text as one message. A compaction summary is model-written text
/// stored as a user record; it must not become a source of truth.
fn user_messages(rec: &Record) -> Option<Vec<Message>> {
    let role = if rec.is_compact_summary == Some(true) {
        Role::System
    } else {
        Role::User
    };
    let text_message = |content: String| Message {
        role,
        content,
        tool_call_id: None,
        tool_calls: Vec::new(),
    };
    match rec.message.get("content") {
        Some(Value::String(s)) => Some(vec![text_message(s.clone())]),
        Some(Value::Array(blocks)) => {
            let mut messages = Vec::new();
            let mut text = Vec::new();
            for block in blocks {
                match block.get("type").and_then(Value::as_str) {
                    Some("tool_result") => messages.push(Message {
                        role: Role::Tool,
                        content: content_text(block.get("content")),
                        tool_call_id: block
                            .get("tool_use_id")
                            .and_then(Value::as_str)
                            .map(String::from),
                        tool_calls: Vec::new(),
                    }),
                    Some("text") => {
                        if let Some(t) = block.get("text").and_then(Value::as_str) {
                            text.push(t.to_string());
                        }
                    }
                    _ => {}
                }
            }
            if !text.is_empty() {
                messages.push(text_message(text.join("\n")));
            }
            Some(messages)
        }
        _ => None,
    }
}

/// Normalize one shipped slice of a transcript into events, in file order.
/// Records that are not messages (attachments, mode changes, cost state, …)
/// and subagent side chains are ignored; a record that should be a message
/// but cannot be read is counted as malformed.
pub fn normalize(ingest: &Ingest, config: &Config) -> Normalized {
    let mut b = Builder {
        config,
        context_limit_hint: ingest.context_limit,
        out: Normalized::default(),
        pending: Vec::new(),
        current: None,
        deferred: Vec::new(),
    };
    for value in &ingest.records {
        let Ok(rec) = serde_json::from_value::<Record>(value.clone()) else {
            b.out.malformed += 1;
            continue;
        };
        if rec.is_sidechain == Some(true) {
            continue;
        }
        match rec.kind.as_str() {
            "user" => b.user(&rec),
            "assistant" => b.assistant(&rec),
            _ => {}
        }
    }
    b.flush();
    b.out
}

fn parse_time(s: &str) -> Option<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(s)
        .ok()
        .map(|d| d.with_timezone(&Utc))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const SESSION: &str = "880138cf-78cd-4d41-9940-a4aa38c2aaec";

    fn user(uuid: &str, content: Value) -> Value {
        json!({
            "type": "user", "uuid": uuid, "sessionId": SESSION, "isSidechain": false,
            "timestamp": "2026-09-12T18:47:10.000Z",
            "message": {"role": "user", "content": content}
        })
    }

    fn assistant(uuid: &str, request_id: &str, block: Value) -> Value {
        json!({
            "type": "assistant", "uuid": uuid, "requestId": request_id, "sessionId": SESSION,
            "isSidechain": false, "timestamp": "2026-09-12T18:47:12.000Z",
            "message": {
                "model": "claude-haiku-4-5-20251001", "role": "assistant", "content": [block],
                "usage": {"input_tokens": 10, "cache_read_input_tokens": 13607,
                          "cache_creation_input_tokens": 7762, "output_tokens": 129}
            }
        })
    }

    fn ingest(records: Vec<Value>) -> Ingest {
        Ingest {
            records,
            context_limit: Some(200_000),
        }
    }

    #[test]
    fn groups_assistant_records_by_request_and_builds_deltas() {
        let records = vec![
            json!({"type": "attachment", "attachment": {"type": "environment"}}),
            user("u1", json!("Read notes.txt and tell me the port")),
            assistant(
                "a1",
                "req_A",
                json!({"type": "thinking", "thinking": "..."}),
            ),
            assistant(
                "a2",
                "req_A",
                json!({"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"file_path": "/x/notes.txt"}}),
            ),
            user(
                "u2",
                json!([{"type": "tool_result", "tool_use_id": "toolu_1", "content": "llama.cpp is on port 8080"}]),
            ),
            assistant(
                "a3",
                "req_B",
                json!({"type": "text", "text": "llama.cpp uses port 8080."}),
            ),
        ];
        let n = normalize(&ingest(records), &Config::default());
        assert_eq!(n.malformed, 0);
        assert_eq!(n.events.len(), 2);

        let a = &n.events[0];
        assert_eq!(a.event_id, "req_A");
        assert_eq!(a.conversation_id, SESSION);
        assert_eq!(a.conversation_id_source, IdSource::Session);
        assert_eq!(a.message_id.as_deref(), Some("a1"));
        assert_eq!(a.model, "claude-haiku-4-5-20251001");
        assert_eq!(a.kind, EventKind::Chat);
        assert!(a.messages_are_delta);
        assert!(a.starts_prompt, "the delta carries the user's prompt");
        assert_eq!(a.prompt_tokens, Some(10 + 13607 + 7762));
        assert_eq!(a.completion_tokens, Some(129));
        assert_eq!(a.context_limit, Some(200_000));
        assert_eq!(a.messages.len(), 1);
        assert_eq!(a.messages[0].role, Role::User);
        assert_eq!(a.response_text, None, "a tool-only reply has no text");
        assert_eq!(a.tool_calls.len(), 1);
        assert_eq!(a.tool_calls[0].id.as_deref(), Some("toolu_1"));
        assert_eq!(a.tool_calls[0].name, "Read");
        assert_eq!(a.tool_calls[0].arguments, r#"{"file_path":"/x/notes.txt"}"#);

        let b = &n.events[1];
        assert_eq!(b.event_id, "req_B");
        assert!(!b.starts_prompt, "a tool round-trip continues the prompt");
        assert_eq!(
            b.response_text.as_deref(),
            Some("llama.cpp uses port 8080.")
        );
        let roles: Vec<Role> = b.messages.iter().map(|m| m.role).collect();
        assert_eq!(roles, vec![Role::Assistant, Role::Tool]);
        assert_eq!(
            b.messages[0].tool_calls.len(),
            1,
            "the previous reply carries its calls"
        );
        assert_eq!(b.messages[1].tool_call_id.as_deref(), Some("toolu_1"));
        assert_eq!(b.tool_results.len(), 1);
        assert_eq!(b.tool_results[0].content, "llama.cpp is on port 8080");
    }

    /// Claude Code executes a tool as soon as its block streams in, so a tool
    /// result can be written between two `tool_use` records of one API call.
    /// The call is still one completion with both calls; the result answers it.
    #[test]
    fn tool_results_inside_one_response_do_not_split_the_turn() {
        let records = vec![
            user("u1", json!("read notes.txt then list the directory")),
            assistant(
                "a1",
                "req_A",
                json!({"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"file_path": "/x/notes.txt"}}),
            ),
            user(
                "u2",
                json!([{"type": "tool_result", "tool_use_id": "toolu_1", "content": "port 9090"}]),
            ),
            assistant(
                "a2",
                "req_A",
                json!({"type": "tool_use", "id": "toolu_2", "name": "Bash", "input": {"command": "ls"}}),
            ),
            user(
                "u3",
                json!([{"type": "tool_result", "tool_use_id": "toolu_2", "content": "notes.txt"}]),
            ),
            assistant(
                "a3",
                "req_B",
                json!({"type": "text", "text": "The port is 9090."}),
            ),
        ];
        let n = normalize(&ingest(records), &Config::default());
        assert_eq!(
            n.events.len(),
            2,
            "{:?}",
            n.events.iter().map(|e| &e.event_id).collect::<Vec<_>>()
        );
        let a = &n.events[0];
        assert_eq!(a.event_id, "req_A");
        assert_eq!(
            a.tool_calls
                .iter()
                .map(|t| t.name.as_str())
                .collect::<Vec<_>>(),
            vec!["Read", "Bash"]
        );
        assert_eq!(a.messages.len(), 1);
        let b = &n.events[1];
        let roles: Vec<Role> = b.messages.iter().map(|m| m.role).collect();
        assert_eq!(roles, vec![Role::Assistant, Role::Tool, Role::Tool]);
        assert_eq!(b.messages[0].tool_calls.len(), 2);
        assert_eq!(b.tool_results.len(), 2);
        assert_eq!(b.tool_results[0].content, "port 9090");
    }

    #[test]
    fn config_limit_beats_the_shipper_hint() {
        let mut config = Config::default();
        config
            .model_limits
            .insert("claude-haiku-4-5-20251001".into(), 1_000_000);
        let n = normalize(
            &ingest(vec![assistant(
                "a1",
                "req_A",
                json!({"type": "text", "text": "hi"}),
            )]),
            &config,
        );
        assert_eq!(n.events[0].context_limit, Some(1_000_000));
        let n = normalize(
            &Ingest {
                records: vec![assistant(
                    "a1",
                    "req_A",
                    json!({"type": "text", "text": "hi"}),
                )],
                context_limit: None,
            },
            &Config::default(),
        );
        assert_eq!(n.events[0].context_limit, None);
    }

    #[test]
    fn compaction_summaries_and_side_chains_do_not_become_truth() {
        let mut summary = user("u1", json!("Summary: the port is 9999"));
        summary["isCompactSummary"] = json!(true);
        let mut side = user("u2", json!("subagent prompt"));
        side["isSidechain"] = json!(true);
        let records = vec![
            summary,
            side,
            assistant("a1", "req_A", json!({"type": "text", "text": "ok"})),
        ];
        let n = normalize(&ingest(records), &Config::default());
        assert_eq!(n.events.len(), 1);
        let roles: Vec<Role> = n.events[0].messages.iter().map(|m| m.role).collect();
        assert_eq!(roles, vec![Role::System]);
    }

    #[test]
    fn api_errors_are_failures_that_keep_the_pending_delta() {
        let mut err = assistant(
            "e1",
            "req_E",
            json!({"type": "text", "text": "API Error: prompt is too long: 213265 tokens > 200000 maximum"}),
        );
        err["isApiErrorMessage"] = json!(true);
        let records = vec![
            user("u1", json!("hello")),
            err,
            assistant("a1", "req_A", json!({"type": "text", "text": "hi"})),
        ];
        let n = normalize(&ingest(records), &Config::default());
        assert_eq!(n.events.len(), 2);
        assert_eq!(n.events[0].kind, EventKind::Failure);
        assert!(!n.events[0].starts_prompt);
        assert!(n.events[1].starts_prompt);
        assert_eq!(n.events[0].event_id, "req_E");
        assert!(n.events[0]
            .error
            .as_deref()
            .unwrap()
            .contains("prompt is too long"));
        assert_eq!(n.events[0].messages.len(), 1);
        assert_eq!(n.events[1].kind, EventKind::Chat);
        assert_eq!(
            n.events[1].messages.len(),
            1,
            "the retry still sees the prompt"
        );
    }

    #[test]
    fn unusable_records_are_counted_not_fatal() {
        let records = vec![
            json!("not an object"),
            json!({"type": "user", "sessionId": SESSION, "message": {"content": 42}}),
            json!({"type": "assistant", "sessionId": SESSION, "message": {"content": []}}), // no request id, no uuid
            json!({"type": "assistant", "uuid": "a1", "requestId": "req_A", "message": {"content": [{"type": "text", "text": "hi"}]}}), // no session
            assistant("a2", "req_B", json!({"type": "text", "text": "fine"})),
        ];
        let n = normalize(&ingest(records), &Config::default());
        assert_eq!(n.malformed, 4);
        assert_eq!(n.events.len(), 1);
        assert_eq!(n.events[0].event_id, "req_B");
    }

    #[test]
    fn parse_body_accepts_envelope_array_and_ndjson() {
        let env =
            br#"{"records": [{"type": "user"}, {"type": "assistant"}], "context_limit": 200000}"#;
        let i = parse_body(env).unwrap();
        assert_eq!(i.records.len(), 2);
        assert_eq!(i.context_limit, Some(200_000));
        let arr = parse_body(br#"[{"type": "user"}]"#).unwrap();
        assert_eq!((arr.records.len(), arr.context_limit), (1, None));
        let nd = parse_body(b"{\"type\": \"user\"}\n{\"type\": \"assistant\"}\n").unwrap();
        assert_eq!(nd.records.len(), 2);
        assert!(parse_body(b"nope").is_err());
    }
}
