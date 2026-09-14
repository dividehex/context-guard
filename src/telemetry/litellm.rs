//! Parsing LiteLLM's `generic_api` callback bodies (`StandardLoggingPayload`).
//!
//! Everything is optional: LiteLLM's payload shape drifts between versions and
//! model backends omit fields. Missing data becomes `None`, never a guess.

use chrono::{DateTime, TimeZone, Utc};
use serde::Deserialize;
use serde_json::Value;
use thiserror::Error;

use super::content_text;
use super::event::{ConversationEvent, EventKind, Message, Role, ToolCall, ToolResult};
use super::identity::{self, TAG_MESSAGE_ID, TAG_TASK, TAG_USER_ID};
use crate::config::Config;

#[derive(Debug, Error)]
pub enum ParseError {
    #[error("body is not JSON: {0}")]
    Json(String),
    #[error("body is neither a JSON object, array nor NDJSON")]
    Shape,
    #[error("body has no usable {0}")]
    MissingField(&'static str),
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum NormalizeError {
    #[error("payload is not a JSON object")]
    NotObject,
    #[error("payload has no id")]
    MissingId,
    #[error("payload has no model")]
    MissingModel,
    #[error("payload call_type {0:?} is not a chat completion")]
    UnsupportedCallType(String),
}

/// Split a request body into individual payload values. Accepts a JSON array
/// (what LiteLLM sends), a single object, or newline-delimited objects.
pub fn split_body(body: &[u8]) -> Result<Vec<Value>, ParseError> {
    let text = std::str::from_utf8(body).map_err(|e| ParseError::Json(e.to_string()))?;
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Ok(Vec::new());
    }
    match serde_json::from_str::<Value>(trimmed) {
        Ok(Value::Array(items)) => Ok(items),
        Ok(obj @ Value::Object(_)) => Ok(vec![obj]),
        Ok(_) => Err(ParseError::Shape),
        Err(first_error) => {
            // NDJSON: every non-empty line must be an object.
            let mut items = Vec::new();
            for line in trimmed.lines().map(str::trim).filter(|l| !l.is_empty()) {
                match serde_json::from_str::<Value>(line) {
                    Ok(obj @ Value::Object(_)) => items.push(obj),
                    _ => return Err(ParseError::Json(first_error.to_string())),
                }
            }
            Ok(items)
        }
    }
}

/// The subset of `StandardLoggingPayload` we read. Unknown fields are ignored.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct RawPayload {
    id: Option<String>,
    trace_id: Option<String>,
    litellm_call_id: Option<String>,
    call_type: Option<String>,
    stream: Option<bool>,
    status: Option<String>,
    error_str: Option<String>,
    model: Option<String>,
    model_group: Option<String>,
    prompt_tokens: Option<Value>,
    completion_tokens: Option<Value>,
    #[serde(rename = "startTime")]
    start_time: Option<Value>,
    #[serde(rename = "endTime")]
    end_time: Option<Value>,
    messages: Value,
    response: Value,
    model_map_information: Value,
    request_tags: Value,
    end_user: Option<String>,
    metadata: Value,
}

const CHAT_CALL_TYPES: &[&str] = &["completion", "acompletion"];

/// Normalize one payload into a `ConversationEvent`.
pub fn normalize(value: &Value, config: &Config) -> Result<ConversationEvent, NormalizeError> {
    if !value.is_object() {
        return Err(NormalizeError::NotObject);
    }
    let raw: RawPayload =
        serde_json::from_value(value.clone()).map_err(|_| NormalizeError::NotObject)?;

    let event_id = raw
        .id
        .clone()
        .filter(|s| !s.is_empty())
        .ok_or(NormalizeError::MissingId)?;
    let call_type = raw.call_type.clone().unwrap_or_default();
    if !CHAT_CALL_TYPES.contains(&call_type.as_str()) {
        return Err(NormalizeError::UnsupportedCallType(call_type));
    }
    let model = raw
        .model_group
        .clone()
        .filter(|s| !s.is_empty())
        .or_else(|| raw.model.clone())
        .filter(|s| !s.is_empty())
        .ok_or(NormalizeError::MissingModel)?;

    let tags = identity::parse_tags(&raw.request_tags);
    let messages = parse_messages(&raw.messages);
    let first_user_message = messages
        .iter()
        .find(|m| m.role == Role::User)
        .map(|m| m.content.as_str());
    let user_id = tags
        .get(TAG_USER_ID)
        .cloned()
        .or_else(|| {
            raw.metadata
                .get("user_api_key_end_user_id")
                .and_then(Value::as_str)
                .map(String::from)
        })
        .or_else(|| raw.end_user.clone().filter(|s| !s.is_empty()));
    let resolved = identity::resolve_conversation(
        &tags,
        raw.trace_id.as_deref(),
        config.trust_trace_id,
        user_id.as_deref(),
        &model,
        first_user_message,
    );

    let (response_text, tool_calls) = parse_response(&raw.response);
    let tool_results = messages
        .iter()
        .filter(|m| m.role == Role::Tool)
        .map(|m| ToolResult {
            tool_call_id: m.tool_call_id.clone(),
            content: m.content.clone(),
        })
        .collect();

    let kind = if raw.status.as_deref() == Some("failure")
        || raw.error_str.as_deref().is_some_and(|e| !e.is_empty())
    {
        EventKind::Failure
    } else if tags.get(TAG_TASK).is_some_and(|t| !t.is_empty())
        || looks_like_task(raw.stream, &messages)
    {
        EventKind::Task
    } else {
        EventKind::Chat
    };

    let context_limit = config
        .model_limit(&model)
        .or_else(|| payload_context_limit(&raw.model_map_information));
    let timestamp = parse_time(raw.end_time.as_ref()).unwrap_or_else(Utc::now);
    let started_at = parse_time(raw.start_time.as_ref()).unwrap_or(timestamp);

    Ok(ConversationEvent {
        event_id,
        conversation_id: resolved.conversation_id,
        conversation_id_source: resolved.source,
        user_id,
        message_id: tags.get(TAG_MESSAGE_ID).cloned(),
        model,
        request_id: raw.litellm_call_id,
        started_at,
        timestamp,
        kind,
        stream: raw.stream,
        prompt_tokens: raw.prompt_tokens.as_ref().and_then(as_u64),
        completion_tokens: raw.completion_tokens.as_ref().and_then(as_u64),
        context_limit,
        messages,
        messages_are_delta: false,
        starts_prompt: true,
        response_text,
        tool_calls,
        tool_results,
        error: raw.error_str.filter(|e| !e.is_empty()),
    })
}

/// Open WebUI background tasks are non-streaming single-prompt calls. This is
/// only used when the task header is not configured.
fn looks_like_task(stream: Option<bool>, messages: &[Message]) -> bool {
    stream == Some(false)
        && messages.iter().filter(|m| m.role == Role::User).count() == 1
        && !messages.iter().any(|m| m.role == Role::Assistant)
}

fn payload_context_limit(model_map: &Value) -> Option<u64> {
    let info = model_map.get("model_map_value")?;
    info.get("max_input_tokens")
        .and_then(as_u64)
        .or_else(|| info.get("max_tokens").and_then(as_u64))
}

fn as_u64(v: &Value) -> Option<u64> {
    match v {
        Value::Number(n) => n
            .as_u64()
            .or_else(|| n.as_f64().filter(|f| *f >= 0.0).map(|f| f as u64)),
        Value::String(s) => s.trim().parse().ok(),
        _ => None,
    }
}

fn parse_time(v: Option<&Value>) -> Option<DateTime<Utc>> {
    match v? {
        Value::Number(n) => {
            let secs = n.as_f64()?;
            if !secs.is_finite() || secs < 0.0 {
                return None;
            }
            Utc.timestamp_opt(secs.trunc() as i64, ((secs.fract()) * 1e9) as u32)
                .single()
        }
        Value::String(s) => DateTime::parse_from_rfc3339(s)
            .ok()
            .map(|d| d.with_timezone(&Utc)),
        _ => None,
    }
}

fn parse_messages(v: &Value) -> Vec<Message> {
    let Some(items) = v.as_array() else {
        return Vec::new();
    };
    items.iter().filter_map(parse_message).collect()
}

fn parse_message(v: &Value) -> Option<Message> {
    let obj = v.as_object()?;
    let role = Role::parse(obj.get("role").and_then(Value::as_str).unwrap_or(""));
    Some(Message {
        role,
        content: content_text(obj.get("content")),
        tool_call_id: obj
            .get("tool_call_id")
            .and_then(Value::as_str)
            .map(String::from),
        tool_calls: obj
            .get("tool_calls")
            .map(parse_tool_calls)
            .unwrap_or_default(),
    })
}

fn parse_tool_calls(v: &Value) -> Vec<ToolCall> {
    let Some(items) = v.as_array() else {
        return Vec::new();
    };
    items
        .iter()
        .filter_map(|tc| {
            let function = tc.get("function")?;
            let name = function.get("name").and_then(Value::as_str)?.to_string();
            let arguments = match function.get("arguments") {
                Some(Value::String(s)) => s.clone(),
                Some(other) => other.to_string(),
                None => String::new(),
            };
            Some(ToolCall {
                id: tc.get("id").and_then(Value::as_str).map(String::from),
                name,
                arguments,
            })
        })
        .collect()
}

fn parse_response(v: &Value) -> (Option<String>, Vec<ToolCall>) {
    let Some(message) = v
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|c| c.first())
        .and_then(|c| c.get("message"))
    else {
        return (None, Vec::new());
    };
    let text = match message.get("content") {
        Some(Value::String(s)) => Some(s.clone()),
        Some(Value::Array(_)) => Some(content_text(message.get("content"))),
        _ => None,
    };
    let tool_calls = message
        .get("tool_calls")
        .map(parse_tool_calls)
        .unwrap_or_default();
    (text, tool_calls)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn cfg() -> Config {
        Config::default()
    }

    fn payload() -> Value {
        json!({
            "id": "chatcmpl-1",
            "trace_id": "t-1",
            "litellm_call_id": "call-1",
            "call_type": "acompletion",
            "stream": true,
            "status": "success",
            "model": "openai/qwen3-30b-a3b",
            "model_group": "qwen3-30b-a3b",
            "prompt_tokens": 1200,
            "completion_tokens": 30,
            "startTime": 1757600000.5,
            "endTime": 1757600002.25,
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": [{"type": "text", "text": "llama.cpp runs on port 8080"}, {"type": "image_url", "image_url": {"url": "data:..."}}]},
                {"role": "assistant", "content": null, "tool_calls": [{"id": "call_abc", "type": "function", "function": {"name": "lookup", "arguments": "{\"q\": 1}"}}]},
                {"role": "tool", "tool_call_id": "call_abc", "content": "result"}
            ],
            "response": {"choices": [{"message": {"role": "assistant", "content": "Done.", "tool_calls": [{"id": "call_def", "function": {"name": "ping", "arguments": "{}"}}]}}]},
            "model_map_information": {"model_map_key": "qwen3-30b-a3b", "model_map_value": {"max_input_tokens": 122880}},
            "request_tags": ["User-Agent: OpenAI", "x-openwebui-chat-id: chat-42", "x-openwebui-user-id: user-7", "x-openwebui-message-id: msg-9"],
            "metadata": {},
            "end_user": ""
        })
    }

    #[test]
    fn split_body_accepts_array_object_and_ndjson() {
        assert_eq!(split_body(b"[{\"a\":1},{\"b\":2}]").unwrap().len(), 2);
        assert_eq!(split_body(b"{\"a\":1}").unwrap().len(), 1);
        assert_eq!(split_body(b"{\"a\":1}\n{\"b\":2}\n").unwrap().len(), 2);
        assert!(split_body(b"   ").unwrap().is_empty());
        assert!(split_body(b"42").is_err());
        assert!(split_body(b"not json").is_err());
        assert!(split_body(&[0xff, 0xfe]).is_err());
    }

    #[test]
    fn normalizes_a_full_payload() {
        let ev = normalize(&payload(), &cfg()).unwrap();
        assert_eq!(ev.event_id, "chatcmpl-1");
        assert_eq!(ev.conversation_id, "chat-42");
        assert_eq!(
            ev.conversation_id_source,
            super::super::event::IdSource::ChatTag
        );
        assert_eq!(ev.user_id.as_deref(), Some("user-7"));
        assert_eq!(ev.message_id.as_deref(), Some("msg-9"));
        assert_eq!(ev.model, "qwen3-30b-a3b");
        assert_eq!(ev.kind, EventKind::Chat);
        assert_eq!(ev.prompt_tokens, Some(1200));
        assert_eq!(ev.context_limit, Some(122_880));
        assert_eq!(ev.messages.len(), 4);
        assert_eq!(ev.messages[1].content, "llama.cpp runs on port 8080");
        assert_eq!(ev.messages[2].tool_calls[0].name, "lookup");
        assert_eq!(ev.tool_results.len(), 1);
        assert_eq!(ev.response_text.as_deref(), Some("Done."));
        assert_eq!(ev.tool_calls[0].id.as_deref(), Some("call_def"));
        assert_eq!(ev.timestamp.timestamp(), 1_757_600_002);
    }

    #[test]
    fn config_limit_overrides_payload_limit() {
        let mut c = cfg();
        c.model_limits.insert("qwen3-30b-a3b".into(), 4096);
        assert_eq!(normalize(&payload(), &c).unwrap().context_limit, Some(4096));
    }

    #[test]
    fn task_tag_and_heuristic_mark_tasks() {
        let mut p = payload();
        p["request_tags"]
            .as_array_mut()
            .unwrap()
            .push(json!("x-openwebui-task: title_generation"));
        assert_eq!(normalize(&p, &cfg()).unwrap().kind, EventKind::Task);

        let heuristic = json!({
            "id": "x", "call_type": "acompletion", "model": "m", "stream": false,
            "messages": [{"role": "user", "content": "Generate a title"}],
            "response": {"choices": [{"message": {"content": "Title"}}]}
        });
        assert_eq!(normalize(&heuristic, &cfg()).unwrap().kind, EventKind::Task);
    }

    #[test]
    fn failures_and_missing_fields_are_handled() {
        let mut p = payload();
        p["status"] = json!("failure");
        p["error_str"] = json!("boom");
        let ev = normalize(&p, &cfg()).unwrap();
        assert_eq!(ev.kind, EventKind::Failure);
        assert_eq!(ev.error.as_deref(), Some("boom"));

        assert_eq!(
            normalize(&json!([]), &cfg()).unwrap_err(),
            NormalizeError::NotObject
        );
        assert_eq!(
            normalize(&json!({"call_type": "acompletion", "model": "m"}), &cfg()).unwrap_err(),
            NormalizeError::MissingId
        );
        assert_eq!(
            normalize(&json!({"id": "1", "call_type": "acompletion"}), &cfg()).unwrap_err(),
            NormalizeError::MissingModel
        );
        assert!(matches!(
            normalize(
                &json!({"id": "1", "call_type": "embedding", "model": "m"}),
                &cfg()
            )
            .unwrap_err(),
            NormalizeError::UnsupportedCallType(_)
        ));

        let sparse = json!({"id": "1", "call_type": "completion", "model": "m", "messages": "not-a-list", "response": 5, "prompt_tokens": "12", "request_tags": null});
        let ev = normalize(&sparse, &cfg()).unwrap();
        assert!(ev.messages.is_empty());
        assert!(ev.response_text.is_none());
        assert_eq!(ev.prompt_tokens, Some(12));
        assert!(ev.context_limit.is_none());
        assert!(ev.conversation_id.starts_with("fallback:"));
    }
}
