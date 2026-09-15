//! Turning telemetry into normalized conversation events. Each source has its
//! own module with a `normalize` entry point; nothing downstream of
//! [`event::ConversationEvent`] knows which source an event came from.

pub mod claude_code;
pub mod codex;
pub mod event;
pub mod identity;
pub mod litellm;
pub mod opencode;

use chrono::{DateTime, Utc};
use serde_json::Value;

use event::{ConversationEvent, EventKind, IdSource, Message, Role, ToolCall, ToolResult};

/// The events one ingest body produced, and how many records were unusable.
#[derive(Debug, Default)]
pub struct Normalized {
    pub events: Vec<ConversationEvent>,
    pub malformed: usize,
}

/// One completed API response as a transcript source (Claude Code, Codex)
/// knows it. What differs between sources is how these fields are found;
/// how they become an event does not.
pub(crate) struct DeltaReply {
    pub event_id: String,
    pub conversation_id: String,
    pub message_id: Option<String>,
    pub model: String,
    pub context_limit: Option<u64>,
    pub started_at: Option<DateTime<Utc>>,
    pub timestamp: Option<DateTime<Utc>>,
    pub prompt_tokens: Option<u64>,
    pub completion_tokens: Option<u64>,
    pub text: Vec<String>,
    pub tool_calls: Vec<ToolCall>,
}

/// The completion event of an incrementally shipped transcript. `pending` is
/// the request delta since the previous completion and is taken; the reply
/// (with its tool calls) and the tool results deferred while it streamed are
/// left behind as the start of the next request.
pub(crate) fn delta_completion(
    pending: &mut Vec<Message>,
    deferred: &mut Vec<Message>,
    reply: DeltaReply,
) -> ConversationEvent {
    let messages = std::mem::take(pending);
    let starts_prompt = messages.iter().any(|m| m.role == Role::User);
    let response_text = if reply.text.is_empty() {
        None
    } else {
        Some(reply.text.join("\n"))
    };
    let timestamp = reply.timestamp.unwrap_or_else(Utc::now);
    pending.push(Message {
        role: Role::Assistant,
        content: response_text.clone().unwrap_or_default(),
        tool_call_id: None,
        tool_calls: reply.tool_calls.clone(),
    });
    pending.append(deferred);
    ConversationEvent {
        event_id: reply.event_id,
        conversation_id: reply.conversation_id,
        conversation_id_source: IdSource::Session,
        user_id: None,
        message_id: reply.message_id,
        request_id: None,
        started_at: reply.started_at.unwrap_or(timestamp),
        timestamp,
        kind: EventKind::Chat,
        stream: None,
        prompt_tokens: reply.prompt_tokens,
        completion_tokens: reply.completion_tokens,
        context_limit: reply.context_limit,
        model: reply.model,
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
        tool_calls: reply.tool_calls,
        error: None,
    }
}

/// A failed completion of the pending request.
pub(crate) struct DeltaFailure {
    pub event_id: String,
    pub conversation_id: String,
    pub message_id: Option<String>,
    pub model: String,
    pub context_limit: Option<u64>,
    pub timestamp: DateTime<Utc>,
    pub error: String,
}

/// The failure event of an incrementally shipped transcript. The pending
/// delta is copied, not taken: the agent retries the same request, and its
/// prompt is counted once (`starts_prompt` stays false here).
pub(crate) fn delta_failure(pending: &[Message], failure: DeltaFailure) -> ConversationEvent {
    ConversationEvent {
        event_id: failure.event_id,
        conversation_id: failure.conversation_id,
        conversation_id_source: IdSource::Session,
        user_id: None,
        message_id: failure.message_id,
        request_id: None,
        started_at: failure.timestamp,
        timestamp: failure.timestamp,
        kind: EventKind::Failure,
        stream: None,
        prompt_tokens: None,
        completion_tokens: None,
        context_limit: failure.context_limit,
        model: failure.model,
        messages: pending.to_vec(),
        messages_are_delta: true,
        starts_prompt: false,
        response_text: None,
        tool_calls: Vec::new(),
        tool_results: Vec::new(),
        error: Some(failure.error).filter(|e| !e.is_empty()),
    }
}

/// Flatten message content to text: strings as-is, arrays keep `text` parts.
/// Shared by the OpenAI (LiteLLM) and Anthropic (Claude Code) content shapes.
pub(crate) fn content_text(v: Option<&Value>) -> String {
    match v {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|p| p.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

/// Transcript timestamps (Claude Code and Codex both write RFC 3339 UTC).
pub(crate) fn parse_rfc3339(s: &str) -> Option<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(s)
        .ok()
        .map(|d| d.with_timezone(&Utc))
}
