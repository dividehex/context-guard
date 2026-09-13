//! The normalized representation of one LLM completion as seen by LiteLLM.

use chrono::{DateTime, Utc};
use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
    Other,
}

impl Role {
    pub fn parse(s: &str) -> Role {
        match s {
            "system" | "developer" => Role::System,
            "user" => Role::User,
            "assistant" => Role::Assistant,
            "tool" | "function" => Role::Tool,
            _ => Role::Other,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
            Role::Other => "other",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ToolCall {
    pub id: Option<String>,
    pub name: String,
    /// Raw JSON text of the arguments, as the model emitted it.
    pub arguments: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
    pub tool_call_id: Option<String>,
    pub tool_calls: Vec<ToolCall>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ToolResult {
    pub tool_call_id: Option<String>,
    pub content: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EventKind {
    /// A real chat completion: counts as a turn and is scored.
    Chat,
    /// An Open WebUI background task (title, tags, follow-ups, queries): recorded, never scored.
    Task,
    /// LiteLLM reported a failed completion: recorded, never scored.
    Failure,
}

impl EventKind {
    pub fn as_str(self) -> &'static str {
        match self {
            EventKind::Chat => "chat",
            EventKind::Task => "task",
            EventKind::Failure => "failure",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum IdSource {
    ChatTag,
    TraceId,
    /// The Claude Code session id, which names the transcript file.
    Session,
    Fallback,
}

impl IdSource {
    pub fn as_str(self) -> &'static str {
        match self {
            IdSource::ChatTag => "chat_tag",
            IdSource::TraceId => "trace_id",
            IdSource::Session => "session",
            IdSource::Fallback => "fallback",
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ConversationEvent {
    /// LiteLLM's payload id; unique per completion, used for deduplication.
    pub event_id: String,
    pub conversation_id: String,
    pub conversation_id_source: IdSource,
    pub user_id: Option<String>,
    /// Open WebUI assistant message id, when the connection header is configured.
    pub message_id: Option<String>,
    pub model: String,
    pub request_id: Option<String>,
    pub started_at: DateTime<Utc>,
    pub timestamp: DateTime<Utc>,
    pub kind: EventKind,
    pub stream: Option<bool>,
    pub prompt_tokens: Option<u64>,
    pub completion_tokens: Option<u64>,
    pub context_limit: Option<u64>,
    /// The request as the model saw it (LiteLLM), or, when `messages_are_delta`
    /// is set, only the messages added since this conversation's previous
    /// completion (Claude Code transcripts, which are shipped incrementally).
    pub messages: Vec<Message>,
    pub messages_are_delta: bool,
    /// This completion begins a new prompt, the unit the anomaly window is
    /// counted in. LiteLLM: every completion. Claude Code: a completion whose
    /// delta carries a user message, so a tool loop is one prompt.
    pub starts_prompt: bool,
    pub response_text: Option<String>,
    pub tool_calls: Vec<ToolCall>,
    pub tool_results: Vec<ToolResult>,
    pub error: Option<String>,
}
