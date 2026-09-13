//! Turning telemetry into normalized conversation events. Each source has its
//! own module with a `normalize` entry point; nothing downstream of
//! [`event::ConversationEvent`] knows which source an event came from.

pub mod claude_code;
pub mod event;
pub mod identity;
pub mod litellm;

use serde_json::Value;

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
