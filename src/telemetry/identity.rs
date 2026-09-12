//! Deciding which conversation (and user, message, task) an event belongs to.
//!
//! LiteLLM turns configured request headers into `request_tags` entries of the
//! form `"<header-name>: <value>"`. Open WebUI sends `X-OpenWebUI-Chat-Id` and
//! `X-OpenWebUI-User-Id` on its own; the message id and task headers come from
//! the connection's custom headers (see README).

use std::collections::HashMap;

use serde_json::Value;
use sha2::{Digest, Sha256};

use super::event::IdSource;

pub const TAG_CHAT_ID: &str = "x-openwebui-chat-id";
pub const TAG_USER_ID: &str = "x-openwebui-user-id";
pub const TAG_MESSAGE_ID: &str = "x-openwebui-message-id";
pub const TAG_TASK: &str = "x-openwebui-task";

/// Parse `request_tags` into a lowercase-keyed map. Anything that is not a
/// list of `"key: value"` strings is ignored rather than rejected.
pub fn parse_tags(tags: &Value) -> HashMap<String, String> {
    let mut out = HashMap::new();
    for tag in tags.as_array().map(Vec::as_slice).unwrap_or_default() {
        let Some(text) = tag.as_str() else { continue };
        let Some((key, value)) = text.split_once(':') else {
            continue;
        };
        let value = value.trim();
        if value.is_empty() {
            continue;
        }
        out.insert(key.trim().to_ascii_lowercase(), value.to_string());
    }
    out
}

pub struct Resolved {
    pub conversation_id: String,
    pub source: IdSource,
}

/// Pick the conversation id by precedence: chat tag, then (opt-in) trace id,
/// then a hash of user, model and first user message.
pub fn resolve_conversation(
    tags: &HashMap<String, String>,
    trace_id: Option<&str>,
    trust_trace_id: bool,
    user_id: Option<&str>,
    model: &str,
    first_user_message: Option<&str>,
) -> Resolved {
    if let Some(id) = tags.get(TAG_CHAT_ID).filter(|v| is_plausible_id(v)) {
        return Resolved {
            conversation_id: id.clone(),
            source: IdSource::ChatTag,
        };
    }
    if trust_trace_id {
        if let Some(id) = trace_id.filter(|v| is_plausible_id(v)) {
            return Resolved {
                conversation_id: id.to_string(),
                source: IdSource::TraceId,
            };
        }
    }
    let mut hasher = Sha256::new();
    hasher.update(user_id.unwrap_or("").as_bytes());
    hasher.update(b"|");
    hasher.update(model.as_bytes());
    hasher.update(b"|");
    hasher.update(first_user_message.unwrap_or("").as_bytes());
    let digest = hex::encode(hasher.finalize());
    Resolved {
        conversation_id: format!("fallback:{}", &digest[..16]),
        source: IdSource::Fallback,
    }
}

fn is_plausible_id(value: &str) -> bool {
    let len = value.chars().count();
    (4..=200).contains(&len)
        && value
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | ':' | '.'))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn tags_parse_case_insensitively_and_skip_junk() {
        let tags = parse_tags(&json!([
            "X-OpenWebUI-Chat-Id: abc-123",
            "User-Agent: OpenAI",
            "garbage",
            42,
            "x-openwebui-task: ",
        ]));
        assert_eq!(tags.get(TAG_CHAT_ID).map(String::as_str), Some("abc-123"));
        assert_eq!(tags.get("user-agent").map(String::as_str), Some("OpenAI"));
        assert!(!tags.contains_key(TAG_TASK));
        assert!(parse_tags(&Value::Null).is_empty());
        assert!(parse_tags(&json!("nope")).is_empty());
    }

    #[test]
    fn chat_tag_wins_over_trace_id() {
        let mut tags = HashMap::new();
        tags.insert(TAG_CHAT_ID.to_string(), "chat-1".to_string());
        let r = resolve_conversation(&tags, Some("trace-1234"), true, None, "m", None);
        assert_eq!(r.conversation_id, "chat-1");
        assert_eq!(r.source, IdSource::ChatTag);
    }

    #[test]
    fn trace_id_only_when_trusted() {
        let tags = HashMap::new();
        let trusted = resolve_conversation(&tags, Some("trace-1234"), true, None, "m", None);
        assert_eq!(trusted.source, IdSource::TraceId);
        let untrusted = resolve_conversation(&tags, Some("trace-1234"), false, None, "m", None);
        assert_eq!(untrusted.source, IdSource::Fallback);
    }

    #[test]
    fn fallback_is_stable_and_sensitive_to_inputs() {
        let tags = HashMap::new();
        let a = resolve_conversation(&tags, None, false, Some("u"), "m", Some("hello"));
        let b = resolve_conversation(&tags, None, false, Some("u"), "m", Some("hello"));
        let c = resolve_conversation(&tags, None, false, Some("u"), "m", Some("hi"));
        assert_eq!(a.conversation_id, b.conversation_id);
        assert_ne!(a.conversation_id, c.conversation_id);
        assert!(a.conversation_id.starts_with("fallback:"));
    }
}
