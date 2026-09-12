//! Tool-call ledger checks that need nothing beyond what the payload contains.

use std::sync::LazyLock;

use regex::Regex;

use crate::telemetry::event::{Message, Role};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResultStatus {
    Ok,
    Failed,
}

impl ResultStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            ResultStatus::Ok => "ok",
            ResultStatus::Failed => "failed",
        }
    }
}

static FAILURE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r#"(?i)^\s*[\{\["]?\s*(?:"?errors?"?\s*:|error\b|exception\b|failed\b|failure\b|traceback\b|permission denied|not found|timed? ?out|HTTP/?\s?[45]\d\d\b|[45]\d\d\s+(?:bad|unauthorized|forbidden|not|internal|service|gateway))"#,
    )
    .unwrap()
});

/// Classify a tool result by its leading text. Conservative: only obvious
/// failure markers count as failures.
pub fn result_status(content: &str) -> ResultStatus {
    let head: String = content.chars().take(200).collect();
    if FAILURE_RE.is_match(&head) {
        ResultStatus::Failed
    } else {
        ResultStatus::Ok
    }
}

/// `tool` messages whose `tool_call_id` was not issued by an earlier assistant
/// message in the same request. Self-contained per request, so it is correct
/// even when monitoring started mid-conversation.
pub fn orphan_results(messages: &[Message]) -> Vec<String> {
    let mut issued: Vec<&str> = Vec::new();
    let mut orphans = Vec::new();
    for m in messages {
        match m.role {
            Role::Assistant => issued.extend(m.tool_calls.iter().filter_map(|tc| tc.id.as_deref())),
            Role::Tool => {
                let id = m.tool_call_id.as_deref().unwrap_or("");
                if !issued.contains(&id) {
                    orphans.push(if id.is_empty() {
                        "<missing>".to_string()
                    } else {
                        id.to_string()
                    });
                }
            }
            _ => {}
        }
    }
    orphans
}

/// Strings in assistant text that look like this conversation's tool-call ids
/// (same prefix as the ids actually seen) but were never issued.
pub fn unknown_call_id_references(text: &str, known_ids: &[String]) -> Vec<String> {
    let Some(prefix) = id_prefix(known_ids) else {
        return Vec::new();
    };
    let Ok(re) = Regex::new(&format!(
        r"\b{}[A-Za-z0-9_-]{{4,}}\b",
        regex::escape(&prefix)
    )) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for m in re.find_iter(text) {
        let candidate = m.as_str();
        if !known_ids.iter().any(|k| k == candidate) && !out.iter().any(|o| o == candidate) {
            out.push(candidate.to_string());
        }
    }
    out
}

/// Common prefix of the known ids, cut back to the last separator (`call_`).
fn id_prefix(ids: &[String]) -> Option<String> {
    let first = ids.first()?;
    let mut prefix: String = first.clone();
    for id in &ids[1..] {
        let common: String = prefix
            .chars()
            .zip(id.chars())
            .take_while(|(a, b)| a == b)
            .map(|(a, _)| a)
            .collect();
        prefix = common;
    }
    let cut = prefix.rfind(['_', '-'])? + 1;
    let prefix = &prefix[..cut];
    (prefix.len() >= 3).then(|| prefix.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::telemetry::event::ToolCall;

    fn assistant(ids: &[&str]) -> Message {
        Message {
            role: Role::Assistant,
            content: String::new(),
            tool_call_id: None,
            tool_calls: ids
                .iter()
                .map(|id| ToolCall {
                    id: Some(id.to_string()),
                    name: "t".into(),
                    arguments: "{}".into(),
                })
                .collect(),
        }
    }

    fn tool(id: &str) -> Message {
        Message {
            role: Role::Tool,
            content: "ok".into(),
            tool_call_id: Some(id.into()),
            tool_calls: vec![],
        }
    }

    #[test]
    fn orphan_detection_is_per_request() {
        assert!(orphan_results(&[assistant(&["call_1"]), tool("call_1")]).is_empty());
        assert_eq!(
            orphan_results(&[tool("call_9")]),
            vec!["call_9".to_string()]
        );
        assert_eq!(
            orphan_results(&[assistant(&["call_1"]), tool("call_2")]),
            vec!["call_2".to_string()]
        );
    }

    #[test]
    fn unknown_reference_uses_learned_prefix() {
        let known = vec!["call_abc123".to_string(), "call_def456".to_string()];
        assert_eq!(
            unknown_call_id_references("see call_abc123 and call_zzz999", &known),
            vec!["call_zzz999".to_string()]
        );
        assert!(unknown_call_id_references("no ids here", &known).is_empty());
        assert!(unknown_call_id_references("call_zzz999", &[]).is_empty());
    }

    #[test]
    fn failure_markers() {
        assert_eq!(
            result_status("Error: connection refused"),
            ResultStatus::Failed
        );
        assert_eq!(result_status("{\"error\": \"nope\"}"), ResultStatus::Failed);
        assert_eq!(
            result_status("HTTP 500 Internal Server Error"),
            ResultStatus::Failed
        );
        assert_eq!(
            result_status("Traceback (most recent call last)"),
            ResultStatus::Failed
        );
        assert_eq!(
            result_status("The weather is fine, no errors expected"),
            ResultStatus::Ok
        );
    }
}
