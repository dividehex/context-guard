//! Looping detection: repeated tool calls and near-identical responses.

use super::text::{canonical_json, jaccard, sha256_hex, shingles, word_count};

pub const TOOL_WINDOW: usize = 5;
pub const TOOL_REPEAT_THRESHOLD: usize = 3;
pub const RESPONSE_HISTORY: usize = 3;
pub const RESPONSE_SIMILARITY: f64 = 0.90;
pub const RESPONSE_MIN_WORDS: usize = 20;
const SHINGLE_SIZE: usize = 3;

/// Stable key for "the same operation": tool name plus canonicalized arguments.
pub fn tool_call_key(name: &str, arguments: &str) -> String {
    sha256_hex(&format!("{}\n{}", name.trim(), canonical_json(arguments)))
}

/// Keys that occur at least `TOOL_REPEAT_THRESHOLD` times within the last
/// `TOOL_WINDOW` calls. `keys` is oldest-first and must already include the
/// calls of the current turn.
pub fn repeated_tool_calls(keys: &[String]) -> Vec<String> {
    let start = keys.len().saturating_sub(TOOL_WINDOW);
    let window = &keys[start..];
    let mut flagged = Vec::new();
    for key in window {
        if flagged.contains(key) {
            continue;
        }
        if window.iter().filter(|k| *k == key).count() >= TOOL_REPEAT_THRESHOLD {
            flagged.push(key.clone());
        }
    }
    flagged
}

/// Highest similarity when the current response is near-identical to at least
/// two of the previous responses; `None` otherwise.
pub fn response_loop(current: &str, previous: &[String]) -> Option<f64> {
    if word_count(current) < RESPONSE_MIN_WORDS {
        return None;
    }
    let cur = shingles(current, SHINGLE_SIZE);
    let scores: Vec<f64> = previous
        .iter()
        .rev()
        .take(RESPONSE_HISTORY)
        .filter(|p| word_count(p) >= RESPONSE_MIN_WORDS)
        .map(|p| jaccard(&cur, &shingles(p, SHINGLE_SIZE)))
        .filter(|s| *s >= RESPONSE_SIMILARITY)
        .collect();
    if scores.len() >= 2 {
        scores
            .into_iter()
            .fold(None, |acc, s| Some(acc.map_or(s, |a: f64| a.max(s))))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_key_ignores_argument_formatting() {
        assert_eq!(
            tool_call_key("search", "{\"q\": \"x\", \"n\": 1}"),
            tool_call_key("search", "{\"n\":1,\"q\":\"x\"}")
        );
        assert_ne!(
            tool_call_key("search", "{\"q\": \"x\"}"),
            tool_call_key("search", "{\"q\": \"y\"}")
        );
        assert_ne!(tool_call_key("search", "{}"), tool_call_key("fetch", "{}"));
    }

    #[test]
    fn three_of_five_is_repetition_two_is_not() {
        let a = tool_call_key("a", "{}");
        let b = tool_call_key("b", "{}");
        assert_eq!(
            repeated_tool_calls(&[a.clone(), b.clone(), a.clone(), b.clone(), a.clone()]),
            vec![a.clone()]
        );
        assert!(repeated_tool_calls(&[a.clone(), b.clone(), a.clone()]).is_empty());
        // Old repeats fall out of the window.
        let c = tool_call_key("c", "{}");
        let d = tool_call_key("d", "{}");
        assert!(repeated_tool_calls(&[
            a.clone(),
            a.clone(),
            a.clone(),
            b.clone(),
            c.clone(),
            d.clone(),
            a.clone()
        ])
        .is_empty());
    }

    #[test]
    fn identical_long_responses_loop_but_short_or_varied_do_not() {
        let long = "I have checked the configuration file and the service is running on the expected port with no errors reported in the logs at this time.".to_string();
        assert!(response_loop(&long, &[long.clone(), long.clone()]).is_some());
        assert!(response_loop(&long, std::slice::from_ref(&long)).is_none());
        let varied = "The weather today is sunny with a light breeze and temperatures in the low twenties, ideal for a walk in the park this afternoon.".to_string();
        assert!(response_loop(&long, &[varied.clone(), varied]).is_none());
        assert!(response_loop("ok", &["ok".into(), "ok".into()]).is_none());
    }
}
