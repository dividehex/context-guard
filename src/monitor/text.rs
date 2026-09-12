//! Deterministic text helpers shared by the signal modules.

use std::collections::HashSet;
use std::sync::LazyLock;

use regex::Regex;
use serde_json::Value;
use sha2::{Digest, Sha256};

pub fn sha256_hex(input: &str) -> String {
    hex::encode(Sha256::digest(input.as_bytes()))
}

static SECRET_NAME_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)(?:^|[_-])(?:api[_-]?key|key|token|secret|password|passwd|pwd|credentials?|auth)(?:$|[_-])")
        .unwrap()
});

/// `NAME=value` / `name: value` with a secret-looking name, and well-known
/// token shapes (OpenAI, GitHub, Slack, AWS, Google keys, JWTs, bearer tokens).
static SECRET_VALUE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(concat!(
        r#"(?i)\b((?:[a-z0-9]+[_-])*(?:api[_-]?key|key|token|secret|password|passwd|pwd|credentials?|auth)(?:[_-][a-z0-9]+)*\s*[:=]\s*)("[^"\n]*"|'[^'\n]*'|[^\s"',;]+)"#,
        r"|\b(sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|xox[abprs]-[A-Za-z0-9-]{10,}|AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_-]{30,}|eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,})\b",
        r"|(?i)\b(bearer\s+)([A-Za-z0-9._~+/=-]{16,})",
    ))
    .unwrap()
});

pub const REDACTED: &str = "[redacted]";

/// Does an identifier look like it names a key, token or password?
pub fn is_secret_name(name: &str) -> bool {
    SECRET_NAME_RE.is_match(name)
}

/// Replace values that look like keys, tokens or passwords with `[redacted]`,
/// so a string is safe to log or to show outside the conversation.
pub fn redact_secrets(text: &str) -> String {
    SECRET_VALUE_RE
        .replace_all(text, |c: &regex::Captures| {
            let prefix = c.get(1).or_else(|| c.get(4)).map_or("", |m| m.as_str());
            format!("{prefix}{REDACTED}")
        })
        .into_owned()
}

/// Lowercase, punctuation stripped, whitespace collapsed.
pub fn normalize(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut last_space = true;
    for c in text.chars() {
        if c.is_alphanumeric() {
            out.extend(c.to_lowercase());
            last_space = false;
        } else if !last_space {
            out.push(' ');
            last_space = true;
        }
    }
    out.trim_end().to_string()
}

pub fn word_count(text: &str) -> usize {
    normalize(text).split_whitespace().count()
}

/// Word n-gram shingles of the normalized text.
pub fn shingles(text: &str, n: usize) -> HashSet<String> {
    let normalized = normalize(text);
    let words: Vec<&str> = normalized.split_whitespace().collect();
    if words.len() < n {
        return words.iter().map(|w| w.to_string()).collect();
    }
    words.windows(n).map(|w| w.join(" ")).collect()
}

pub fn jaccard(a: &HashSet<String>, b: &HashSet<String>) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    let intersection = a.intersection(b).count();
    let union = a.len() + b.len() - intersection;
    if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    }
}

pub fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    if a.is_empty() {
        return b.len();
    }
    if b.is_empty() {
        return a.len();
    }
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0; b.len() + 1];
    for (i, ca) in a.iter().enumerate() {
        cur[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let cost = usize::from(ca != cb);
            cur[j + 1] = (prev[j + 1] + 1).min(cur[j] + 1).min(prev[j] + cost);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

/// JSON with object keys sorted and no insignificant whitespace, so that two
/// argument objects that differ only in key order or formatting hash the same.
/// Non-JSON input is trimmed and returned as-is.
pub fn canonical_json(text: &str) -> String {
    match serde_json::from_str::<Value>(text) {
        Ok(v) => canonical_value(&v).to_string(),
        Err(_) => text.trim().to_string(),
    }
}

fn canonical_value(v: &Value) -> Value {
    match v {
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            let mut out = serde_json::Map::new();
            for k in keys {
                out.insert(k.clone(), canonical_value(&map[k]));
            }
            Value::Object(out)
        }
        Value::Array(items) => Value::Array(items.iter().map(canonical_value).collect()),
        Value::String(s) => Value::String(s.trim().to_string()),
        other => other.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_collapses_case_punctuation_and_space() {
        assert_eq!(
            normalize("Hello,   World!  It's ME."),
            "hello world it s me"
        );
    }

    #[test]
    fn jaccard_of_identical_is_one_and_disjoint_is_zero() {
        let a = shingles("the quick brown fox jumps", 3);
        let b = shingles("the quick brown fox jumps", 3);
        let c = shingles("completely different words here now", 3);
        assert_eq!(jaccard(&a, &b), 1.0);
        assert_eq!(jaccard(&a, &c), 0.0);
    }

    #[test]
    fn levenshtein_basics() {
        assert_eq!(levenshtein("kitten", "sitting"), 3);
        assert_eq!(levenshtein("", "abc"), 3);
        assert_eq!(levenshtein("same", "same"), 0);
    }

    #[test]
    fn secret_names_are_recognized() {
        for name in [
            "OPENAI_API_KEY",
            "LITELLM_MASTER_KEY",
            "token",
            "DB_PASSWORD",
            "aws-secret-access-key",
            "AUTH_HEADER",
            "passwd",
        ] {
            assert!(is_secret_name(name), "{name}");
        }
        for name in [
            "PORT",
            "CONTEXT_GUARD_RETENTION_DAYS",
            "monkey",
            "keyboard",
            "HOSTNAME",
        ] {
            assert!(!is_secret_name(name), "{name}");
        }
    }

    #[test]
    fn secret_values_are_redacted_and_ordinary_text_is_kept() {
        let cases = [
            (
                "set OPENAI_API_KEY=sk-live-abcdefghijklmnopqrstuvwxyz0123 and PORT=8080",
                "set OPENAI_API_KEY=[redacted] and PORT=8080",
            ),
            ("api_key: \"abc\" then token = x-y-z", "api_key: [redacted] then token = [redacted]"),
            (
                "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0In0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c",
                "Authorization: Bearer [redacted]",
            ),
            ("ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123 and AKIAIOSFODNN7EXAMPLE", "[redacted] and [redacted]"),
            ("llama.cpp listens on port 8080 at /etc/llama-swap/config.yaml", "llama.cpp listens on port 8080 at /etc/llama-swap/config.yaml"),
            ("the keyboard=us layout and monkey: 3", "the keyboard=us layout and monkey: 3"),
        ];
        for (input, expected) in cases {
            assert_eq!(redact_secrets(input), expected, "{input}");
        }
    }

    #[test]
    fn canonical_json_ignores_key_order_and_whitespace() {
        assert_eq!(
            canonical_json("{\"b\": 1, \"a\": \" x \"}"),
            canonical_json("{\"a\":\"x\",\"b\":1}")
        );
        assert_eq!(canonical_json("  raw text "), "raw text");
    }
}
