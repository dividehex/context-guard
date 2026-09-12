//! Deterministic text helpers shared by the signal modules.

use std::collections::HashSet;

use serde_json::Value;
use sha2::{Digest, Sha256};

pub fn sha256_hex(input: &str) -> String {
    hex::encode(Sha256::digest(input.as_bytes()))
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
    fn canonical_json_ignores_key_order_and_whitespace() {
        assert_eq!(
            canonical_json("{\"b\": 1, \"a\": \" x \"}"),
            canonical_json("{\"a\":\"x\",\"b\":1}")
        );
        assert_eq!(canonical_json("  raw text "), "raw text");
    }
}
