//! `cargo run --example extract`: drive the known-value extractor and the drift
//! rule directly, the way `Monitor::process` does, without LiteLLM, the worker
//! or a database. Used by `scripts/extraction_recall/` to measure extraction
//! recall over thousands of synthetic statements per second.
//!
//! Stdin: one JSON case per line, `{"id": "...", "messages": [{"role": "user" |
//! "tool" | "assistant", "text": "..."}]}`. User and tool messages build the
//! registry and, through their identifiers, the anchor lexicon, exactly as the
//! monitor does; assistant messages are claims. Optional argv[1]:
//! comma-separated container-name prefixes (default `ai-`).
//!
//! Stdout: one JSON result per case, `{"id", "registry", "claims", "drift"}`.
//! A line that is not a valid case is reported on stderr and skipped, so one bad
//! line never loses a batch.

use std::io::{self, BufRead, Write};

use context_guard::monitor::identifiers;
use context_guard::monitor::known_values::{
    detect_drift, extract, extract_claims, Drift, Extracted, KnownValue, Lexicon,
};
use serde::Deserialize;
use serde_json::{json, Value};

#[derive(Deserialize)]
struct Case {
    id: String,
    messages: Vec<CaseMessage>,
}

#[derive(Deserialize)]
struct CaseMessage {
    role: String,
    text: String,
}

fn main() -> anyhow::Result<()> {
    let prefixes: Vec<String> = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "ai-".to_string())
        .split(',')
        .filter(|p| !p.is_empty())
        .map(str::to_string)
        .collect();

    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = io::BufWriter::new(stdout.lock());
    for (line_no, line) in stdin.lock().lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let case: Case = match serde_json::from_str(&line) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("line {}: not a case: {e}", line_no + 1);
                continue;
            }
        };
        let result = run_case(&case, &prefixes);
        serde_json::to_writer(&mut out, &result)?;
        out.write_all(b"\n")?;
    }
    out.flush()?;
    Ok(())
}

fn run_case(case: &Case, prefixes: &[String]) -> Value {
    let mut lexicon = Lexicon::default();
    for m in case.messages.iter().filter(|m| m.role != "assistant") {
        for id in identifiers::extract(&m.text, prefixes) {
            lexicon.add_identifier(&id.value);
        }
        lexicon.add_text(&m.text);
    }
    let mut registry: Vec<KnownValue> = Vec::new();
    let mut claims: Vec<Extracted> = Vec::new();
    for m in &case.messages {
        match m.role.as_str() {
            "user" | "tool" => {
                for e in extract(&m.text, prefixes, &lexicon) {
                    let k = KnownValue {
                        kind: e.kind,
                        anchor: e.anchor,
                        value: e.value,
                    };
                    if !registry.contains(&k) {
                        registry.push(k);
                    }
                }
            }
            "assistant" => claims.extend(extract_claims(&m.text, prefixes, &lexicon)),
            _ => {}
        }
    }
    let drift = detect_drift(&claims, &registry);
    json!({
        "id": case.id,
        "registry": registry.iter().map(known_json).collect::<Vec<_>>(),
        "claims": claims.iter().map(extracted_json).collect::<Vec<_>>(),
        "drift": drift.iter().map(drift_json).collect::<Vec<_>>(),
    })
}

fn known_json(k: &KnownValue) -> Value {
    json!({"kind": k.kind.as_str(), "anchor": k.anchor, "value": k.value})
}

fn extracted_json(e: &Extracted) -> Value {
    json!({"kind": e.kind.as_str(), "anchor": e.anchor, "value": e.value})
}

fn drift_json(d: &Drift) -> Value {
    json!({"kind": d.kind.as_str(), "anchor": d.anchor, "known": d.known, "claimed": d.claimed})
}
