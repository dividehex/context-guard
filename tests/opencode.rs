//! opencode session messages through the real pipeline: the full
//! `{info, parts}` ship of one session as a fixture (built to the shapes
//! opencode's own SDK types declare, since a headless session needs a live
//! provider), an overlapping re-ship the way the TUI plugin's cursor rule
//! does it, and synthetic records for what the fixture did not exercise.

mod common;

use common::{harness, harness_with};
use serde_json::{json, Value};

const SESSION: &str = "opc_01a8f2a23ede4b2aa7f4c12f9d5a6b7c";
const FULL_BODY: &str = include_str!("fixtures/opencode-session.json");
const FULL_RECORDS: usize = 13;
const FULL_TURNS: i64 = 7;

fn fixture_records(body: &str) -> Vec<Value> {
    let body: Value = serde_json::from_str(body).unwrap();
    body["records"].as_array().unwrap().clone()
}

fn signals(result: &Value) -> Vec<String> {
    result["reasons"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["signal"].as_str().unwrap().to_string())
        .collect()
}

fn user(id: &str, text: &str) -> Value {
    json!({"info": {"id": id, "role": "user",
                    "time": {"created": 1_755_000_000_000_i64}},
           "parts": [{"type": "text", "text": text}]})
}

fn reply(id: &str, text: &str, input: u64, model: &str) -> Value {
    json!({"info": {"id": id, "role": "assistant", "modelID": model,
                    "time": {"created": 1_755_000_000_000_i64, "completed": 1_755_000_000_500_i64},
                    "tokens": {"input": input, "output": 20, "cache": {"read": 0, "write": 0}}},
           "parts": [{"type": "text", "text": text}]})
}

/// A first turn that reads notes.txt: establishes llama.cpp on port 8080 so
/// a later contradictory reply is a drift.
fn fact_turn(model: &str) -> Vec<Value> {
    vec![
        user("msg_u1", "Read notes.txt and tell me the port"),
        json!({"info": {"id": "msg_1", "role": "assistant", "modelID": model,
                 "time": {"created": 1_755_000_000_000_i64, "completed": 1_755_000_000_500_i64},
                 "tokens": {"input": 1000, "output": 5, "cache": {"read": 0, "write": 0}}},
        "parts": [
            {"type": "text", "text": "On it."},
            {"type": "tool", "callID": "call_1", "tool": "bash",
             "state": {"status": "completed",
                       "input": {"command": "cat notes.txt"},
                       "output": "llama.cpp is running on port 8080 behind the gateway.\n"}},
        ]}),
    ]
}

/// The fixture is one session end to end: user facts, a tool call with its
/// result, an assistant drift, three identical deploy calls that trip the
/// repetition signal, a compaction summary the monitor must skip, and a
/// final API error that is the backend rejecting the request for its
/// context window. Every assistant message is one turn; the failure also
/// scores as an overflow.
#[tokio::test]
async fn captured_session_scores_every_assistant_message_as_a_turn() {
    let h = harness().await;
    let (status, body) = h
        .post("/api/v1/ingest/opencode", FULL_BODY.as_bytes().to_vec())
        .await;
    assert_eq!(status, 202, "{body}");
    assert_eq!(body, json!({ "accepted": FULL_RECORDS, "dropped": 0 }));

    let first = h.wait_for_message(SESSION, "msg_1").await;
    assert_eq!(first["turn"], json!(1));
    assert_eq!(first["model"], json!("claude-sonnet-4-5"));
    assert_eq!(first["score"], json!(100));
    assert_eq!(first["status"], json!("healthy"));
    assert_eq!(first["context"]["prompt_tokens"], json!(2000));
    assert_eq!(first["context"]["limit"], json!(200_000));
    assert_eq!(first["signals"]["tool_anomalies"], json!(0));

    // The tool result and the drift: the reply contradicts what reading the
    // file established.
    let drift = h.wait_for_message(SESSION, "msg_2").await;
    assert_eq!(drift["turn"], json!(2));
    assert_eq!(signals(&drift), vec!["known_value_drift"], "{drift}");
    assert_eq!(drift["score"], json!(85));
    assert_eq!(
        drift["reasons"][0]["detail"],
        json!("assistant said port of llama.cpp 8000 but the conversation established 8080")
    );

    // Three identical deploy calls in the last five trip repetition.
    let repeated = h.wait_for_message(SESSION, "msg_5").await;
    assert_eq!(
        signals(&repeated),
        vec!["known_value_drift", "repeated_tool_call"],
        "{repeated}"
    );
    assert_eq!(repeated["score"], json!(80));

    // The rejected request is an overflow, not an anonymous failure; the
    // drift and the repetition from earlier in the window still count.
    let latest = h.wait_for_turns(SESSION, FULL_TURNS).await;
    assert_eq!(latest["message_id"], json!("msg_7:failure"));
    assert_eq!(latest["turn"], json!(FULL_TURNS));
    assert_eq!(
        signals(&latest),
        vec!["context_90", "known_value_drift", "repeated_tool_call"],
        "{latest}"
    );
    assert_eq!(latest["score"], json!(60));
    assert_eq!(latest["context"]["prompt_tokens"], json!(250000));
    assert!(
        latest["summary"]
            .as_str()
            .unwrap()
            .contains("🔴 context overflow"),
        "{latest}"
    );

    // One prompt per user message; tool-only turns share their prompt; the
    // compaction summary never became a turn.
    let (_, history) = h
        .get(&format!("/api/v1/conversations/{SESSION}/history"))
        .await;
    let results = history["results"].as_array().unwrap();
    let scores: Vec<u64> = results
        .iter()
        .map(|r| r["score"].as_u64().unwrap())
        .collect();
    assert_eq!(scores, vec![100, 85, 85, 85, 80, 80, 60]);
    let prompts: Vec<u64> = results
        .iter()
        .map(|r| r["prompt"].as_u64().unwrap())
        .collect();
    assert_eq!(prompts, vec![1, 2, 3, 3, 3, 4, 4]);
    assert_eq!(
        history["anomalies"].as_array().unwrap().len(),
        2,
        "{history}"
    );

    // Redelivery of the whole body is harmless.
    h.ingest_opencode(SESSION, &fixture_records(FULL_BODY), Some(8192))
        .await;
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    let again = h.wait_for_turns(SESSION, FULL_TURNS).await;
    assert_eq!(again["turn"], json!(FULL_TURNS));
}

/// The plugin re-ships from the last shipped message; its slice overlaps the
/// previous body. The overlap is deduplicated, the rest still appends, and a
/// duplicate never extends an event or double-counts an anomaly.
#[tokio::test]
async fn an_overlapping_reship_is_deduplicated_not_replayed() {
    let h = harness().await;
    let records = fixture_records(FULL_BODY);

    let first_ship = records[0..4].to_vec(); // through msg_2
    h.ingest_opencode(SESSION, &first_ship, Some(8192)).await;
    let third = h.wait_for_turns(SESSION, 2).await;
    assert_eq!(third["turn"], json!(2));
    assert_eq!(signals(&third), vec!["known_value_drift"]);

    // Re-ship from the last acked message (msg_2) forward, as the cursor rule
    // dictates.
    let reship = records[3..].to_vec();
    h.ingest_opencode(SESSION, &reship, Some(8192)).await;
    let latest = h.wait_for_turns(SESSION, FULL_TURNS).await;
    assert_eq!(latest["message_id"], json!("msg_7:failure"));

    let (_, history) = h
        .get(&format!("/api/v1/conversations/{SESSION}/history"))
        .await;
    let anomaly_signals: Vec<&str> = history["anomalies"]
        .as_array()
        .unwrap()
        .iter()
        .map(|a| a["signal"].as_str().unwrap())
        .collect();
    assert_eq!(
        anomaly_signals,
        vec!["known_value_drift", "repeated_tool_call"],
        "the overlap must not duplicate anomalies: {history}"
    );
}

/// A record that cannot be read is dropped and counted, not fatal; system
/// records are ignored; the envelope context limit reaches the scored turn.
#[tokio::test]
async fn bookkeeping_and_bad_records_do_not_stop_the_batch() {
    let h = harness().await;
    let records = vec![
        json!("not an object"),
        json!({"info": "not an object either", "parts": []}),
        json!({"info": {"id": "msg_sys", "role": "system", "text": "ignored"}, "parts": []}),
        user("msg_u_s", "now what port"),
        reply("msg_2", "hi", 100, "gpt-6-astra"),
    ];
    h.ingest_opencode(SESSION, &records, Some(8192)).await;
    let r = h.wait_for_message(SESSION, "msg_2").await;
    assert_eq!(r["turn"], json!(1));
    assert_eq!(r["model"], json!("gpt-6-astra"));
    assert_eq!(r["context"]["prompt_tokens"], json!(100));
    assert_eq!(r["context"]["limit"], json!(8192));
    let text = h.state.metrics.encode();
    assert!(
        text.contains("context_guard_events_dropped_total{reason=\"malformed\"} 2"),
        "{text}"
    );
}

/// A configured model limit beats the plugin's resolved envelope, and the
/// reply lands in the high context band alongside the drift.
#[tokio::test]
async fn configured_model_limit_beats_the_envelope() {
    let h = harness_with(|c| {
        c.model_limits.insert("gpt-6-astra".into(), 10_000);
    })
    .await;
    let mut records = fact_turn("gpt-6-astra");
    records.push(user("msg_u2", "what port?"));
    records.push(reply(
        "msg_d",
        "llama.cpp is on port 8000.",
        7800,
        "gpt-6-astra",
    ));
    h.ingest_opencode(SESSION, &records, Some(258_400)).await;
    let r = h.wait_for_message(SESSION, "msg_d").await;
    assert_eq!(r["context"]["limit"], json!(10_000), "{r}");
    assert_eq!(signals(&r), vec!["context_70", "known_value_drift"], "{r}");
    assert_eq!(r["score"], json!(80));
}
