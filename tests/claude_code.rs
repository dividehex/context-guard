//! Claude Code transcripts through the real pipeline: the hook's captured body
//! as a fixture, the cursor rule's overlap, and synthetic records for the
//! signals the captured session did not trigger.

mod common;

use common::{harness, harness_with};
use serde_json::{json, Value};

const SESSION: &str = "880138cf-78cd-4d41-9940-a4aa38c2aaec";
const FIXTURE: &str = include_str!("fixtures/claude-code-2.1.270-session.json");

fn fixture_records() -> Vec<Value> {
    let body: Value = serde_json::from_str(FIXTURE).unwrap();
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

/// A headless `claude -p` session of two prompts: a Read of notes.txt, then
/// three identical `ls` calls in one reply. Captured with
/// CONTEXT_GUARD_CAPTURE_DIR from what the hook actually shipped.
#[tokio::test]
async fn captured_session_scores_every_api_call_as_a_turn() {
    let h = harness().await;
    let (status, body) = h
        .post("/api/v1/ingest/claude-code", FIXTURE.as_bytes().to_vec())
        .await;
    assert_eq!(status, 202, "{body}");
    assert_eq!(body, json!({ "accepted": 16, "dropped": 0 }));

    let latest = h.wait_for_turns(SESSION, 4).await;
    assert_eq!(latest["model"], json!("claude-haiku-4-5-20251001"));
    assert_eq!(latest["score"], json!(95));
    assert_eq!(latest["status"], json!("healthy"));
    // input + cache_read + cache_creation of the last call, against the hook's window hint.
    assert_eq!(
        latest["context"],
        json!({ "prompt_tokens": 22207, "limit": 200000, "percent": 11.1035 })
    );
    assert_eq!(
        latest["summary"],
        json!("🟢 Context Guard 95 · healthy · 🟢 context 11% (22,207/200,000) · 1 repeated call")
    );

    let (_, history) = h
        .get(&format!("/api/v1/conversations/{SESSION}/history"))
        .await;
    let scores: Vec<u64> = history["results"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["score"].as_u64().unwrap())
        .collect();
    assert_eq!(scores, vec![100, 100, 95, 95]);
    let anomalies = history["anomalies"].as_array().unwrap();
    assert_eq!(anomalies.len(), 1, "{history}");
    assert_eq!(anomalies[0]["signal"], json!("repeated_tool_call"));
    assert_eq!(anomalies[0]["turn"], json!(3));
    assert_eq!(
        anomalies[0]["detail"],
        json!("Bash called with identical arguments 3 times")
    );

    let (_, list) = h.get("/api/v1/conversations").await;
    let c = &list["conversations"][0];
    assert_eq!(c["conversation_id"], json!(SESSION));
    assert_eq!(c["id_source"], json!("session"));

    // Redelivery of the whole body is harmless.
    h.ingest_claude_code(&fixture_records(), Some(200_000))
        .await;
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    let again = h.wait_for_turns(SESSION, 4).await;
    assert_eq!(again["turn"], json!(4));
}

/// The hook resends from the first record of the last assistant group, so the
/// second ship overlaps the first by one completion. Same result as one ship.
#[tokio::test]
async fn overlapping_ships_score_the_same_as_one() {
    let records = fixture_records();
    // Split after the first tool result: the second ship restarts at req_A's first record.
    let second_group_start = records
        .iter()
        .position(|r| {
            r["requestId"]
                .as_str()
                .is_some_and(|id| id.ends_with("Suucqy"))
        })
        .unwrap();
    let first_tool_result = records
        .iter()
        .position(|r| r["type"] == "user" && r["message"]["content"].is_array())
        .unwrap();
    let first = &records[..=first_tool_result];
    let second = &records[second_group_start..];

    let h = harness().await;
    h.ingest_claude_code(first, Some(200_000)).await;
    h.wait_for_turns(SESSION, 1).await;
    h.ingest_claude_code(second, Some(200_000)).await;
    let latest = h.wait_for_turns(SESSION, 4).await;
    assert_eq!(latest["score"], json!(95));
    let (_, history) = h
        .get(&format!("/api/v1/conversations/{SESSION}/history"))
        .await;
    assert_eq!(history["anomalies"].as_array().unwrap().len(), 1);
    // The tool result shipped twice was matched to its call once: no orphan, no unknown id.
    assert_eq!(latest["signals"]["tool_anomalies"], json!(0));
}

/// A headless session whose first API call streamed two tool calls with the
/// first tool's result written between them (Claude Code runs tools as their
/// blocks arrive). Three API calls, three turns; the interleaved result is
/// part of the second request, not a split of the first.
#[tokio::test]
async fn interleaved_tool_results_stay_one_turn_per_api_call() {
    const INTERLEAVED: &str = include_str!("fixtures/claude-code-2.1.270-interleaved.json");
    let body: Value = serde_json::from_str(INTERLEAVED).unwrap();
    let records = body["records"].as_array().unwrap();
    let api_calls: std::collections::BTreeSet<&str> = records
        .iter()
        .filter_map(|r| r["requestId"].as_str())
        .collect();
    let (session, first_group) = records
        .iter()
        .find(|r| r["type"] == "assistant")
        .map(|r| {
            (
                r["sessionId"].as_str().unwrap(),
                r["requestId"].as_str().unwrap(),
            )
        })
        .unwrap();
    let interleaved = records
        .iter()
        .filter(|r| r["requestId"].as_str() == Some(first_group))
        .count();
    assert!(
        interleaved >= 3,
        "the fixture's first call has several records"
    );

    let h = harness().await;
    let (status, _) = h
        .post(
            "/api/v1/ingest/claude-code",
            INTERLEAVED.as_bytes().to_vec(),
        )
        .await;
    assert_eq!(status, 202);
    let latest = h.wait_for_turns(session, api_calls.len() as i64).await;
    assert_eq!(latest["turn"], json!(3));
    assert_eq!(latest["score"], json!(100));
    assert_eq!(latest["signals"]["tool_anomalies"], json!(0), "{latest}");
    let (_, history) = h
        .get(&format!("/api/v1/conversations/{session}/history"))
        .await;
    assert_eq!(
        history["anomalies"].as_array().unwrap().len(),
        0,
        "{history}"
    );
}

fn user(uuid: &str, content: Value) -> Value {
    json!({
        "type": "user", "uuid": uuid, "sessionId": SESSION, "isSidechain": false,
        "timestamp": "2026-09-12T18:47:10.000Z",
        "message": {"role": "user", "content": content}
    })
}

fn assistant(uuid: &str, request_id: &str, blocks: Value, prompt_tokens: u64) -> Value {
    json!({
        "type": "assistant", "uuid": uuid, "requestId": request_id, "sessionId": SESSION,
        "isSidechain": false, "timestamp": "2026-09-12T18:47:12.000Z",
        "message": {
            "model": "claude-haiku-4-5-20251001", "role": "assistant", "content": blocks,
            "usage": {"input_tokens": prompt_tokens, "cache_read_input_tokens": 0,
                      "cache_creation_input_tokens": 0, "output_tokens": 20}
        }
    })
}

/// A fact learned from a Read tool result (with the tool's line-number
/// prefix) is contradicted by the assistant: drift. The value the user never
/// mentioned is what makes it drift.
#[tokio::test]
async fn drift_from_a_tool_result_and_a_configured_model_limit() {
    let h = harness_with(|c| {
        c.model_limits
            .insert("claude-haiku-4-5-20251001".into(), 10_000);
    })
    .await;
    let records = vec![
        user("u1", json!("Read notes.txt and tell me the port")),
        assistant(
            "a1",
            "req_A",
            json!([{"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"file_path": "/x/notes.txt"}}]),
            2000,
        ),
        user(
            "u2",
            json!([{"type": "tool_result", "tool_use_id": "toolu_1", "content": "     1\tllama.cpp is running on port 8080 behind the gateway.\n     2\t"}]),
        ),
        assistant(
            "a2",
            "req_B",
            json!([{"type": "text", "text": "llama.cpp is on port 8000."}]),
            7800,
        ),
    ];
    h.ingest_claude_code(&records, Some(200_000)).await;
    let r = h.wait_for_message(SESSION, "a2").await;
    assert_eq!(r["turn"], json!(2));
    assert_eq!(signals(&r), vec!["context_70", "known_value_drift"], "{r}");
    assert_eq!(
        r["context"]["limit"],
        json!(10_000),
        "config beats the hook's hint"
    );
    assert_eq!(r["score"], json!(80));
    assert_eq!(
        r["reasons"][1]["detail"],
        json!("assistant said port of llama.cpp 8000 but the conversation established 8080")
    );
}

/// The anomaly window is counted in user prompts, not API calls: a drift
/// caught early in a long tool loop still counts at the end of that loop,
/// and ages out only after enough later prompts.
#[tokio::test]
async fn window_counts_user_prompts_not_api_calls() {
    let h = harness_with(|c| c.scoring.window_turns = 2).await;
    let mut records = vec![
        user(
            "u1",
            json!("Read notes.txt, then list the directory many times"),
        ),
        assistant(
            "a1",
            "req_1",
            json!([{"type": "tool_use", "id": "t1", "name": "Read", "input": {"file_path": "/x/notes.txt"}}]),
            100,
        ),
        user(
            "r1",
            json!([{"type": "tool_result", "tool_use_id": "t1", "content": "llama.cpp is running on port 8080"}]),
        ),
        assistant(
            "a2",
            "req_2",
            json!([{"type": "text", "text": "llama.cpp is on port 8000, checking further."}, {"type": "tool_use", "id": "t2", "name": "Bash", "input": {"command": "ls -a"}}]),
            100,
        ),
    ];
    // Twelve more tool round-trips inside the same prompt, each with different arguments.
    for i in 3..15 {
        records.push(user(
            &format!("r{i}"),
            json!([{"type": "tool_result", "tool_use_id": format!("t{}", i - 1), "content": "notes.txt"}]),
        ));
        records.push(assistant(
            &format!("a{i}"),
            &format!("req_{i}"),
            json!([{"type": "tool_use", "id": format!("t{i}"), "name": "Bash", "input": {"command": format!("ls -{i}")}}]),
            100,
        ));
    }
    records.push(user(
        "r15",
        json!([{"type": "tool_result", "tool_use_id": "t14", "content": "notes.txt"}]),
    ));
    records.push(assistant(
        "a15",
        "req_15",
        json!([{"type": "text", "text": "Done."}]),
        100,
    ));
    h.ingest_claude_code(&records, Some(200_000)).await;
    let end_of_loop = h.wait_for_message(SESSION, "a15").await;
    assert_eq!(end_of_loop["turn"], json!(15));
    assert_eq!(end_of_loop["prompt"], json!(1));
    assert_eq!(
        end_of_loop["score"],
        json!(85),
        "the drift from turn 2 still counts: same prompt"
    );

    // Two more user prompts: the drift is now outside a two-prompt window.
    let later = vec![
        user("u2", json!("thanks")),
        assistant(
            "a16",
            "req_16",
            json!([{"type": "text", "text": "You're welcome."}]),
            100,
        ),
        user("u3", json!("bye")),
        assistant(
            "a17",
            "req_17",
            json!([{"type": "text", "text": "Bye."}]),
            100,
        ),
    ];
    h.ingest_claude_code(&later, Some(200_000)).await;
    let after = h.wait_for_message(SESSION, "a17").await;
    assert_eq!(after["prompt"], json!(3));
    assert_eq!(after["score"], json!(100), "{after}");
    let (_, explain) = h
        .get(&format!("/api/v1/conversations/{SESSION}/explain"))
        .await;
    assert_eq!(explain["scoring"]["window_from_prompt"], json!(2));
    assert_eq!(explain["issues"][0]["prompt"], json!(1));
    assert_eq!(explain["issues"][0]["counting"], json!(false));
}

/// Claude Code writes a rejected request as an assistant record flagged
/// `isApiErrorMessage`; a context overflow is scored as one.
#[tokio::test]
async fn api_error_for_an_oversized_prompt_scores_as_overflow() {
    let h = harness().await;
    let mut err = assistant(
        "e1",
        "req_E",
        json!([{"type": "text", "text": "API Error: 400 {\"type\":\"error\",\"error\":{\"type\":\"invalid_request_error\",\"message\":\"prompt is too long: 213265 tokens > 200000 maximum\"}}"}]),
        0,
    );
    err["isApiErrorMessage"] = json!(true);
    let records = vec![user("u1", json!("carry on")), err];
    h.ingest_claude_code(&records, Some(200_000)).await;
    let r = h.wait_for_message(SESSION, "e1").await;
    assert_eq!(signals(&r), vec!["context_90"], "{r}");
    assert_eq!(
        r["reasons"][0]["detail"],
        json!("request of 213265 tokens exceeded the model's context window")
    );
    assert_eq!(r["context"]["prompt_tokens"], json!(213_265));
    assert_eq!(r["score"], json!(80));
    assert!(
        r["summary"]
            .as_str()
            .unwrap()
            .contains("🔴 context overflow (213,265/200,000)"),
        "{r}"
    );
}

/// Metadata records and side chains are ignored; a record that should be a
/// message but is unreadable is dropped and counted, not fatal.
#[tokio::test]
async fn bookkeeping_and_bad_records_do_not_stop_the_batch() {
    let h = harness().await;
    let mut side = user("s1", json!("subagent prompt"));
    side["isSidechain"] = json!(true);
    let records = vec![
        json!({"type": "attachment", "attachment": {"type": "environment"}}),
        side,
        json!({"type": "user", "sessionId": SESSION, "message": {"content": 42}}),
        user("u1", json!("hello")),
        assistant("a1", "req_A", json!([{"type": "text", "text": "hi"}]), 100),
    ];
    h.ingest_claude_code(&records, None).await;
    let r = h.wait_for_message(SESSION, "a1").await;
    assert_eq!(r["turn"], json!(1));
    assert_eq!(
        r["context"],
        json!({ "prompt_tokens": 100, "limit": Value::Null, "percent": Value::Null })
    );
    let text = h.state.metrics.encode();
    assert!(
        text.contains("context_guard_events_dropped_total{reason=\"malformed\"} 1"),
        "{text}"
    );
}
