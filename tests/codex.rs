//! Codex CLI rollouts through the real pipeline: the two bodies the hook
//! shipped for one `codex exec` run as fixtures (a `PostToolUse` ship and the
//! `Stop` ship that overlaps it), and synthetic records for what that session
//! did not exercise.

mod common;

use common::{harness, harness_with};
use serde_json::{json, Value};

const SESSION: &str = "01a0a08f-68b2-70b0-b3aa-6fa4b6d73962";
const POST_TOOL_USE: &str = include_str!("fixtures/codex-0.154.0-posttooluse.json");
const STOP: &str = include_str!("fixtures/codex-0.154.0-stop.json");
const FIRST_REPLY: &str = "msg_022ec2bde5e79c9c016aa81458ac8c87d1ab433e2bbb9adee0";
const LAST_REPLY: &str = "msg_022ec2bde5e79c9c016aa8145d7acc87d18c1aff674b36dc72";

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

/// A `codex exec` run of one prompt: the model read notes.txt and ran `ls`
/// three times inside one code-mode `exec` call, then answered. Two API
/// responses, two turns. The first body is what the hook shipped on
/// `PostToolUse` (the prompt and the closed first response), the second what
/// `Stop` shipped, restarting at that response as the cursor rule says.
#[tokio::test]
async fn captured_session_scores_every_api_response_as_a_turn() {
    let h = harness().await;
    let (status, body) = h
        .post("/api/v1/ingest/codex", POST_TOOL_USE.as_bytes().to_vec())
        .await;
    assert_eq!(status, 202, "{body}");
    assert_eq!(body, json!({ "accepted": 10, "dropped": 0 }));
    let first = h.wait_for_turns(SESSION, 1).await;
    assert_eq!(first["message_id"], json!(FIRST_REPLY));
    assert_eq!(first["context"]["prompt_tokens"], json!(14_376));

    let (status, body) = h
        .post("/api/v1/ingest/codex", STOP.as_bytes().to_vec())
        .await;
    assert_eq!(status, 202, "{body}");
    assert_eq!(body, json!({ "accepted": 8, "dropped": 0 }));
    let latest = h.wait_for_turns(SESSION, 2).await;
    assert_eq!(latest["model"], json!("gpt-6-astra"));
    assert_eq!(latest["message_id"], json!(LAST_REPLY));
    assert_eq!(latest["score"], json!(100));
    assert_eq!(latest["status"], json!("healthy"));
    // input_tokens of the last response against the window from `task_started`.
    assert_eq!(latest["context"]["prompt_tokens"], json!(14_683));
    assert_eq!(latest["context"]["limit"], json!(258_400));
    assert_eq!(
        latest["summary"],
        json!("🟢 Context Guard 100 · healthy · 🟢 context 6% (14,683/258,400)")
    );
    // The tool output shipped twice was matched to its call once.
    assert_eq!(latest["signals"]["tool_anomalies"], json!(0), "{latest}");

    let (_, history) = h
        .get(&format!("/api/v1/conversations/{SESSION}/history"))
        .await;
    let results = history["results"].as_array().unwrap();
    let scores: Vec<u64> = results
        .iter()
        .map(|r| r["score"].as_u64().unwrap())
        .collect();
    assert_eq!(scores, vec![100, 100]);
    let prompts: Vec<u64> = results
        .iter()
        .map(|r| r["prompt"].as_u64().unwrap())
        .collect();
    assert_eq!(prompts, vec![1, 1], "one user prompt, two API responses");
    assert_eq!(
        history["anomalies"].as_array().unwrap().len(),
        0,
        "{history}"
    );

    let (_, list) = h.get("/api/v1/conversations").await;
    let c = &list["conversations"][0];
    assert_eq!(c["conversation_id"], json!(SESSION));
    assert_eq!(c["id_source"], json!("session"));

    // Redelivery of a whole body is harmless.
    h.ingest_codex(
        SESSION,
        &fixture_records(STOP),
        Some("gpt-6-astra"),
        Some(258_400),
    )
    .await;
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    let again = h.wait_for_turns(SESSION, 2).await;
    assert_eq!(again["turn"], json!(2));
}

fn record(kind: &str, payload: Value) -> Value {
    json!({"timestamp": "2026-09-14T15:35:55.000Z", "type": kind, "payload": payload})
}

fn turn_context() -> Value {
    record(
        "turn_context",
        json!({"turn_id": "turn_1", "model": "gpt-6-astra", "cwd": "/w"}),
    )
}

fn user(id: &str, text: &str) -> Value {
    record(
        "response_item",
        json!({"type": "message", "id": id, "role": "user",
               "content": [{"type": "input_text", "text": text}],
               "internal_chat_message_metadata_passthrough": {"content_item_kinds": ["user.text"]}}),
    )
}

fn reply(id: &str, text: &str) -> Value {
    record(
        "response_item",
        json!({"type": "message", "id": id, "role": "assistant", "phase": "final_answer",
               "content": [{"type": "output_text", "text": text}]}),
    )
}

fn call(id: &str, call_id: &str, input: &str) -> Value {
    record(
        "response_item",
        json!({"type": "custom_tool_call", "id": id, "call_id": call_id, "name": "exec",
               "status": "completed", "input": input}),
    )
}

fn output(call_id: &str, text: &str) -> Value {
    record(
        "response_item",
        json!({"type": "custom_tool_call_output", "id": "ctco_x", "call_id": call_id,
               "output": [{"type": "input_text", "text": text}]}),
    )
}

fn usage(response_id: &str, input_tokens: u64) -> Value {
    record(
        "token_usage_record",
        json!({"thread_id": SESSION, "turn_id": "turn_1", "response_id": response_id,
               "usage": {"input_tokens": input_tokens, "cached_input_tokens": 0,
                         "output_tokens": 20, "total_tokens": input_tokens + 20}}),
    )
}

/// A fact learned from a tool output is contradicted by the assistant: drift.
#[tokio::test]
async fn drift_from_a_tool_output_and_a_configured_model_limit() {
    let h = harness_with(|c| {
        c.model_limits.insert("gpt-6-astra".into(), 10_000);
    })
    .await;
    let records = vec![
        turn_context(),
        user("msg_u1", "Read notes.txt and tell me the port"),
        call("ctc_1", "call_1", "cat notes.txt"),
        usage("resp_1", 2000),
        output(
            "call_1",
            "llama.cpp is running on port 8080 behind the gateway.\n",
        ),
        reply("msg_2", "llama.cpp is on port 8000."),
        usage("resp_2", 7800),
    ];
    h.ingest_codex(SESSION, &records, Some("gpt-6-astra"), Some(258_400))
        .await;
    let r = h.wait_for_message(SESSION, "msg_2").await;
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

/// The anomaly window is counted in user prompts, not API responses: a drift
/// caught early in a long tool loop still counts at the end of that loop,
/// and ages out only after enough later prompts.
#[tokio::test]
async fn window_counts_user_prompts_not_api_responses() {
    let h = harness_with(|c| c.scoring.window_turns = 2).await;
    let mut records = vec![
        turn_context(),
        user(
            "msg_u1",
            "Read notes.txt, then list the directory many times",
        ),
        call("ctc_1", "call_1", "cat notes.txt"),
        usage("resp_1", 100),
        output("call_1", "llama.cpp is running on port 8080"),
        reply("msg_2", "llama.cpp is on port 8000, checking further."),
        call("ctc_2", "call_2", "ls -a"),
        usage("resp_2", 100),
    ];
    // Twelve more tool round-trips inside the same prompt, each with different arguments.
    for i in 3..15 {
        records.push(output(&format!("call_{}", i - 1), "notes.txt"));
        records.push(call(
            &format!("ctc_{i}"),
            &format!("call_{i}"),
            &format!("ls -{i}"),
        ));
        records.push(usage(&format!("resp_{i}"), 100));
    }
    records.push(output("call_14", "notes.txt"));
    records.push(reply("msg_15", "Done."));
    records.push(usage("resp_15", 100));
    h.ingest_codex(SESSION, &records, Some("gpt-6-astra"), Some(258_400))
        .await;
    let end_of_loop = h.wait_for_message(SESSION, "msg_15").await;
    assert_eq!(end_of_loop["turn"], json!(15));
    assert_eq!(end_of_loop["prompt"], json!(1));
    assert_eq!(
        end_of_loop["score"],
        json!(85),
        "the drift from turn 2 still counts: same prompt"
    );

    // Two more user prompts: the drift is now outside a two-prompt window.
    let later = vec![
        turn_context(),
        user("msg_u2", "thanks"),
        reply("msg_16", "You're welcome."),
        usage("resp_16", 100),
        turn_context(),
        user("msg_u3", "bye"),
        reply("msg_17", "Bye."),
        usage("resp_17", 100),
    ];
    h.ingest_codex(SESSION, &later, Some("gpt-6-astra"), Some(258_400))
        .await;
    let after = h.wait_for_message(SESSION, "msg_17").await;
    assert_eq!(after["prompt"], json!(3));
    assert_eq!(after["score"], json!(100), "{after}");
    let (_, explain) = h
        .get(&format!("/api/v1/conversations/{SESSION}/explain"))
        .await;
    assert_eq!(explain["scoring"]["window_from_prompt"], json!(2));
    assert_eq!(explain["issues"][0]["prompt"], json!(1));
    assert_eq!(explain["issues"][0]["counting"], json!(false));
}

/// Codex ends a turn that hit the context window with a `task_complete`
/// event carrying the error; it is scored as an overflow.
#[tokio::test]
async fn a_turn_that_ran_out_of_context_scores_as_overflow() {
    let h = harness().await;
    let records = vec![
        turn_context(),
        user("msg_u1", "carry on"),
        record(
            "event_msg",
            json!({"type": "task_complete", "turn_id": "turn_1", "last_agent_message": null,
                   "error": {"message": "Codex ran out of room in the model's context window. Start a new thread or clear earlier history before retrying.",
                             "codex_error_info": "context_window_exceeded"}}),
        ),
    ];
    h.ingest_codex(SESSION, &records, Some("gpt-6-astra"), Some(258_400))
        .await;
    let r = h.wait_for_message(SESSION, "turn_1:failure").await;
    assert_eq!(signals(&r), vec!["context_90"], "{r}");
    assert_eq!(r["score"], json!(80));
    assert!(
        r["summary"]
            .as_str()
            .unwrap()
            .contains("🔴 context overflow"),
        "{r}"
    );
}

/// Session metadata and world state are ignored; a record that cannot be
/// read is dropped and counted, not fatal.
#[tokio::test]
async fn bookkeeping_and_bad_records_do_not_stop_the_batch() {
    let h = harness().await;
    let records = vec![
        record(
            "session_meta",
            json!({"id": SESSION, "base_instructions": {"text": "…"}}),
        ),
        record("world_state", json!({"full": true})),
        json!("not an object"),
        record("response_item", json!({"id": "no type"})),
        turn_context(),
        user("msg_u1", "hello"),
        reply("msg_1", "hi"),
        usage("resp_1", 100),
    ];
    h.ingest_codex(SESSION, &records, None, None).await;
    let r = h.wait_for_message(SESSION, "msg_1").await;
    assert_eq!(r["turn"], json!(1));
    assert_eq!(
        r["context"],
        json!({ "prompt_tokens": 100, "limit": Value::Null, "percent": Value::Null })
    );
    let text = h.state.metrics.encode();
    assert!(
        text.contains("context_guard_events_dropped_total{reason=\"malformed\"} 2"),
        "{text}"
    );
}
