mod common;

use axum::http::StatusCode;
use common::{assistant, harness, tool_call, user, PayloadBuilder};
use serde_json::json;

const CHAT: &str = "chat-abc";

#[tokio::test]
async fn scores_turns_with_context_pressure_drift_and_tool_signals() {
    let h = harness().await;

    // Turn 1: the user establishes a fact; plenty of context left.
    let t1 = PayloadBuilder::new("ev-1", CHAT, "msg-1")
        .messages(json!([user(
            "llama.cpp is running on port 8080, please remember that"
        )]))
        .response("Noted: llama.cpp is on port 8080.")
        .prompt_tokens(2000)
        .times(1.0, 2.0)
        .build();
    h.ingest(vec![t1]).await;
    let r1 = h.wait_for_message(CHAT, "msg-1").await;
    assert_eq!(r1["score"], json!(100));
    assert_eq!(r1["status"], json!("healthy"));
    assert_eq!(r1["turn"], json!(1));
    assert_eq!(r1["context"]["percent"], json!(20.0));
    assert_eq!(
        r1["summary"],
        json!("🟢 Context Guard 100 · healthy · 🟢 context 20% (2,000/10,000)")
    );

    // Turn 2: 78% context and the assistant contradicts the port.
    let t2 = PayloadBuilder::new("ev-2", CHAT, "msg-2")
        .messages(json!([
            user("llama.cpp is running on port 8080, please remember that"),
            assistant("Noted: llama.cpp is on port 8080."),
            user("restart it please")
        ]))
        .response("Restarting your llama.cpp server on port 8000 now.")
        .prompt_tokens(7800)
        .times(3.0, 4.0)
        .build();
    h.ingest(vec![t2.clone()]).await;
    let r2 = h.wait_for_message(CHAT, "msg-2").await;
    assert_eq!(r2["risk"], json!(20), "{r2}");
    assert_eq!(r2["score"], json!(80));
    assert_eq!(r2["status"], json!("good"));
    assert_eq!(r2["signals"]["known_value_drift"], json!(1));
    let signals: Vec<&str> = r2["reasons"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["signal"].as_str().unwrap())
        .collect();
    assert_eq!(signals, vec!["context_70", "known_value_drift"]);
    assert_eq!(
        r2["summary"],
        json!("🟢 Context Guard 80 · good · 🟡 context 78% (7,800/10,000) · 1 drift")
    );

    // Turn 3: three identical tool calls in one reply and an orphan tool result in the request.
    let t3 = PayloadBuilder::new("ev-3", CHAT, "msg-3")
        .messages(json!([
            user("llama.cpp is running on port 8080, please remember that"),
            assistant("Noted: llama.cpp is on port 8080."),
            user("restart it please"),
            assistant("Restarting your llama.cpp server on port 8000 now."),
            {"role": "tool", "tool_call_id": "call_ghost", "content": "Error: no such call"},
            user("check the status")
        ]))
        .response("")
        .tool_calls(json!([
            tool_call("call_a1", "restart", "{\"service\": \"llama.cpp\"}"),
            tool_call("call_a2", "restart", "{\"service\":\"llama.cpp\"}"),
            tool_call("call_a3", "restart", "{ \"service\" : \"llama.cpp\" }")
        ]))
        .prompt_tokens(3000)
        .times(5.0, 6.0)
        .build();
    h.ingest(vec![t3]).await;
    let r3 = h.wait_for_message(CHAT, "msg-3").await;
    // Window still holds the drift (15) + repeated call (5) + orphan result (20).
    assert_eq!(r3["risk"], json!(40), "{r3}");
    assert_eq!(r3["status"], json!("watch"));
    assert_eq!(r3["signals"]["tool_anomalies"], json!(1));
    assert_eq!(r3["signals"]["loop_events"], json!(1));
    assert_eq!(r3["signals"]["known_value_drift"], json!(1));

    // A background task with the same chat id is recorded but does not become a turn.
    let task = PayloadBuilder::new("ev-task", CHAT, "")
        .tag("x-openwebui-task: title_generation")
        .messages(json!([user("Create a title")]))
        .response("Port question")
        .set("stream", json!(false))
        .build();
    // Re-delivering turn 2 is a no-op.
    h.ingest(vec![task, t2]).await;
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    let latest = h.wait_for_turns(CHAT, 3).await;
    assert_eq!(latest["turn"], json!(3));
    assert_eq!(latest["message_id"], json!("msg-3"));

    // History and listing.
    let (status, history) = h
        .get(&format!("/api/v1/conversations/{CHAT}/history"))
        .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(history["results"].as_array().unwrap().len(), 3);
    assert_eq!(history["anomalies"].as_array().unwrap().len(), 3);
    let (status, list) = h.get("/api/v1/conversations").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(list["conversations"][0]["conversation_id"], json!(CHAT));
    assert_eq!(list["conversations"][0]["turns"], json!(3));
    let (status, filtered) = h.get("/api/v1/conversations?status=healthy").await;
    assert_eq!(status, StatusCode::OK);
    assert!(filtered["conversations"].as_array().unwrap().is_empty());

    // Lookup modes.
    let (status, body) = h
        .get(&format!(
            "/api/v1/conversations/{CHAT}/health?message_id=nope"
        ))
        .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(body["error"]["code"], json!("not_scored_yet"));
    let (status, body) = h
        .get(&format!("/api/v1/conversations/{CHAT}/health?after=5"))
        .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["turn"], json!(3));
    let (status, _) = h
        .get(&format!("/api/v1/conversations/{CHAT}/health?after=999"))
        .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let (status, body) = h.get("/api/v1/conversations/unknown/health").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(body["error"]["code"], json!("unknown_conversation"));

    // Metrics carry aggregate counters only.
    let (status, _) = h.get("/metrics").await;
    assert_eq!(status, StatusCode::OK);
    let text = h.state.metrics.encode();
    assert!(
        text.contains("context_guard_events_received_total{kind=\"chat\"} 4"),
        "received counts the redelivered duplicate too: {text}"
    );
    assert!(text.contains("context_guard_events_received_total{kind=\"task\"} 1"));
    assert!(text.contains("context_guard_known_value_drift_total{model=\"test-model\"} 1"));
    assert!(!text.contains(CHAT));
}

#[tokio::test]
async fn window_lets_a_conversation_recover() {
    let h = harness_with_window(2).await;
    let drift = PayloadBuilder::new("w-1", "chat-w", "m1")
        .messages(json!([user("the API is on port 4000")]))
        .response("Your API on port 4100 is fine.")
        .times(1.0, 2.0)
        .build();
    h.ingest(vec![drift]).await;
    assert_eq!(h.wait_for_message("chat-w", "m1").await["risk"], json!(15));
    for i in 2..=3 {
        let clean = PayloadBuilder::new(&format!("w-{i}"), "chat-w", &format!("m{i}"))
            .messages(json!([
                user("the API is on port 4000"),
                assistant("Your API on port 4100 is fine."),
                user(&format!("thanks {i}"))
            ]))
            .response(&format!(
                "You are welcome, message number {i} of this chat."
            ))
            .times(f64::from(i) * 2.0, f64::from(i) * 2.0 + 1.0)
            .build();
        h.ingest(vec![clean]).await;
    }
    assert_eq!(
        h.wait_for_message("chat-w", "m2").await["risk"],
        json!(15),
        "still inside the 2-turn window"
    );
    assert_eq!(
        h.wait_for_message("chat-w", "m3").await["risk"],
        json!(0),
        "drift aged out"
    );
}

async fn harness_with_window(turns: u32) -> common::Harness {
    common::harness_with(|c| c.scoring.window_turns = turns).await
}
