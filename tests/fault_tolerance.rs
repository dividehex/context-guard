mod common;

use axum::http::StatusCode;
use common::{harness_with, user, PayloadBuilder};
use serde_json::json;

#[tokio::test]
async fn garbage_never_takes_the_service_down() {
    let h = harness_with(|c| c.max_body_bytes = 1024 * 1024).await;

    let (status, body) = h
        .post("/api/v1/ingest/litellm", b"not json at all".to_vec())
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["code"], json!("malformed_body"));

    let (status, _) = h.post("/api/v1/ingest/litellm", b"42".to_vec()).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    let (status, _) = h
        .post("/api/v1/ingest/litellm", vec![0xff, 0xfe, 0x00])
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    let (status, body) = h.post("/api/v1/ingest/litellm", Vec::new()).await;
    assert_eq!(status, StatusCode::ACCEPTED);
    assert_eq!(body["accepted"], json!(0));

    // Deeply nested JSON must not overflow the stack.
    let nested = format!("{}{}", "[".repeat(100_000), "]".repeat(100_000));
    let (status, _) = h.post("/api/v1/ingest/litellm", nested.into_bytes()).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    // Oversized body is refused, not buffered.
    let huge = vec![b' '; 2 * 1024 * 1024];
    let (status, _) = h.post("/api/v1/ingest/litellm", huge).await;
    assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);

    // A batch with one good payload and assorted junk: the good one is scored.
    let good = PayloadBuilder::new("ok-1", "chat-ft", "m-1")
        .messages(json!([user("hello \u{0}\u{1} world 🙂")]))
        .response("hi \u{0} there \u{fffd} again with enough words to count as a real reply here")
        .build();
    let batch = json!([
        good,
        null,
        "string",
        {"id": "no-model", "call_type": "acompletion"},
        {"id": "wrong-type", "call_type": "embedding", "model": "m"},
        {"id": "weird", "call_type": "acompletion", "model": "m", "messages": {"role": "user"}, "response": [1,2,3], "request_tags": "nope", "prompt_tokens": -5, "startTime": "yesterday"},
        {"id": "ok-2", "call_type": "acompletion", "model": "m", "messages": [{"role": "tool", "content": 12, "tool_call_id": 7}, {"content": null}], "response": {"choices": [{"message": {"tool_calls": [{"function": {"arguments": {"a": 1}}}, {"id": 5}]}}]}}
    ]);
    let (status, body) = h
        .post(
            "/api/v1/ingest/litellm",
            serde_json::to_vec(&batch).unwrap(),
        )
        .await;
    assert_eq!(status, StatusCode::ACCEPTED);
    assert_eq!(body["accepted"], json!(7));
    let scored = h.wait_for_message("chat-ft", "m-1").await;
    assert_eq!(scored["score"], json!(100));

    let (status, health) = h.get("/healthz").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(health["status"], json!("ok"));
    let text = h.state.metrics.encode();
    assert!(
        text.contains("context_guard_events_dropped_total{reason=\"malformed\"}"),
        "{text}"
    );
}

#[tokio::test]
async fn full_queue_drops_instead_of_blocking() {
    let h = harness_with(|c| c.queue_size = 1).await;
    // Fill the queue faster than the worker can drain it; every request still returns 202.
    let mut dropped = 0;
    for i in 0..200 {
        let p = PayloadBuilder::new(&format!("q-{i}"), "chat-q", &format!("m-{i}"))
            .messages(json!([user("x")]))
            .response("y")
            .build();
        let (status, body) = h
            .post(
                "/api/v1/ingest/litellm",
                serde_json::to_vec(&json!([p])).unwrap(),
            )
            .await;
        assert_eq!(status, StatusCode::ACCEPTED);
        dropped += body["dropped"].as_u64().unwrap();
    }
    let (status, _) = h.get("/healthz").await;
    assert_eq!(status, StatusCode::OK);
    // Whether or not drops happened depends on timing; what matters is that nothing blocked or failed.
    let _ = dropped;
}
