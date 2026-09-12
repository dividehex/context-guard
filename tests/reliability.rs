mod common;

use axum::http::StatusCode;
use common::{harness_with, harness_without_worker, user, PayloadBuilder};
use serde_json::json;

fn payload(i: usize) -> serde_json::Value {
    PayloadBuilder::new(&format!("r-{i}"), "chat-r", &format!("m-{i}"))
        .messages(json!([user("x")]))
        .response("y")
        .build()
}

#[tokio::test]
async fn full_queue_drops_and_counts_without_blocking() {
    let h = harness_without_worker(|c| c.queue_size = 1).await;
    let (status, body) = h
        .post(
            "/api/v1/ingest/litellm",
            serde_json::to_vec(&json!([payload(1)])).unwrap(),
        )
        .await;
    assert_eq!(status, StatusCode::ACCEPTED);
    assert_eq!(body, json!({"accepted": 1, "dropped": 0}));

    let (status, body) = h
        .post(
            "/api/v1/ingest/litellm",
            serde_json::to_vec(&json!([payload(2), payload(3)])).unwrap(),
        )
        .await;
    assert_eq!(
        status,
        StatusCode::ACCEPTED,
        "a full queue must still answer immediately"
    );
    assert_eq!(body, json!({"accepted": 0, "dropped": 2}));

    let text = h.state.metrics.encode();
    assert!(
        text.contains("context_guard_events_dropped_total{reason=\"queue_full\"} 2"),
        "{text}"
    );
    let (status, health) = h.get("/healthz").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(health["queue_depth"], json!(1));
}

#[tokio::test]
async fn capture_dir_stores_raw_bodies_verbatim() {
    let h = harness_with(|c| {
        c.capture_dir =
            Some(std::env::temp_dir().join(format!("cg-capture-{}", std::process::id())))
    })
    .await;
    let dir = h.state.config.capture_dir.clone().unwrap();
    let body = serde_json::to_vec(&json!([payload(1)])).unwrap();
    let (status, _) = h.post("/api/v1/ingest/litellm", body.clone()).await;
    assert_eq!(status, StatusCode::ACCEPTED);
    let files: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    assert_eq!(files.len(), 1);
    assert_eq!(std::fs::read(&files[0]).unwrap(), body);
    std::fs::remove_dir_all(&dir).ok();
}

#[tokio::test]
async fn sixty_concurrent_posts_are_all_scored_in_order() {
    let h = harness_with(|c| c.queue_size = 256).await;
    let mut tasks = Vec::new();
    for i in 1..=60 {
        let router = h.router.clone();
        tasks.push(tokio::spawn(async move {
            use axum::body::Body;
            use axum::http::Request;
            use tower::ServiceExt;
            let p = PayloadBuilder::new(&format!("c-{i}"), "chat-c", &format!("m-{i}"))
                .messages(json!([user(&format!("message {i}"))]))
                .response(&format!(
                    "reply number {i} with enough words to be a real answer in this test"
                ))
                .times(i as f64, i as f64 + 0.5)
                .build();
            let req = Request::post("/api/v1/ingest/litellm")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&json!([p])).unwrap()))
                .unwrap();
            router.oneshot(req).await.unwrap().status()
        }));
    }
    for t in tasks {
        assert_eq!(t.await.unwrap(), StatusCode::ACCEPTED);
    }
    let latest = h.wait_for_turns("chat-c", 60).await;
    assert_eq!(latest["turns"], json!(60));
    let (_, history) = h
        .get("/api/v1/conversations/chat-c/history?limit=1000")
        .await;
    let turns: Vec<i64> = history["results"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["turn"].as_i64().unwrap())
        .collect();
    assert_eq!(
        turns,
        (1..=60).collect::<Vec<_>>(),
        "every event became exactly one turn, in order"
    );
    assert!(!h.state.metrics.encode().contains("queue_full"));
}

#[tokio::test]
async fn unsupported_call_types_are_counted_not_scored() {
    let h = harness_with(|_| {}).await;
    let batch = json!([
        {"id": "emb-1", "call_type": "aembedding", "model": "embed-model", "messages": [], "response": {}},
        payload(9)
    ]);
    h.post(
        "/api/v1/ingest/litellm",
        serde_json::to_vec(&batch).unwrap(),
    )
    .await;
    h.wait_for_message("chat-r", "m-9").await;
    let text = h.state.metrics.encode();
    assert!(
        text.contains("context_guard_events_dropped_total{reason=\"unsupported\"} 1"),
        "{text}"
    );
    let (_, list) = h.get("/api/v1/conversations").await;
    assert_eq!(list["conversations"].as_array().unwrap().len(), 1);
}

#[tokio::test]
async fn payload_logging_flag_does_not_change_results() {
    let h = harness_with(|c| c.log_payloads = true).await;
    h.ingest(vec![payload(1)]).await;
    assert_eq!(
        h.wait_for_message("chat-r", "m-1").await["score"],
        json!(100)
    );
}

#[tokio::test]
async fn api_rejects_bad_queries_cleanly() {
    let h = harness_with(|_| {}).await;
    h.ingest(vec![payload(1)]).await;
    h.wait_for_message("chat-r", "m-1").await;

    let (status, body) = h.get("/api/v1/conversations/chat-r/health?after=-5").await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["code"], json!("invalid_after"));
    let (status, _) = h
        .get("/api/v1/conversations/chat-r/health?after=notanumber")
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let (status, body) = h.get("/api/v1/conversations/nope/history").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(body["error"]["code"], json!("unknown_conversation"));
    let (status, list) = h.get("/api/v1/conversations?limit=0").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        list["conversations"].as_array().unwrap().len(),
        1,
        "limit is clamped to at least 1"
    );
    let (status, _) = h
        .get("/api/v1/conversations/chat-r/history?limit=99999")
        .await;
    assert_eq!(status, StatusCode::OK);
    let (status, _) = h.get("/nothing/here").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}
