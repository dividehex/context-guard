//! The explanation surface: the signal catalog, the per-conversation explain
//! document a front end renders, and the bundled page that renders it.

mod common;

use axum::http::StatusCode;
use common::{assistant, harness, user, PayloadBuilder};
use context_guard::config::Config;
use context_guard::monitor::scoring::Signal;
use serde_json::json;

const CHAT: &str = "chat-explain";

#[tokio::test]
async fn signal_catalog_lists_every_signal_with_its_configured_penalty() {
    let h = harness().await;
    let (status, body) = h.get("/api/v1/signals").await;
    assert_eq!(status, StatusCode::OK);
    let signals = body["signals"].as_array().unwrap();
    assert_eq!(signals.len(), Signal::ALL.len());
    let defaults = Config::default();
    for s in Signal::ALL {
        let entry = signals
            .iter()
            .find(|v| v["signal"] == s.as_str())
            .unwrap_or_else(|| panic!("{} missing", s.as_str()));
        assert_eq!(entry["penalty"], json!(defaults.penalties.for_signal(s)));
        assert_eq!(entry["title"], json!(s.title()));
        assert!(entry["explanation"].as_str().unwrap().len() > 100);
    }
    assert_eq!(body["scoring"]["window_turns"], json!(10));
    assert_eq!(body["thresholds"]["good"], json!(75));
}

#[tokio::test]
async fn explain_document_carries_reasons_issues_and_history() {
    let h = harness().await;
    let t1 = PayloadBuilder::new("ev-1", CHAT, "msg-1")
        .messages(json!([user("llama.cpp is running on port 8080")]))
        .response("Noted: llama.cpp is on port 8080.")
        .prompt_tokens(2000)
        .times(1.0, 2.0)
        .build();
    let t2 = PayloadBuilder::new("ev-2", CHAT, "msg-2")
        .messages(json!([
            user("llama.cpp is running on port 8080"),
            assistant("Noted: llama.cpp is on port 8080."),
            user("restart it please")
        ]))
        .response("Restarting your llama.cpp server on port 8000 now.")
        .prompt_tokens(7800)
        .times(3.0, 4.0)
        .build();
    h.ingest(vec![t1, t2]).await;
    h.wait_for_message(CHAT, "msg-2").await;

    let (status, d) = h
        .get(&format!("/api/v1/conversations/{CHAT}/explain"))
        .await;
    assert_eq!(status, StatusCode::OK, "{d}");
    assert_eq!(d["conversation_id"], json!(CHAT));
    assert_eq!(d["score"], json!(80));
    assert_eq!(d["status"], json!("good"));
    assert_eq!(d["turn"], json!(2));
    assert_eq!(d["context"]["percent"], json!(78.0));
    assert_eq!(d["scoring"]["window_turns"], json!(10));
    assert_eq!(d["scoring"]["window_from_prompt"], json!(1));
    assert_eq!(d["prompt"], json!(2), "LiteLLM: one prompt per completion");
    assert!(d["summary"].as_str().unwrap().contains("1 drift"));

    let reasons = d["reasons"].as_array().unwrap();
    assert_eq!(reasons.len(), 2, "{d}");
    assert_eq!(reasons[0]["signal"], json!("context_70"));
    assert_eq!(reasons[0]["penalty"], json!(5));
    assert_eq!(reasons[1]["signal"], json!("known_value_drift"));
    assert_eq!(reasons[1]["title"], json!(Signal::KnownValueDrift.title()));
    assert_eq!(
        reasons[1]["detail"],
        json!("assistant said port of llama.cpp 8000 but the conversation established 8080")
    );
    assert!(reasons[1]["explanation"]
        .as_str()
        .unwrap()
        .contains("diverged"));

    let issues = d["issues"].as_array().unwrap();
    assert_eq!(issues.len(), 1);
    assert_eq!(issues[0]["signal"], json!("known_value_drift"));
    assert_eq!(issues[0]["turn"], json!(2));
    assert_eq!(issues[0]["severity"], json!("medium"));
    assert_eq!(issues[0]["counting"], json!(true));
    assert_eq!(issues[0]["explanation"], reasons[1]["explanation"]);

    let history = d["history"].as_array().unwrap();
    let scores: Vec<u64> = history
        .iter()
        .map(|r| r["score"].as_u64().unwrap())
        .collect();
    assert_eq!(scores, vec![100, 80]);

    let (status, err) = h.get("/api/v1/conversations/nope/explain").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(err["error"]["code"], json!("unknown_conversation"));
}

#[tokio::test]
async fn the_page_is_served_and_reads_the_explain_endpoint() {
    let h = harness().await;
    let req = axum::http::Request::get("/ui/conversations/anything")
        .body(axum::body::Body::empty())
        .unwrap();
    let res = tower::ServiceExt::oneshot(h.router.clone(), req)
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let content_type = res
        .headers()
        .get("content-type")
        .unwrap()
        .to_str()
        .unwrap()
        .to_string();
    assert!(content_type.starts_with("text/html"), "{content_type}");
    let body = http_body_util::BodyExt::collect(res.into_body())
        .await
        .unwrap()
        .to_bytes();
    let html = std::str::from_utf8(&body).unwrap();
    assert!(html.contains("/api/v1/conversations/"));
    assert!(html.contains("/explain"));
    for needle in [
        "src=\"http",
        "href=\"http",
        "@import",
        "url(http",
        "fetch('http",
    ] {
        assert!(
            !html.contains(needle),
            "the page loads nothing from the network: {needle}"
        );
    }
}
