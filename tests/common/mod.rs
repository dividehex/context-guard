//! In-process test harness: temp SQLite, real worker, router driven via tower.

#![allow(dead_code)]

use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::Router;
use context_guard::api::{self, AppState};
use context_guard::config::Config;
use context_guard::database::Database;
use context_guard::metrics::Metrics;
use context_guard::monitor::Monitor;
use context_guard::worker;
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tower::ServiceExt;

pub struct Harness {
    pub router: Router,
    pub state: AppState,
    _dir: tempfile::TempDir,
}

pub async fn harness() -> Harness {
    harness_with(|_| {}).await
}

pub async fn harness_with(tweak: impl FnOnce(&mut Config)) -> Harness {
    let dir = tempfile::tempdir().unwrap();
    let mut config = Config {
        database: dir.path().join("cg.db"),
        queue_size: 16,
        ..Config::default()
    };
    tweak(&mut config);
    let config = Arc::new(config);
    let db = Database::connect(&config.database).await.unwrap();
    let metrics = Arc::new(Metrics::new());
    let (tx, rx) = tokio::sync::mpsc::channel(config.queue_size);
    let monitor = Monitor::new(db.clone(), config.clone(), metrics.clone());
    tokio::spawn(worker::run(rx, monitor, config.clone(), metrics.clone()));
    let state = AppState {
        db,
        tx,
        metrics,
        config,
        started: Instant::now(),
    };
    Harness {
        router: api::router(state.clone()),
        state,
        _dir: dir,
    }
}

impl Harness {
    pub async fn post(&self, path: &str, body: Vec<u8>) -> (StatusCode, Value) {
        let req = Request::post(path)
            .header("content-type", "application/json")
            .body(Body::from(body))
            .unwrap();
        self.send(req).await
    }

    pub async fn get(&self, path: &str) -> (StatusCode, Value) {
        self.send(Request::get(path).body(Body::empty()).unwrap())
            .await
    }

    async fn send(&self, req: Request<Body>) -> (StatusCode, Value) {
        let res = self.router.clone().oneshot(req).await.unwrap();
        let status = res.status();
        let bytes = res.into_body().collect().await.unwrap().to_bytes();
        let value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
        (status, value)
    }

    pub async fn ingest(&self, payloads: Vec<Value>) {
        let (status, body) = self
            .post(
                "/api/v1/ingest/litellm",
                serde_json::to_vec(&payloads).unwrap(),
            )
            .await;
        assert_eq!(status, StatusCode::ACCEPTED, "{body}");
    }

    /// Poll until the health endpoint returns 200 for the message id.
    pub async fn wait_for_message(&self, conversation: &str, message_id: &str) -> Value {
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            let (status, body) = self
                .get(&format!(
                    "/api/v1/conversations/{conversation}/health?message_id={message_id}"
                ))
                .await;
            if status == StatusCode::OK {
                return body;
            }
            assert!(
                Instant::now() < deadline,
                "timed out waiting for {conversation}/{message_id}: {body}"
            );
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }

    pub async fn wait_for_turns(&self, conversation: &str, turns: i64) -> Value {
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            let (status, body) = self
                .get(&format!("/api/v1/conversations/{conversation}/health"))
                .await;
            if status == StatusCode::OK && body["turns"] == json!(turns) {
                return body;
            }
            assert!(
                Instant::now() < deadline,
                "timed out waiting for {conversation} turns={turns}: {body}"
            );
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }
}

/// A LiteLLM `StandardLoggingPayload` shaped like the ones the generic_api
/// logger sends, with only the fields Context Guard reads.
pub struct PayloadBuilder {
    value: Value,
}

impl PayloadBuilder {
    pub fn new(id: &str, chat_id: &str, message_id: &str) -> PayloadBuilder {
        PayloadBuilder {
            value: json!({
                "id": id,
                "trace_id": format!("trace-{id}"),
                "litellm_call_id": format!("call-{id}"),
                "call_type": "acompletion",
                "stream": true,
                "status": "success",
                "model": "openai/test-model",
                "model_group": "test-model",
                "prompt_tokens": 1000,
                "completion_tokens": 50,
                "startTime": 1_757_600_000.0,
                "endTime": 1_757_600_001.0,
                "messages": [],
                "response": {"choices": [{"message": {"role": "assistant", "content": ""}}]},
                "model_map_information": {"model_map_key": "test-model", "model_map_value": {"max_input_tokens": 10000}},
                "request_tags": [
                    "User-Agent: OpenAI",
                    format!("x-openwebui-chat-id: {chat_id}"),
                    "x-openwebui-user-id: user-1",
                    format!("x-openwebui-message-id: {message_id}"),
                ],
                "metadata": {},
                "end_user": ""
            }),
        }
    }

    pub fn messages(mut self, messages: Value) -> Self {
        self.value["messages"] = messages;
        self
    }

    pub fn response(mut self, content: &str) -> Self {
        self.value["response"]["choices"][0]["message"]["content"] = json!(content);
        self
    }

    pub fn tool_calls(mut self, calls: Value) -> Self {
        self.value["response"]["choices"][0]["message"]["tool_calls"] = calls;
        self
    }

    pub fn prompt_tokens(mut self, n: u64) -> Self {
        self.value["prompt_tokens"] = json!(n);
        self
    }

    pub fn times(mut self, start: f64, end: f64) -> Self {
        self.value["startTime"] = json!(start);
        self.value["endTime"] = json!(end);
        self
    }

    pub fn tag(mut self, tag: &str) -> Self {
        self.value["request_tags"]
            .as_array_mut()
            .unwrap()
            .push(json!(tag));
        self
    }

    pub fn set(mut self, key: &str, v: Value) -> Self {
        self.value[key] = v;
        self
    }

    pub fn build(self) -> Value {
        self.value
    }
}

pub fn user(text: &str) -> Value {
    json!({"role": "user", "content": text})
}

pub fn assistant(text: &str) -> Value {
    json!({"role": "assistant", "content": text})
}

pub fn tool_call(id: &str, name: &str, args: &str) -> Value {
    json!({"id": id, "type": "function", "function": {"name": name, "arguments": args}})
}
