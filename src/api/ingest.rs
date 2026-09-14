//! `POST /api/v1/ingest/{litellm,claude-code,codex}`: validate, enqueue, answer.
//! Never waits for the database.

use axum::body::Bytes;
use axum::extract::State;
use axum::http::StatusCode;
use axum::Json;
use serde_json::json;

use super::{ApiError, AppState};
use crate::telemetry::{claude_code, codex, litellm};
use crate::worker::Batch;

type Accepted = Result<(StatusCode, Json<serde_json::Value>), ApiError>;

pub async fn litellm(State(state): State<AppState>, body: Bytes) -> Accepted {
    let batch = match litellm::split_body(&body) {
        Ok(items) => Batch::LiteLlm(items),
        Err(e) => return Err(reject(&state, &body, e)),
    };
    Ok(enqueue(&state, &body, batch))
}

pub async fn claude_code(State(state): State<AppState>, body: Bytes) -> Accepted {
    let batch = match claude_code::parse_body(&body) {
        Ok(ingest) => Batch::ClaudeCode(ingest),
        Err(e) => return Err(reject(&state, &body, e)),
    };
    Ok(enqueue(&state, &body, batch))
}

pub async fn codex(State(state): State<AppState>, body: Bytes) -> Accepted {
    let batch = match codex::parse_body(&body) {
        Ok(ingest) => Batch::Codex(ingest),
        Err(e) => return Err(reject(&state, &body, e)),
    };
    Ok(enqueue(&state, &body, batch))
}

fn reject(state: &AppState, body: &[u8], e: litellm::ParseError) -> ApiError {
    if let Some(dir) = &state.config.capture_dir {
        capture(dir, body);
    }
    state
        .metrics
        .events_dropped
        .with_label_values(&["malformed"])
        .inc();
    tracing::warn!(error = %e, bytes = body.len(), "rejected ingest body");
    ApiError::bad_request("malformed_body", e.to_string())
}

fn enqueue(state: &AppState, body: &[u8], batch: Batch) -> (StatusCode, Json<serde_json::Value>) {
    if let Some(dir) = &state.config.capture_dir {
        capture(dir, body);
    }
    let count = batch.len();
    state.metrics.ingest_batch_size.observe(count as f64);
    if count == 0 {
        return (
            StatusCode::ACCEPTED,
            Json(json!({ "accepted": 0, "dropped": 0 })),
        );
    }
    match state.tx.try_send(batch) {
        Ok(()) => {
            state
                .metrics
                .queue_depth
                .set(state.tx.max_capacity().saturating_sub(state.tx.capacity()) as i64);
            (
                StatusCode::ACCEPTED,
                Json(json!({ "accepted": count, "dropped": 0 })),
            )
        }
        Err(_) => {
            state
                .metrics
                .events_dropped
                .with_label_values(&["queue_full"])
                .inc_by(count as u64);
            tracing::warn!(count, "ingest queue full; batch dropped");
            (
                StatusCode::ACCEPTED,
                Json(json!({ "accepted": 0, "dropped": count })),
            )
        }
    }
}

/// Opt-in raw capture of request bodies for fixture building and debugging.
fn capture(dir: &std::path::Path, body: &[u8]) {
    let name = format!(
        "{}-{}.json",
        chrono::Utc::now().format("%Y%m%dT%H%M%S%.3fZ"),
        body.len()
    );
    if let Err(e) = std::fs::create_dir_all(dir).and_then(|_| std::fs::write(dir.join(name), body))
    {
        tracing::warn!(error = %e, "payload capture failed");
    }
}
