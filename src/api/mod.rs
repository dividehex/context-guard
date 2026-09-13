//! HTTP surface: ingest, health, conversations, metrics.

pub mod conversations;
pub mod explain;
pub mod ingest;
pub mod signals;
pub mod system;
pub mod ui;

use std::sync::Arc;
use std::time::Instant;

use axum::extract::DefaultBodyLimit;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::json;
use tokio::sync::mpsc::Sender;

use crate::config::Config;
use crate::database::Database;
use crate::metrics::Metrics;
use crate::worker::Batch;

#[derive(Clone)]
pub struct AppState {
    pub db: Database,
    pub tx: Sender<Batch>,
    pub metrics: Arc<Metrics>,
    pub config: Arc<Config>,
    pub started: Instant,
}

pub fn router(state: AppState) -> Router {
    let limit = state.config.max_body_bytes;
    Router::new()
        .route("/healthz", get(system::healthz))
        .route("/metrics", get(system::metrics))
        .route("/api/v1/ingest/litellm", post(ingest::litellm))
        .route("/api/v1/ingest/claude-code", post(ingest::claude_code))
        .route("/api/v1/signals", get(signals::catalog))
        .route("/ui/conversations/{id}", get(ui::conversation))
        .route("/api/v1/conversations", get(conversations::list))
        .route(
            "/api/v1/conversations/{id}/health",
            get(conversations::health),
        )
        .route(
            "/api/v1/conversations/{id}/history",
            get(conversations::history),
        )
        .route("/api/v1/conversations/{id}/explain", get(explain::explain))
        .layer(DefaultBodyLimit::max(limit))
        .with_state(state)
}

/// Metrics-only listener, for scraping without exposing conversation data.
pub fn metrics_router(state: AppState) -> Router {
    Router::new()
        .route("/healthz", get(system::healthz))
        .route("/metrics", get(system::metrics))
        .with_state(state)
}

/// `{ "error": { "code", "message" } }`
pub struct ApiError {
    pub status: StatusCode,
    pub code: &'static str,
    pub message: String,
}

impl ApiError {
    pub fn not_found(code: &'static str, message: impl Into<String>) -> ApiError {
        ApiError {
            status: StatusCode::NOT_FOUND,
            code,
            message: message.into(),
        }
    }

    pub fn bad_request(code: &'static str, message: impl Into<String>) -> ApiError {
        ApiError {
            status: StatusCode::BAD_REQUEST,
            code,
            message: message.into(),
        }
    }
}

impl From<sqlx::Error> for ApiError {
    fn from(e: sqlx::Error) -> Self {
        tracing::error!(error = %e, "database query failed");
        ApiError {
            status: StatusCode::SERVICE_UNAVAILABLE,
            code: "database_unavailable",
            message: "database query failed".into(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(json!({ "error": { "code": self.code, "message": self.message } })),
        )
            .into_response()
    }
}
