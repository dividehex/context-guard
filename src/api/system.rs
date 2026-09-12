//! `/healthz` and `/metrics`.

use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::IntoResponse;
use axum::Json;
use serde_json::json;

use super::AppState;

pub async fn healthz(State(state): State<AppState>) -> impl IntoResponse {
    let db_ok = state.db.ping().await;
    let body = json!({
        "status": if db_ok { "ok" } else { "degraded" },
        "database": if db_ok { "ok" } else { "unavailable" },
        "queue_depth": state.tx.max_capacity().saturating_sub(state.tx.capacity()),
        "uptime_s": state.started.elapsed().as_secs(),
        "version": env!("CARGO_PKG_VERSION"),
    });
    // The process is alive either way; a database outage degrades monitoring, not the service.
    (StatusCode::OK, Json(body))
}

pub async fn metrics(State(state): State<AppState>) -> impl IntoResponse {
    refresh_status_gauge(&state).await;
    (
        [(
            header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        state.metrics.encode(),
    )
}

async fn refresh_status_gauge(state: &AppState) {
    let since = chrono::Utc::now() - chrono::Duration::hours(24);
    let Ok(rows) = state.db.status_counts_since(since).await else {
        return;
    };
    for status in ["healthy", "good", "watch", "degraded", "reset_recommended"] {
        let n = rows
            .iter()
            .find(|r| r.status == status)
            .map(|r| r.count)
            .unwrap_or(0);
        state
            .metrics
            .conversations_by_status
            .with_label_values(&[status])
            .set(n);
    }
}
