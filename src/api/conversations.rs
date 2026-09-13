//! Conversation listing, current health, and history.

use axum::extract::{Path, Query, State};
use axum::Json;
use chrono::{DateTime, TimeZone, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use super::{ApiError, AppState};
use crate::database::repo::{AnomalyRow, HealthRow};
use crate::monitor::scoring::{count_signals, Reason, Signal, SignalCounts, WindowAnomaly};

#[derive(Deserialize)]
pub struct ListQuery {
    pub limit: Option<u32>,
    pub status: Option<String>,
}

#[derive(Deserialize)]
pub struct HealthQuery {
    pub message_id: Option<String>,
    /// Unix seconds; return the latest result at or after this time.
    pub after: Option<f64>,
}

#[derive(Deserialize)]
pub struct HistoryQuery {
    pub limit: Option<u32>,
}

#[derive(Serialize)]
struct ContextView {
    prompt_tokens: Option<i64>,
    limit: Option<i64>,
    percent: Option<f64>,
}

pub async fn list(
    State(state): State<AppState>,
    Query(q): Query<ListQuery>,
) -> Result<Json<Value>, ApiError> {
    let limit = q.limit.unwrap_or(50).clamp(1, 500);
    let rows = state
        .db
        .list_conversations(limit, q.status.as_deref())
        .await?;
    let items: Vec<Value> = rows
        .iter()
        .map(|c| {
            json!({
                "conversation_id": c.id,
                "id_source": c.id_source,
                "user_id": c.user_id,
                "model": c.model,
                "first_seen": c.first_seen,
                "last_seen": c.last_seen,
                "turns": c.turns,
                "score": c.last_health,
                "risk": c.last_risk,
                "status": c.last_status,
                "context": ContextView { prompt_tokens: c.last_prompt_tokens, limit: c.last_context_limit, percent: c.last_context_percent },
            })
        })
        .collect();
    Ok(Json(json!({ "conversations": items })))
}

pub async fn health(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Query(q): Query<HealthQuery>,
) -> Result<Json<Value>, ApiError> {
    let conversation = state
        .db
        .get_conversation(&id)
        .await?
        .ok_or_else(|| ApiError::not_found("unknown_conversation", "no such conversation"))?;
    let result = if let Some(message_id) = q.message_id.as_deref().filter(|m| !m.is_empty()) {
        state.db.health_for_message(&id, message_id).await?
    } else if let Some(after) = q.after {
        let at = to_datetime(after)
            .ok_or_else(|| ApiError::bad_request("invalid_after", "after must be unix seconds"))?;
        state.db.health_after(&id, at).await?
    } else {
        state.db.latest_health(&id).await?
    };
    let Some(result) = result else {
        return Err(ApiError::not_found(
            "not_scored_yet",
            "no health result matches yet",
        ));
    };
    let reasons = parse_reasons(&result.reasons_json);
    let counts = signal_counts(&reasons);
    Ok(Json(json!({
        "conversation_id": id,
        "model": conversation.model,
        "user_id": conversation.user_id,
        "score": result.health,
        "risk": result.risk,
        "status": result.status,
        "turns": conversation.turns,
        "turn": result.turn,
        "prompts": conversation.prompts,
        "prompt": result.prompt,
        "message_id": result.message_id,
        "ts": result.ts,
        "context": ContextView { prompt_tokens: result.prompt_tokens, limit: result.context_limit, percent: result.context_percent },
        "signals": counts,
        "reasons": reasons,
        "summary": result.summary,
    })))
}

pub async fn history(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Query(q): Query<HistoryQuery>,
) -> Result<Json<Value>, ApiError> {
    let conversation = state
        .db
        .get_conversation(&id)
        .await?
        .ok_or_else(|| ApiError::not_found("unknown_conversation", "no such conversation"))?;
    let limit = q.limit.unwrap_or(200).clamp(1, 5000);
    let results: Vec<Value> = state
        .db
        .health_history(&id, limit)
        .await?
        .iter()
        .map(health_view)
        .collect();
    let anomalies: Vec<Value> = state
        .db
        .anomalies(&id)
        .await?
        .iter()
        .map(anomaly_view)
        .collect();
    Ok(Json(json!({
        "conversation_id": id,
        "model": conversation.model,
        "turns": conversation.turns,
        "results": results,
        "anomalies": anomalies,
    })))
}

fn health_view(r: &HealthRow) -> Value {
    json!({
        "turn": r.turn,
        "prompt": r.prompt,
        "ts": r.ts,
        "message_id": r.message_id,
        "score": r.health,
        "risk": r.risk,
        "status": r.status,
        "context_percent": r.context_percent,
        "reasons": parse_reasons(&r.reasons_json),
        "summary": r.summary,
    })
}

fn anomaly_view(a: &AnomalyRow) -> Value {
    json!({
        "turn": a.turn,
        "prompt": a.prompt,
        "ts": a.ts,
        "signal": a.signal,
        "penalty": a.penalty,
        "severity": a.severity,
        "detail": a.detail,
    })
}

fn parse_reasons(json: &str) -> Vec<Reason> {
    serde_json::from_str(json).unwrap_or_default()
}

fn signal_counts(reasons: &[Reason]) -> SignalCounts {
    let window: Vec<WindowAnomaly> = reasons
        .iter()
        .filter_map(|r| {
            Some(WindowAnomaly {
                signal: Signal::parse(&r.signal)?,
                penalty: r.penalty,
                detail: String::new(),
                turn: 0,
            })
        })
        .collect();
    count_signals(&window)
}

fn to_datetime(unix_seconds: f64) -> Option<DateTime<Utc>> {
    if !unix_seconds.is_finite() || unix_seconds < 0.0 {
        return None;
    }
    Utc.timestamp_opt(
        unix_seconds.trunc() as i64,
        (unix_seconds.fract() * 1e9) as u32,
    )
    .single()
}
