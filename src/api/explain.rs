//! `GET /api/v1/conversations/{id}/explain`: everything a front end needs to
//! show why a conversation scores what it scores, in one JSON document:
//! the latest result, each reason with its explanation, every issue ever
//! caught (and whether it still counts), and the score over turns.

use axum::extract::{Path, State};
use axum::Json;
use serde_json::{json, Value};

use super::signals::describe;
use super::{ApiError, AppState};
use crate::monitor::scoring::{Reason, Signal};

pub async fn explain(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    let conversation = state
        .db
        .get_conversation(&id)
        .await?
        .ok_or_else(|| ApiError::not_found("unknown_conversation", "no such conversation"))?;
    let Some(latest) = state.db.latest_health(&id).await? else {
        return Err(ApiError::not_found(
            "not_scored_yet",
            "no health result yet",
        ));
    };
    let window_turns = state.config.scoring.window_turns;
    let latest_prompt = u32::try_from(latest.prompt).unwrap_or(0);
    // Same arithmetic as the monitor: the window is counted in prompts and
    // ends at the latest one.
    let window_from_prompt = latest_prompt
        .saturating_sub(window_turns.saturating_sub(1))
        .max(1);

    let reasons: Vec<Value> = serde_json::from_str::<Vec<Reason>>(&latest.reasons_json)
        .unwrap_or_default()
        .iter()
        .map(|r| {
            let mut v = describe_named(&r.signal, r.penalty);
            v["detail"] = json!(r.detail);
            v
        })
        .collect();

    let issues: Vec<Value> = state
        .db
        .anomalies(&id)
        .await?
        .iter()
        .map(|a| {
            let prompt = u32::try_from(a.prompt).unwrap_or(0);
            let mut v = describe_named(&a.signal, u32::try_from(a.penalty).unwrap_or(0));
            v["turn"] = json!(a.turn);
            v["prompt"] = json!(prompt);
            v["ts"] = json!(a.ts);
            v["detail"] = json!(a.detail);
            v["counting"] = json!(prompt >= window_from_prompt);
            v
        })
        .collect();

    let history: Vec<Value> = state
        .db
        .health_history(&id, 5000)
        .await?
        .iter()
        .map(|h| {
            json!({
                "turn": h.turn,
                "prompt": h.prompt,
                "ts": h.ts,
                "score": h.health,
                "risk": h.risk,
                "status": h.status,
                "context_percent": h.context_percent,
            })
        })
        .collect();

    Ok(Json(json!({
        "conversation_id": id,
        "id_source": conversation.id_source,
        "model": conversation.model,
        "user_id": conversation.user_id,
        "first_seen": conversation.first_seen,
        "last_seen": conversation.last_seen,
        "turns": conversation.turns,
        "turn": latest.turn,
        "prompts": conversation.prompts,
        "prompt": latest.prompt,
        "ts": latest.ts,
        "score": latest.health,
        "risk": latest.risk,
        "status": latest.status,
        "summary": latest.summary,
        "context": {
            "prompt_tokens": latest.prompt_tokens,
            "limit": latest.context_limit,
            "percent": latest.context_percent,
        },
        "scoring": {
            "formula": "health = 100 - risk; risk = context penalty of the latest request + penalties of every issue recorded in the last window_turns prompts, capped at 100. A prompt is one completion for LiteLLM and one user message for Claude Code.",
            "window_turns": window_turns,
            "window_from_prompt": window_from_prompt,
            "thresholds": {
                "healthy": 90,
                "good": state.config.thresholds.good,
                "watch": state.config.thresholds.watch,
                "degraded": state.config.thresholds.degraded,
            },
        },
        "reasons": reasons,
        "issues": issues,
        "history": history,
    })))
}

/// Signal metadata by its stored name; a name this build does not know (a
/// newer or older database) still gets a usable entry.
fn describe_named(signal: &str, penalty: u32) -> Value {
    match Signal::parse(signal) {
        Some(s) => describe(s, penalty),
        None => json!({
            "signal": signal,
            "penalty": penalty,
            "title": signal,
            "explanation": "",
        }),
    }
}
