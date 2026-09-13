//! `GET /api/v1/signals`: the catalog of signals with their configured
//! penalties and explanations, plus the scoring parameters. What the
//! explanation page needs to say why a score is what it is.

use axum::extract::State;
use axum::Json;
use serde_json::{json, Value};

use super::AppState;
use crate::monitor::scoring::Signal;

/// One signal as the API describes it, with the penalty that applies.
pub(super) fn describe(signal: Signal, penalty: u32) -> Value {
    json!({
        "signal": signal.as_str(),
        "family": signal.family().as_str(),
        "severity": signal.severity(),
        "penalty": penalty,
        "short": signal.short(),
        "phrase": signal.phrase(),
        "title": signal.title(),
        "explanation": signal.explanation(),
    })
}

pub async fn catalog(State(state): State<AppState>) -> Json<Value> {
    let signals: Vec<Value> = Signal::ALL
        .iter()
        .map(|s| describe(*s, state.config.penalties.for_signal(*s)))
        .collect();
    Json(json!({
        "signals": signals,
        "scoring": { "window_turns": state.config.scoring.window_turns },
        "thresholds": {
            "healthy": 90,
            "good": state.config.thresholds.good,
            "watch": state.config.thresholds.watch,
            "degraded": state.config.thresholds.degraded,
        },
    }))
}
