//! The single consumer of the ingest queue. Runs forever; never lets one bad
//! payload stop the next one.

use std::sync::Arc;

use serde_json::Value;
use tokio::sync::mpsc::Receiver;

use crate::config::Config;
use crate::metrics::Metrics;
use crate::monitor::{Monitor, Outcome};
use crate::telemetry::litellm;

pub type Batch = Vec<Value>;

pub async fn run(
    mut rx: Receiver<Batch>,
    monitor: Monitor,
    config: Arc<Config>,
    metrics: Arc<Metrics>,
) {
    while let Some(batch) = rx.recv().await {
        metrics.queue_depth.set(rx.len() as i64);
        let mut events = Vec::with_capacity(batch.len());
        for value in &batch {
            match litellm::normalize(value, &config) {
                Ok(event) => events.push(event),
                Err(litellm::NormalizeError::UnsupportedCallType(t)) => {
                    tracing::debug!(call_type = %t, "ignoring non-chat payload");
                    metrics
                        .events_dropped
                        .with_label_values(&["unsupported"])
                        .inc();
                }
                Err(e) => {
                    tracing::warn!(error = %e, "malformed payload rejected");
                    metrics
                        .events_dropped
                        .with_label_values(&["malformed"])
                        .inc();
                }
            }
        }
        events.sort_by_key(|e| e.started_at);
        for event in &events {
            metrics
                .events_received
                .with_label_values(&[event.kind.as_str()])
                .inc();
            if config.log_payloads {
                tracing::info!(event = ?event, "payload");
            }
            match monitor.process(event).await {
                Ok(Outcome::Scored(s)) => tracing::info!(
                    conversation = %s.conversation_id, model = %s.model, turn = s.turn,
                    health = s.health, risk = s.risk, status = %s.status, "scored"
                ),
                Ok(Outcome::Recorded(kind)) => {
                    tracing::debug!(conversation = %event.conversation_id, kind = kind.as_str(), "recorded without scoring")
                }
                Ok(Outcome::Duplicate) => {
                    tracing::debug!(event = %event.event_id, "duplicate event ignored")
                }
                Err(e) => {
                    tracing::error!(error = %e, event = %event.event_id, "processing failed");
                    metrics
                        .processing_errors
                        .with_label_values(&["process"])
                        .inc();
                }
            }
        }
    }
    tracing::info!("ingest queue closed; worker exiting");
}
