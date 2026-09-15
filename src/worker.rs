//! The single consumer of the ingest queue. Runs forever; never lets one bad
//! payload stop the next one.

use std::sync::Arc;

use serde_json::Value;
use tokio::sync::mpsc::Receiver;

use crate::config::Config;
use crate::metrics::Metrics;
use crate::monitor::{Monitor, Outcome};
use crate::telemetry::event::ConversationEvent;
use crate::telemetry::litellm;
use crate::telemetry::{claude_code, codex, opencode};

/// One accepted ingest body, tagged with the source whose normalizer reads it.
#[derive(Debug)]
pub enum Batch {
    LiteLlm(Vec<Value>),
    ClaudeCode(claude_code::Ingest),
    Codex(codex::Ingest),
    OpenCode(opencode::Ingest),
}

impl Batch {
    /// Number of raw items in the body, for metrics and the ingest response.
    pub fn len(&self) -> usize {
        match self {
            Batch::LiteLlm(items) => items.len(),
            Batch::ClaudeCode(ingest) => ingest.records.len(),
            Batch::Codex(ingest) => ingest.records.len(),
            Batch::OpenCode(ingest) => ingest.records.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub async fn run(
    mut rx: Receiver<Batch>,
    monitor: Monitor,
    config: Arc<Config>,
    metrics: Arc<Metrics>,
) {
    while let Some(batch) = rx.recv().await {
        metrics.queue_depth.set(rx.len() as i64);
        let mut events = normalize(&batch, &config, &metrics);
        // Transcript slices are already in file order and must stay that way:
        // the delta of each completion depends on the one before it.
        if matches!(batch, Batch::LiteLlm(_)) {
            events.sort_by_key(|e| e.started_at);
        }
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

fn normalize(batch: &Batch, config: &Config, metrics: &Metrics) -> Vec<ConversationEvent> {
    match batch {
        Batch::LiteLlm(items) => {
            let mut events = Vec::with_capacity(items.len());
            for value in items {
                match litellm::normalize(value, config) {
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
            events
        }
        Batch::ClaudeCode(ingest) => {
            let normalized = claude_code::normalize(ingest, config);
            count_malformed(metrics, normalized.malformed);
            normalized.events
        }
        Batch::Codex(ingest) => {
            let normalized = codex::normalize(ingest, config);
            count_malformed(metrics, normalized.malformed);
            normalized.events
        }
        Batch::OpenCode(ingest) => {
            let normalized = opencode::normalize(ingest, config);
            count_malformed(metrics, normalized.malformed);
            normalized.events
        }
    }
}

fn count_malformed(metrics: &Metrics, malformed: usize) {
    if malformed > 0 {
        tracing::warn!(count = malformed, "malformed transcript records skipped");
        metrics
            .events_dropped
            .with_label_values(&["malformed"])
            .inc_by(malformed as u64);
    }
}
