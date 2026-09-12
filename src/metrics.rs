//! Prometheus metrics. Aggregate only: no conversation ids as labels.

use prometheus::{
    Encoder, GaugeVec, Histogram, HistogramOpts, IntCounterVec, IntGauge, IntGaugeVec, Opts,
    Registry, TextEncoder,
};

use crate::monitor::scoring::{Signal, SignalFamily};

pub struct Metrics {
    registry: Registry,
    pub events_received: IntCounterVec,
    pub events_dropped: IntCounterVec,
    pub processing_errors: IntCounterVec,
    pub health_score: GaugeVec,
    pub risk_score: GaugeVec,
    pub context_utilization: GaugeVec,
    pub conversations_by_status: IntGaugeVec,
    pub known_value_drift: IntCounterVec,
    pub tool_anomalies: IntCounterVec,
    pub loop_events: IntCounterVec,
    pub suspicious_identifiers: IntCounterVec,
    pub ingest_batch_size: Histogram,
    pub queue_depth: IntGauge,
}

impl Metrics {
    pub fn new() -> Metrics {
        let registry = Registry::new();
        let counter = |name: &str, help: &str, labels: &[&str]| {
            let c = IntCounterVec::new(Opts::new(name, help), labels).expect("valid metric");
            registry
                .register(Box::new(c.clone()))
                .expect("unique metric");
            c
        };
        let gauge = |name: &str, help: &str, labels: &[&str]| {
            let g = GaugeVec::new(Opts::new(name, help), labels).expect("valid metric");
            registry
                .register(Box::new(g.clone()))
                .expect("unique metric");
            g
        };
        let conversations_by_status = IntGaugeVec::new(
            Opts::new(
                "context_guard_conversations_by_status",
                "Conversations active in the last 24h by latest status",
            ),
            &["status"],
        )
        .expect("valid metric");
        registry
            .register(Box::new(conversations_by_status.clone()))
            .expect("unique metric");
        let ingest_batch_size = Histogram::with_opts(
            HistogramOpts::new(
                "context_guard_ingest_batch_size",
                "Payloads per ingest request",
            )
            .buckets(vec![1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 512.0]),
        )
        .expect("valid metric");
        registry
            .register(Box::new(ingest_batch_size.clone()))
            .expect("unique metric");
        let queue_depth = IntGauge::new(
            "context_guard_queue_depth",
            "Ingest batches waiting for the worker",
        )
        .expect("valid metric");
        registry
            .register(Box::new(queue_depth.clone()))
            .expect("unique metric");

        Metrics {
            events_received: counter(
                "context_guard_events_received_total",
                "Telemetry events received by kind",
                &["kind"],
            ),
            events_dropped: counter(
                "context_guard_events_dropped_total",
                "Telemetry events dropped by reason",
                &["reason"],
            ),
            processing_errors: counter(
                "context_guard_processing_errors_total",
                "Processing errors by stage",
                &["stage"],
            ),
            health_score: gauge(
                "context_guard_health_score",
                "Health of the most recently scored turn per model",
                &["model"],
            ),
            risk_score: gauge(
                "context_guard_risk_score",
                "Risk of the most recently scored turn per model",
                &["model"],
            ),
            context_utilization: gauge(
                "context_guard_context_utilization_ratio",
                "Prompt tokens / context limit of the most recent turn per model",
                &["model"],
            ),
            conversations_by_status,
            known_value_drift: counter(
                "context_guard_known_value_drift_total",
                "Known-value drift anomalies",
                &["model"],
            ),
            tool_anomalies: counter(
                "context_guard_tool_anomalies_total",
                "Tool anomalies",
                &["model", "signal"],
            ),
            loop_events: counter(
                "context_guard_loop_events_total",
                "Repetition anomalies",
                &["model", "signal"],
            ),
            suspicious_identifiers: counter(
                "context_guard_suspicious_identifiers_total",
                "Suspicious identifier anomalies",
                &["model"],
            ),
            ingest_batch_size,
            queue_depth,
            registry,
        }
    }

    pub fn record_anomaly(&self, model: &str, signal: Signal) {
        match signal.family() {
            SignalFamily::KnownValueDrift => {
                self.known_value_drift.with_label_values(&[model]).inc()
            }
            SignalFamily::ToolAnomaly => self
                .tool_anomalies
                .with_label_values(&[model, signal.as_str()])
                .inc(),
            SignalFamily::Loop => self
                .loop_events
                .with_label_values(&[model, signal.as_str()])
                .inc(),
            SignalFamily::SuspiciousIdentifier => self
                .suspicious_identifiers
                .with_label_values(&[model])
                .inc(),
            SignalFamily::Context => {}
        }
    }

    pub fn record_score(&self, model: &str, health: u32, risk: u32, utilization: Option<f64>) {
        self.health_score
            .with_label_values(&[model])
            .set(f64::from(health));
        self.risk_score
            .with_label_values(&[model])
            .set(f64::from(risk));
        if let Some(ratio) = utilization {
            self.context_utilization
                .with_label_values(&[model])
                .set(ratio);
        }
    }

    pub fn encode(&self) -> String {
        let mut buf = Vec::new();
        TextEncoder::new()
            .encode(&self.registry.gather(), &mut buf)
            .ok();
        String::from_utf8(buf).unwrap_or_default()
    }
}

impl Default for Metrics {
    fn default() -> Self {
        Metrics::new()
    }
}
