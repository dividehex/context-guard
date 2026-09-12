//! Risk → health conversion with transparent reasons.

use serde::Serialize;

use super::context::ContextAssessment;
use crate::config::{Penalties, Thresholds};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Signal {
    Context70,
    Context80,
    Context90,
    RepeatedToolCall,
    ResponseLoop,
    KnownValueDrift,
    ToolResultWithoutCall,
    ToolCallIdReferenceUnknown,
    SuspiciousIdentifier,
}

impl Signal {
    pub fn as_str(self) -> &'static str {
        match self {
            Signal::Context70 => "context_70",
            Signal::Context80 => "context_80",
            Signal::Context90 => "context_90",
            Signal::RepeatedToolCall => "repeated_tool_call",
            Signal::ResponseLoop => "response_loop",
            Signal::KnownValueDrift => "known_value_drift",
            Signal::ToolResultWithoutCall => "tool_result_without_call",
            Signal::ToolCallIdReferenceUnknown => "tool_call_id_reference_unknown",
            Signal::SuspiciousIdentifier => "suspicious_identifier",
        }
    }

    pub fn parse(s: &str) -> Option<Signal> {
        Some(match s {
            "context_70" => Signal::Context70,
            "context_80" => Signal::Context80,
            "context_90" => Signal::Context90,
            "repeated_tool_call" => Signal::RepeatedToolCall,
            "response_loop" => Signal::ResponseLoop,
            "known_value_drift" => Signal::KnownValueDrift,
            "tool_result_without_call" => Signal::ToolResultWithoutCall,
            "tool_call_id_reference_unknown" => Signal::ToolCallIdReferenceUnknown,
            "suspicious_identifier" => Signal::SuspiciousIdentifier,
            _ => return None,
        })
    }

    pub fn severity(self) -> &'static str {
        match self {
            Signal::ToolResultWithoutCall | Signal::ToolCallIdReferenceUnknown => "high",
            Signal::KnownValueDrift | Signal::Context90 => "medium",
            _ => "low",
        }
    }

    /// Human phrase used in summaries and reason strings.
    pub fn phrase(self) -> &'static str {
        match self {
            Signal::Context70 | Signal::Context80 | Signal::Context90 => "context pressure",
            Signal::RepeatedToolCall => "repeated operation",
            Signal::ResponseLoop => "response loop",
            Signal::KnownValueDrift => "known-value drift",
            Signal::ToolResultWithoutCall => "orphan tool result",
            Signal::ToolCallIdReferenceUnknown => "unknown tool-call reference",
            Signal::SuspiciousIdentifier => "suspicious identifier",
        }
    }

    /// Short label for the one-line status summary.
    pub fn short(self) -> &'static str {
        match self {
            Signal::Context70 | Signal::Context80 | Signal::Context90 => "context",
            Signal::RepeatedToolCall => "repeated call",
            Signal::ResponseLoop => "loop",
            Signal::KnownValueDrift => "drift",
            Signal::ToolResultWithoutCall => "orphan result",
            Signal::ToolCallIdReferenceUnknown => "unknown call id",
            Signal::SuspiciousIdentifier => "suspicious id",
        }
    }

    /// Bucket used by the API's `signals` counts and by Prometheus.
    pub fn family(self) -> SignalFamily {
        match self {
            Signal::Context70 | Signal::Context80 | Signal::Context90 => SignalFamily::Context,
            Signal::RepeatedToolCall | Signal::ResponseLoop => SignalFamily::Loop,
            Signal::KnownValueDrift => SignalFamily::KnownValueDrift,
            Signal::ToolResultWithoutCall | Signal::ToolCallIdReferenceUnknown => {
                SignalFamily::ToolAnomaly
            }
            Signal::SuspiciousIdentifier => SignalFamily::SuspiciousIdentifier,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignalFamily {
    Context,
    Loop,
    KnownValueDrift,
    ToolAnomaly,
    SuspiciousIdentifier,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    Healthy,
    Good,
    Watch,
    Degraded,
    ResetRecommended,
}

impl Status {
    pub fn as_str(self) -> &'static str {
        match self {
            Status::Healthy => "healthy",
            Status::Good => "good",
            Status::Watch => "watch",
            Status::Degraded => "degraded",
            Status::ResetRecommended => "reset_recommended",
        }
    }

    pub fn for_health(health: u32, t: &Thresholds) -> Status {
        if health >= 90 {
            Status::Healthy
        } else if health >= t.good {
            Status::Good
        } else if health >= t.watch {
            Status::Watch
        } else if health >= t.degraded {
            Status::Degraded
        } else {
            Status::ResetRecommended
        }
    }
}

/// One contribution to the risk score.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, serde::Deserialize)]
pub struct Reason {
    pub signal: String,
    pub penalty: u32,
    pub detail: String,
}

/// An anomaly still inside the scoring window, with the penalty it was recorded with.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WindowAnomaly {
    pub signal: Signal,
    pub penalty: u32,
    pub detail: String,
    pub turn: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Score {
    pub health: u32,
    pub risk: u32,
    pub status: Status,
    pub reasons: Vec<Reason>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct SignalCounts {
    pub known_value_drift: u32,
    pub tool_anomalies: u32,
    pub loop_events: u32,
    pub suspicious_identifiers: u32,
}

pub fn count_signals(anomalies: &[WindowAnomaly]) -> SignalCounts {
    let mut c = SignalCounts::default();
    for a in anomalies {
        match a.signal.family() {
            SignalFamily::KnownValueDrift => c.known_value_drift += 1,
            SignalFamily::ToolAnomaly => c.tool_anomalies += 1,
            SignalFamily::Loop => c.loop_events += 1,
            SignalFamily::SuspiciousIdentifier => c.suspicious_identifiers += 1,
            SignalFamily::Context => {}
        }
    }
    c
}

pub fn score(
    context: &ContextAssessment,
    anomalies: &[WindowAnomaly],
    penalties: &Penalties,
    thresholds: &Thresholds,
) -> Score {
    let mut reasons = Vec::new();
    if let Some(signal) = context.signal {
        let detail = if context.overflow {
            match context.prompt_tokens {
                Some(n) => format!("request of {n} tokens exceeded the model's context window"),
                None => "request exceeded the model's context window".to_string(),
            }
        } else {
            format!(
                "context utilization {:.1}% of {} tokens",
                context.percent.unwrap_or(0.0),
                context.limit.unwrap_or(0)
            )
        };
        reasons.push(Reason {
            signal: signal.as_str().to_string(),
            penalty: penalties.for_signal(signal),
            detail,
        });
    }
    for a in anomalies {
        reasons.push(Reason {
            signal: a.signal.as_str().to_string(),
            penalty: a.penalty,
            detail: a.detail.clone(),
        });
    }
    let risk: u32 = reasons.iter().map(|r| r.penalty).sum::<u32>().min(100);
    let health = 100 - risk;
    Score {
        health,
        risk,
        status: Status::for_health(health, thresholds),
        reasons,
    }
}

/// One line for the Open WebUI status widget, which renders a single line
/// with an ellipsis (`line-clamp-1`), e.g.
/// `🟢 Context Guard 92 · healthy · 🟡 context 74% (90,800/122,880) · 1 drift · 1 loop`
///
/// Two lights: the first is the health status, the second the context-window
/// pressure, on the same colour scale. Anomalies use short labels and are
/// added while the line stays within `SUMMARY_BUDGET` characters; the rest
/// collapse into `+N more`. The full reasons are in the API response.
pub fn summary(score: &Score, context: &ContextAssessment, anomalies: &[WindowAnomaly]) -> String {
    let mut line = format!(
        "{} Context Guard {} · {} · {}",
        health_light(score.status),
        score.health,
        score.status.as_str().replace('_', " "),
        context_phrase(context)
    );
    let mut counts: Vec<(Signal, usize)> = Vec::new();
    for a in anomalies {
        match counts.iter_mut().find(|(s, _)| *s == a.signal) {
            Some((_, n)) => *n += 1,
            None => counts.push((a.signal, 1)),
        }
    }
    let total = counts.len();
    for (i, (signal, n)) in counts.iter().enumerate() {
        let part = format!(" · {n} {}", signal.short());
        let remaining = total - i;
        let more = if remaining > 1 {
            format!(" · +{} more", remaining - 1)
        } else {
            String::new()
        };
        if line.chars().count() + part.chars().count() + more.chars().count() > SUMMARY_BUDGET
            && i > 0
        {
            line.push_str(&format!(" · +{remaining} more"));
            return line;
        }
        line.push_str(&part);
    }
    line
}

/// Characters that fit on one status line at Open WebUI's default chat width.
const SUMMARY_BUDGET: usize = 96;

fn health_light(status: Status) -> &'static str {
    match status {
        Status::Healthy | Status::Good => "🟢",
        Status::Watch => "🟡",
        Status::Degraded => "🟠",
        Status::ResetRecommended => "🔴",
    }
}

/// Context light follows the penalty bands: green below 70 %, yellow 70–80 %,
/// orange 80–90 %, red above 90 %; white when the limit is unknown.
fn context_phrase(context: &ContextAssessment) -> String {
    if context.overflow {
        return match (context.prompt_tokens, context.limit) {
            (Some(used), Some(limit)) => format!(
                "🔴 context overflow ({}/{})",
                with_commas(used),
                with_commas(limit)
            ),
            (Some(used), None) => format!("🔴 context overflow ({} tokens)", with_commas(used)),
            _ => "🔴 context overflow".to_string(),
        };
    }
    match (context.percent, context.prompt_tokens, context.limit) {
        (Some(pct), Some(used), Some(limit)) => {
            let light = match context.signal {
                None => "🟢",
                Some(Signal::Context70) => "🟡",
                Some(Signal::Context80) => "🟠",
                _ => "🔴",
            };
            format!(
                "{light} context {pct:.0}% ({}/{})",
                with_commas(used),
                with_commas(limit)
            )
        }
        (_, Some(used), None) => {
            format!("⚪ context ? ({} tokens, limit unknown)", with_commas(used))
        }
        _ => "⚪ context unknown".to_string(),
    }
}

fn with_commas(n: u64) -> String {
    let digits = n.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monitor::context::assess;

    fn anomaly(signal: Signal, penalty: u32, turn: u32) -> WindowAnomaly {
        WindowAnomaly {
            signal,
            penalty,
            detail: format!("{} at turn {turn}", signal.as_str()),
            turn,
        }
    }

    #[test]
    fn risk_is_sum_of_reasons_and_deterministic() {
        let p = Penalties::default();
        let t = Thresholds::default();
        let ctx = assess(Some(7820), Some(10_000));
        let anomalies = vec![
            anomaly(Signal::KnownValueDrift, 15, 3),
            anomaly(Signal::RepeatedToolCall, 5, 4),
        ];
        let a = score(&ctx, &anomalies, &p, &t);
        let b = score(&ctx, &anomalies, &p, &t);
        assert_eq!(a, b);
        assert_eq!(a.risk, 25);
        assert_eq!(a.health, 75);
        assert_eq!(a.status, Status::Good);
        assert_eq!(a.reasons.iter().map(|r| r.penalty).sum::<u32>(), a.risk);
        assert_eq!(a.reasons[0].signal, "context_70");
        assert_eq!(
            summary(&a, &ctx, &anomalies),
            "🟢 Context Guard 75 · good · 🟡 context 78% (7,820/10,000) · 1 drift · 1 repeated call"
        );
    }

    #[test]
    fn overflow_scores_as_red_context() {
        let ctx = crate::monitor::context::overflow(Some(16_456), Some(12_288));
        let s = score(&ctx, &[], &Penalties::default(), &Thresholds::default());
        assert_eq!(s.risk, 20);
        assert_eq!(
            s.reasons[0].detail,
            "request of 16456 tokens exceeded the model's context window"
        );
        assert_eq!(
            summary(&s, &ctx, &[]),
            "🟢 Context Guard 80 · good · 🔴 context overflow (16,456/12,288)"
        );
    }

    #[test]
    fn health_is_clamped_to_zero() {
        let p = Penalties::default();
        let t = Thresholds::default();
        let anomalies: Vec<_> = (0..10)
            .map(|i| anomaly(Signal::ToolCallIdReferenceUnknown, 25, i))
            .collect();
        let s = score(&assess(None, None), &anomalies, &p, &t);
        assert_eq!(s.health, 0);
        assert_eq!(s.risk, 100);
        assert_eq!(s.status, Status::ResetRecommended);
        assert!(summary(&s, &assess(Some(9500), Some(10_000)), &anomalies)
            .starts_with("🔴 Context Guard 0 · reset recommended · 🔴 context 95% (9,500/10,000)"));
    }

    #[test]
    fn status_boundaries() {
        let t = Thresholds::default();
        assert_eq!(Status::for_health(100, &t), Status::Healthy);
        assert_eq!(Status::for_health(90, &t), Status::Healthy);
        assert_eq!(Status::for_health(89, &t), Status::Good);
        assert_eq!(Status::for_health(75, &t), Status::Good);
        assert_eq!(Status::for_health(74, &t), Status::Watch);
        assert_eq!(Status::for_health(60, &t), Status::Watch);
        assert_eq!(Status::for_health(59, &t), Status::Degraded);
        assert_eq!(Status::for_health(40, &t), Status::Degraded);
        assert_eq!(Status::for_health(39, &t), Status::ResetRecommended);
    }

    #[test]
    fn summary_stays_on_one_line_and_folds_the_rest() {
        let p = Penalties::default();
        let t = Thresholds::default();
        let all = vec![
            anomaly(Signal::KnownValueDrift, 15, 1),
            anomaly(Signal::SuspiciousIdentifier, 5, 2),
            anomaly(Signal::ResponseLoop, 5, 3),
            anomaly(Signal::ToolResultWithoutCall, 20, 4),
            anomaly(Signal::RepeatedToolCall, 5, 5),
            anomaly(Signal::ToolCallIdReferenceUnknown, 25, 6),
        ];
        let ctx = assess(Some(446), Some(122_880));
        let line = summary(&score(&ctx, &all, &p, &t), &ctx, &all);
        assert!(line.chars().count() <= SUMMARY_BUDGET, "{line}");
        assert_eq!(line, "🔴 Context Guard 25 · reset recommended · 🟢 context 0% (446/122,880) · 1 drift · +5 more");
        let three = &all[..3];
        let line = summary(&score(&ctx, three, &p, &t), &ctx, three);
        assert_eq!(line, "🟢 Context Guard 75 · good · 🟢 context 0% (446/122,880) · 1 drift · 1 suspicious id · 1 loop");
    }

    /// Two hundred seeded random anomaly sets: the accounting identities hold for all of them.
    #[test]
    fn invariants_hold_for_random_anomaly_sets() {
        let all = [
            Signal::RepeatedToolCall,
            Signal::ResponseLoop,
            Signal::KnownValueDrift,
            Signal::ToolResultWithoutCall,
            Signal::ToolCallIdReferenceUnknown,
            Signal::SuspiciousIdentifier,
        ];
        let p = Penalties::default();
        let t = Thresholds::default();
        let mut seed: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for _ in 0..200 {
            let n = (next() % 12) as usize;
            let anomalies: Vec<WindowAnomaly> = (0..n)
                .map(|i| {
                    let s = all[(next() % all.len() as u64) as usize];
                    anomaly(s, p.for_signal(s), i as u32)
                })
                .collect();
            let ctx = assess(Some(next() % 12_000), Some(10_000));
            let s = score(&ctx, &anomalies, &p, &t);
            let raw: u32 = s.reasons.iter().map(|r| r.penalty).sum();
            assert_eq!(
                raw.min(100),
                s.risk,
                "risk is the reasons' sum, capped at 100"
            );
            assert_eq!(s.health + s.risk, 100);
            assert!(s.risk <= 100);
            assert_eq!(s.status, Status::for_health(s.health, &t));
            assert_eq!(s, score(&ctx, &anomalies, &p, &t), "deterministic");
            assert!(summary(&s, &ctx, &anomalies).chars().count() <= SUMMARY_BUDGET + 12);
        }
    }

    #[test]
    fn no_signals_means_healthy_with_no_reasons() {
        let s = score(
            &assess(Some(10), Some(1000)),
            &[],
            &Penalties::default(),
            &Thresholds::default(),
        );
        assert_eq!(s.health, 100);
        assert!(s.reasons.is_empty());
        assert_eq!(
            summary(&s, &assess(Some(1234567), None), &[]),
            "🟢 Context Guard 100 · healthy · ⚪ context ? (1,234,567 tokens, limit unknown)"
        );
        assert_eq!(with_commas(999), "999");
        assert_eq!(with_commas(1000), "1,000");
        assert_eq!(
            summary(&s, &assess(None, None), &[]),
            "🟢 Context Guard 100 · healthy · ⚪ context unknown"
        );
    }
}
