//! Context-window utilization.

use super::scoring::Signal;

#[derive(Debug, Clone, PartialEq)]
pub struct ContextAssessment {
    pub prompt_tokens: Option<u64>,
    pub limit: Option<u64>,
    /// `None` when either side is unknown.
    pub percent: Option<f64>,
    pub signal: Option<Signal>,
    /// The backend rejected the request because it exceeded the context window.
    pub overflow: bool,
}

pub fn assess(prompt_tokens: Option<u64>, limit: Option<u64>) -> ContextAssessment {
    let percent = match (prompt_tokens, limit) {
        (Some(p), Some(l)) if l > 0 => Some(p as f64 / l as f64 * 100.0),
        _ => None,
    };
    let signal = percent.and_then(|pct| {
        if pct > 90.0 {
            Some(Signal::Context90)
        } else if pct >= 80.0 {
            Some(Signal::Context80)
        } else if pct >= 70.0 {
            Some(Signal::Context70)
        } else {
            None
        }
    });
    ContextAssessment {
        prompt_tokens,
        limit,
        percent,
        signal,
        overflow: false,
    }
}

/// The request was rejected for exceeding the window: always the top band,
/// with the percentage when the backend reported the request size.
pub fn overflow(prompt_tokens: Option<u64>, limit: Option<u64>) -> ContextAssessment {
    let percent = match (prompt_tokens, limit) {
        (Some(p), Some(l)) if l > 0 => Some(p as f64 / l as f64 * 100.0),
        _ => None,
    };
    ContextAssessment {
        prompt_tokens,
        limit,
        percent,
        signal: Some(Signal::Context90),
        overflow: true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn signal_at(pct: u64) -> Option<Signal> {
        assess(Some(pct * 100), Some(10_000)).signal
    }

    #[test]
    fn bands_match_specification() {
        assert_eq!(signal_at(50), None);
        assert_eq!(signal_at(69), None);
        assert_eq!(signal_at(75), Some(Signal::Context70));
        assert_eq!(signal_at(85), Some(Signal::Context80));
        assert_eq!(signal_at(95), Some(Signal::Context90));
    }

    #[test]
    fn overflow_is_always_the_top_band() {
        let o = overflow(Some(16_456), Some(12_288));
        assert_eq!(o.signal, Some(Signal::Context90));
        assert!(o.overflow);
        assert!(o.percent.unwrap() > 100.0);
        assert_eq!(overflow(None, Some(1)).signal, Some(Signal::Context90));
    }

    #[test]
    fn unknown_limit_reports_unknown_not_a_guess() {
        let a = assess(Some(5000), None);
        assert_eq!(a.percent, None);
        assert_eq!(a.signal, None);
        assert_eq!(assess(Some(1), Some(0)).percent, None);
        assert_eq!(assess(None, Some(100)).percent, None);
    }
}
