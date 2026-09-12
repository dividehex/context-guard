//! Process configuration: environment variables layered over an optional TOML
//! file layered over compiled defaults. Nothing else in the crate reads the
//! environment.

use std::collections::HashMap;
use std::path::PathBuf;

use serde::Deserialize;
use thiserror::Error;

use crate::monitor::scoring::Signal;

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("invalid value for {name}: {value:?} ({reason})")]
    Invalid {
        name: &'static str,
        value: String,
        reason: String,
    },
    #[error("cannot read config file {path}: {source}")]
    Read {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("cannot parse config file {path}: {source}")]
    Parse {
        path: PathBuf,
        source: toml::de::Error,
    },
}

/// Penalty points per signal. `health = 100 - Σ penalties`.
#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct Penalties {
    pub context_70: u32,
    pub context_80: u32,
    pub context_90: u32,
    pub repeated_tool_call: u32,
    pub response_loop: u32,
    pub known_value_drift: u32,
    pub tool_result_without_call: u32,
    pub tool_call_id_reference_unknown: u32,
    pub suspicious_identifier: u32,
}

impl Default for Penalties {
    fn default() -> Self {
        Self {
            context_70: 5,
            context_80: 10,
            context_90: 20,
            repeated_tool_call: 5,
            response_loop: 5,
            known_value_drift: 15,
            tool_result_without_call: 20,
            tool_call_id_reference_unknown: 25,
            suspicious_identifier: 5,
        }
    }
}

impl Penalties {
    pub fn for_signal(&self, signal: Signal) -> u32 {
        match signal {
            Signal::Context70 => self.context_70,
            Signal::Context80 => self.context_80,
            Signal::Context90 => self.context_90,
            Signal::RepeatedToolCall => self.repeated_tool_call,
            Signal::ResponseLoop => self.response_loop,
            Signal::KnownValueDrift => self.known_value_drift,
            Signal::ToolResultWithoutCall => self.tool_result_without_call,
            Signal::ToolCallIdReferenceUnknown => self.tool_call_id_reference_unknown,
            Signal::SuspiciousIdentifier => self.suspicious_identifier,
        }
    }
}

/// Lower bounds (inclusive) of the health statuses below `healthy` (≥ 90).
#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct Thresholds {
    pub good: u32,
    pub watch: u32,
    pub degraded: u32,
}

impl Default for Thresholds {
    fn default() -> Self {
        Self {
            good: 75,
            watch: 60,
            degraded: 40,
        }
    }
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct Scoring {
    /// Anomalies recorded within this many most recent turns count toward risk.
    pub window_turns: u32,
}

impl Default for Scoring {
    fn default() -> Self {
        Self { window_turns: 10 }
    }
}

/// The optional TOML file. Every section is optional.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct FileConfig {
    penalties: Penalties,
    thresholds: Thresholds,
    scoring: Scoring,
    model_limits: HashMap<String, u64>,
}

#[derive(Debug, Clone)]
pub struct Config {
    pub listen: String,
    pub metrics_listen: Option<String>,
    pub database: PathBuf,
    pub retention_days: u32,
    pub queue_size: usize,
    pub max_body_bytes: usize,
    pub store_messages: bool,
    pub log_payloads: bool,
    pub log_json: bool,
    pub capture_dir: Option<PathBuf>,
    pub trust_trace_id: bool,
    pub container_prefixes: Vec<String>,
    pub model_limits: HashMap<String, u64>,
    pub penalties: Penalties,
    pub thresholds: Thresholds,
    pub scoring: Scoring,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            listen: "0.0.0.0:7432".to_string(),
            metrics_listen: None,
            database: PathBuf::from("/data/context-guard.db"),
            retention_days: 30,
            queue_size: 1024,
            max_body_bytes: 32 * 1024 * 1024,
            store_messages: false,
            log_payloads: false,
            log_json: false,
            capture_dir: None,
            trust_trace_id: false,
            container_prefixes: vec!["ai-".to_string()],
            model_limits: HashMap::new(),
            penalties: Penalties::default(),
            thresholds: Thresholds::default(),
            scoring: Scoring::default(),
        }
    }
}

impl Config {
    /// Load from the process environment (and the TOML file it may point at).
    pub fn load() -> Result<Self, ConfigError> {
        let env: HashMap<String, String> = std::env::vars().collect();
        Self::from_env(&env)
    }

    pub fn from_env(env: &HashMap<String, String>) -> Result<Self, ConfigError> {
        let mut cfg = Config::default();

        if let Some(path) = env.get("CONTEXT_GUARD_CONFIG").filter(|p| !p.is_empty()) {
            let path = PathBuf::from(path);
            let text = std::fs::read_to_string(&path).map_err(|source| ConfigError::Read {
                path: path.clone(),
                source,
            })?;
            let file: FileConfig = toml::from_str(&text).map_err(|source| ConfigError::Parse {
                path: path.clone(),
                source,
            })?;
            cfg.penalties = file.penalties;
            cfg.thresholds = file.thresholds;
            cfg.scoring = file.scoring;
            cfg.model_limits = file.model_limits;
        }

        if let Some(v) = non_empty(env, "CONTEXT_GUARD_LISTEN") {
            cfg.listen = v.to_string();
        }
        cfg.metrics_listen = non_empty(env, "CONTEXT_GUARD_METRICS_LISTEN").map(str::to_string);
        if let Some(v) = non_empty(env, "CONTEXT_GUARD_DATABASE") {
            cfg.database = PathBuf::from(v);
        }
        if let Some(v) = non_empty(env, "CONTEXT_GUARD_RETENTION_DAYS") {
            cfg.retention_days = parse_number("CONTEXT_GUARD_RETENTION_DAYS", v)?;
        }
        if let Some(v) = non_empty(env, "CONTEXT_GUARD_QUEUE_SIZE") {
            cfg.queue_size = parse_number("CONTEXT_GUARD_QUEUE_SIZE", v)?;
            if cfg.queue_size == 0 {
                return Err(invalid("CONTEXT_GUARD_QUEUE_SIZE", v, "must be at least 1"));
            }
        }
        if let Some(v) = non_empty(env, "CONTEXT_GUARD_MAX_BODY_BYTES") {
            cfg.max_body_bytes = parse_number("CONTEXT_GUARD_MAX_BODY_BYTES", v)?;
        }
        if let Some(v) = non_empty(env, "CONTEXT_GUARD_STORE_MESSAGES") {
            cfg.store_messages = parse_bool("CONTEXT_GUARD_STORE_MESSAGES", v)?;
        }
        if let Some(v) = non_empty(env, "CONTEXT_GUARD_LOG_PAYLOADS") {
            cfg.log_payloads = parse_bool("CONTEXT_GUARD_LOG_PAYLOADS", v)?;
        }
        if let Some(v) = non_empty(env, "CONTEXT_GUARD_LOG_JSON") {
            cfg.log_json = parse_bool("CONTEXT_GUARD_LOG_JSON", v)?;
        }
        cfg.capture_dir = non_empty(env, "CONTEXT_GUARD_CAPTURE_DIR").map(PathBuf::from);
        if let Some(v) = non_empty(env, "CONTEXT_GUARD_TRUST_TRACE_ID") {
            cfg.trust_trace_id = parse_bool("CONTEXT_GUARD_TRUST_TRACE_ID", v)?;
        }
        if let Some(v) = non_empty(env, "CONTEXT_GUARD_CONTAINER_PREFIXES") {
            cfg.container_prefixes = v
                .split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(String::from)
                .collect();
        }
        if let Some(v) = non_empty(env, "CONTEXT_GUARD_MODEL_LIMITS") {
            for (model, limit) in parse_model_limits(v)? {
                cfg.model_limits.insert(model, limit);
            }
        }
        Ok(cfg)
    }

    /// Configured context limit for a model, if any.
    pub fn model_limit(&self, model: &str) -> Option<u64> {
        self.model_limits.get(model).copied()
    }
}

fn non_empty<'a>(env: &'a HashMap<String, String>, name: &str) -> Option<&'a str> {
    env.get(name)
        .map(String::as_str)
        .map(str::trim)
        .filter(|v| !v.is_empty())
}

fn invalid(name: &'static str, value: &str, reason: &str) -> ConfigError {
    ConfigError::Invalid {
        name,
        value: value.to_string(),
        reason: reason.to_string(),
    }
}

fn parse_number<T: std::str::FromStr>(name: &'static str, value: &str) -> Result<T, ConfigError> {
    value
        .parse::<T>()
        .map_err(|_| invalid(name, value, "expected a non-negative integer"))
}

fn parse_bool(name: &'static str, value: &str) -> Result<bool, ConfigError> {
    match value.to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        _ => Err(invalid(name, value, "expected true or false")),
    }
}

/// `model=limit,model=limit`
fn parse_model_limits(value: &str) -> Result<Vec<(String, u64)>, ConfigError> {
    let mut out = Vec::new();
    for item in value.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let (model, limit) = item.split_once('=').ok_or_else(|| {
            invalid(
                "CONTEXT_GUARD_MODEL_LIMITS",
                value,
                "expected model=tokens pairs",
            )
        })?;
        let limit: u64 = limit.trim().parse().map_err(|_| {
            invalid(
                "CONTEXT_GUARD_MODEL_LIMITS",
                value,
                "token limits must be integers",
            )
        })?;
        out.push((model.trim().to_string(), limit));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn defaults_apply_without_env() {
        let cfg = Config::from_env(&env(&[])).unwrap();
        assert_eq!(cfg.listen, "0.0.0.0:7432");
        assert_eq!(cfg.penalties, Penalties::default());
        assert_eq!(cfg.scoring.window_turns, 10);
    }

    #[test]
    fn env_overrides_and_model_limits_parse() {
        let cfg = Config::from_env(&env(&[
            ("CONTEXT_GUARD_LISTEN", "127.0.0.1:1"),
            (
                "CONTEXT_GUARD_MODEL_LIMITS",
                "qwen3-30b-a3b=122880, fury/qwen3-8b=12288",
            ),
            ("CONTEXT_GUARD_STORE_MESSAGES", "true"),
        ]))
        .unwrap();
        assert_eq!(cfg.listen, "127.0.0.1:1");
        assert_eq!(cfg.model_limit("qwen3-30b-a3b"), Some(122_880));
        assert_eq!(cfg.model_limit("fury/qwen3-8b"), Some(12_288));
        assert!(cfg.store_messages);
    }

    #[test]
    fn bad_values_are_errors_not_panics() {
        assert!(Config::from_env(&env(&[("CONTEXT_GUARD_RETENTION_DAYS", "soon")])).is_err());
        assert!(Config::from_env(&env(&[("CONTEXT_GUARD_MODEL_LIMITS", "qwen3")])).is_err());
        assert!(Config::from_env(&env(&[("CONTEXT_GUARD_QUEUE_SIZE", "0")])).is_err());
    }

    #[test]
    fn every_variable_parses_and_every_bad_value_is_an_error() {
        let cfg = Config::from_env(&env(&[
            ("CONTEXT_GUARD_METRICS_LISTEN", "0.0.0.0:7433"),
            ("CONTEXT_GUARD_DATABASE", "/tmp/x.db"),
            ("CONTEXT_GUARD_MAX_BODY_BYTES", "1024"),
            ("CONTEXT_GUARD_LOG_PAYLOADS", "yes"),
            ("CONTEXT_GUARD_LOG_JSON", "on"),
            ("CONTEXT_GUARD_CAPTURE_DIR", "/tmp/cap"),
            ("CONTEXT_GUARD_TRUST_TRACE_ID", "1"),
            ("CONTEXT_GUARD_CONTAINER_PREFIXES", "ai-, svc-, ,"),
            ("CONTEXT_GUARD_QUEUE_SIZE", "5"),
        ]))
        .unwrap();
        assert_eq!(cfg.metrics_listen.as_deref(), Some("0.0.0.0:7433"));
        assert_eq!(cfg.database, PathBuf::from("/tmp/x.db"));
        assert_eq!(cfg.max_body_bytes, 1024);
        assert!(cfg.log_payloads && cfg.log_json && cfg.trust_trace_id);
        assert_eq!(cfg.capture_dir, Some(PathBuf::from("/tmp/cap")));
        assert_eq!(
            cfg.container_prefixes,
            vec!["ai-".to_string(), "svc-".to_string()]
        );
        assert_eq!(cfg.queue_size, 5);

        for (name, value) in [
            ("CONTEXT_GUARD_MAX_BODY_BYTES", "big"),
            ("CONTEXT_GUARD_LOG_PAYLOADS", "maybe"),
            ("CONTEXT_GUARD_LOG_JSON", "2"),
            ("CONTEXT_GUARD_TRUST_TRACE_ID", "sure"),
            ("CONTEXT_GUARD_STORE_MESSAGES", ""),
            ("CONTEXT_GUARD_MODEL_LIMITS", "m=lots"),
            ("CONTEXT_GUARD_CONFIG", "/definitely/missing.toml"),
        ] {
            let result = Config::from_env(&env(&[(name, value)]));
            if value.is_empty() {
                assert!(result.is_ok(), "empty values mean unset");
            } else {
                let err = result
                    .err()
                    .unwrap_or_else(|| panic!("{name}={value} should fail"));
                assert!(!err.to_string().is_empty());
            }
        }
        let dir = tempfile::tempdir().unwrap();
        let bad = dir.path().join("bad.toml");
        std::fs::write(&bad, "[penalties]\nnot_a_signal = 1\n").unwrap();
        assert!(
            Config::from_env(&env(&[("CONTEXT_GUARD_CONFIG", bad.to_str().unwrap())])).is_err(),
            "unknown keys are rejected"
        );
    }

    #[test]
    fn toml_file_sets_weights_and_env_wins_for_limits() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cg.toml");
        std::fs::write(
            &path,
            "[penalties]\nknown_value_drift = 30\n[model_limits]\n\"a\" = 100\n\"b\" = 200\n",
        )
        .unwrap();
        let cfg = Config::from_env(&env(&[
            ("CONTEXT_GUARD_CONFIG", path.to_str().unwrap()),
            ("CONTEXT_GUARD_MODEL_LIMITS", "b=300"),
        ]))
        .unwrap();
        assert_eq!(cfg.penalties.known_value_drift, 30);
        assert_eq!(cfg.penalties.context_70, 5);
        assert_eq!(cfg.model_limit("a"), Some(100));
        assert_eq!(cfg.model_limit("b"), Some(300));
    }
}
