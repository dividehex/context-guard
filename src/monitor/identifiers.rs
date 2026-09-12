//! Identifier registry and near-duplicate detection (`qwen3-general` vs
//! `qwen3-general-v2`). Low-severity by design.

use std::sync::LazyLock;

use regex::Regex;
use serde::Serialize;

use super::text::levenshtein;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum IdKind {
    Model,
    Container,
    Host,
    Path,
    EnvVar,
    Tool,
    Name,
}

impl IdKind {
    pub fn as_str(self) -> &'static str {
        match self {
            IdKind::Model => "model",
            IdKind::Container => "container",
            IdKind::Host => "host",
            IdKind::Path => "path",
            IdKind::EnvVar => "env_var",
            IdKind::Tool => "tool",
            IdKind::Name => "name",
        }
    }

    pub fn parse(s: &str) -> Option<IdKind> {
        Some(match s {
            "model" => IdKind::Model,
            "container" => IdKind::Container,
            "host" => IdKind::Host,
            "path" => IdKind::Path,
            "env_var" => IdKind::EnvVar,
            "tool" => IdKind::Tool,
            "name" => IdKind::Name,
            _ => return None,
        })
    }

    /// Identifiers compared against each other: names of things (models,
    /// containers, hosts, tools) form one group; paths and env vars stand alone.
    fn group(self) -> u8 {
        match self {
            IdKind::Model | IdKind::Container | IdKind::Host | IdKind::Tool | IdKind::Name => 0,
            IdKind::Path => 1,
            IdKind::EnvVar => 2,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Identifier {
    pub kind: IdKind,
    pub value: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Suspicious {
    pub claimed: Identifier,
    pub similar_to: Identifier,
}

const MIN_LEN: usize = 4;
const MAX_RELATIVE_DISTANCE: f64 = 0.2;
const MIN_PREFIX_LEN: usize = 6;

static HOST_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b((?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+(?:com|net|org|io|dev|ai|app|cloud|tech|local|lan|home|internal))\b").unwrap()
});
static PATH_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(?:^|[\s"'`(=:,<>])(/[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)+)"#).unwrap()
});
static ENV_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?:\$\{?|\b)([A-Z][A-Z0-9_]{2,})(?:\}|=|\b)").unwrap());
static NAME_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?:^|[\s`'\x22(,;:])([a-z][a-z0-9]*(?:[-_.][a-z0-9]+)+)").unwrap()
});

/// Identifiers mentioned in text. Hyphenated lowercase names (`llama-swap`,
/// `qwen3-general`) are `Name`; prefixed ones (`ai-litellm`) are `Container`.
pub fn extract(text: &str, container_prefixes: &[String]) -> Vec<Identifier> {
    let mut out: Vec<Identifier> = Vec::new();
    let mut push = |id: Identifier| {
        if id.value.len() >= MIN_LEN && !out.contains(&id) {
            out.push(id);
        }
    };
    for c in HOST_RE.captures_iter(text) {
        push(Identifier {
            kind: IdKind::Host,
            value: c[1].to_ascii_lowercase(),
        });
    }
    for c in PATH_RE.captures_iter(text) {
        push(Identifier {
            kind: IdKind::Path,
            value: c[1].trim_end_matches(['.', ',', ';', ':', ')']).to_string(),
        });
    }
    for c in ENV_RE.captures_iter(text) {
        let name = &c[1];
        if name.contains('_')
            || text[..c.get(1).unwrap().start()].ends_with('$')
            || text[..c.get(1).unwrap().start()].ends_with("${")
        {
            push(Identifier {
                kind: IdKind::EnvVar,
                value: name.to_string(),
            });
        }
    }
    for c in NAME_RE.captures_iter(text) {
        let name = c[1].trim_end_matches('.');
        if name.contains('/')
            || name.ends_with(".yaml")
            || name.ends_with(".yml")
            || name.ends_with(".json")
            || name.ends_with(".py")
            || name.ends_with(".rs")
        {
            continue;
        }
        let kind = if container_prefixes
            .iter()
            .any(|p| !p.is_empty() && name.starts_with(p.as_str()))
        {
            IdKind::Container
        } else {
            IdKind::Name
        };
        push(Identifier {
            kind,
            value: name.to_string(),
        });
    }
    out
}

/// Claims that are new to the registry but closely resemble a known identifier.
pub fn detect_suspicious(claims: &[Identifier], registry: &[Identifier]) -> Vec<Suspicious> {
    let mut out: Vec<Suspicious> = Vec::new();
    for claim in claims {
        let group = claim.kind.group();
        let known_exactly = registry
            .iter()
            .any(|k| k.kind.group() == group && k.value.eq_ignore_ascii_case(&claim.value));
        if known_exactly {
            continue;
        }
        let similar = registry
            .iter()
            .filter(|k| k.kind.group() == group)
            .find(|k| is_similar(&claim.value, &k.value, group == 0));
        if let Some(known) = similar {
            let s = Suspicious {
                claimed: claim.clone(),
                similar_to: known.clone(),
            };
            if !out.iter().any(|x| x.claimed == s.claimed) {
                out.push(s);
            }
        }
    }
    out
}

fn is_similar(a: &str, b: &str, allow_prefix: bool) -> bool {
    let a = a.to_ascii_lowercase();
    let b = b.to_ascii_lowercase();
    let longer = a.chars().count().max(b.chars().count());
    if longer == 0 {
        return false;
    }
    let one_extends_other = a.starts_with(&b) || b.starts_with(&a);
    if !allow_prefix && one_extends_other {
        return false; // a longer path or variable built on a known one is normal
    }
    let distance = levenshtein(&a, &b) as f64 / longer as f64;
    if distance <= MAX_RELATIVE_DISTANCE {
        return true;
    }
    if allow_prefix {
        let shorter = a.chars().count().min(b.chars().count());
        return shorter >= MIN_PREFIX_LEN && one_extends_other;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(kind: IdKind, v: &str) -> Identifier {
        Identifier {
            kind,
            value: v.into(),
        }
    }

    #[test]
    fn extracts_names_containers_hosts_paths_env() {
        let ids = extract("Run ai-litellm and llama-swap with $LLAMA_SWAP_API_KEY, see /app/config.yaml on host nas.home", &["ai-".into()]);
        assert!(ids.contains(&id(IdKind::Container, "ai-litellm")));
        assert!(ids.contains(&id(IdKind::Name, "llama-swap")));
        assert!(ids.contains(&id(IdKind::EnvVar, "LLAMA_SWAP_API_KEY")));
        assert!(ids.contains(&id(IdKind::Path, "/app/config.yaml")));
        assert!(ids.contains(&id(IdKind::Host, "nas.home")));
    }

    #[test]
    fn near_duplicate_of_known_model_is_suspicious() {
        let registry = vec![
            id(IdKind::Model, "qwen3-general"),
            id(IdKind::Container, "ai-litellm"),
        ];
        let s = detect_suspicious(
            &extract("switch to qwen3-general-v2 now", &["ai-".into()]),
            &registry,
        );
        assert_eq!(s.len(), 1);
        assert_eq!(s[0].similar_to.value, "qwen3-general");
        // Exact known name: nothing.
        assert!(detect_suspicious(&extract("use qwen3-general", &[]), &registry).is_empty());
        // Unrelated name: nothing.
        assert!(detect_suspicious(&extract("use mistral-small", &[]), &registry).is_empty());
    }

    #[test]
    fn paths_only_use_edit_distance_not_prefix() {
        let registry = vec![id(IdKind::Path, "/data/context-guard.db")];
        assert!(
            detect_suspicious(&[id(IdKind::Path, "/data/context-guard.db-wal")], &registry)
                .is_empty()
        );
        assert_eq!(
            detect_suspicious(&[id(IdKind::Path, "/data/context-guard.bd")], &registry).len(),
            1
        );
    }
}
