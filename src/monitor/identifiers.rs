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
        let peers: Vec<&Identifier> = registry
            .iter()
            .filter(|k| k.kind.group() == group)
            .collect();
        if peers
            .iter()
            .any(|k| comparable(claim, k) == comparable(k, claim))
        {
            continue;
        }
        let similar = peers
            .iter()
            .find(|k| is_similar(&comparable(claim, k), &comparable(k, claim), group == 0));
        if let Some(known) = similar {
            let s = Suspicious {
                claimed: claim.clone(),
                similar_to: (*known).clone(),
            };
            if !out.iter().any(|x| x.claimed == s.claimed) {
                out.push(s);
            }
        }
    }
    out
}

/// The form an identifier is compared in against `other`. Plain names use
/// hyphen and underscore interchangeably in prose (`daemon-reload` for
/// Ansible's `daemon_reload`); container, model and host names are looked up
/// verbatim, so for them the separator is part of the name.
fn comparable(id: &Identifier, other: &Identifier) -> String {
    let lower = id.value.to_ascii_lowercase();
    if id.kind == IdKind::Name && other.kind == IdKind::Name {
        lower.replace('_', "-")
    } else {
        lower
    }
}

/// `word` is `stem` with an English plural ending: prose about "the
/// auto-respawns" names the known `auto-respawn`, it does not invent a new one.
fn is_plural_of(word: &str, stem: &str) -> bool {
    ["es", "s"]
        .iter()
        .any(|suffix| word.strip_suffix(suffix) == Some(stem))
}

/// `claim` resembles `known` closely enough to be a slip or an invention:
/// within the edit budget, or (names only) `known` extended by a suffix such
/// as `-v2`. A plural, a shortened form beyond the edit budget (`re-auth` for
/// `re-authenticate`) and a dotted attribute (`ansible_facts.env`) all name
/// the known thing rather than a new one. Both inputs are already lowercased.
fn is_similar(claim: &str, known: &str, allow_prefix: bool) -> bool {
    let longer = claim.chars().count().max(known.chars().count());
    if longer == 0 || is_plural_of(claim, known) || is_plural_of(known, claim) {
        return false;
    }
    let claim_extends_known = claim.starts_with(known);
    if !allow_prefix && (claim_extends_known || known.starts_with(claim)) {
        return false; // a longer path or variable built on a known one is normal
    }
    if claim_extends_known && claim[known.len()..].starts_with('.') {
        return false; // an attribute of a known name, not a new name
    }
    let distance = levenshtein(claim, known) as f64 / longer as f64;
    if distance <= MAX_RELATIVE_DISTANCE {
        return true;
    }
    allow_prefix && claim_extends_known && known.chars().count() >= MIN_PREFIX_LEN
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
    fn plurals_of_known_names_are_not_suspicious() {
        let registry = vec![Identifier {
            kind: IdKind::Name,
            value: "auto-respawn".into(),
        }];
        let plural = vec![Identifier {
            kind: IdKind::Name,
            value: "auto-respawns".into(),
        }];
        assert!(detect_suspicious(&plural, &registry).is_empty());
        // A stem ending in "e" takes a plain "s"; the "es" rule must not shadow it.
        let registry_e = vec![id(IdKind::Name, "known-value")];
        assert!(detect_suspicious(&[id(IdKind::Name, "known-values")], &registry_e).is_empty());
        let typo = vec![Identifier {
            kind: IdKind::Name,
            value: "auto-respwan".into(),
        }];
        assert_eq!(
            detect_suspicious(&typo, &registry).len(),
            1,
            "a real near-miss still fires"
        );
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

    #[test]
    fn shortened_forms_of_known_names_are_not_suspicious() {
        // Prose shortens real names; only an extension of a known name is invention.
        for (claim, known) in [
            ("re-run", "re-running"),
            ("re-auth", "re-authenticate"),
            ("gnome-keyring", "gnome-keyring-daemon"),
            ("tool_result", "tool_result_without_call"),
        ] {
            let registry = vec![id(IdKind::Name, known)];
            assert!(
                detect_suspicious(&[id(IdKind::Name, claim)], &registry).is_empty(),
                "{claim} vs {known}"
            );
        }
        // Dropping a short tail stays within the edit budget and still fires.
        let registry = vec![id(IdKind::Model, "qwen3-general-v2")];
        assert_eq!(
            detect_suspicious(&[id(IdKind::Model, "qwen3-general")], &registry).len(),
            1
        );
        // An extension beyond the edit budget fires through the suffix rule.
        let registry = vec![id(IdKind::Name, "auto-respawn")];
        assert_eq!(
            detect_suspicious(&[id(IdKind::Name, "auto-respawn-service")], &registry).len(),
            1
        );
    }

    #[test]
    fn separator_only_differences_name_the_same_thing() {
        let registry = vec![
            id(IdKind::Name, "daemon_reload"),
            id(IdKind::Name, "tool_results"),
            id(IdKind::Name, "known_values"),
        ];
        for claim in ["daemon-reload", "tool-result", "known-value"] {
            assert!(
                detect_suspicious(&[id(IdKind::Name, claim)], &registry).is_empty(),
                "{claim}"
            );
        }
        // Containers and models are looked up verbatim: the separator is part of the name.
        let registry = vec![
            id(IdKind::Container, "ai-litellm"),
            id(IdKind::Model, "qwen3-general"),
        ];
        assert_eq!(
            detect_suspicious(&[id(IdKind::Name, "ai_litellm")], &registry).len(),
            1
        );
        assert_eq!(
            detect_suspicious(&[id(IdKind::Model, "qwen3_general")], &registry).len(),
            1
        );
    }

    #[test]
    fn dotted_extension_of_known_name_is_attribute_access() {
        let registry = vec![id(IdKind::Name, "ansible_facts")];
        for claim in ["ansible_facts.env", "ansible_facts.os"] {
            assert!(
                detect_suspicious(&[id(IdKind::Name, claim)], &registry).is_empty(),
                "{claim}"
            );
        }
        assert_eq!(
            detect_suspicious(&[id(IdKind::Name, "ansible_facts-v2")], &registry).len(),
            1
        );
    }
}
