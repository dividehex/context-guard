//! Passive fact registry: structured values stated by the user (or returned by
//! tools) and the conservative rule for flagging an assistant that contradicts
//! them. No prose semantics; only recognizable technical values.

use std::net::{Ipv4Addr, Ipv6Addr};
use std::sync::LazyLock;

use regex::Regex;
use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ValueKind {
    Ipv4,
    Ipv6,
    Port,
    Url,
    Path,
    EnvVar,
    Version,
    Hostname,
    Container,
    NumericCfg,
}

impl ValueKind {
    pub fn as_str(self) -> &'static str {
        match self {
            ValueKind::Ipv4 => "ipv4",
            ValueKind::Ipv6 => "ipv6",
            ValueKind::Port => "port",
            ValueKind::Url => "url",
            ValueKind::Path => "path",
            ValueKind::EnvVar => "env_var",
            ValueKind::Version => "version",
            ValueKind::Hostname => "hostname",
            ValueKind::Container => "container",
            ValueKind::NumericCfg => "numeric_cfg",
        }
    }

    pub fn parse(s: &str) -> Option<ValueKind> {
        Some(match s {
            "ipv4" => ValueKind::Ipv4,
            "ipv6" => ValueKind::Ipv6,
            "port" => ValueKind::Port,
            "url" => ValueKind::Url,
            "path" => ValueKind::Path,
            "env_var" => ValueKind::EnvVar,
            "version" => ValueKind::Version,
            "hostname" => ValueKind::Hostname,
            "container" => ValueKind::Container,
            "numeric_cfg" => ValueKind::NumericCfg,
            _ => return None,
        })
    }
}

/// A value found in text. `anchor` is the entity it belongs to when one is
/// recognizable (a host for a port, a key for a setting), else empty.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Extracted {
    pub kind: ValueKind,
    pub anchor: String,
    pub value: String,
}

/// A registry entry: a value established by a user or tool message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KnownValue {
    pub kind: ValueKind,
    pub anchor: String,
    pub value: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Drift {
    pub kind: ValueKind,
    pub anchor: String,
    pub known: String,
    pub claimed: String,
}

static URL_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#"https?://[^\s<>"'`)\]]+"#).unwrap());
static IPV4_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?:^|[^\w.])((?:\d{1,3}\.){3}\d{1,3})(?:[^\w.]|$)").unwrap());
static IPV6_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?:^|[\s\[=])([0-9A-Fa-f]{0,4}(?::[0-9A-Fa-f]{0,4}){2,7})(?:[\s\]/,;]|$)").unwrap()
});
static PORT_WORD_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)(?:\bport(?:\s+number)?\s*[:=#]?\s*|--port[= ]|-p\s+)(\d{1,5})\b").unwrap()
});
static HOST_PORT_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?:^|[\s(\[\x22'`=@])([A-Za-z][A-Za-z0-9.-]*):(\d{2,5})(?:[^\w.]|$)").unwrap()
});
static PATH_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(?:^|[\s"'`(=:,<>])(/[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)+)"#).unwrap()
});
static ENV_ASSIGN_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"\b([A-Z][A-Z0-9_]{2,})=("[^"\n]*"|'[^'\n]*'|[^\s"',;]+)"#).unwrap()
});
static VERSION_WORD_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b(?:version|v\.?)\s*(\d+\.\d+(?:\.\d+)?(?:[-+][A-Za-z0-9.]+)?)\b").unwrap()
});
static NAME_VERSION_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\b([A-Za-z][A-Za-z0-9._-]*[A-Za-z])\s+v?(\d+\.\d+\.\d+(?:[-+][A-Za-z0-9.]+)?)\b")
        .unwrap()
});
static HOSTNAME_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b((?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+([a-z]{2,}))\b").unwrap()
});
static CONTAINER_WORD_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)\bcontainer\s+[`']?([a-z0-9][a-z0-9_.-]+)").unwrap());
static NUMERIC_CFG_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r#"\b([a-z][a-z0-9]*(?:[_-][a-z0-9]+)+)\s*[:=]\s*["']?(\d+(?:\.\d+)?)["']?(?:[^\w.]|$)"#,
    )
    .unwrap()
});
static IDENT_TOKEN_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^[A-Za-z][A-Za-z0-9._/-]*$").unwrap());

/// Hostname TLDs we accept for bare (non-URL) hostnames, so that `config.yaml`
/// and `llama.cpp` are not mistaken for hosts.
const HOST_TLDS: &[&str] = &[
    "com",
    "net",
    "org",
    "io",
    "dev",
    "ai",
    "app",
    "cloud",
    "tech",
    "me",
    "us",
    "uk",
    "de",
    "eu",
    "edu",
    "gov",
    "co",
    "xyz",
    "local",
    "lan",
    "home",
    "internal",
    "localdomain",
];

pub fn extract(text: &str, container_prefixes: &[String]) -> Vec<Extracted> {
    let mut out: Vec<Extracted> = Vec::new();
    let mut push = |e: Extracted| {
        if !out.contains(&e) {
            out.push(e);
        }
    };

    for m in URL_RE.find_iter(text) {
        let url = m.as_str().trim_end_matches(['.', ',', ';', ':', '!', '?']);
        push(Extracted {
            kind: ValueKind::Url,
            anchor: String::new(),
            value: url.to_string(),
        });
        if let Some((host, port)) = url_host_port(url) {
            if let Ok(ip) = host.parse::<Ipv4Addr>() {
                push(Extracted {
                    kind: ValueKind::Ipv4,
                    anchor: String::new(),
                    value: ip.to_string(),
                });
            } else if host.contains('.')
                || host == "localhost"
                || has_prefix(&host, container_prefixes)
            {
                push(Extracted {
                    kind: ValueKind::Hostname,
                    anchor: String::new(),
                    value: host.to_ascii_lowercase(),
                });
            }
            if let Some(port) = port {
                push(Extracted {
                    kind: ValueKind::Port,
                    anchor: host.to_ascii_lowercase(),
                    value: port,
                });
            }
        }
    }

    for c in IPV4_RE.captures_iter(text) {
        let m = c.get(1).unwrap();
        let preceded_by_v = text[..m.start()]
            .chars()
            .last()
            .is_some_and(|ch| ch == 'v' || ch == 'V');
        if preceded_by_v {
            continue;
        }
        if let Ok(ip) = m.as_str().parse::<Ipv4Addr>() {
            push(Extracted {
                kind: ValueKind::Ipv4,
                anchor: anchor_before(text, m.start()),
                value: ip.to_string(),
            });
        }
    }

    for c in IPV6_RE.captures_iter(text) {
        let m = c.get(1).unwrap();
        let candidate = m.as_str();
        if candidate.matches(':').count() < 2 {
            continue;
        }
        if let Ok(ip) = candidate.parse::<Ipv6Addr>() {
            push(Extracted {
                kind: ValueKind::Ipv6,
                anchor: anchor_before(text, m.start()),
                value: ip.to_string(),
            });
        }
    }

    for c in PORT_WORD_RE.captures_iter(text) {
        let whole = c.get(0).unwrap();
        if let Some(port) = valid_port(&c[1]) {
            push(Extracted {
                kind: ValueKind::Port,
                anchor: anchor_before(text, whole.start()),
                value: port,
            });
        }
    }

    for c in HOST_PORT_RE.captures_iter(text) {
        let host = &c[1];
        let looks_like_host = host.contains('.')
            || host.eq_ignore_ascii_case("localhost")
            || has_prefix(host, container_prefixes);
        if !looks_like_host || text[..c.get(1).unwrap().start()].ends_with("//") {
            continue;
        }
        if let Some(port) = valid_port(&c[2]) {
            push(Extracted {
                kind: ValueKind::Port,
                anchor: host.to_ascii_lowercase(),
                value: port,
            });
        }
    }

    for c in PATH_RE.captures_iter(text) {
        let path = c[1].trim_end_matches(['.', ',', ';', ':', ')']);
        push(Extracted {
            kind: ValueKind::Path,
            anchor: String::new(),
            value: path.to_string(),
        });
    }

    for c in ENV_ASSIGN_RE.captures_iter(text) {
        let value = c[2].trim_matches(['"', '\'']).to_string();
        push(Extracted {
            kind: ValueKind::EnvVar,
            anchor: c[1].to_string(),
            value,
        });
    }

    for c in VERSION_WORD_RE.captures_iter(text) {
        let whole = c.get(0).unwrap();
        push(Extracted {
            kind: ValueKind::Version,
            anchor: anchor_before(text, whole.start()),
            value: c[1].to_string(),
        });
    }
    for c in NAME_VERSION_RE.captures_iter(text) {
        let name = c[1].to_ascii_lowercase();
        if name == "version" || name == "v" {
            continue;
        }
        push(Extracted {
            kind: ValueKind::Version,
            anchor: name,
            value: c[2].to_string(),
        });
    }

    for c in HOSTNAME_RE.captures_iter(text) {
        let tld = c[2].to_ascii_lowercase();
        if !HOST_TLDS.contains(&tld.as_str()) {
            continue;
        }
        let m = c.get(1).unwrap();
        if text[..m.start()].ends_with("//") || text[..m.start()].ends_with('@') {
            continue; // part of a URL or e-mail address, handled elsewhere / ignored
        }
        push(Extracted {
            kind: ValueKind::Hostname,
            anchor: String::new(),
            value: m.as_str().to_ascii_lowercase(),
        });
    }

    for c in CONTAINER_WORD_RE.captures_iter(text) {
        push(Extracted {
            kind: ValueKind::Container,
            anchor: String::new(),
            value: c[1].to_ascii_lowercase(),
        });
    }
    for token in text.split(|ch: char| {
        ch.is_whitespace() || matches!(ch, '`' | '\'' | '"' | '(' | ')' | ',' | ';' | ':')
    }) {
        let token = token.trim_matches(['.', '!', '?']);
        if token.len() > 3
            && has_prefix(token, container_prefixes)
            && IDENT_TOKEN_RE.is_match(token)
            && !token.contains('/')
        {
            push(Extracted {
                kind: ValueKind::Container,
                anchor: String::new(),
                value: token.to_ascii_lowercase(),
            });
        }
    }

    for c in NUMERIC_CFG_RE.captures_iter(text) {
        push(Extracted {
            kind: ValueKind::NumericCfg,
            anchor: c[1].to_string(),
            value: c[2].to_string(),
        });
    }

    out
}

/// Flag assistant claims that contradict an unambiguous user-established value.
pub fn detect_drift(claims: &[Extracted], registry: &[KnownValue]) -> Vec<Drift> {
    let mut drifts: Vec<Drift> = Vec::new();
    for claim in claims {
        let same_entity: Vec<&KnownValue> = registry
            .iter()
            .filter(|k| k.kind == claim.kind && k.anchor == claim.anchor)
            .collect();
        let [known] = same_entity.as_slice() else {
            continue;
        }; // must be exactly one
        if known.value == claim.value {
            continue;
        }
        let claimed_value_known_elsewhere = registry
            .iter()
            .any(|k| k.kind == claim.kind && k.value == claim.value);
        if claimed_value_known_elsewhere {
            continue;
        }
        let drift = Drift {
            kind: claim.kind,
            anchor: claim.anchor.clone(),
            known: known.value.clone(),
            claimed: claim.value.clone(),
        };
        if !drifts.contains(&drift) {
            drifts.push(drift);
        }
    }
    drifts
}

fn valid_port(digits: &str) -> Option<String> {
    let n: u32 = digits.parse().ok()?;
    (1..=65535).contains(&n).then(|| n.to_string())
}

fn has_prefix(token: &str, prefixes: &[String]) -> bool {
    prefixes.iter().any(|p| {
        !p.is_empty()
            && token
                .to_ascii_lowercase()
                .starts_with(&p.to_ascii_lowercase())
    })
}

fn url_host_port(url: &str) -> Option<(String, Option<String>)> {
    let rest = url.split_once("://")?.1;
    let authority = rest.split(['/', '?', '#']).next()?;
    let authority = authority.rsplit('@').next()?;
    if let Some(stripped) = authority.strip_prefix('[') {
        let (host, tail) = stripped.split_once(']')?;
        let port = tail.strip_prefix(':').and_then(valid_port);
        return Some((host.to_string(), port));
    }
    match authority.rsplit_once(':') {
        Some((host, port)) if port.chars().all(|c| c.is_ascii_digit()) && !port.is_empty() => {
            Some((host.to_string(), valid_port(port)))
        }
        _ => Some((authority.to_string(), None)),
    }
}

/// The nearest identifier-like token (something with a dot, dash, underscore,
/// slash or digit in it) within the six tokens before `pos`; empty if none.
fn anchor_before(text: &str, pos: usize) -> String {
    let before = &text[..pos];
    for token in before.split_whitespace().rev().take(6) {
        let token = token.trim_matches(|c: char| !c.is_alphanumeric() && c != '/');
        if token.len() < 3 || !IDENT_TOKEN_RE.is_match(token) {
            continue;
        }
        let is_identifier = token
            .chars()
            .any(|c| matches!(c, '.' | '-' | '_' | '/') || c.is_ascii_digit());
        let is_value = token
            .trim_start_matches(['v', 'V'])
            .chars()
            .all(|c| c.is_ascii_digit() || c == '.');
        if is_identifier && !is_value {
            return token.to_ascii_lowercase();
        }
    }
    String::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ex(text: &str) -> Vec<Extracted> {
        extract(text, &["ai-".to_string()])
    }

    fn has(values: &[Extracted], kind: ValueKind, anchor: &str, value: &str) -> bool {
        values
            .iter()
            .any(|e| e.kind == kind && e.anchor == anchor && e.value == value)
    }

    #[test]
    fn extracts_ports_with_anchors() {
        let v = ex("llama.cpp is running on port 8080 and the API is on port 4000.");
        assert!(has(&v, ValueKind::Port, "llama.cpp", "8080"));
        assert!(has(&v, ValueKind::Port, "", "4000"));
        assert!(has(
            &ex("start with --port=9292 please"),
            ValueKind::Port,
            "",
            "9292"
        ));
        assert!(ex("port 70000 is invalid")
            .iter()
            .all(|e| e.kind != ValueKind::Port));
    }

    #[test]
    fn extracts_urls_hosts_ips_and_paths() {
        let v = ex("Use http://llama-swap:8080/v1 or 192.0.2.10:9292 and see /etc/llama-swap/config.yaml; docs at docs.example.com.");
        assert!(has(&v, ValueKind::Url, "", "http://llama-swap:8080/v1"));
        assert!(has(&v, ValueKind::Port, "llama-swap", "8080"));
        assert!(has(&v, ValueKind::Ipv4, "", "192.0.2.10"));
        assert!(has(&v, ValueKind::Path, "", "/etc/llama-swap/config.yaml"));
        assert!(has(&v, ValueKind::Hostname, "", "docs.example.com"));
        assert!(!has(&v, ValueKind::Hostname, "", "config.yaml"));
        assert!(has(
            &ex("bind to [fe80::1]:53 or ::1"),
            ValueKind::Ipv6,
            "",
            "fe80::1"
        ));
        assert!(ex("meet at 12:30:45 tomorrow")
            .iter()
            .all(|e| e.kind != ValueKind::Ipv6));
    }

    #[test]
    fn versions_are_not_ip_addresses_and_need_a_marker() {
        let v = ex("litellm v1.94.1 and Open WebUI version 0.11.3; note 1.2.3.4 is an address, but v1.2.3.4 is not");
        assert!(has(&v, ValueKind::Version, "litellm", "1.94.1"));
        assert!(has(&v, ValueKind::Version, "", "0.11.3")); // "WebUI" is a plain word, not an identifier anchor
        assert!(v
            .iter()
            .any(|e| e.kind == ValueKind::Ipv4 && e.value == "1.2.3.4"));
        assert_eq!(v.iter().filter(|e| e.kind == ValueKind::Ipv4).count(), 1);
        assert!(ex("the number 3.14 alone")
            .iter()
            .all(|e| e.kind != ValueKind::Version));
    }

    #[test]
    fn env_vars_containers_and_numeric_settings() {
        let v = ex("Set CONTEXT_GUARD_RETENTION_DAYS=30 in container ai-litellm; max_input_tokens: 12288 and ai-openwebui too.");
        assert!(has(
            &v,
            ValueKind::EnvVar,
            "CONTEXT_GUARD_RETENTION_DAYS",
            "30"
        ));
        assert!(has(&v, ValueKind::Container, "", "ai-litellm"));
        assert!(has(&v, ValueKind::Container, "", "ai-openwebui"));
        assert!(has(&v, ValueKind::NumericCfg, "max_input_tokens", "12288"));
    }

    #[test]
    fn url_variants_and_port_forms() {
        let v = ex("see http://[fe80::1]:8443/x and https://user:pw@api.example.com:9443/v1 plus http://10.0.0.5/ then run -p 5432 and PORT=6379");
        assert!(
            has(&v, ValueKind::Ipv6, "", "fe80::1")
                || v.iter()
                    .any(|e| e.kind == ValueKind::Port && e.value == "8443")
        );
        assert!(has(&v, ValueKind::Hostname, "", "api.example.com"));
        assert!(has(&v, ValueKind::Port, "api.example.com", "9443"));
        assert!(has(&v, ValueKind::Ipv4, "", "10.0.0.5"));
        assert!(has(&v, ValueKind::Port, "", "5432"));
        assert!(has(&v, ValueKind::EnvVar, "PORT", "6379"));
        assert!(
            ex("mail me at someone@mail.example.com")
                .iter()
                .all(|e| e.kind != ValueKind::Hostname),
            "e-mail hosts are not hostnames"
        );
        assert!(has(
            &ex("the container proton-bridge restarted"),
            ValueKind::Container,
            "",
            "proton-bridge"
        ));
        assert!(has(
            &ex("Open WebUI v.0.11.3 is current"),
            ValueKind::Version,
            "",
            "0.11.3"
        ));
    }

    #[test]
    fn drift_dedupes_repeated_claims_and_ignores_unknown_kinds() {
        let registry = vec![known(ValueKind::Port, "", "4000")];
        let d = detect_drift(&ex("port 4100 ... again port 4100"), &registry);
        assert_eq!(d.len(), 1, "one drift per distinct claim");
        assert!(detect_drift(&ex("path /var/log/x.log"), &registry).is_empty());
        assert_eq!(ValueKind::parse("nope"), None);
        for k in [
            ValueKind::Ipv4,
            ValueKind::Ipv6,
            ValueKind::Port,
            ValueKind::Url,
            ValueKind::Path,
            ValueKind::EnvVar,
            ValueKind::Version,
            ValueKind::Hostname,
            ValueKind::Container,
            ValueKind::NumericCfg,
        ] {
            assert_eq!(ValueKind::parse(k.as_str()), Some(k));
        }
    }

    fn known(kind: ValueKind, anchor: &str, value: &str) -> KnownValue {
        KnownValue {
            kind,
            anchor: anchor.into(),
            value: value.into(),
        }
    }

    #[test]
    fn drift_fires_only_when_unambiguous() {
        let registry = vec![known(ValueKind::Port, "llama.cpp", "8080")];
        // Same value: nothing.
        assert!(detect_drift(&ex("your llama.cpp server on port 8080"), &registry).is_empty());
        // Different value for the same entity: drift.
        let d = detect_drift(&ex("your llama.cpp server on port 8000"), &registry);
        assert_eq!(d.len(), 1);
        assert_eq!(d[0].known, "8080");
        assert_eq!(d[0].claimed, "8000");
        // Different entity: nothing (no registry entry for that anchor).
        assert!(detect_drift(&ex("the ai-litellm service on port 4000"), &registry).is_empty());
        // Number without a port marker: nothing.
        assert!(detect_drift(&ex("llama.cpp has 8000 users"), &registry).is_empty());
    }

    #[test]
    fn drift_is_silent_when_ambiguous_or_value_known_elsewhere() {
        let two_ports = vec![
            known(ValueKind::Port, "", "4000"),
            known(ValueKind::Port, "", "4100"),
        ];
        assert!(detect_drift(&ex("The API on port 4200"), &two_ports).is_empty());

        let registry = vec![
            known(ValueKind::Port, "llama.cpp", "8080"),
            known(ValueKind::Port, "ai-litellm", "4000"),
        ];
        // 4000 was established (for another anchor): the assistant may be talking about that.
        assert!(detect_drift(&ex("llama.cpp listens on port 4000"), &registry).is_empty());
    }

    #[test]
    fn spec_example_without_anchor() {
        let registry = vec![known(ValueKind::Port, "", "4000")];
        let d = detect_drift(&ex("The API on port 4100 is ready"), &registry);
        assert_eq!(d.len(), 1);
        assert_eq!(d[0].claimed, "4100");
    }
}
