//! Passive fact registry: structured values stated by the user (or returned by
//! tools) and the conservative rule for flagging an assistant that contradicts
//! them. No prose semantics; only recognizable technical values.
//!
//! Facts and claims are recognized differently. Everything a user or tool says
//! in any recognizable form establishes a fact (`extract`). An assistant reply
//! counts as a claim only in explicit marker forms (`extract_claims`): prose
//! such as `raise max_tokens to 8192` or `try --port 8081` is a suggestion far
//! more often than a statement, and false positives are worse than misses.

use std::collections::HashSet;
use std::net::{Ipv4Addr, Ipv6Addr};
use std::sync::LazyLock;

use regex::Regex;
use serde::Serialize;

use super::text::is_secret_name;

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

    /// Kinds whose value describes an attribute of some entity (the port *of*
    /// a host, the version *of* a package), as opposed to kinds whose value
    /// *is* the entity: a path, URL, hostname or container name identifies a
    /// thing, so a different one is a different thing, not a contradiction.
    /// Only attributes can drift; near-duplicate names are the
    /// suspicious-identifier signal's job.
    fn describes_attribute(self) -> bool {
        match self {
            ValueKind::Ipv4
            | ValueKind::Ipv6
            | ValueKind::Port
            | ValueKind::EnvVar
            | ValueKind::Version
            | ValueKind::NumericCfg => true,
            ValueKind::Url | ValueKind::Path | ValueKind::Hostname | ValueKind::Container => false,
        }
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

/// The words a conversation uses to name the things it is about, taken from
/// its identifiers: `nginx` from `/etc/nginx/nginx.conf`, `ai-nginx` or
/// `nginx.lan`; `api` from `/api/v1`. A plain word can anchor a value only
/// when it is in the lexicon, so `the API is on port 4000` anchors to `api`
/// in a chat that has mentioned `/api/v1`, and to nothing otherwise. The
/// monitor fills it from the conversation's identifier registry; the text
/// being extracted always contributes its own identifiers.
#[derive(Debug, Clone, Default)]
pub struct Lexicon {
    words: HashSet<String>,
}

impl Lexicon {
    /// Add the components of one identifier (`ai-llama-swap` gives `llama`
    /// and `swap`; `qwen3-30b-a3b` gives `qwen3`, `30b`, `a3b`).
    pub fn add_identifier(&mut self, value: &str) {
        for part in value.split(|c: char| !c.is_alphanumeric()) {
            let lower = part.to_ascii_lowercase();
            if part.len() >= 3 && !part.chars().all(|c| c.is_ascii_digit()) && !never_a_name(&lower)
            {
                self.words.insert(lower);
            }
        }
    }

    /// Add every identifier-like token in `text`, and the subject of every
    /// `X is …` / `X runs …` sentence: a user who writes "nginx is on port
    /// 8080" has named nginx, identifier or not.
    pub fn add_text(&mut self, text: &str) {
        for raw in text.split_whitespace() {
            let token = trim_token(raw);
            if IDENT_TOKEN_RE.is_match(token) && is_identifier_like(token) {
                self.add_identifier(token);
            }
        }
        // `The postgres database is …` names postgres: the head noun is often
        // generic, so the modifier before it counts as well.
        for c in SUBJECT_RE.captures_iter(text) {
            for g in [c.get(1), c.get(2)].into_iter().flatten() {
                let lower = g.as_str().to_ascii_lowercase();
                if !never_a_name(&lower) {
                    self.words.insert(lower);
                }
            }
        }
    }

    pub fn contains(&self, word_lower: &str) -> bool {
        self.words.contains(word_lower)
    }
}

/// Which forms count. See the module docs.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Scope {
    Facts,
    Claims,
}

static URL_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#"https?://[^\s<>"'`)\]]+"#).unwrap());
// A value may end a sentence: a trailing period followed by whitespace or the
// end of the text is not part of it.
static IPV4_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?:^|[^\w.])((?:\d{1,3}\.){3}\d{1,3})(?:[^\w.]|\.(?:\s|$)|$)").unwrap()
});
static IPV6_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r#"(?:^|[\s\[=("'`])([0-9A-Fa-f]{0,4}(?::[0-9A-Fa-f]{0,4}){2,7})(?:[\s\]/,;)"'`]|\.(?:\s|$)|$)"#,
    )
    .unwrap()
});
// `port 8080`, `port: 8080`, `"port": 8080`, `port is 8080`, `port `8080``,
// `--port 8080`, `-p 8080`.
static PORT_WORD_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r#"(?i)(?:\bport(?:\s+number)?["']?(?:\s+is)?\s*[:=#]?\s*["'`]?|--port[= ]["']?|-p\s+)(\d{1,5})\b"#,
    )
    .unwrap()
});
// `8080 is the port for X`.
static PORT_FIRST_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)\b(\d{2,5})\s+is\s+the\s+port\b").unwrap());
// `listen 8080;` (nginx), `listens on 8080`. Facts only.
static LISTEN_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)\blisten(?:s|ing)?(?:\s+on)?\s+([1-9]\d{1,4})\b").unwrap());
static PORTS_WORD_AFTER_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)^\s+ports?\b").unwrap());
// `host:8080` where the host is a dotted name, `localhost`, a container name or
// an IPv4 literal (`0.0.0.0:8080`, as `ss` and `docker ps` print it).
static HOST_PORT_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?:^|[\s(\[\x22'`=@])((?:\d{1,3}\.){3}\d{1,3}|[A-Za-z][A-Za-z0-9.-]*):(\d{2,5})(?:[^\w.]|\.(?:\s|$)|$)",
    )
    .unwrap()
});
// `8080/tcp` (not the container side of a `->8080/tcp` mapping) and a quoted
// compose mapping `"8080:80"` near the word `ports`, whose first number is the
// published port. The mapping form is facts only.
static TCP_PORT_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)(?:^|[^\w.:>-])(\d{2,5})/(?:tcp|udp)\b").unwrap());
static PORT_MAP_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#"["']([1-9]\d{1,4}):\d{2,5}(?:/(?:tcp|udp))?["']"#).unwrap());
static PATH_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(?:^|[\s"'`(=:,<>])(/[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)+)"#).unwrap()
});
// An unquoted value stops at whitespace, quotes, separators and the closing
// side of any bracket or backtick that may wrap the assignment.
const ENV_VALUE: &str = r#"("[^"\n]*"|'[^'\n]*'|[^\s"',;`()\[\]{}<>]+)"#;
static ENV_ASSIGN_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(&format!(r"\b([A-Z][A-Z0-9_]{{2,}})\s*=\s*{ENV_VALUE}")).unwrap());
// Facts only, and only for underscored names, so that `NOTE: three things`
// and `TODO: fix` are not environment variables: `NAME: value` (compose
// `environment:` maps, JSON), `NAME is set to value`, `NAME equals value`,
// `NAME is 42`.
static ENV_COLON_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(&format!(
        r#"\b([A-Z][A-Z0-9]*_[A-Z0-9_]*)["']?\s*:\s*{ENV_VALUE}"#
    ))
    .unwrap()
});
static ENV_PROSE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(&format!(
        r"\b([A-Z][A-Z0-9]*_[A-Z0-9_]*)\s+(?:is\s+set\s+to|equals)\s+{ENV_VALUE}"
    ))
    .unwrap()
});
static ENV_IS_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b([A-Z][A-Z0-9]*_[A-Z0-9_]*)\s+is\s+(\d[\w./:-]*)\b").unwrap());
// `const MAX_RETRIES: u32 = 5;`, `static URL_RE: LazyLock<Regex>`: a type
// annotation, not an environment variable.
static DECLARATION_LINE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:const|static|let|var|val|final|export\s+const|readonly)\b")
        .unwrap()
});
// `version 1.2.3`, `version: 1.2.3`, `"version": "1.2.3"`, `version = "1.2.3"`,
// `version is 1.2.3`, `v1.2.3`.
static VERSION_WORD_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r#"(?i)(?:\bversion["']?\s*(?:is\s+|[:=]\s*)?["']?|\bv\.?\s*)(\d+\.\d+(?:\.\d+)?(?:[-+][A-Za-z0-9.]+)?)\b"#,
    )
    .unwrap()
});
// `litellm 1.94.1`, `litellm v1.94.1`, `litellm==1.94.1`, `context-guard@0.2.0`,
// `litellm: 1.94.1` (three components: a two-part `key: 1.2` is a setting).
static NAME_VERSION_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"\b([A-Za-z][A-Za-z0-9._-]*[A-Za-z0-9])(?:\s+v?|==|@v?|:\s+v?)(\d+\.\d+\.\d+(?:[-+][A-Za-z0-9.]+)?)\b",
    )
    .unwrap()
});
static HOSTNAME_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b((?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+([a-z]{2,}))\b").unwrap()
});
static CONTAINER_WORD_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)\bcontainer\s+[`']?([a-z0-9][a-z0-9_.-]+)").unwrap());
// `max_tokens: 4096`, `"max_tokens": 4096`, `max-tokens = 4096`. The colon needs
// a following space, so `open-webui:2881` is a host and port.
static NUMERIC_CFG_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r#"["']?\b([a-z][a-z0-9]*(?:[_-][a-z0-9]+)+)["']?\s*(?:=\s*|:\s+)["']?(\d+(?:\.\d+)?)["']?(?:[^\w.]|\.(?:\s|$)|$)"#,
    )
    .unwrap()
});
// Facts only: `set max_tokens to 4096`, `max_tokens is 4096`, `the max_tokens
// setting is 4096`, `--max-tokens 4096`.
static NUMERIC_PROSE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+)(?:\s+setting)?\s+(?:is(?:\s+set)?\s+to|is|equals|to)\s+(\d+(?:\.\d+)?)\b",
    )
    .unwrap()
});
static NUMERIC_FLAG_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"--([a-z][a-z0-9]*(?:[_-][a-z0-9]+)+)[= ](\d+(?:\.\d+)?)\b").unwrap()
});
static IDENT_TOKEN_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^[A-Za-z][A-Za-z0-9._/-]*$").unwrap());
// A reply is not stating what is when the words governing the value hedge,
// condition, suggest, negate or propose a change: `you could move it to port
// 8081`, `if it were on 8081`, `by default it listens on 3000`, `not 8081`,
// `bump regex to 1.11`, `earlier it was on 8081`, `I will put it on 8081`. Only
// the eight tokens before the value in its own sentence count, so `is on port
// 8000, so the request should go through` is still a claim.
static HEDGE_BEFORE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)\b(?:could|would|might|try|trying|consider|instead|alternatively|suggest|suggests|suggested|recommend|recommended|if|unless|whether|move|moving|switch|switching|raise|raising|lower|lowering|bump|bumping|upgrade|upgrading|downgrade|change|changing|was|were|previously|earlier|will|plan|let's|let me|going to|used to|by default|for example|such as|defaults? to|e\.g\.)\b|\bnot\s+(?:on\s+|at\s+|to\s+)?(?:port\s+|version\s+)?[`\x22']?$",
    )
    .unwrap()
});
// Hedges that govern a value from after it: `X 1.95.0 fixes that; consider
// upgrading`, `port 8081 would double throughput`, `localhost:3000 if you use
// the dev server`. Kept to the few words that only ever hedge, so `is on port
// 8000, so the request should go through` stays a claim.
static HEDGE_AFTER_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)\b(?:consider|would|if|instead|by default)\b").unwrap());
// A hedge line that introduces a block (`Alternatively:`, `**What you could
// do:**`) governs the lines that follow it, until a blank line.
static HEDGE_ANYWHERE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)\b(?:could|would|might|try|consider|instead|alternatively|suggest|recommend|if|example|option|alternative|proposal|plan)\b",
    )
    .unwrap()
});
// The subject of a fact sentence (`nginx is on port 8080`, `grafana runs on
// 3000`) names a thing even when it is a plain word; see `Lexicon::add_text`.
static SUBJECT_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)\b(?:([a-z][a-z0-9]{2,})\s+)?([a-z][a-z0-9]{2,})(?:\s+(?:is|are|runs|running|listens|listening|sits|lives|serves|answers)\b|\s*→)",
    )
    .unwrap()
});

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

/// Source-file extensions: `README.md:355` in compiler or grep output is a
/// file and line, not a host and port.
const SOURCE_EXTENSIONS: &[&str] = &[
    "c", "cc", "cfg", "conf", "cpp", "cs", "css", "go", "h", "html", "ini", "java", "js", "json",
    "jsx", "kt", "lock", "log", "md", "php", "py", "rb", "rs", "sh", "sql", "toml", "ts", "tsx",
    "txt", "xml", "yaml", "yml",
];

/// Verbs that take `port N` as their object (`open port 8080`, `expose port
/// 80`): the word before the marker, never the thing it belongs to.
const PORT_VERBS: &[&str] = &[
    "allow",
    "allowed",
    "allows",
    "bind",
    "binds",
    "block",
    "blocked",
    "blocks",
    "bound",
    "close",
    "closed",
    "closes",
    "expose",
    "exposed",
    "exposes",
    "forward",
    "forwarded",
    "forwards",
    "listen",
    "listening",
    "listens",
    "map",
    "mapped",
    "maps",
    "open",
    "opened",
    "opens",
    "publish",
    "published",
    "publishes",
    "use",
    "used",
    "uses",
    "using",
];

/// Path roots and other segments that name no thing (`/var/log/app` is about
/// `app`, not `var` or `log`).
const PATH_ROOTS: &[&str] = &[
    "bin", "boot", "data", "dev", "etc", "home", "lib", "lib64", "local", "mnt", "opt", "proc",
    "root", "run", "sbin", "share", "srv", "sys", "tmp", "usr", "var", "log", "logs", "www",
];

/// Type names a `NAME: type` member declares (`MAX_RETRIES: number;`).
const TYPE_WORDS: &[&str] = &[
    "any", "bool", "boolean", "bytes", "char", "dict", "f32", "f64", "float", "i8", "i16", "i32",
    "i64", "i128", "int", "isize", "list", "never", "none", "null", "number", "object", "option",
    "str", "string", "u8", "u16", "u32", "u64", "u128", "unknown", "usize", "void",
];

/// Generic nouns that stand for a thing without naming it (`the database is
/// at`, `the other server`, `that one`).
const GENERIC_NOUNS: &[&str] = &[
    "backend",
    "box",
    "boxes",
    "dashboard",
    "database",
    "databases",
    "everything",
    "frontend",
    "gui",
    "here",
    "instance",
    "instances",
    "machine",
    "machines",
    "node",
    "nodes",
    "one",
    "ones",
    "server",
    "servers",
    "service",
    "services",
    "setup",
    "thing",
    "things",
    "ui",
    "webui",
];

/// Closed-class words that can never be an anchor, even if a chat's
/// identifiers happen to contain them.
const FUNCTION_WORDS: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "do", "does", "for", "from",
    "has", "have", "he", "her", "his", "how", "i", "if", "in", "into", "is", "it", "its", "may",
    "my", "no", "not", "now", "of", "on", "or", "our", "per", "she", "so", "than", "that", "the",
    "their", "then", "there", "they", "this", "to", "too", "up", "via", "was", "we", "were",
    "what", "when", "where", "which", "while", "who", "will", "with", "would", "yes", "you",
    "your",
];

/// The labels of the values themselves, which name a kind, not a thing.
const VALUE_LABELS: &[&str] = &[
    "address",
    "config",
    "container",
    "env",
    "environment",
    "file",
    "host",
    "hostname",
    "image",
    "ip",
    "ipaddress",
    "ips",
    "ipv4",
    "ipv6",
    "key",
    "localhost",
    "model",
    "number",
    "path",
    "port",
    "ports",
    "setting",
    "settings",
    "tcp",
    "udp",
    "url",
    "value",
    "var",
    "variable",
    "version",
    "versions",
];

/// Conjunctions and connectors at which the forward anchor search stops, so a
/// value is never attributed to the next clause's subject.
const CLAUSE_BREAKS: &[&str] = &[
    "and", "but", "or", "then", "versus", "vs", "whereas", "while",
];

/// Every value a user or tool message establishes.
pub fn extract(text: &str, container_prefixes: &[String], lexicon: &Lexicon) -> Vec<Extracted> {
    extract_with(text, container_prefixes, lexicon, Scope::Facts)
}

/// The values an assistant reply states in an explicit marker form. Prose
/// forms are left out: in a reply they are usually suggestions.
pub fn extract_claims(
    text: &str,
    container_prefixes: &[String],
    lexicon: &Lexicon,
) -> Vec<Extracted> {
    extract_with(text, container_prefixes, lexicon, Scope::Claims)
}

fn extract_with(
    text: &str,
    container_prefixes: &[String],
    lexicon: &Lexicon,
    scope: Scope,
) -> Vec<Extracted> {
    let facts = scope == Scope::Facts;
    // A fact sentence may name its own subject; a reply anchors plain words
    // only to things the conversation has already named.
    let mut words = lexicon.clone();
    if facts {
        words.add_text(text);
    }
    let anchor = |start: usize, end: usize| anchor_for(text, start, end, &words);

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
                if !machine_local_v4(ip) {
                    push(Extracted {
                        kind: ValueKind::Ipv4,
                        anchor: String::new(),
                        value: ip.to_string(),
                    });
                }
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
            if machine_local_v4(ip) {
                continue; // 0.0.0.0 and 127.0.0.1 name this machine, not a fact about anything
            }
            if ip.octets()[3] == 0 && is_prefix_length(text, m.end()) {
                continue; // `10.0.0.0/24` is a network, not a host
            }
            push(Extracted {
                kind: ValueKind::Ipv4,
                anchor: anchor(m.start(), m.end()),
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
            if ip.is_loopback() || ip.is_unspecified() {
                continue;
            }
            if ip.segments()[7] == 0 && is_prefix_length(text, m.end()) {
                continue; // `fd10:c222::/64` is a network
            }
            push(Extracted {
                kind: ValueKind::Ipv6,
                anchor: anchor(m.start(), m.end()),
                value: ip.to_string(),
            });
        }
    }

    for c in PORT_WORD_RE.captures_iter(text) {
        let whole = c.get(0).unwrap();
        let digits = c.get(1).unwrap();
        if !ends_value(text, digits.end()) {
            continue; // `port 10.0.0.5`: not a port
        }
        if let Some(port) = valid_port(digits.as_str()) {
            push(Extracted {
                kind: ValueKind::Port,
                anchor: anchor(whole.start(), digits.end()),
                value: port,
            });
        }
    }
    if facts {
        for c in LISTEN_RE.captures_iter(text) {
            let whole = c.get(0).unwrap();
            let digits = c.get(1).unwrap();
            if !ends_value(text, digits.end())
                || PORTS_WORD_AFTER_RE.is_match(&text[digits.end()..])
            {
                continue; // `listens on 10.0.0.5:8080`, `listening on 80 ports`
            }
            if let Some(port) = valid_port(digits.as_str()) {
                push(Extracted {
                    kind: ValueKind::Port,
                    anchor: anchor(whole.start(), digits.end()),
                    value: port,
                });
            }
        }
    }

    for c in HOST_PORT_RE.captures_iter(text) {
        let host = c.get(1).unwrap();
        let host_str = host.as_str();
        let numeric = host_str.parse::<Ipv4Addr>().is_ok();
        let local = numeric || host_str.eq_ignore_ascii_case("localhost");
        let looks_like_host =
            local || host_str.contains('.') || has_prefix(host_str, container_prefixes);
        if !looks_like_host || text[..host.start()].ends_with("//") || is_source_file(host_str) {
            continue;
        }
        if let Some(port) = valid_port(&c[2]) {
            // `localhost` and IPv4 literals name this machine, not a service:
            // the anchor is whatever the sentence is about.
            let port_end = c.get(2).unwrap().end();
            push(Extracted {
                kind: ValueKind::Port,
                anchor: if local {
                    anchor(host.start(), port_end)
                } else {
                    host_str.to_ascii_lowercase()
                },
                value: port,
            });
        }
    }

    for c in TCP_PORT_RE
        .captures_iter(text)
        .chain(PORT_FIRST_RE.captures_iter(text))
    {
        let digits = c.get(1).unwrap();
        if let Some(port) = valid_port(digits.as_str()) {
            push(Extracted {
                kind: ValueKind::Port,
                anchor: anchor(digits.start(), digits.end()),
                value: port,
            });
        }
    }
    if facts {
        for c in PORT_MAP_RE.captures_iter(text) {
            let digits = c.get(1).unwrap();
            if !near_word_port(text, digits.start()) {
                continue; // `"02:30"` is a time
            }
            if let Some(port) = valid_port(digits.as_str()) {
                push(Extracted {
                    kind: ValueKind::Port,
                    anchor: anchor(digits.start(), digits.end()),
                    value: port,
                });
            }
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

    let env_forms: &[&Regex] = if facts {
        &[&ENV_ASSIGN_RE, &ENV_COLON_RE, &ENV_PROSE_RE, &ENV_IS_RE]
    } else {
        &[&ENV_ASSIGN_RE]
    };
    for re in env_forms {
        for c in re.captures_iter(text) {
            if is_secret_name(&c[1]) {
                continue; // a key or password is not a fact to check claims against, and must not be quoted
            }
            let raw = c.get(2).unwrap();
            let value = raw.as_str().trim_matches(['"', '\'']).to_string();
            if value.is_empty() || value.starts_with('=') {
                continue; // `MAX == 5` is a comparison, not an assignment
            }
            if std::ptr::eq(*re, &*ENV_COLON_RE)
                && is_type_annotation(text, c.get(0).unwrap().start(), raw.end(), &value)
            {
                continue;
            }
            push(Extracted {
                kind: ValueKind::EnvVar,
                anchor: c[1].to_string(),
                value,
            });
        }
    }

    for c in VERSION_WORD_RE.captures_iter(text) {
        let whole = c.get(0).unwrap();
        let version = c.get(1).unwrap();
        if !ends_value(text, version.end()) {
            continue; // `v1.2.3.4` is not a version
        }
        push(Extracted {
            kind: ValueKind::Version,
            anchor: anchor(whole.start(), version.end()),
            value: version.as_str().to_string(),
        });
    }
    for c in NAME_VERSION_RE.captures_iter(text) {
        let whole = c.get(0).unwrap();
        let version = c.get(2).unwrap();
        if !ends_value(text, version.end()) {
            continue; // `at 10.213.99.112` is an address
        }
        let name = c[1].to_ascii_lowercase();
        if name == "version" || name == "v" {
            continue;
        }
        // `litellm 1.94.1` names its subject. `upgraded X to 1.2.3` does not:
        // as a fact the subject is looked up; as a claim it is a bare number.
        let anchor = if is_name(&name) {
            name
        } else if facts {
            anchor(whole.start(), version.end())
        } else {
            continue;
        };
        push(Extracted {
            kind: ValueKind::Version,
            anchor,
            value: version.as_str().to_string(),
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

    let numeric_forms: &[&Regex] = if facts {
        &[&NUMERIC_CFG_RE, &NUMERIC_FLAG_RE, &NUMERIC_PROSE_RE]
    } else {
        &[&NUMERIC_CFG_RE]
    };
    for re in numeric_forms {
        for c in re.captures_iter(text) {
            let number = c.get(2).unwrap();
            if !ends_value(text, number.end()) {
                continue;
            }
            push(Extracted {
                kind: ValueKind::NumericCfg,
                // `--max-tokens` on the command line is `max_tokens` in the file.
                anchor: c[1].replace('-', "_"),
                value: number.as_str().to_string(),
            });
        }
    }

    if !facts {
        out.retain(|e| !only_in_hedged_positions(text, &e.value));
    }
    let anchored: Vec<(ValueKind, String)> = out
        .iter()
        .filter(|e| !e.anchor.is_empty())
        .map(|e| (e.kind, e.value.clone()))
        .collect();
    out.retain(|e| !e.anchor.is_empty() || !anchored.contains(&(e.kind, e.value.clone())));
    out
}

/// Flag assistant claims that contradict an unambiguous user-established value.
pub fn detect_drift(claims: &[Extracted], registry: &[KnownValue]) -> Vec<Drift> {
    let mut drifts: Vec<Drift> = Vec::new();
    for claim in claims.iter().filter(|c| c.kind.describes_attribute()) {
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

/// True when the text after a numeric value does not continue it: a `.`
/// followed by a digit means the match was a fragment of an address.
fn ends_value(text: &str, end: usize) -> bool {
    let mut rest = text[end..].chars();
    match (rest.next(), rest.next()) {
        (Some('.'), Some(d)) => !d.is_ascii_digit(),
        _ => true,
    }
}

/// `/24` right after an address.
fn is_prefix_length(text: &str, end: usize) -> bool {
    let mut rest = text[end..].chars();
    rest.next() == Some('/') && rest.next().is_some_and(|c| c.is_ascii_digit())
}

fn machine_local_v4(ip: Ipv4Addr) -> bool {
    ip.is_loopback() || ip.is_unspecified()
}

fn is_source_file(host: &str) -> bool {
    host.rsplit_once('.')
        .is_some_and(|(_, ext)| SOURCE_EXTENSIONS.contains(&ext.to_ascii_lowercase().as_str()))
}

/// Whether the word `port` occurs shortly before `pos` (a compose `ports:` list).
fn near_word_port(text: &str, pos: usize) -> bool {
    let from = text[..pos]
        .char_indices()
        .rev()
        .nth(80)
        .map_or(0, |(i, _)| i);
    text[from..pos].to_ascii_lowercase().contains("port")
}

/// `const MAX_RETRIES: u32 = 5;`, `MAX_WORKERS: int = 4`: the line declares a
/// typed name, so the "value" is a type.
fn is_type_annotation(text: &str, start: usize, value_end: usize, value: &str) -> bool {
    let line_start = text[..start].rfind('\n').map_or(0, |i| i + 1);
    DECLARATION_LINE_RE.is_match(&text[line_start..])
        || text[value_end..].trim_start().starts_with('=')
        || TYPE_WORDS.contains(&value.to_ascii_lowercase().as_str())
}

/// True when every occurrence of `value` in `text` is hedged (see `hedged_at`).
/// A value the text does not contain verbatim is kept.
fn only_in_hedged_positions(text: &str, value: &str) -> bool {
    let mut seen = false;
    for (pos, _) in text.match_indices(value) {
        let end = pos + value.len();
        let before = text[..pos].chars().last();
        let mut rest = text[end..].chars();
        let (after, after_next) = (rest.next(), rest.next());
        let bounded = |c: Option<char>| !c.is_some_and(|c| c.is_alphanumeric() || c == '.');
        let version_prefix = before.is_some_and(|c| c == 'v' || c == 'V');
        // `8000.` ends a sentence; `10.0.0` continues an address.
        let sentence_end = after == Some('.') && after_next.is_none_or(char::is_whitespace);
        if !(bounded(before) || version_prefix) || !(bounded(after) || sentence_end) {
            continue; // `80` inside `8080`
        }
        seen = true;
        if !hedged_at(text, pos) {
            return false;
        }
    }
    seen
}

/// Whether the value at `pos` is governed by a hedge: it sits in a fenced code
/// block, in a question, after a hedge word within its clause, or under a
/// hedge line that ends with a colon.
fn hedged_at(text: &str, pos: usize) -> bool {
    if text[..pos].matches("```").count() % 2 == 1 {
        return true;
    }
    let (start, end) = sentence_bounds(text, pos);
    if text[start..end].trim_end().ends_with('?') {
        return true;
    }
    let before = &text[start..pos];
    let window_start = before
        .split_whitespace()
        .rev()
        .take(8)
        .last()
        .map_or(pos, |tok| tok.as_ptr() as usize - text.as_ptr() as usize);
    if HEDGE_BEFORE_RE.is_match(&text[window_start..pos]) {
        return true;
    }
    let after_tokens: Vec<&str> = text[pos..end].split_whitespace().skip(1).take(8).collect();
    if HEDGE_AFTER_RE.is_match(&after_tokens.join(" ")) {
        return true;
    }
    // A hedge heading governs its block: every line up to the next blank one,
    // and the heading itself may be set off by a blank line.
    let mut lines = text[..start]
        .lines()
        .rev()
        .map(str::trim)
        .skip_while(|l| l.is_empty());
    for l in lines.by_ref() {
        if l.is_empty() {
            break;
        }
        if is_hedge_heading(l) {
            return true;
        }
    }
    lines.find(|l| !l.is_empty()).is_some_and(is_hedge_heading)
}

fn is_hedge_heading(line: &str) -> bool {
    line.trim_end_matches(['*', '_', '`']).ends_with(':') && HEDGE_ANYWHERE_RE.is_match(line)
}

/// The sentence around `pos`: bounded by `!`, `?`, a newline, or a period
/// followed by whitespace (so `10.0.0.5` and `v1.2.3` do not split it). A
/// semicolon joins clauses that share one hedge (`…; consider upgrading`).
fn sentence_bounds(text: &str, pos: usize) -> (usize, usize) {
    let is_boundary = |i: usize, c: char| {
        matches!(c, '!' | '?' | '\n')
            || (c == '.' && text[i + 1..].chars().next().is_none_or(char::is_whitespace))
    };
    let start = text[..pos]
        .char_indices()
        .rev()
        .find(|&(i, c)| is_boundary(i, c))
        .map_or(0, |(i, c)| i + c.len_utf8());
    let end = text[pos..]
        .char_indices()
        .find(|&(i, c)| is_boundary(pos + i, c))
        .map_or(text.len(), |(i, c)| pos + i + c.len_utf8());
    (start, end)
}

/// Words that never name a thing: function words, value labels, the verbs
/// that take a port as object, and path roots.
fn never_a_name(word_lower: &str) -> bool {
    FUNCTION_WORDS.contains(&word_lower)
        || VALUE_LABELS.contains(&word_lower)
        || PORT_VERBS.contains(&word_lower)
        || PATH_ROOTS.contains(&word_lower)
        || GENERIC_NOUNS.contains(&word_lower)
}

/// The entity a value belongs to: the nearest usable token before it, else the
/// first one after it within the same clause, else none.
fn anchor_for(text: &str, start: usize, end: usize, words: &Lexicon) -> String {
    let before = anchor_before(text, start, words);
    if before.is_empty() {
        anchor_after(text, end, words)
    } else {
        before
    }
}

fn trim_token(raw: &str) -> &str {
    raw.trim_matches(|c: char| !c.is_alphanumeric() && c != '/')
}

fn is_identifier_like(token: &str) -> bool {
    token
        .chars()
        .any(|c| matches!(c, '.' | '-' | '_' | '/') || c.is_ascii_digit())
}

fn is_number_like(token: &str) -> bool {
    token
        .trim_start_matches(['v', 'V'])
        .chars()
        .all(|c| c.is_ascii_digit() || c == '.')
}

/// Whether a word immediately before a version (`litellm 1.94.1`) names its
/// subject: anything but a function word or a value label.
fn is_name(word_lower: &str) -> bool {
    word_lower.len() >= 2 && !is_number_like(word_lower) && !never_a_name(word_lower)
}

/// Whether `token` can name the thing a value belongs to. Identifier-like
/// tokens (a dot, dash, underscore, slash or digit in them: `llama.cpp`,
/// `ai-litellm`, `qwen3`) can, unless they are a value label (`ipv6`). A plain
/// word can only when `allow_plain` and the conversation's lexicon has it.
fn is_anchor_token(token: &str, allow_plain: bool, words: &Lexicon) -> bool {
    if token.len() < 2 || !IDENT_TOKEN_RE.is_match(token) || is_number_like(token) {
        return false;
    }
    let lower = token.to_ascii_lowercase();
    if never_a_name(&lower) {
        return false;
    }
    is_identifier_like(token) || (allow_plain && words.contains(&lower))
}

/// The nearest anchor token within the six tokens before `pos`. An
/// identifier-like token may sit in the previous sentence (`llama.cpp is up.
/// It listens on port 8080`); a plain word must be in the same one.
fn anchor_before(text: &str, pos: usize, words: &Lexicon) -> String {
    // A line break ends the sentence too: in a Markdown list each bullet has
    // its own subject.
    let (earlier, line) = text[..pos].rsplit_once('\n').unwrap_or(("", &text[..pos]));
    let tokens = line
        .split_whitespace()
        .rev()
        .map(|t| (t, true))
        .chain(earlier.split_whitespace().rev().map(|t| (t, false)));
    let mut same_sentence = true;
    for (raw, same_line) in tokens.take(6) {
        if !same_line || raw.ends_with(['.', '!', '?', ';']) {
            same_sentence = false; // this token closes the previous sentence
        }
        let token = trim_token(raw);
        if is_anchor_token(token, same_sentence, words) {
            return token.to_ascii_lowercase();
        }
    }
    String::new()
}

/// The first anchor token within the six tokens after `end`, stopping at the
/// end of the clause so `port 8080 and the UI on port 3000` does not attribute
/// 8080 to the UI.
fn anchor_after(text: &str, end: usize, words: &Lexicon) -> String {
    for raw in text[end..].split_whitespace().take(6) {
        let token = trim_token(raw);
        if CLAUSE_BREAKS.contains(&token.to_ascii_lowercase().as_str()) {
            break;
        }
        if is_anchor_token(token, true, words) {
            return token.to_ascii_lowercase();
        }
        if raw.ends_with(['.', '!', '?', ';', ',']) {
            break;
        }
    }
    String::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    const PREFIXES: &[&str] = &["ai-"];

    fn prefixes() -> Vec<String> {
        PREFIXES.iter().map(|p| p.to_string()).collect()
    }

    fn lexicon(ids: &[&str]) -> Lexicon {
        let mut l = Lexicon::default();
        for id in ids {
            l.add_identifier(id);
        }
        l
    }

    fn ex(text: &str) -> Vec<Extracted> {
        extract(text, &prefixes(), &Lexicon::default())
    }

    fn ex_with(text: &str, ids: &[&str]) -> Vec<Extracted> {
        extract(text, &prefixes(), &lexicon(ids))
    }

    fn claims(text: &str) -> Vec<Extracted> {
        extract_claims(text, &prefixes(), &Lexicon::default())
    }

    fn has(values: &[Extracted], kind: ValueKind, anchor: &str, value: &str) -> bool {
        values
            .iter()
            .any(|e| e.kind == kind && e.anchor == anchor && e.value == value)
    }

    #[test]
    fn env_values_stop_at_backticks_and_brackets() {
        let v = ex("run `CLAUDECODE=1` first, then (RUST_LOG=debug) and <FOO=bar>");
        assert!(has(&v, ValueKind::EnvVar, "CLAUDECODE", "1"));
        assert!(has(&v, ValueKind::EnvVar, "RUST_LOG", "debug"));
        assert!(has(&v, ValueKind::EnvVar, "FOO", "bar"));
        assert!(v
            .iter()
            .all(|e| e.kind != ValueKind::EnvVar || !e.value.ends_with(['`', ')', '>'])));
        // The same assignment quoted in a reply is therefore not drift.
        let registry: Vec<KnownValue> = v
            .iter()
            .map(|e| known(e.kind, &e.anchor, &e.value))
            .collect();
        let reply = claims("set `CLAUDECODE=1` in your shell");
        assert!(detect_drift(&reply, &registry).is_empty());
    }

    #[test]
    fn extracts_ports_with_anchors() {
        let v = ex("llama.cpp is running on port 8080 and the API is on port 4000.");
        assert!(has(&v, ValueKind::Port, "llama.cpp", "8080"));
        assert!(has(&v, ValueKind::Port, "api", "4000"));
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
        assert!(has(&v, ValueKind::Version, "", "0.11.3")); // "WebUI" is a plain word the chat has not used as an identifier
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
    fn secret_env_vars_are_never_learned() {
        let v =
            ex("export OPENAI_API_KEY=sk-live-abcdef0123456789 DB_PASSWORD='hunter2' PORT=6379");
        assert!(has(&v, ValueKind::EnvVar, "PORT", "6379"));
        assert!(
            v.iter()
                .all(|e| e.anchor != "OPENAI_API_KEY" && e.anchor != "DB_PASSWORD"),
            "{v:?}"
        );
        // So an assistant showing a placeholder for the key is not "drift".
        let registry: Vec<KnownValue> = v
            .iter()
            .map(|e| known(e.kind, &e.anchor, &e.value))
            .collect();
        assert!(detect_drift(&ex("set OPENAI_API_KEY=your-key-here"), &registry).is_empty());
    }

    #[test]
    fn a_different_path_url_host_or_container_is_not_drift() {
        let facts = ex("llama.cpp runs in container ai-llama-swap, config /etc/llama-swap/config.yaml, docs at https://docs.example.com/x on host docs.example.com");
        let registry: Vec<KnownValue> = facts
            .iter()
            .map(|e| known(e.kind, &e.anchor, &e.value))
            .collect();
        assert!(registry.iter().any(|k| k.kind == ValueKind::Path));
        assert!(registry.iter().any(|k| k.kind == ValueKind::Container));
        let claims = ex("Also check /var/log/syslog, restart ai-litellm, and see https://github.com/ggml-org/llama.cpp or wiki.example.org");
        assert!(
            detect_drift(&claims, &registry).is_empty(),
            "{:?}",
            detect_drift(&claims, &registry)
        );
        // Attributes still drift: the one known port for that anchor.
        let d = detect_drift(
            &ex("llama.cpp is on port 8000"),
            &[known(ValueKind::Port, "llama.cpp", "8080")],
        );
        assert_eq!(d.len(), 1);
    }

    #[test]
    fn spec_example_subject_anchor() {
        let registry: Vec<KnownValue> = ex("the API is on port 4000")
            .iter()
            .map(|e| known(e.kind, &e.anchor, &e.value))
            .collect();
        assert!(registry.contains(&known(ValueKind::Port, "api", "4000")));
        let mut lexicon = Lexicon::default();
        lexicon.add_text("the API is on port 4000"); // as the monitor does for the delta
        let d = detect_drift(
            &extract_claims("The API on port 4100 is ready", &prefixes(), &lexicon),
            &registry,
        );
        assert_eq!(d.len(), 1);
        assert_eq!(d[0].claimed, "4100");
    }

    fn registry_of(text: &str) -> Vec<KnownValue> {
        ex(text)
            .iter()
            .map(|e| known(e.kind, &e.anchor, &e.value))
            .collect()
    }

    #[test]
    fn sentence_final_punctuation_does_not_hide_values() {
        let v = ex("llama.cpp is at 10.213.99.112.");
        assert!(has(&v, ValueKind::Ipv4, "llama.cpp", "10.213.99.112"));
        assert!(v.iter().all(|e| e.kind != ValueKind::Version), "{v:?}");
        assert!(has(
            &ex("node_exporter is at fd10:c222::82."),
            ValueKind::Ipv6,
            "node_exporter",
            "fd10:c222::82"
        ));
        assert!(has(
            &ex("Use open-webui at localhost:2881."),
            ValueKind::Port,
            "open-webui",
            "2881"
        ));
        // Still not values: a fourth component, or a sentence continuing with a digit.
        assert!(ex("see 1.2.3.4")
            .iter()
            .all(|e| e.kind != ValueKind::Version));
        assert!(ex("version 1.2.3.4 is odd")
            .iter()
            .all(|e| e.kind != ValueKind::Version));
    }

    #[test]
    fn plain_words_anchor_only_through_the_lexicon() {
        // The subject of a fact sentence names a thing, identifier or not.
        let v = ex("grafana is on port 3000 and litellm is on port 4000.");
        assert!(
            has(&v, ValueKind::Port, "grafana", "3000")
                && has(&v, ValueKind::Port, "litellm", "4000"),
            "{v:?}"
        );
        // A plain word that is not a subject needs an identifier elsewhere in the conversation.
        let v = ex("the port for grafana is 3000 and for litellm 4000");
        assert!(
            v.iter()
                .filter(|e| e.kind == ValueKind::Port)
                .all(|e| e.anchor.is_empty()),
            "{v:?}"
        );
        let ids = ["ai-grafana", "/etc/litellm/config.yaml"];
        let v = ex_with("grafana is on port 3000 and litellm is on port 4000.", &ids);
        assert!(has(&v, ValueKind::Port, "grafana", "3000"), "{v:?}");
        assert!(has(&v, ValueKind::Port, "litellm", "4000"));
        let registry: Vec<KnownValue> = v
            .iter()
            .map(|e| known(e.kind, &e.anchor, &e.value))
            .collect();
        let reply = extract_claims(
            "grafana is listening on port 3001",
            &prefixes(),
            &lexicon(&ids),
        );
        let d = detect_drift(&reply, &registry);
        assert_eq!(d.len(), 1, "{d:?}");
        assert_eq!(
            (d[0].anchor.as_str(), d[0].known.as_str()),
            ("grafana", "3000")
        );
        // The text's own identifiers count too.
        assert!(has(
            &ex("container ai-nginx: nginx is on port 80"),
            ValueKind::Port,
            "nginx",
            "80"
        ));
        // Port verbs and path roots are never names, even when an identifier contains them.
        assert!(has(
            &ex("open port 3000 for the UI"),
            ValueKind::Port,
            "",
            "3000"
        ));
        assert!(has(
            &ex_with("Also open port 8080 on the firewall", &["open-webui"]),
            ValueKind::Port,
            "",
            "8080"
        ));
        assert!(has(
            &ex_with("the log shipper is on port 5044", &["/var/log/app"]),
            ValueKind::Port,
            "shipper",
            "5044"
        ));
        // The subject may follow the value, but not across a conjunction.
        assert!(has(
            &ex("port 2881 is where open-webui listens"),
            ValueKind::Port,
            "open-webui",
            "2881"
        ));
        assert!(has(
            &ex("port 8080 and the ai-webui on port 3000"),
            ValueKind::Port,
            "",
            "8080"
        ));
        // A plain word in the previous sentence is not the subject; an identifier is.
        assert!(has(
            &ex_with("Thanks for the nginx help. The port is 8080", &["ai-nginx"]),
            ValueKind::Port,
            "",
            "8080"
        ));
        assert!(has(
            &ex("llama.cpp is up. It listens on port 8080."),
            ValueKind::Port,
            "llama.cpp",
            "8080"
        ));
        // Labels and verbs never anchor.
        assert!(has(
            &ex("node_exporter ipv6: fd10:c222::82"),
            ValueKind::Ipv6,
            "node_exporter",
            "fd10:c222::82"
        ));
        assert!(has(
            &ex("start with --port=9292 please"),
            ValueKind::Port,
            "",
            "9292"
        ));
        assert!(has(
            &ex_with("nginx is hosted at 10.0.0.9", &["ai-nginx"]),
            ValueKind::Ipv4,
            "nginx",
            "10.0.0.9"
        ));
    }

    #[test]
    fn lexicon_splits_identifiers_into_words() {
        let l = lexicon(&[
            "/etc/nginx/nginx.conf",
            "ai-litellm",
            "qwen3-30b-a3b",
            "/api/v1",
        ]);
        for w in ["nginx", "conf", "litellm", "qwen3", "30b", "a3b", "api"] {
            assert!(l.contains(w), "{w}");
        }
        assert!(!l.contains("v1"), "two characters are too short");
        assert!(!l.contains("ai"));
        assert!(!l.contains("etc"), "path roots name nothing");
    }

    #[test]
    fn version_anchor_skips_function_words_and_accepts_pin_forms() {
        assert!(has(
            &ex("The qwen3-general version is 3.75.13."),
            ValueKind::Version,
            "qwen3-general",
            "3.75.13"
        ));
        assert!(has(
            &ex("upgraded qwen3-general to 3.75.13"),
            ValueKind::Version,
            "qwen3-general",
            "3.75.13"
        ));
        assert!(has(
            &ex("nginx 5.3.2 is installed"),
            ValueKind::Version,
            "nginx",
            "5.3.2"
        ));
        assert!(has(
            &ex("litellm==1.94.1"),
            ValueKind::Version,
            "litellm",
            "1.94.1"
        ));
        assert!(has(
            &ex("context-guard@0.2.0"),
            ValueKind::Version,
            "context-guard",
            "0.2.0"
        ));
        // A name on the previous line is out of reach: lines are sentences too.
        assert!(has(
            &ex("Name: litellm\nVersion: 1.94.1\n"),
            ValueKind::Version,
            "",
            "1.94.1"
        ));
        assert!(has(
            &ex_with("Name: litellm\nVersion: 1.94.1\n", &["ai-litellm"]),
            ValueKind::Version,
            "",
            "1.94.1"
        ));
        let d = detect_drift(
            &claims("You're running qwen3-general v6.95.91."),
            &registry_of("The qwen3-general version is 3.75.13."),
        );
        assert_eq!(d.len(), 1, "{d:?}");
        assert!(ex("the number 3.14 alone")
            .iter()
            .all(|e| e.kind != ValueKind::Version));
    }

    #[test]
    fn quoted_and_colon_forms_from_tool_output() {
        let v = ex(r#"{"open-webui": {"host": "0.0.0.0", "port": 2881}}"#);
        assert!(has(&v, ValueKind::Port, "open-webui", "2881"), "{v:?}");
        assert!(
            v.iter().all(|e| e.kind != ValueKind::Ipv4),
            "0.0.0.0 is not a fact"
        );
        assert!(has(
            &ex(r#"version = "0.2.0""#),
            ValueKind::Version,
            "",
            "0.2.0"
        ));
        assert!(has(
            &ex(r#"{"name": "context-guard", "version": "0.2.0"}"#),
            ValueKind::Version,
            "context-guard",
            "0.2.0"
        ));
        assert!(has(
            &ex(r#"{"max_tokens": 4096}"#),
            ValueKind::NumericCfg,
            "max_tokens",
            "4096"
        ));
        assert!(has(
            &ex("environment:\n  LOG_LEVEL: \"debug\"\n  OPENWEBUI_ATTACH_RESULTS: \"true\"\n"),
            ValueKind::EnvVar,
            "LOG_LEVEL",
            "debug"
        ));
        assert!(has(
            &ex("`open-webui` runs on port `2881`."),
            ValueKind::Port,
            "open-webui",
            "2881"
        ));
        assert!(has(
            &ex("The open-webui port is 2881."),
            ValueKind::Port,
            "open-webui",
            "2881"
        ));
    }

    #[test]
    fn numeric_hosts_listen_directives_and_docker_forms() {
        assert!(has(
            &ex("LISTEN 0 4096 0.0.0.0:2881 0.0.0.0:* users:((\"open-webui\",pid=4242,fd=3))"),
            ValueKind::Port,
            "",
            "2881"
        ));
        assert!(has(
            &ex("BRIDGE_IMAP_ADDR: 127.0.0.1:1143"),
            ValueKind::Port,
            "bridge_imap_addr",
            "1143"
        ));
        let compose = ex("services:\n  open-webui:\n    ports:\n      - \"2881:8080\"\n");
        assert!(
            has(&compose, ValueKind::Port, "open-webui", "2881"),
            "{compose:?}"
        );
        assert!(
            !compose
                .iter()
                .any(|e| e.kind == ValueKind::Port && e.value == "8080"),
            "container side is not published"
        );
        let ps = ex("0.0.0.0:2881->8080/tcp   open-webui");
        assert!(ps
            .iter()
            .any(|e| e.kind == ValueKind::Port && e.value == "2881"));
        assert!(!ps
            .iter()
            .any(|e| e.kind == ValueKind::Port && e.value == "8080"));
        assert!(has(
            &ex("server { listen 9101; }"),
            ValueKind::Port,
            "",
            "9101"
        ));
        assert!(has(
            &ex("nginx listens on 8080"),
            ValueKind::Port,
            "nginx",
            "8080"
        ));
        assert!(
            claims("nginx listens on 8081")
                .iter()
                .all(|e| e.kind != ValueKind::Port),
            "listen is a fact form only"
        );
        assert!(has(
            &ex("open-webui is on 2881/tcp"),
            ValueKind::Port,
            "open-webui",
            "2881"
        ));
        assert!(has(
            &ex("docker run -p 5432:5432 postgres"),
            ValueKind::Port,
            "",
            "5432"
        ));
        // Loopback and unspecified addresses are not facts; a real address next to them is.
        let v = ex("bind to 127.0.0.1 and 0.0.0.0; llama.cpp is 10.0.0.5");
        assert_eq!(v.iter().filter(|e| e.kind == ValueKind::Ipv4).count(), 1);
        assert!(has(&v, ValueKind::Ipv4, "llama.cpp", "10.0.0.5"));
        // `listens on 10.0.0.5:8080`: 10 is not a port.
        assert!(!ex("nginx listens on 10.0.0.5:8080")
            .iter()
            .any(|e| e.kind == ValueKind::Port && e.value == "10"));
    }

    #[test]
    fn env_var_prose_and_yaml_forms_need_an_underscored_name() {
        assert!(has(
            &ex("LLAMA_PORT is set to 42"),
            ValueKind::EnvVar,
            "LLAMA_PORT",
            "42"
        ));
        assert!(has(
            &ex("MAX_WORKERS = 8"),
            ValueKind::EnvVar,
            "MAX_WORKERS",
            "8"
        ));
        assert!(has(
            &ex("LOG_LEVEL: debug"),
            ValueKind::EnvVar,
            "LOG_LEVEL",
            "debug"
        ));
        assert!(has(
            &ex("the env var GPU_LAYERS equals 40"),
            ValueKind::EnvVar,
            "GPU_LAYERS",
            "40"
        ));
        assert!(has(
            &ex("MAX_WORKERS is 8"),
            ValueKind::EnvVar,
            "MAX_WORKERS",
            "8"
        ));
        // Still ignored: prose labels, comparisons, non-numeric `is`, secrets.
        for text in [
            "NOTE: 3 things to do",
            "TODO: fix the parser",
            "if MAX_WORKERS == 5 then",
            "MAX_WORKERS is too high",
            "API_KEY: abc123",
            "DB_PASSWORD is set to hunter2",
            "pub const MAX_RETRIES: u32 = 5;",
            "static URL_RE: LazyLock<Regex> = LazyLock::new(|| ..);",
            "    MAX_WORKERS: int = 4",
            "interface Config { MAX_RETRIES: number; }",
            "class C(TypedDict):
    MAX_WORKERS: int
",
        ] {
            assert!(
                ex(text).iter().all(|e| e.kind != ValueKind::EnvVar),
                "{text}"
            );
        }
        // In a reply these forms are suggestions, not claims.
        for text in [
            "LLAMA_PORT is set to 42",
            "LOG_LEVEL: debug",
            "if MAX_WORKERS is 16 or higher",
        ] {
            assert!(
                claims(text).iter().all(|e| e.kind != ValueKind::EnvVar),
                "{text}"
            );
        }
        assert!(has(
            &claims("LLAMA_PORT=42 is set"),
            ValueKind::EnvVar,
            "LLAMA_PORT",
            "42"
        ));
    }

    #[test]
    fn command_line_flags_are_settings() {
        assert!(has(
            &ex("--max_tokens 4096"),
            ValueKind::NumericCfg,
            "max_tokens",
            "4096"
        ));
        assert!(has(
            &ex("run --context-length 8192 now"),
            ValueKind::NumericCfg,
            "context_length",
            "8192"
        ));
        let d = detect_drift(
            &ex("context_length: 4096"),
            &registry_of("start with --context-length 8192"),
        );
        assert_eq!(d.len(), 1);
        assert!(ex("--port 8080")
            .iter()
            .all(|e| e.kind != ValueKind::NumericCfg));
        assert!(claims("try --context-length 4096")
            .iter()
            .all(|e| e.kind != ValueKind::NumericCfg));
        assert!(claims("set context_length to 4096")
            .iter()
            .all(|e| e.kind != ValueKind::NumericCfg));
    }

    #[test]
    fn value_first_prose_and_colon_version_forms() {
        assert!(has(
            &ex("2881 is the port for open-webui."),
            ValueKind::Port,
            "open-webui",
            "2881"
        ));
        assert!(has(
            &ex("set context_length to 55328"),
            ValueKind::NumericCfg,
            "context_length",
            "55328"
        ));
        assert!(has(
            &ex("the context_length setting is 55328"),
            ValueKind::NumericCfg,
            "context_length",
            "55328"
        ));
        assert!(has(
            &ex("context_length is 55328"),
            ValueKind::NumericCfg,
            "context_length",
            "55328"
        ));
        assert!(has(
            &ex("vllm: 0.89.57"),
            ValueKind::Version,
            "vllm",
            "0.89.57"
        ));
        assert!(has(
            &ex("image_tag: 1.2"),
            ValueKind::NumericCfg,
            "image_tag",
            "1.2"
        ));
        assert!(has(
            &ex("Set max_tokens: 4096. Then restart."),
            ValueKind::NumericCfg,
            "max_tokens",
            "4096"
        ));
        assert!(has(
            &ex("\"fd7e:ab81::d3\" is where llama.cpp is hosted."),
            ValueKind::Ipv6,
            "llama.cpp",
            "fd7e:ab81::d3"
        ));
        assert!(has(
            &ex_with("postgres is hosted at 10.228.213.101.", &["ai-postgres"]),
            ValueKind::Ipv4,
            "postgres",
            "10.228.213.101"
        ));
        assert!(has(
            &ex("open-webui is active on port 2881."),
            ValueKind::Port,
            "open-webui",
            "2881"
        ));
        // A host:port with a plain name is neither a setting nor (without a prefix) a port fact.
        assert!(ex("open-webui:2881 and postgres:59568")
            .iter()
            .all(|e| e.kind != ValueKind::NumericCfg));
        // Bare numbers next to a key are still nothing.
        assert!(ex("13400 timeout_seconds")
            .iter()
            .all(|e| e.kind != ValueKind::NumericCfg));
    }

    #[test]
    fn suggestions_in_replies_are_not_claims() {
        // Each pair: a fact the user or a tool established, then a benign reply.
        let benign = [
            (
                "set max_tokens to 4096",
                "you could raise max_tokens to 8192",
            ),
            (
                "start with --context-length 8192",
                "Try passing --context-length 16384 instead.",
            ),
            (
                "environment:\n  LOG_LEVEL: info\n",
                "set LOG_LEVEL: debug temporarily",
            ),
            (
                "MAX_WORKERS is 8",
                "If MAX_WORKERS is 16 or higher, the pool doubles.",
            ),
            (
                "[[package]]\nname = \"regex\"\nversion = \"1.10.6\"\n",
                "bump regex to 1.11.1 when you can",
            ),
            (
                "nginx listens on 8080",
                "nginx listens on 8081 in the example",
            ),
        ];
        for (fact, reply) in benign {
            let d = detect_drift(&claims(reply), &registry_of(fact));
            assert!(d.is_empty(), "{fact:?} then {reply:?}: {d:?}");
        }
        // The same facts still catch a marker-form contradiction.
        assert_eq!(
            detect_drift(
                &claims("max_tokens: 8192"),
                &registry_of("set max_tokens to 4096")
            )
            .len(),
            1
        );
        assert_eq!(
            detect_drift(
                &claims("regex version 1.11.1 is what you have"),
                &registry_of("[[package]]\nname = \"regex\"\nversion = \"1.10.6\"\n")
            )
            .len(),
            1
        );
    }

    #[test]
    fn times_counts_and_file_lines_are_not_ports() {
        for text in [
            r#"{"at": "02:30"}"#,
            "the server is listening on 3 ports",
            "it is listening on 80 ports",
            "README.md:355: warning: unused import",
            "src/monitor/known_values.rs:1193:9",
        ] {
            assert!(
                ex(text).iter().all(|e| e.kind != ValueKind::Port),
                "{text}: {:?}",
                ex(text)
            );
        }
        assert!(has(
            &ex("ports:\n  - \"80:80\"\n"),
            ValueKind::Port,
            "",
            "80"
        ));
    }

    #[test]
    fn hedges_governing_the_value_suppress_the_claim() {
        let registry = registry_of("llama.cpp is running on port 8080, litellm v1.94.1, LOG_LEVEL=info, max_tokens: 4096. ai-box is at 10.0.0.5");
        for reply in [
            "You could move llama.cpp to port 8081 if that one is busy.",
            "By default llama.cpp listens on port 8000, but your compose overrides that.",
            "If llama.cpp were on port 8000, the proxy would need updating.",
            "For example, port 8000 would also work for llama.cpp.",
            "Try `--port 8000` for llama.cpp temporarily.",
            "Keep llama.cpp where it is; do not switch it to port 8000.",
            "Newer releases such as litellm v1.95.0 changed the default.",
            "You could set LOG_LEVEL=debug to test.",
            "Try pinging 10.0.0.9 to see whether ai-box answers there.",
            "Is llama.cpp on port 8000 in staging too?",
            "Earlier llama.cpp was on port 8000.",
            "I will put llama.cpp on port 8000 tomorrow.",
            "You could use this instead:\n```yaml\nllama.cpp:\n  port: 8000\n```",
            "Alternatively:\n\nllama.cpp port: 8000",
            "llama.cpp is not on port 8000.",
        ] {
            let d = detect_drift(&claims(reply), &registry);
            assert!(d.is_empty(), "{reply:?}: {d:?}");
        }
        // Statements of what is still drift, even with a modal or negation elsewhere.
        for reply in [
            "Your llama.cpp server on port 8000 looks healthy.",
            "Restarting your llama.cpp server on port 8000 now.",
            "llama.cpp is on port 8000, so the request should go through.",
            "llama.cpp listens on port 8000; you may need to restart it.",
            "llama.cpp is on port 8000 and it can't be reached.",
            "litellm 1.95.0 is what you have installed.",
            "LOG_LEVEL=debug is set.",
            "max_tokens: 8192",
            "ai-box is at 10.0.0.9.",
        ] {
            let d = detect_drift(&claims(reply), &registry);
            assert_eq!(d.len(), 1, "{reply:?}: {d:?}");
        }
        // `not port 8081` is negated; the 8000 beside it is not.
        let d = detect_drift(
            &claims("Actually llama.cpp is on port 8000, not port 8081."),
            &registry,
        );
        assert_eq!(
            d.iter().map(|x| x.claimed.as_str()).collect::<Vec<_>>(),
            ["8000"],
            "{d:?}"
        );
        // The hedge is per clause: a hedged sentence next to a plain one.
        let d = detect_drift(
            &claims("You could try port 9000. Your llama.cpp server is on port 8000 today."),
            &registry,
        );
        assert_eq!(d.len(), 1, "{d:?}");
        assert_eq!(d[0].claimed, "8000");
        // A value is matched whole: `80` is not inside `8080`.
        let d = detect_drift(
            &claims("You could use port 80. llama.cpp is on port 8000"),
            &registry,
        );
        assert_eq!(d.len(), 1, "{d:?}");
    }

    #[test]
    fn empty_anchors_do_not_collide_across_subjects() {
        // Round-one regression from the review: the user names nginx; the reply's
        // 3000 belongs to something else and must not be compared with 8080.
        let registry = registry_of("nginx is on port 8080.");
        assert!(
            registry.contains(&known(ValueKind::Port, "nginx", "8080")),
            "{registry:?}"
        );
        let mut lexicon = Lexicon::default();
        lexicon.add_text("nginx is on port 8080.");
        let reply = extract_claims(
            "nginx forwards requests to the app on port 3000.",
            &prefixes(),
            &lexicon,
        );
        assert!(detect_drift(&reply, &registry).is_empty(), "{reply:?}");
        let reply = extract_claims("nginx is listening on port 3000.", &prefixes(), &lexicon);
        assert_eq!(detect_drift(&reply, &registry).len(), 1, "{reply:?}");
    }

    #[test]
    fn markdown_lists_generic_subjects_and_headings() {
        // Each bullet has its own subject; the fact about nginx does not leak into grafana's line.
        let registry = registry_of("nginx is on port 8080.");
        let mut lexicon = Lexicon::default();
        lexicon.add_text("nginx is on port 8080.");
        let reply = extract_claims(
            "Current ports:\n- nginx: unchanged\n- grafana port: 3000",
            &prefixes(),
            &lexicon,
        );
        assert!(detect_drift(&reply, &registry).is_empty(), "{reply:?}");
        // A hedge heading governs the whole block, not only its first line.
        let reply = extract_claims(
            "Alternatively:\n- nginx port: 8081\n- grafana port: 3000",
            &prefixes(),
            &lexicon,
        );
        assert!(reply.iter().all(|e| e.kind != ValueKind::Port), "{reply:?}");
        let reply = extract_claims(
            "**What you could do:**\n- nginx port: 8081",
            &prefixes(),
            &lexicon,
        );
        assert!(reply.iter().all(|e| e.kind != ValueKind::Port), "{reply:?}");
        // Generic nouns never name a thing; the modifier before them does.
        let v = ex("The postgres database is at 10.0.0.5");
        assert!(has(&v, ValueKind::Ipv4, "postgres", "10.0.0.5"), "{v:?}");
        let registry = registry_of("The postgres database is at 10.0.0.5");
        let mut lexicon = Lexicon::default();
        lexicon.add_text("The postgres database is at 10.0.0.5");
        assert!(detect_drift(
            &extract_claims("The redis database is at 10.0.0.9", &prefixes(), &lexicon),
            &registry
        )
        .is_empty());
        assert_eq!(
            detect_drift(
                &extract_claims("postgres is at 10.0.0.9", &prefixes(), &lexicon),
                &registry
            )
            .len(),
            1
        );
        let registry = registry_of("The nginx server is on port 8080");
        let mut lexicon = Lexicon::default();
        lexicon.add_text("The nginx server is on port 8080");
        assert!(detect_drift(
            &extract_claims("The upstream server is on port 3000", &prefixes(), &lexicon),
            &registry
        )
        .is_empty());
        assert!(has(
            &ex("Here is my setup: nginx port: 8080"),
            ValueKind::Port,
            "",
            "8080"
        ));
        assert!(has(
            &ex("nginx → port 8080"),
            ValueKind::Port,
            "nginx",
            "8080"
        ));
        // `defaults to` and hedges after the value.
        let registry =
            registry_of("open-webui is on port 2881; litellm v1.94.1; retention_days: 30");
        for reply in [
            "open-webui defaults to port 8080; your override is fine.",
            "litellm 1.95.0 fixes that bug; consider upgrading.",
            "retention_days: 14 would double the churn.",
            "open-webui on port 8080 if you use the dev server.",
        ] {
            let d = detect_drift(&claims(reply), &registry);
            assert!(d.is_empty(), "{reply:?}: {d:?}");
        }
        assert_eq!(
            detect_drift(
                &claims("open-webui is on port 8080, so the request should go through."),
                &registry
            )
            .len(),
            1
        );
    }

    #[test]
    fn network_prefixes_are_not_addresses() {
        let v = ex("ai-box is 10.0.0.5 on 10.0.0.0/24, ipv6 fd10:c222::/64 and fd10:c222::82");
        assert_eq!(
            v.iter().filter(|e| e.kind == ValueKind::Ipv4).count(),
            1,
            "{v:?}"
        );
        assert!(has(&v, ValueKind::Ipv4, "ai-box", "10.0.0.5"));
        assert_eq!(
            v.iter().filter(|e| e.kind == ValueKind::Ipv6).count(),
            1,
            "{v:?}"
        );
        // An interface address in CIDR notation is still a host.
        assert!(has(
            &ex("inet 10.213.99.112/24 brd 10.213.99.255 scope global eth0"),
            ValueKind::Ipv4,
            "eth0",
            "10.213.99.112"
        ));
    }
}
