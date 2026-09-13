"""Planted facts: the ground truth every measurement compares against.

A `FactSheet` is generated from a seed, so a run is reproducible and two runs
with different seeds exercise different values. Kinds and anchors follow
`src/monitor/known_values.rs`: attribute kinds (port, ipv4, ipv6, version,
env_var, numeric_cfg) can drift and carry an anchor; entity kinds (path, url,
hostname, container) cannot drift and have an empty anchor.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

ATTRIBUTE_KINDS = ("port", "ipv4", "ipv6", "version", "env_var", "numeric_cfg")
ENTITY_KINDS = ("path", "url", "hostname", "container")
ALL_KINDS = ATTRIBUTE_KINDS + ENTITY_KINDS

# Things a fact can be about. Identifier-like names contain '.', '-', '_' or a
# digit, which is what the extractor's anchor rule requires; plain names do
# not, so the extractor records those facts with an empty anchor by design.
IDENTIFIER_ENTITIES = (
    "llama.cpp",
    "ai-litellm",
    "open-webui",
    "qwen3-general",
    "llama-swap",
    "ai-openwebui",
    "node_exporter",
    "context-guard",
)
PLAIN_ENTITIES = ("nginx", "redis", "postgres", "litellm", "grafana", "prometheus", "vllm", "caddy")
ENTITIES = IDENTIFIER_ENTITIES + PLAIN_ENTITIES

# Environment variable names that `is_secret_name` does not treat as secrets.
ENV_NAMES = (
    "LLAMA_PORT",
    "OLLAMA_HOST",
    "LOG_LEVEL",
    "MAX_WORKERS",
    "CONTEXT_GUARD_PORT",
    "DATABASE_URL",
    "MODEL_NAME",
    "FLUSH_INTERVAL",
    "GPU_LAYERS",
    "HTTP_PROXY",
)
NUMERIC_KEYS = (
    "max_tokens",
    "n_ctx",
    "flush_interval",
    "context_length",
    "num_workers",
    "batch_size",
    "retention_days",
    "window_turns",
    "gpu_layers",
    "timeout_seconds",
)
HOST_TLDS = ("lan", "home", "local", "internal", "com", "io", "dev")
PATH_DIRS = ("/etc", "/opt", "/var/lib", "/srv", "/home/ops", "/data")
PATH_FILES = ("config.yaml", "settings.toml", "model.gguf", "app.env", "server.json", "context-guard.db")
DEFAULT_CONTAINER_PREFIX = "ai-"


@dataclass(frozen=True)
class Fact:
    """One planted value. `anchor` is what the extractor should record for it."""

    kind: str
    entity: str
    value: str
    anchor: str
    # Byproducts the extractor legitimately registers from the same text
    # (a URL yields its host and port), as (kind, anchor, value) triples.
    derived: tuple[tuple[str, str, str], ...] = field(default=())

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "entity": self.entity,
            "value": self.value,
            "anchor": self.anchor,
            "derived": [list(d) for d in self.derived],
        }

    @staticmethod
    def from_dict(d: dict) -> "Fact":
        return Fact(
            kind=d["kind"],
            entity=d["entity"],
            value=d["value"],
            anchor=d["anchor"],
            derived=tuple(tuple(x) for x in d.get("derived", [])),
        )


def attribute_anchor(entity: str) -> str:
    """The anchor the extractor is expected to record for a fact about `entity`.

    Mirrors `anchor_before` in known_values.rs: an identifier-like token is kept,
    lowercased; a plain word yields no anchor.
    """
    if any(ch in entity for ch in "._-/") or any(ch.isdigit() for ch in entity):
        return entity.lower()
    return ""


class FactSheet:
    """A seeded set of facts, one or more per kind, with disjoint values."""

    def __init__(self, seed: int, per_kind: int = 2, container_prefix: str = DEFAULT_CONTAINER_PREFIX):
        self.seed = seed
        self.rng = random.Random(seed)
        self.container_prefix = container_prefix
        self._used: set[tuple[str, str]] = set()
        self.facts: list[Fact] = []
        entities = list(ENTITIES)
        self.rng.shuffle(entities)
        for kind in ALL_KINDS:
            for i in range(per_kind):
                entity = entities[(i + ALL_KINDS.index(kind) * per_kind) % len(entities)]
                self.facts.append(self._make(kind, entity))

    # ---- generation ----------------------------------------------------------------

    def _fresh(self, kind: str, gen) -> str:
        for _ in range(1000):
            v = gen()
            if (kind, v) not in self._used:
                self._used.add((kind, v))
                return v
        raise RuntimeError(f"could not generate a fresh {kind}")

    def _make(self, kind: str, entity: str) -> Fact:
        r = self.rng
        if kind == "port":
            return Fact(kind, entity, self._fresh(kind, lambda: str(r.randint(1024, 65000))), attribute_anchor(entity))
        if kind == "ipv4":
            return Fact(
                kind,
                entity,
                self._fresh(kind, lambda: f"10.{r.randint(0, 250)}.{r.randint(0, 250)}.{r.randint(2, 250)}"),
                attribute_anchor(entity),
            )
        if kind == "ipv6":
            # Canonical form (lowercase, no leading zeros, one `::`), which is
            # how Ipv6Addr's Display writes it back.
            return Fact(
                kind,
                entity,
                self._fresh(kind, lambda: f"fd{r.randint(16, 255):x}:{r.randint(4096, 65535):x}::{r.randint(16, 255):x}"),
                attribute_anchor(entity),
            )
        if kind == "version":
            return Fact(
                kind,
                entity,
                self._fresh(kind, lambda: f"{r.randint(0, 9)}.{r.randint(0, 99)}.{r.randint(0, 99)}"),
                attribute_anchor(entity),
            )
        if kind == "env_var":
            name = r.choice(ENV_NAMES)
            return Fact(kind, name, self._fresh(kind, lambda: str(r.randint(2, 9999))), name)
        if kind == "numeric_cfg":
            key = r.choice(NUMERIC_KEYS)
            return Fact(kind, key, self._fresh(kind, lambda: str(r.randint(1, 65536))), key)
        if kind == "path":
            return Fact(kind, entity, self._fresh(kind, lambda: f"{r.choice(PATH_DIRS)}/{entity}/{r.choice(PATH_FILES)}"), "")
        if kind == "url":
            host = f"{entity}.{r.choice(HOST_TLDS)}"
            port = str(r.randint(1024, 65000))
            url = self._fresh(kind, lambda: f"http://{host}:{port}/v1")
            return Fact(kind, entity, url, "", derived=(("hostname", "", host), ("port", host, port)))
        if kind == "hostname":
            return Fact(kind, entity, self._fresh(kind, lambda: f"{entity.replace('.', '-')}.{r.choice(HOST_TLDS)}"), "")
        if kind == "container":
            base = entity if entity.startswith(self.container_prefix) else f"{self.container_prefix}{entity.replace('.', '-')}"
            return Fact(kind, entity, self._fresh(kind, lambda: base), "")
        raise ValueError(kind)

    # ---- queries ------------------------------------------------------------------

    def by_kind(self, kind: str) -> list[Fact]:
        return [f for f in self.facts if f.kind == kind]

    def wrong_value(self, fact: Fact) -> str:
        """A value of the same kind that appears nowhere on the sheet, so a claim
        carrying it satisfies the drift rule's "never seen elsewhere" condition."""
        probe = FactSheet.__new__(FactSheet)
        probe.rng = random.Random(f"{self.seed}:{fact.kind}:{fact.value}")
        probe.container_prefix = self.container_prefix
        probe._used = set(self._used)
        probe._used.add((fact.kind, fact.value))
        return probe._make(fact.kind, fact.entity).value


def expected_triples(fact: Fact) -> set[tuple[str, str, str]]:
    """Registry triples that count as correct for `fact` (the fact plus its byproducts)."""
    return {(fact.kind, fact.anchor, fact.value), *fact.derived}
