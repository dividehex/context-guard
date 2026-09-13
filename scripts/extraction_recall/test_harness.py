"""Unit tests for the harness's own logic. The extractor itself is tested in Rust;
the one end-to-end test here runs only when the example binary has been built."""

from __future__ import annotations

import pytest

from .buckets import benign_bucket, claim_bucket, registry_bucket, spurious_entries
from .corpus import Corpus, render
from .facts import ALL_KINDS, ATTRIBUTE_KINDS, Fact, FactSheet, attribute_anchor
from .paraphrase import keep, parse_rewrites
from .runner import EXTRACT_BINARY, run_cases
from .survey import Source, loose_candidates, missed_candidates
from .templates import BENIGN, CLAIMS, MULTI, STATEMENTS, TOOL


def entry(kind, anchor, value):
    return {"kind": kind, "anchor": anchor, "value": value}


# ---- facts ----------------------------------------------------------------------------


def test_sheet_is_deterministic_per_seed_and_covers_every_kind():
    a, b, c = FactSheet(7), FactSheet(7), FactSheet(8)
    assert [f.value for f in a.facts] == [f.value for f in b.facts]
    assert [f.value for f in a.facts] != [f.value for f in c.facts]
    assert {f.kind for f in a.facts} == set(ALL_KINDS)
    values = [(f.kind, f.value) for f in a.facts]
    assert len(values) == len(set(values))


def test_wrong_value_differs_and_is_stable():
    sheet = FactSheet(3)
    for fact in sheet.facts:
        if fact.kind in ATTRIBUTE_KINDS:
            wrong = sheet.wrong_value(fact)
            assert wrong != fact.value
            assert (fact.kind, wrong) not in {(f.kind, f.value) for f in sheet.facts}
            assert wrong == sheet.wrong_value(fact)


def test_expected_anchor_follows_identifier_rule():
    assert attribute_anchor("llama.cpp") == "llama.cpp"
    assert attribute_anchor("Open-WebUI") == "open-webui"
    assert attribute_anchor("qwen3") == "qwen3"
    assert attribute_anchor("nginx") == ""


# ---- templates ------------------------------------------------------------------------


def test_every_template_renders_with_the_planted_tokens_verbatim():
    sheet = FactSheet(1)
    for kind, templates in STATEMENTS.items():
        for t in templates:
            for fact in sheet.by_kind(kind):
                text = render(t, fact)
                assert fact.value in text, (t.id, text)
    for t in TOOL:
        for fact in sheet.by_kind(t.kind):
            assert fact.value in render(t, fact), t.id
    for kind, templates in MULTI.items():
        first, second = sheet.by_kind(kind)[:2]
        for t in templates:
            text = render(t, first, second)
            assert first.value in text and second.value in text, t.id
    for kind, templates in CLAIMS.items():
        assert kind in ATTRIBUTE_KINDS
        for t in templates:
            fact = sheet.by_kind(kind)[0]
            assert fact.value in render(t, fact), t.id


def test_corpus_has_statement_and_claim_cases_only_where_drift_applies():
    corpus = Corpus.build([FactSheet(1)], ["template", "multi", "tool", "benign"], None)
    statements, claims, benign = corpus.statements(), corpus.claims(), corpus.benign()
    assert statements and claims and benign
    assert all(c.fact.kind in ATTRIBUTE_KINDS for c in claims + benign)
    assert all(c.wrong_value != c.fact.value for c in claims + benign)
    assert all(c.wrong_value in c.claim for c in benign)
    assert not any(c.benign for c in claims)
    assert len({c.id for c in corpus.cases}) == len(corpus.cases)
    for kind, templates in BENIGN.items():
        for t in templates:
            assert "{value}" in t.text and t.kind == kind


# ---- buckets --------------------------------------------------------------------------

PORT = Fact("port", "llama.cpp", "8080", "llama.cpp")
PLAIN_PORT = Fact("port", "nginx", "8080", "")


def test_registry_buckets():
    assert registry_bucket(PORT, [entry("port", "llama.cpp", "8080")]) == "anchored"
    assert registry_bucket(PORT, [entry("port", "", "8080")]) == "unanchored"
    assert registry_bucket(PORT, [entry("port", "localhost", "8080")]) == "misanchored"
    assert registry_bucket(PORT, [entry("port", "llama.cpp", "8000")]) == "missing"
    assert registry_bucket(PLAIN_PORT, [entry("port", "", "8080")]) == "anchored"
    assert registry_bucket(PLAIN_PORT, [entry("port", "nginx", "8080")]) == "anchored"


def test_spurious_ignores_planted_derived_and_entity_kinds():
    url = Fact("url", "x", "http://h.lan:9000/v1", "", derived=(("hostname", "", "h.lan"), ("port", "h.lan", "9000")))
    registry = [
        entry("url", "", "http://h.lan:9000/v1"),
        entry("hostname", "", "h.lan"),
        entry("port", "h.lan", "9000"),
        entry("container", "", "ai-x"),
        entry("ipv4", "", "0.0.0.0"),
    ]
    assert [e["value"] for e in spurious_entries(url, (), registry)] == ["0.0.0.0"]


def test_benign_bucket_flags_any_fire_as_a_false_positive():
    drift = [{"kind": "port", "anchor": "llama.cpp", "known": "8080", "claimed": "8000"}]
    assert benign_bucket(PORT, "8000", drift) == "false_positive"
    assert benign_bucket(PORT, "8000", []) == "silent"
    assert benign_bucket(PORT, "8000", [{**drift[0], "claimed": "9000"}]) == "silent"


def test_claim_buckets_follow_the_drift_rule():
    reg = [entry("port", "llama.cpp", "8080")]
    claim = [entry("port", "llama.cpp", "8000")]
    fired = [{"kind": "port", "anchor": "llama.cpp", "known": "8080", "claimed": "8000"}]
    assert claim_bucket(PORT, "8000", reg, claim, fired) == "fired"
    assert claim_bucket(PORT, "8000", reg, claim, [{**fired[0], "known": "3000"}]) == "fired_misattributed"
    assert claim_bucket(PORT, "8000", reg, [], []) == "claim_not_recognized"
    assert claim_bucket(PORT, "8000", [], claim, []) == "fact_missing"
    assert claim_bucket(PORT, "8000", reg + [entry("port", "", "8000")], claim, []) == "suppressed_known_elsewhere"
    assert claim_bucket(PORT, "8000", reg, [entry("port", "", "8000")], []) == "anchor_mismatch"
    two = [entry("port", "", "8080"), entry("port", "", "3000")]
    assert claim_bucket(PLAIN_PORT, "8000", two, [entry("port", "", "8000")], []) == "ambiguous"
    assert claim_bucket(PORT, "8000", reg, claim, []) == "unexplained"


# ---- paraphrase filter ----------------------------------------------------------------


def test_paraphrase_parsing_and_verbatim_filter():
    reply = "<think>hmm</think>\n1. llama.cpp is listening on port 8080\n- port 8080 is llama.cpp's\n\n\"eighty-eighty is llama.cpp's port\"\n"
    rewrites = parse_rewrites(reply)
    assert rewrites == ["llama.cpp is listening on port 8080", "port 8080 is llama.cpp's", "eighty-eighty is llama.cpp's port"]
    tokens = ["8080", "llama.cpp"]
    original = "llama.cpp is running on port 8080."
    assert keep(rewrites[0], original, tokens)
    assert not keep(rewrites[2], original, tokens)
    assert not keep(original, original, tokens)


# ---- survey ---------------------------------------------------------------------------


def test_loose_sweep_finds_what_the_strict_extractor_is_known_to_skip():
    text = 'server { listen 8080; } and {"port": 9000} and LOG_LEVEL: debug and API_KEY=abc and lz4-4.4.5 and timeout: 30'
    found = {(k, v) for k, _s, v, _ in loose_candidates(text)}
    assert {("port", "8080"), ("port", "9000"), ("env_var", "debug"), ("version", "4.4.5"), ("numeric_cfg", "30")} <= found
    assert not any(v == "abc" for _, v in found)


def test_loose_sweep_skips_known_noise():
    text = '{"name":"Pine","x":31,"y":122} src/monitor.py:263: try: NOTE: GUIDs <command-name>/clear</command-name> ~/.pyenv/versions/3.12.1'
    found = {(k, v) for k, _s, v, _ in loose_candidates(text)}
    assert not any(v in ("31", "122", "263") for _, v in found)
    assert ("env_var", "GUIDs") not in found
    assert not any(k == "path" and v == "/clear" for k, v in found)
    assert ("version", "3.12.1") not in found


def test_missed_candidates_drop_registered_values_and_path_prefixes():
    src = Source("f", 1, "user", "config at /etc/llama-swap/config.yaml under /etc/llama-swap on port 8080")
    registry = [entry("path", "", "/etc/llama-swap/config.yaml"), entry("port", "", "8080")]
    assert list(missed_candidates(src, registry)) == []


# ---- end to end -----------------------------------------------------------------------


@pytest.mark.skipif(not EXTRACT_BINARY.exists(), reason="run `cargo build --example extract` first")
def test_binary_round_trip_matches_the_e2e_walkthrough():
    corpus = Corpus.build([FactSheet(1)], ["template"], None)
    results = run_cases(EXTRACT_BINARY, [c.to_request() for c in corpus.cases[:20]])
    assert set(results) == {c.id for c in corpus.cases[:20]}
    hand = run_cases(
        EXTRACT_BINARY,
        [
            {
                "id": "demo",
                "messages": [
                    {"role": "user", "text": "llama.cpp is running on port 8080 on host ai-llama-swap"},
                    {"role": "assistant", "text": "Your llama.cpp server on port 8000 looks healthy."},
                ],
            }
        ],
    )
    assert hand["demo"]["drift"] == [{"kind": "port", "anchor": "llama.cpp", "known": "8080", "claimed": "8000"}]
