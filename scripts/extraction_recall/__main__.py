"""Command-line entry point: `python -m scripts.extraction_recall <command>`.

  report      build the corpus, run the extractor, print the recall report
  paraphrase  generate the paraphrase fixture with a local model via LiteLLM
  survey      list values in real conversations that the extractor missed
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from . import paraphrase, survey
from .corpus import BENIGN_TIER, PARAPHRASE_TIER, Corpus
from .facts import FactSheet
from .report import Report
from .runner import EXTRACT_BINARY, build_extract_binary, extract_texts, run_cases

HERE = Path(__file__).resolve().parent
DEFAULT_PARAPHRASES = HERE / "corpus" / "paraphrases.jsonl"
DEFAULT_OUT = HERE / "out"
ALL_TIERS = ("template", "multi", "tool", BENIGN_TIER, PARAPHRASE_TIER)


def _binary(args) -> Path:
    if args.extract_bin:
        return Path(args.extract_bin)
    return build_extract_binary() if not args.no_build else EXTRACT_BINARY


def cmd_report(args) -> int:
    sheets = [FactSheet(seed, per_kind=args.per_kind, container_prefix=args.container_prefixes.split(",")[0]) for seed in args.seed]
    corpus = Corpus.build(sheets, args.tiers.split(","), Path(args.paraphrases))
    if not corpus.cases:
        print("no cases; check --tiers", file=sys.stderr)
        return 2
    results = run_cases(_binary(args), (c.to_request() for c in corpus.cases), args.container_prefixes)
    report = Report.build(args.seed, corpus, results)
    print(report.to_markdown(args.limit))
    if args.json:
        Path(args.json).write_text(report.to_json())
        print(f"\nJSON written to {args.json}", file=sys.stderr)
    return 0


def cmd_paraphrase(args) -> int:
    key = args.litellm_key or os.environ.get("LITELLM_MASTER_KEY")
    if not key:
        print("need --litellm-key or LITELLM_MASTER_KEY", file=sys.stderr)
        return 2
    client = paraphrase.LiteLLMClient(args.litellm_url, key, args.model, timeout=args.timeout, no_think=not args.think)
    sheet = FactSheet(args.seed, per_kind=args.facts_per_template)
    kinds = args.kinds.split(",") if args.kinds else None
    records = paraphrase.generate(
        sheet,
        client.complete,
        n=args.n,
        facts_per_template=args.facts_per_template,
        kinds=kinds,
        log=lambda msg: print(msg, file=sys.stderr),
    )
    paraphrase.write_fixture(Path(args.out), records)
    print(f"{len(records)} paraphrases written to {args.out}", file=sys.stderr)
    return 0


def cmd_survey(args) -> int:
    sources = []
    if args.transcripts:
        sources.extend(survey.iter_transcripts(Path(args.transcripts).expanduser()))
    if args.db:
        sources.extend(survey.iter_database(Path(args.db)))
    sources = [s for s in sources if s.text.strip()]
    if not sources:
        print("no user or tool text found", file=sys.stderr)
        return 2
    registries = extract_texts(_binary(args), ((str(i), s.role, s.text) for i, s in enumerate(sources)), args.container_prefixes)
    per_origin: dict[str, int] = {}
    candidates = []
    for i, s in enumerate(sources):
        for c in survey.missed_candidates(s, registries.get(str(i), [])):
            if per_origin.get(c.origin, 0) >= args.max_per_source:
                break
            per_origin[c.origin] = per_origin.get(c.origin, 0) + 1
            candidates.append(c)
    n = survey.write_csv(Path(args.out), candidates)
    print(survey.shape_summary(candidates, args.top))
    print(f"{len(sources)} texts scanned, {n} missed candidates written to {args.out}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m scripts.extraction_recall", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--extract-bin", help="path to a built examples/extract binary (default: cargo build it)")
        p.add_argument("--no-build", action="store_true", help="use target/debug/examples/extract without rebuilding")
        p.add_argument("--container-prefixes", default="ai-", help="comma-separated, as in the Context Guard config")

    p = sub.add_parser("report", help="run the corpus and print the recall report")
    common(p)
    p.add_argument("--seed", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--per-kind", type=int, default=2, help="facts per kind on each sheet")
    p.add_argument("--tiers", default=",".join(ALL_TIERS), help=f"comma-separated subset of {','.join(ALL_TIERS)}")
    p.add_argument("--paraphrases", default=str(DEFAULT_PARAPHRASES), help="paraphrase fixture (skipped if absent)")
    p.add_argument("--limit", type=int, default=12, help="misses listed per bucket")
    p.add_argument("--json", help="also write the full report as JSON here")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("paraphrase", help="generate the paraphrase fixture with a local model")
    p.add_argument("--litellm-url", default="http://127.0.0.1:4000")
    p.add_argument("--litellm-key", help="default: $LITELLM_MASTER_KEY")
    p.add_argument("--model", default="qwen3-30b-a3b")
    p.add_argument("--n", type=int, default=8, help="rewrites requested per template")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--facts-per-template", type=int, default=1)
    p.add_argument("--kinds", help="comma-separated subset of kinds")
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--think", action="store_true", help="do not append /no_think to the prompt")
    p.add_argument("--out", default=str(DEFAULT_PARAPHRASES))
    p.set_defaults(func=cmd_paraphrase)

    p = sub.add_parser("survey", help="find values in real conversations that the extractor missed")
    common(p)
    p.add_argument("--transcripts", default="~/.claude/projects", help="Claude Code transcripts root ('' to skip)")
    p.add_argument("--db", help="Context Guard SQLite file (optional)")
    p.add_argument("--max-per-source", type=int, default=50)
    p.add_argument("--top", type=int, default=25, help="shapes listed per kind in the summary")
    p.add_argument("--out", default=str(DEFAULT_OUT / "survey.csv"))
    p.set_defaults(func=cmd_survey)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
