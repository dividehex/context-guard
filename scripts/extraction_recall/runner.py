"""Run cases through `examples/extract.rs`, the binary that calls the real
extractor and drift rule. One process handles a whole corpus over stdin."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
EXTRACT_BINARY = REPO_ROOT / "target" / "debug" / "examples" / "extract"


def cargo_env() -> dict[str, str]:
    """Non-interactive shells on the dev machine lack ~/.cargo/bin on PATH."""
    env = dict(os.environ)
    cargo_bin = str(Path.home() / ".cargo" / "bin")
    if cargo_bin not in env.get("PATH", "").split(os.pathsep):
        env["PATH"] = cargo_bin + os.pathsep + env.get("PATH", "")
    return env


def build_extract_binary(repo_root: Path = REPO_ROOT) -> Path:
    subprocess.run(
        ["cargo", "build", "-q", "--example", "extract"],
        cwd=repo_root,
        env=cargo_env(),
        check=True,
    )
    return repo_root / "target" / "debug" / "examples" / "extract"


def run_cases(binary: Path, requests: Iterable[dict], container_prefixes: str = "ai-") -> dict[str, dict]:
    """Feed `{"id", "messages"}` requests to the binary; return results by id."""
    payload = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in requests)
    proc = subprocess.run(
        [str(binary), container_prefixes],
        input=payload,
        text=True,
        capture_output=True,
        check=True,
    )
    results: dict[str, dict] = {}
    for line in proc.stdout.splitlines():
        if line.strip():
            rec = json.loads(line)
            results[rec["id"]] = rec
    return results


def extract_texts(binary: Path, texts: Iterable[tuple[str, str, str]], container_prefixes: str = "ai-") -> dict[str, list[dict]]:
    """Registry for many standalone texts: `(id, role, text)` in, `{id: registry}` out."""
    requests = [{"id": i, "messages": [{"role": role, "text": text}]} for i, role, text in texts]
    return {i: r["registry"] for i, r in run_cases(binary, requests, container_prefixes).items()}
