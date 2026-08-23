"""Data prep pipeline: hitoshura25/cvefixes -> instruction-tuning JSONL.

Pipeline stages, each a pure function over plain dicts/dataclasses so they
can be unit tested without network access (see tests/test_cvefixes.py and
tests/fixtures/sample_raw_records.json, which mirror the real schema
confirmed by streaming live rows on 2026-08-22):

    load_raw_dataset  -> raw dict per row (network; not unit tested)
    filter_records    -> clean_record() over each row, dedup, optional cap
    build_dataset     -> filter_records() + to_example() -> FixDiffExample
    split_examples    -> deterministic seeded train/val/test split
    write_jsonl / read_jsonl

Run end to end with:  python -m lora_bench.data.cvefixes --config configs/default.yaml
Or, without network, sanity-check the wiring against the bundled fixture:
    python -m lora_bench.data.cvefixes --dry-run
"""

from __future__ import annotations

import argparse
import ast
import json
import random
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from lora_bench.config import Config, DataConfig, load_config
from lora_bench.data.schema import FixDiffExample

KNOWN_SEVERITIES = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}

INSTRUCTION_TEMPLATE = (
    "The following {language} code contains a known security vulnerability "
    "({cve_id}, {cwe_id}: {cwe_name}). Rewrite it to fix the vulnerability "
    "while preserving its intended behavior."
)

FIXTURE_PATH = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "sample_raw_records.json"


def parse_cve_description(raw: Any) -> str:
    """Extract readable text from the dataset's `cve_description` field.

    Real rows store this as the *string repr* of a list of dicts, e.g.
    `"[{'lang': 'en', 'value': 'Some CVE text'}]"` (single-quoted, not JSON).
    Falls back to the raw value (stringified) if it doesn't parse as that
    shape, rather than raising — this field is metadata for readability, not
    something filtering decisions depend on.
    """
    if not raw:
        return ""
    if not isinstance(raw, str):
        return str(raw)
    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return raw
    if not isinstance(parsed, list) or not parsed:
        return raw
    entries = [e for e in parsed if isinstance(e, dict) and "value" in e]
    if not entries:
        return raw
    for e in entries:
        if e.get("lang") == "en":
            return str(e["value"])
    return str(entries[0]["value"])


def normalize_severity(raw: Any) -> str:
    """Map the source's messy severity field to a fixed vocabulary.

    Observed values include actual severities ("MEDIUM", "HIGH", ...), the
    literal string "nan" (pandas NaN stringified upstream), None, and empty
    strings. All non-recognized values become "UNKNOWN" rather than being
    passed through, so downstream code has a closed set to branch on.
    """
    if raw is None:
        return "UNKNOWN"
    s = str(raw).strip().upper()
    return s if s in KNOWN_SEVERITIES else "UNKNOWN"


def clean_record(raw: dict, cfg: DataConfig) -> dict | None:
    """Validate + normalize one raw row. Returns None if it should be dropped.

    Drop reasons, in the order checked: missing required identifiers, empty
    code fields, language not in the allowlist, either code field outside
    [min_chars, max_chars], or (if cfg.drop_noop_pairs) vulnerable_code and
    fixed_code identical after stripping whitespace.
    """
    cve_id = raw.get("cve_id")
    vulnerable_code = raw.get("vulnerable_code") or ""
    fixed_code = raw.get("fixed_code") or ""
    language = raw.get("language")
    commit_hash = raw.get("hash")
    repo_url = raw.get("repo_url")

    if not cve_id or not commit_hash or not repo_url:
        return None
    if not vulnerable_code.strip() or not fixed_code.strip():
        return None
    if language not in cfg.languages:
        return None

    vc_len, fc_len = len(vulnerable_code), len(fixed_code)
    if not (cfg.min_chars <= vc_len <= cfg.max_chars):
        return None
    if not (cfg.min_chars <= fc_len <= cfg.max_chars):
        return None

    if cfg.drop_noop_pairs and vulnerable_code.strip() == fixed_code.strip():
        return None

    return {
        "cve_id": cve_id,
        "cwe_id": raw.get("cwe_id") or "UNKNOWN",
        "cwe_name": raw.get("cwe_name") or "unspecified weakness",
        "severity": normalize_severity(raw.get("severity")),
        "language": language,
        "repo_url": repo_url,
        "commit_hash": commit_hash,
        "vulnerable_code": vulnerable_code,
        "fixed_code": fixed_code,
        "diff": raw.get("diff_with_context") or "",
        "cve_description": parse_cve_description(raw.get("cve_description")),
    }


def filter_records(raw_records: Iterable[dict], cfg: DataConfig) -> list[dict]:
    """clean_record() over every row, then dedup and (optionally) cap.

    Dedup key is (commit_hash, repo_url, cve_id): the same fix commit can
    appear more than once if the source dataset has one row per changed
    file. Capping samples uniformly at random (seeded) from the full
    filtered set rather than taking the first N, since source rows are
    grouped by repo and truncating in dataset order would skew the kept
    examples toward whichever repos happen to sort first.
    """
    seen: set[tuple[str, str, str]] = set()
    cleaned: list[dict] = []
    for raw in raw_records:
        rec = clean_record(raw, cfg)
        if rec is None:
            continue
        key = (rec["commit_hash"], rec["repo_url"], rec["cve_id"])
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(rec)

    if cfg.max_examples is not None and len(cleaned) > cfg.max_examples:
        cleaned = random.Random(cfg.seed).sample(cleaned, cfg.max_examples)

    return cleaned


def to_example(rec: dict) -> FixDiffExample:
    """Map one cleaned record to the instruction-tuning example shape."""
    instruction = INSTRUCTION_TEMPLATE.format(
        language=rec["language"],
        cve_id=rec["cve_id"],
        cwe_id=rec["cwe_id"],
        cwe_name=rec["cwe_name"],
    )
    return FixDiffExample(
        cve_id=rec["cve_id"],
        cwe_id=rec["cwe_id"],
        cwe_name=rec["cwe_name"],
        severity=rec["severity"],
        language=rec["language"],
        repo_url=rec["repo_url"],
        commit_hash=rec["commit_hash"],
        instruction=instruction,
        input=rec["vulnerable_code"],
        output=rec["fixed_code"],
        diff=rec["diff"],
    )


def build_dataset(raw_records: Iterable[dict], cfg: DataConfig) -> list[FixDiffExample]:
    return [to_example(rec) for rec in filter_records(raw_records, cfg)]


def split_examples(
    examples: list[FixDiffExample], cfg: DataConfig
) -> tuple[list[FixDiffExample], list[FixDiffExample], list[FixDiffExample]]:
    """Deterministic seeded shuffle-then-slice train/val/test split.

    Same (examples, cfg.seed) always yields the same split — needed so a
    re-run of data prep doesn't silently change what Day 2 trains/evals on.
    The test split absorbs rounding remainder so every example lands
    somewhere and counts always sum to len(examples).
    """
    shuffled = list(examples)
    random.Random(cfg.seed).shuffle(shuffled)

    n = len(shuffled)
    n_train = round(n * cfg.train_ratio)
    n_val = round(n * cfg.val_ratio)
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)

    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]
    return train, val, test


def write_jsonl(examples: Iterable[FixDiffExample], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[FixDiffExample]:
    path = Path(path)
    examples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(FixDiffExample.from_dict(json.loads(line)))
    return examples


def load_raw_dataset(dataset_name: str, split: str) -> Iterator[dict]:
    """Thin wrapper around datasets.load_dataset — the one network call in
    this module. Isolated here so nothing else in the pipeline imports
    `datasets` or touches the network, keeping the rest unit-testable.
    """
    from datasets import load_dataset  # local import: only needed on this path

    ds = load_dataset(dataset_name, split=split)
    for row in ds:
        yield dict(row)


def _load_fixture_records() -> list[dict]:
    with FIXTURE_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_pipeline(cfg: Config, raw_records: Iterable[dict], out_dir: str | Path) -> dict[str, int]:
    examples = build_dataset(raw_records, cfg.data)
    train, val, test = split_examples(examples, cfg.data)

    out_dir = Path(out_dir)
    write_jsonl(train, out_dir / "train.jsonl")
    write_jsonl(val, out_dir / "val.jsonl")
    write_jsonl(test, out_dir / "test.jsonl")

    return {"total": len(examples), "train": len(train), "val": len(val), "test": len(test)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument(
        "--out-dir", default="data/processed", help="Where to write train/val/test.jsonl."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use the bundled tests/fixtures sample instead of the live HF Hub "
        "(no network) to smoke-test the pipeline wiring.",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)

    if args.dry_run:
        raw_records: Iterable[dict] = _load_fixture_records()
    else:
        raw_records = load_raw_dataset(cfg.data.dataset_name, cfg.data.dataset_split)

    stats = run_pipeline(cfg, raw_records, args.out_dir)
    print(f"Wrote {stats['total']} examples to {args.out_dir}/ "
          f"(train={stats['train']}, val={stats['val']}, test={stats['test']})")


if __name__ == "__main__":
    main()
