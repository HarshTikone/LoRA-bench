"""Data prep pipeline: hitoshura25/cvefixes -> instruction-tuning JSONL.

Pipeline stages, each a pure function over plain dicts/dataclasses so they
can be unit tested without network access (see tests/test_cvefixes.py and
tests/fixtures/sample_raw_records.json, which mirror the real schema
confirmed by streaming live rows on 2026-08-22):

    load_raw_dataset  -> raw dict per row (network; not unit tested)
    filter_records    -> clean_record() over each row, dedup, optional cap;
                         also returns a Counter of why rows were dropped
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
from collections import Counter
from collections.abc import Iterable, Iterator
from enum import Enum
from pathlib import Path
from typing import Any

from lora_bench.config import Config, DataConfig, load_config
from lora_bench.data.schema import FixDiffExample

KNOWN_SEVERITIES = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}


class DropReason(str, Enum):
    """Why filter_records dropped a row.

    Values are Counter keys and get printed in the pipeline summary and
    written into the manifest, so they're deliberately stable, lowercase,
    machine-parseable strings rather than free-text.
    """

    INVALID_FIELD_TYPE = "invalid_field_type"
    MISSING_IDENTIFIER = "missing_identifier"
    EMPTY_CODE = "empty_code"
    DISALLOWED_LANGUAGE = "disallowed_language"
    TOO_SHORT = "too_short"
    TOO_LONG = "too_long"
    COMBINED_TOO_LONG = "combined_too_long"
    NOOP_PAIR = "noop_pair"
    DUPLICATE = "duplicate"

# Deliberately excludes the CVE ID -- see ADR-0005. It's kept as metadata on
# FixDiffExample (used for the group-aware split and failure-case analysis)
# but not embedded in the trained-on prompt text: at real inference time
# there is no CVE ID for an unknown vulnerability, and a near-unique
# per-example identifier is exactly the kind of key a model can memorize
# instead of learning the underlying fix. CWE is kept -- it's a coarse,
# shared-across-many-examples category, and a realistic signal a real
# pipeline could supply (e.g. a SAST tool flagging "CWE-79 here").
INSTRUCTION_TEMPLATE = (
    "The following {language} code contains a known security vulnerability "
    "({cwe_id}: {cwe_name}). Rewrite it to fix the vulnerability while "
    "preserving its intended behavior."
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


def clean_record(raw: dict, cfg: DataConfig) -> tuple[dict | None, DropReason | None]:
    """Validate + normalize one raw row.

    Returns (cleaned_record, None) on success, or (None, DropReason) if the
    row should be dropped. Every field this function keys/branches on is
    checked for *type*, not just truthiness: a non-string value in a
    string-shaped field (e.g. a stray float from an upstream column that
    wasn't stringified the way `severity` sometimes arrives as the literal
    string "nan") is dropped, not coerced — silently `str()`-ing a float
    into a code field would train on garbage, not fix a formatting quirk.
    Purely descriptive fields (cwe_id/cwe_name/diff) are coerced with
    `str()` instead of dropped, since they're metadata for readability, not
    training-critical content or join keys.
    """
    cve_id = raw.get("cve_id")
    commit_hash = raw.get("hash")
    repo_url = raw.get("repo_url")
    language = raw.get("language")
    vulnerable_code = raw.get("vulnerable_code")
    fixed_code = raw.get("fixed_code")

    for value in (cve_id, commit_hash, repo_url, language, vulnerable_code, fixed_code):
        if value is not None and not isinstance(value, str):
            return None, DropReason.INVALID_FIELD_TYPE

    if not cve_id or not commit_hash or not repo_url:
        return None, DropReason.MISSING_IDENTIFIER

    vulnerable_code = vulnerable_code or ""
    fixed_code = fixed_code or ""
    if not vulnerable_code.strip() or not fixed_code.strip():
        return None, DropReason.EMPTY_CODE

    if language not in cfg.languages:
        return None, DropReason.DISALLOWED_LANGUAGE

    vc_len, fc_len = len(vulnerable_code), len(fixed_code)
    if vc_len < cfg.min_chars or fc_len < cfg.min_chars:
        return None, DropReason.TOO_SHORT
    if vc_len > cfg.max_chars or fc_len > cfg.max_chars:
        return None, DropReason.TOO_LONG
    if vc_len + fc_len > cfg.max_combined_chars:
        return None, DropReason.COMBINED_TOO_LONG

    if cfg.drop_noop_pairs and vulnerable_code.strip() == fixed_code.strip():
        return None, DropReason.NOOP_PAIR

    cwe_id = raw.get("cwe_id")
    cwe_name = raw.get("cwe_name")
    diff = raw.get("diff_with_context")

    return (
        {
            "cve_id": cve_id,
            "cwe_id": str(cwe_id) if cwe_id else "UNKNOWN",
            "cwe_name": str(cwe_name) if cwe_name else "unspecified weakness",
            "severity": normalize_severity(raw.get("severity")),
            "language": language,
            "repo_url": repo_url,
            "commit_hash": commit_hash,
            "vulnerable_code": vulnerable_code,
            "fixed_code": fixed_code,
            "diff": str(diff) if diff else "",
            "cve_description": parse_cve_description(raw.get("cve_description")),
        },
        None,
    )


def filter_records(
    raw_records: Iterable[dict], cfg: DataConfig
) -> tuple[list[dict], Counter]:
    """clean_record() over every row, then dedup and (optionally) cap.

    Returns the cleaned/deduped/capped records alongside a Counter of why
    every *dropped* row was dropped (DropReason.value -> count). Without
    this, a silent upstream shift (e.g. a schema change that fails 90% of
    rows) would just produce a smaller output file with no signal that
    anything went wrong — see run_pipeline/main, which surface this Counter.

    Dedup key is (commit_hash, repo_url, cve_id): the same fix commit can
    appear more than once if the source dataset has one row per changed
    file. Capping samples uniformly at random (seeded) from the full
    filtered set rather than taking the first N, since source rows are
    grouped by repo and truncating in dataset order would skew the kept
    examples toward whichever repos happen to sort first. Rows dropped by
    this cap are deliberately NOT added to the returned Counter — sampling
    down to a size budget is an intentional choice, not a data-quality
    problem, and mixing the two would make drop-reason counts depend on
    max_examples in a confusing way.
    """
    drop_counts: Counter = Counter()
    seen: set[tuple[str, str, str]] = set()
    cleaned: list[dict] = []
    for raw in raw_records:
        rec, reason = clean_record(raw, cfg)
        if rec is None:
            assert reason is not None
            drop_counts[reason.value] += 1
            continue
        key = (rec["commit_hash"], rec["repo_url"], rec["cve_id"])
        if key in seen:
            drop_counts[DropReason.DUPLICATE.value] += 1
            continue
        seen.add(key)
        cleaned.append(rec)

    if cfg.max_examples is not None and len(cleaned) > cfg.max_examples:
        cleaned = random.Random(cfg.seed).sample(cleaned, cfg.max_examples)

    return cleaned, drop_counts


def to_example(rec: dict) -> FixDiffExample:
    """Map one cleaned record to the instruction-tuning example shape."""
    instruction = INSTRUCTION_TEMPLATE.format(
        language=rec["language"],
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


def build_dataset(
    raw_records: Iterable[dict], cfg: DataConfig
) -> tuple[list[FixDiffExample], Counter]:
    cleaned, drop_counts = filter_records(raw_records, cfg)
    return [to_example(rec) for rec in cleaned], drop_counts


def split_examples(
    examples: list[FixDiffExample], cfg: DataConfig
) -> tuple[list[FixDiffExample], list[FixDiffExample], list[FixDiffExample]]:
    """Deterministic, group-aware train/val/test split, grouped by cve_id.

    Every example sharing a cve_id is assigned to the SAME split, never
    split across the boundary. A single CVE routinely spans several files
    fixed in one commit (or gets a follow-up fix commit later) — row-level
    random splitting would put those near-duplicate rows of the same fix
    on both sides of train/test, inflating whatever quality number gets
    reported off that split. See ADR-0004 for the leakage this closes,
    what it deliberately doesn't (repo-level leakage across *different*
    CVEs), and the measured ratio tolerance this heuristic achieves.

    Same (examples, cfg.seed) always yields the same split — needed so a
    re-run of data prep doesn't silently change what Day 2 trains/evals on.
    Group assignment is a deterministic largest-remaining-need greedy
    heuristic (shuffle group order by seed, then repeatedly hand the next
    group to whichever split is furthest below its target count, ties
    broken train > val > test) — it can't hit the configured ratios
    exactly when group sizes vary, but gets close while guaranteeing zero
    cross-split leakage by construction.
    """
    groups: dict[str, list[FixDiffExample]] = {}
    for ex in examples:
        groups.setdefault(ex.cve_id, []).append(ex)

    group_ids = list(groups)
    random.Random(cfg.seed).shuffle(group_ids)

    total = len(examples)
    targets = {
        "train": total * cfg.train_ratio,
        "val": total * cfg.val_ratio,
        "test": total * cfg.test_ratio,
    }
    counts = {"train": 0, "val": 0, "test": 0}
    assigned: dict[str, list[FixDiffExample]] = {"train": [], "val": [], "test": []}
    split_order = ("train", "val", "test")  # fixed tie-break order

    for gid in group_ids:
        group = groups[gid]
        chosen = max(split_order, key=lambda name: targets[name] - counts[name])
        assigned[chosen].extend(group)
        counts[chosen] += len(group)

    ratios = {"train": cfg.train_ratio, "val": cfg.val_ratio, "test": cfg.test_ratio}
    for name in split_order:
        if ratios[name] > 0 and not assigned[name]:
            raise ValueError(
                f"{name} split came out empty even though {name}_ratio="
                f"{ratios[name]} > 0 ({len(examples)} examples across "
                f"{len(groups)} CVE groups). This silently produces, e.g., a "
                "fine-tune with no validation signal and no error. Increase the "
                "dataset size (raise max_examples, loosen filters) or set this "
                "ratio to 0 if an empty split is genuinely intentional."
            )

    return assigned["train"], assigned["val"], assigned["test"]


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


def load_raw_dataset(dataset_name: str, split: str, revision: str | None = None) -> Iterator[dict]:
    """Thin wrapper around datasets.load_dataset — the one network call in
    this module. Isolated here so nothing else in the pipeline imports
    `datasets` or touches the network, keeping the rest unit-testable.

    `revision` should normally be DataConfig.revision (a pinned commit SHA,
    per ADR-0002) rather than None/a branch name — every comparison this
    project reports depends on the training data staying reproducible.
    """
    from datasets import load_dataset  # local import: only needed on this path

    ds = load_dataset(dataset_name, split=split, revision=revision)
    for row in ds:
        yield dict(row)


def _load_fixture_records() -> list[dict]:
    with FIXTURE_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_pipeline(cfg: Config, raw_records: Iterable[dict], out_dir: str | Path) -> dict[str, Any]:
    examples, drop_counts = build_dataset(raw_records, cfg.data)
    train, val, test = split_examples(examples, cfg.data)

    out_dir = Path(out_dir)
    write_jsonl(train, out_dir / "train.jsonl")
    write_jsonl(val, out_dir / "val.jsonl")
    write_jsonl(test, out_dir / "test.jsonl")

    return {
        "total": len(examples),
        "train": len(train),
        "val": len(val),
        "test": len(test),
        "drop_counts": dict(drop_counts),
    }


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
        raw_records = load_raw_dataset(
            cfg.data.dataset_name, cfg.data.dataset_split, cfg.data.revision
        )

    stats = run_pipeline(cfg, raw_records, args.out_dir)
    print(f"Wrote {stats['total']} examples to {args.out_dir}/ "
          f"(train={stats['train']}, val={stats['val']}, test={stats['test']})")
    if stats["drop_counts"]:
        print("Dropped rows by reason:")
        for reason, count in sorted(stats["drop_counts"].items(), key=lambda kv: -kv[1]):
            print(f"  {reason}: {count}")


if __name__ == "__main__":
    main()
