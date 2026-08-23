"""One-off diagnostic: reproduce the hitoshura25/cvefixes data-quality
findings that ADR.md's ADR-0002 cites when justifying DataConfig's
filtering defaults (max_chars, drop_noop_pairs, the language allowlist).

Needs live network access to the Hugging Face Hub, so it is deliberately
NOT part of the pytest suite and NOT imported by the pipeline — run it
manually to reproduce or refresh the numbers behind ADR-0002.

Usage:
    python scripts/scan_dataset.py --limit 3000
"""

from __future__ import annotations

import argparse
from collections import Counter

from datasets import load_dataset


def scan(dataset_name: str, split: str, limit: int) -> None:
    ds = load_dataset(dataset_name, split=split, streaming=True)

    lang_counter: Counter[str] = Counter()
    severity_counter: Counter[str] = Counter()
    n = noop = empty_vulnerable = empty_fixed = 0
    max_len_seen = 0

    for rec in ds:
        n += 1
        lang_counter[rec.get("language")] += 1
        severity_counter[str(rec.get("severity"))] += 1

        vulnerable_code = rec.get("vulnerable_code") or ""
        fixed_code = rec.get("fixed_code") or ""
        if vulnerable_code.strip() == fixed_code.strip():
            noop += 1
        if not vulnerable_code.strip():
            empty_vulnerable += 1
        if not fixed_code.strip():
            empty_fixed += 1
        max_len_seen = max(max_len_seen, len(vulnerable_code), len(fixed_code))

        if n >= limit:
            break

    print(f"scanned: {n}")
    print(f"top languages: {lang_counter.most_common(15)}")
    print(f"severity values: {severity_counter.most_common(10)}")
    print(f"noop (vulnerable_code == fixed_code): {noop} ({noop / n:.1%})")
    print(f"empty vulnerable_code: {empty_vulnerable}  empty fixed_code: {empty_fixed}")
    print(f"max code length seen (chars): {max_len_seen}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="hitoshura25/cvefixes")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=3000)
    args = parser.parse_args(argv)
    scan(args.dataset, args.split, args.limit)


if __name__ == "__main__":
    main()
