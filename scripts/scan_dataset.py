"""One-off diagnostic: run the pipeline's OWN filter_records over a live
sample of hitoshura25/cvefixes and report the drop-reason breakdown plus a
few descriptive stats (language/severity distribution, max code length
seen) that ADR-0002 cites to justify DataConfig's filtering defaults.

Calls clean_record/filter_records directly rather than re-implementing
the empty-code/no-op-pair/etc. checks inline, so this script and the
pipeline can't silently disagree about what counts as a "drop" -- an
earlier version of this script had its own inline copy of those checks,
which would have gone stale the moment clean_record's logic changed.

Needs live network access to the Hugging Face Hub, so it is deliberately
NOT part of the pytest suite and NOT imported by the pipeline -- run it
manually to reproduce or refresh the numbers behind ADR-0002.

Usage:
    python scripts/scan_dataset.py --limit 3000
"""

from __future__ import annotations

import argparse
from collections import Counter
from itertools import islice

from datasets import load_dataset

from lora_bench.config import DataConfig
from lora_bench.data.cvefixes import filter_records


def scan(dataset_name: str, split: str, revision: str | None, limit: int) -> None:
    ds = load_dataset(dataset_name, split=split, revision=revision, streaming=True)
    raw_records = list(islice(ds, max(limit, 0)))
    n = len(raw_records)

    if n == 0:
        print("No rows scanned (empty stream or --limit 0) -- nothing to report.")
        return

    lang_counter = Counter(str(r.get("language")) for r in raw_records)
    severity_counter = Counter(str(r.get("severity")) for r in raw_records)
    max_len_seen = max(
        max(len(r.get("vulnerable_code") or ""), len(r.get("fixed_code") or ""))
        for r in raw_records
    )

    # max_examples=None: this diagnostic's job is to report the true
    # keep-rate/drop-reason breakdown among the `--limit` rows scanned. The
    # pipeline's own default (max_examples=3000) would silently downsample
    # `cleaned` to 3000 once more than that survives filtering -- fine for
    # the pipeline (it's an intentional size cap), but it would make
    # len(cleaned) a sample size instead of a keep count here, silently
    # under-reporting the keep rate at any --limit above ~3000-4000.
    cfg = DataConfig(max_examples=None)
    cleaned, drop_counts = filter_records(raw_records, cfg)

    print(f"scanned: {n}")
    print(f"top languages: {lang_counter.most_common(15)}")
    print(f"severity values: {severity_counter.most_common(10)}")
    print(f"max code length seen (chars): {max_len_seen}")
    print(f"kept by current DataConfig defaults: {len(cleaned)} ({len(cleaned) / n:.1%})")
    print("dropped by reason:")
    for reason, count in sorted(drop_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {reason}: {count} ({count / n:.1%})")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="hitoshura25/cvefixes")
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--revision",
        default=DataConfig().revision,
        help="Defaults to the pipeline's pinned revision (DataConfig.revision, "
        "see ADR-0002), so this diagnostic scans the same snapshot the pipeline "
        "actually reads -- a script whose job is producing ADR-0002's cited "
        "numbers must not scan a different, possibly-moved snapshot by default. "
        "Pass --revision main to deliberately scan the current branch head "
        "instead, e.g. to check for upstream drift.",
    )
    parser.add_argument("--limit", type=int, default=3000)
    args = parser.parse_args(argv)
    scan(args.dataset, args.split, args.revision, args.limit)


if __name__ == "__main__":
    main()
