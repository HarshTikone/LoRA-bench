"""One-off diagnostic: measure real tokenized lengths of a produced
train.jsonl against the actual training tokenizer/chat template, to check
(or refresh) the heuristic bound ADR-0003 sets via
DataConfig.max_combined_chars vs. ModelConfig.max_seq_len.

Needs `transformers` + `jinja2` (Day 2/Colab-only deps -- deliberately NOT
in requirements-repo.txt, to keep this repo's CPU-side deps small) and
network access to download the tokenizer. Also imports `lora_bench`
itself (for `to_chat_messages`/`FixDiffExample`), so run it somewhere that
package is importable too: `pip install transformers jinja2 && pip install
-e .` in a throwaway environment, or run this in Colab. NOT part of the
pytest suite -- run manually and report what it actually printed; never
estimate this distribution by hand and present it as measured.

Usage:
    python scripts/token_budget.py data/processed/train.jsonl --max-seq-len 1024
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from lora_bench.data.cvefixes import to_chat_messages
from lora_bench.data.schema import FixDiffExample


def render_example(rec: dict, tokenizer) -> str:
    """Render one example the way Day 2 will actually train on it, through
    the tokenizer's own chat template -- not a hand-rolled prompt string,
    since the real overhead comes from whatever role markers/special
    tokens the template actually inserts. Message structure comes from
    to_chat_messages(), the same function the fine-tuning notebook uses,
    so this measurement and actual training can't silently render examples
    differently.
    """
    messages = to_chat_messages(FixDiffExample.from_dict(rec))
    return tokenizer.apply_chat_template(messages, tokenize=False)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "train_jsonl", type=Path, help="Path to a train.jsonl produced by cvefixes.py"
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--max-seq-len", type=int, default=1024)
    args = parser.parse_args(argv)

    from transformers import AutoTokenizer  # local import: only needed on this path

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    records = []
    with args.train_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        print(f"No records found in {args.train_jsonl} -- nothing to measure.")
        return

    lengths: list[int] = []
    over_budget = 0
    lost_chars_total = 0
    for rec in records:
        rendered = render_example(rec, tokenizer)
        ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
        n_tokens = len(ids)
        lengths.append(n_tokens)
        if n_tokens > args.max_seq_len:
            over_budget += 1
            truncated_text = tokenizer.decode(ids[: args.max_seq_len])
            lost_chars_total += max(0, len(rendered) - len(truncated_text))

    lengths.sort()
    n = len(lengths)

    def pct(p: float) -> int:
        return lengths[min(n - 1, int(p * n))]

    print(f"examples: {n}")
    print(
        f"tokens -- min: {lengths[0]}  median: {statistics.median(lengths):.0f}  "
        f"p90: {pct(0.90)}  p95: {pct(0.95)}  p99: {pct(0.99)}  max: {lengths[-1]}"
    )
    print(f"exceed max_seq_len ({args.max_seq_len}): {over_budget} ({over_budget / n:.1%})")
    if over_budget:
        print(
            "avg characters lost to right-truncation among those examples: "
            f"{lost_chars_total / over_budget:.0f}"
        )


if __name__ == "__main__":
    main()
