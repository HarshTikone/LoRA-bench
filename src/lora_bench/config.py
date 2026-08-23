"""Typed config for the LoRA Bench pipeline.

One YAML file (see configs/default.yaml) drives data prep today and will
drive the Day 2+ fine-tuning/quantization/benchmark steps too, so config
shape is defined once here rather than re-parsed ad hoc in each script.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_EPS = 1e-6

# Conservative (i.e. deliberately low) chars-per-token estimate used only to
# validate that DataConfig.max_combined_chars can't blow ModelConfig.max_seq_len
# in the worst case. Real source code under a BPE tokenizer typically runs
# closer to 3.5-4.5 chars/token; using a lower number here means this bound
# demands MORE token budget than a real tokenizer likely needs, not less --
# it's a safety margin, not a measurement. scripts/token_budget.py measures
# the real distribution with the actual training tokenizer; treat that as
# authoritative over this heuristic once it's been run.
CHARS_PER_TOKEN_ESTIMATE = 3.0

# Rough pad (in characters) for INSTRUCTION_TEMPLATE's fixed text plus
# chat-template role markers/special tokens, none of which are counted by
# max_combined_chars (which only covers the code fields). Not measured --
# same caveat as CHARS_PER_TOKEN_ESTIMATE above.
INSTRUCTION_OVERHEAD_CHARS_ESTIMATE = 300


@dataclass
class DataConfig:
    """Controls dataset sourcing, filtering, and splitting."""

    dataset_name: str = "hitoshura25/cvefixes"
    dataset_split: str = "train"

    # Pinned to a specific commit SHA of the dataset repo (resolved via
    # HfApi().dataset_info(...).sha on 2026-08-22), not left to follow the
    # branch head. hitoshura25/cvefixes is a third-party mirror ADR-0002
    # already flags as able to silently diverge from the CVEfixes source --
    # without a pin, the inputs behind any reported comparison number could
    # change under it, which is indefensible for a project whose whole
    # point is a comparison table. See ADR-0002's amendment.
    revision: str = "d4f5c4ea65329d9ccbb8a3b3149e5d06eda5edb2"

    # Language allowlist. Chosen from a live scan of the source dataset
    # (see ADR.md ADR-0002): these six cover the bulk of non-noise rows
    # (PHP/C dominate, Python/JS/Go/C++/Java each have hundreds+) while
    # excluding "Other"/"Unknown"/"JSON"/"Markdown" buckets that aren't code.
    languages: list[str] = field(
        default_factory=lambda: ["Python", "C", "C++", "JavaScript", "Java", "Go"]
    )

    # Length bounds in characters, applied to both vulnerable_code and
    # fixed_code independently. The live scan found empty code fields and at
    # least one ~55MB outlier row, so both bounds are load-bearing, not
    # cosmetic — without max_chars a single row can blow the token/time
    # budget of a T4 Colab session. Note that under the *default* config,
    # max_combined_chars (below) is the actual binding constraint on length
    # for realistic pairs, since it's tighter than 2x max_chars; max_chars
    # still matters as an independent per-field ceiling if someone raises
    # max_combined_chars without revisiting this value. See ADR-0003.
    min_chars: int = 20
    max_chars: int = 4000

    # Cap on vulnerable_code + fixed_code combined, checked *in addition to*
    # the independent max_chars bound above. max_chars alone doesn't bound
    # what a training example actually costs in tokens: it's applied per
    # field, so two fields each just under max_chars can combine into ~2x
    # that — which is exactly the bug ADR-0003 documents (max_chars=4000
    # per field vs. model.max_seq_len=1024 tokens: a worst-case pair could
    # need ~2,400-2,700 tokens, truncating the *output* since tokenizers
    # truncate from the right, i.e. teaching the model to emit a fix that
    # stops mid-token). Config.__post_init__ enforces that this value can't
    # exceed what model.max_seq_len can plausibly hold; see ADR-0003 for why
    # this field exists instead of raising max_seq_len or halving max_chars.
    max_combined_chars: int = 2600

    # Cap on total examples kept after filtering, so the Day 2 fine-tune has
    # a predictable, bounded run time on a free-tier T4 session. None means
    # no cap. Revisit once Day 2 has real wall-clock numbers.
    max_examples: int | None = 3000

    # Drop pairs where vulnerable_code == fixed_code after stripping
    # whitespace (no-op diffs; ~7% of a 3k-row scan). These teach the model
    # nothing about fixing anything.
    drop_noop_pairs: bool = True

    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42

    def __post_init__(self) -> None:
        if self.min_chars < 0:
            raise ValueError(f"min_chars must be >= 0, got {self.min_chars}")
        if self.max_chars <= self.min_chars:
            raise ValueError(
                f"max_chars ({self.max_chars}) must be > min_chars ({self.min_chars})"
            )
        if self.max_examples is not None and self.max_examples <= 0:
            raise ValueError(f"max_examples must be > 0 or null, got {self.max_examples}")
        if self.max_combined_chars < 2 * self.min_chars:
            raise ValueError(
                f"max_combined_chars ({self.max_combined_chars}) must be >= 2 * min_chars "
                f"({2 * self.min_chars}) — otherwise no pair of fields could ever pass "
                "both the per-field min_chars check and the combined-length check."
            )
        if not self.languages:
            raise ValueError("languages must be a non-empty list")
        ratio_sum = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(ratio_sum - 1.0) > _EPS:
            raise ValueError(
                "train_ratio + val_ratio + test_ratio must sum to 1.0, "
                f"got {ratio_sum} ({self.train_ratio}, {self.val_ratio}, {self.test_ratio})"
            )
        for name in ("train_ratio", "val_ratio", "test_ratio"):
            v = getattr(self, name)
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"{name} must be within [0, 1], got {v}")


@dataclass
class ModelConfig:
    """Base model for Day 2 fine-tuning. See ADR-0001 for why this model."""

    # Qwen2.5-Coder-1.5B-Instruct: Apache-2.0 (no gated-license approval
    # flow to break a one-shot Colab run), code-pretrained (relevant prior
    # for reading/writing diffs), and small enough that QLoRA fine-tuning
    # and later GGUF conversion both fit comfortably in a free T4 session.
    base_model: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    max_seq_len: int = 1024


@dataclass
class LoRAConfig:
    """Starting LoRA hyperparameters.

    These are an informed starting point, not a tuned result — the past
    "just a demo" checklist requires the *final* rank/hyperparameters be a
    defended decision from an actual sweep. That sweep is Day 2 work; its
    winner and reasoning get written into ADR.md then, and this default may
    change as a result.
    """

    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    def __post_init__(self) -> None:
        if self.r <= 0:
            raise ValueError(f"r (LoRA rank) must be > 0, got {self.r}")
        if self.alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {self.alpha}")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout must be within [0, 1), got {self.dropout}")
        if not self.target_modules:
            raise ValueError("target_modules must be a non-empty list")


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)

    def __post_init__(self) -> None:
        """Cross-section invariant: data.max_combined_chars must plausibly
        fit within model.max_seq_len.

        Neither DataConfig nor ModelConfig alone can validate this (it spans
        both sections), and nothing did before ADR-0003 — max_chars=4000
        (applied independently per field) and max_seq_len=1024 shipped
        together despite a worst-case pair needing roughly 2.5x the token
        budget. Since a tokenizer truncates from the right, that silently
        truncated the *output* (the fixed code, i.e. the training label),
        not just the input.
        """
        worst_case_chars = self.data.max_combined_chars + INSTRUCTION_OVERHEAD_CHARS_ESTIMATE
        worst_case_tokens = worst_case_chars / CHARS_PER_TOKEN_ESTIMATE
        if worst_case_tokens > self.model.max_seq_len:
            raise ValueError(
                f"data.max_combined_chars ({self.data.max_combined_chars}) plus the "
                f"instruction overhead estimate ({INSTRUCTION_OVERHEAD_CHARS_ESTIMATE} "
                f"chars) could need ~{worst_case_tokens:.0f} tokens in the worst case "
                f"(at a conservative {CHARS_PER_TOKEN_ESTIMATE} chars/token), which "
                f"exceeds model.max_seq_len ({self.model.max_seq_len}). Raise "
                "max_seq_len, lower data.max_combined_chars, or run "
                "scripts/token_budget.py for the real measured distribution before "
                "deciding which. See ADR-0003."
            )


def _build_section(cls: type, raw: dict[str, Any] | None) -> Any:
    raw = raw or {}
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} field(s): {sorted(unknown)}")
    return cls(**raw)


def load_config(path: str | Path) -> Config:
    """Load and validate a Config from a YAML file.

    Raises ValueError (via each section's __post_init__, or for unknown
    keys) on anything malformed, rather than silently accepting it.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    known_sections = {"data", "model", "lora"}
    unknown_sections = set(raw) - known_sections
    if unknown_sections:
        raise ValueError(f"Unknown config section(s): {sorted(unknown_sections)}")

    return Config(
        data=_build_section(DataConfig, raw.get("data")),
        model=_build_section(ModelConfig, raw.get("model")),
        lora=_build_section(LoRAConfig, raw.get("lora")),
    )
