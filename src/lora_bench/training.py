"""Testable training preprocessing shared by the Colab notebook.

This module deliberately does not import torch or transformers at import
time. Repo-side tests stay CPU/lightweight, while the collator creates torch
tensors lazily when the notebook actually calls it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from lora_bench.data.cvefixes import to_chat_messages
from lora_bench.data.schema import FixDiffExample

IGNORE_INDEX = -100


class TokenizationDropReason(str, Enum):
    """Stable reasons an example cannot safely be used for training."""

    OVER_BUDGET = "over_budget"
    NO_ASSISTANT_TOKENS = "no_assistant_tokens"
    MISSING_EOS = "missing_eos"


@dataclass(frozen=True)
class TokenizationOutcome:
    """Result of tokenizing one example without ever truncating it."""

    record: dict[str, list[int]] | None
    drop_reason: TokenizationDropReason | None
    token_count: int

    @property
    def kept(self) -> bool:
        return self.record is not None


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        padding=False,
    )
    return list(encoded["input_ids"])


def tokenize_training_example(
    example: FixDiffExample,
    tokenizer: Any,
    max_seq_len: int,
) -> TokenizationOutcome:
    """Create completion-only causal-LM labels for one chat example.

    The prompt is rendered with the model's generation marker, then checked
    as an exact text prefix of the full user/assistant conversation. The
    exact assistant suffix is tokenized separately and appended to the prompt
    IDs. This preserves the generation-time prompt tokenization even when a
    BPE tokenizer would otherwise merge the prompt's final token with the
    first character of the assistant response.
    """
    if type(max_seq_len) is not int or max_seq_len <= 0:
        raise ValueError(f"max_seq_len must be a positive integer, got {max_seq_len!r}")

    messages = to_chat_messages(example)
    prompt_text = tokenizer.apply_chat_template(
        messages[:1],
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    if not full_text.startswith(prompt_text):
        raise ValueError(
            "The rendered generation prompt is not an exact text prefix of the full "
            "conversation; refusing to guess the assistant-label boundary."
        )

    assistant_text = full_text[len(prompt_text) :]
    prompt_ids = _token_ids(tokenizer, prompt_text)
    assistant_ids = _token_ids(tokenizer, assistant_text)
    full_ids = prompt_ids + assistant_ids
    token_count = len(full_ids)

    if token_count > max_seq_len:
        return TokenizationOutcome(None, TokenizationDropReason.OVER_BUDGET, token_count)
    if not messages[-1]["content"].strip() or not assistant_ids:
        return TokenizationOutcome(
            None,
            TokenizationDropReason.NO_ASSISTANT_TOKENS,
            token_count,
        )

    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    eos_token = getattr(tokenizer, "eos_token", None)
    if (
        eos_token_id is None
        or not eos_token
        or not assistant_text.rstrip().endswith(eos_token)
        or eos_token_id not in assistant_ids
    ):
        return TokenizationOutcome(None, TokenizationDropReason.MISSING_EOS, token_count)

    labels = [IGNORE_INDEX] * len(prompt_ids) + full_ids[len(prompt_ids) :]
    if not any(label != IGNORE_INDEX for label in labels):
        return TokenizationOutcome(
            None,
            TokenizationDropReason.NO_ASSISTANT_TOKENS,
            token_count,
        )

    return TokenizationOutcome(
        record={
            "input_ids": full_ids,
            "attention_mask": [1] * token_count,
            "labels": labels,
        },
        drop_reason=None,
        token_count=token_count,
    )


@dataclass
class CompletionOnlyDataCollator:
    """Right-pad precomputed completion-only records without replacing labels.

    ``as_tensors=False`` exists for lightweight repo-side tests. Colab uses
    the default and therefore receives the torch tensors expected by Trainer.
    """

    tokenizer: Any
    pad_to_multiple_of: int | None = None
    as_tensors: bool = True

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        if not features:
            raise ValueError("Cannot collate an empty batch")
        if getattr(self.tokenizer, "padding_side", "right") != "right":
            raise ValueError("CompletionOnlyDataCollator requires right padding")

        lengths = {len(feature["input_ids"]) for feature in features}
        for feature in features:
            n = len(feature["input_ids"])
            if len(feature["attention_mask"]) != n or len(feature["labels"]) != n:
                raise ValueError("input_ids, attention_mask, and labels must have equal lengths")
            if not any(label != IGNORE_INDEX for label in feature["labels"]):
                raise ValueError("Every training feature must contain an assistant label")

        max_len = max(lengths)
        if self.pad_to_multiple_of:
            if self.pad_to_multiple_of <= 0:
                raise ValueError("pad_to_multiple_of must be positive")
            multiple = self.pad_to_multiple_of
            max_len = ((max_len + multiple - 1) // multiple) * multiple

        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            raise ValueError("tokenizer.pad_token_id must be set before collation")

        batch: dict[str, list[list[int]]] = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }
        for feature in features:
            pad_len = max_len - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + [pad_token_id] * pad_len)
            batch["attention_mask"].append(feature["attention_mask"] + [0] * pad_len)
            batch["labels"].append(feature["labels"] + [IGNORE_INDEX] * pad_len)

        if not self.as_tensors:
            return batch

        import torch

        return {name: torch.tensor(values, dtype=torch.long) for name, values in batch.items()}
