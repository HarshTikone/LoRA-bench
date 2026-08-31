import pytest

from lora_bench.data.schema import FixDiffExample
from lora_bench.training import (
    IGNORE_INDEX,
    CompletionOnlyDataCollator,
    TokenizationDropReason,
    tokenize_training_example,
)


class FakeTokenizer:
    eos_token_id = 99
    pad_token_id = 0
    padding_side = "right"

    def __init__(self, *, full_ids=None, prompt_ids=None):
        self.prompt_ids = prompt_ids if prompt_ids is not None else [10, 11, 12]
        self.full_ids = full_ids if full_ids is not None else [10, 11, 12, 20, 21, 99]

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        return "prompt" if add_generation_prompt else "full"

    def __call__(self, text, **kwargs):
        assert kwargs == {
            "add_special_tokens": False,
            "truncation": False,
            "padding": False,
        }
        return {"input_ids": self.prompt_ids if text == "prompt" else self.full_ids}


def example() -> FixDiffExample:
    return FixDiffExample(
        cve_id="CVE-1",
        cwe_id="CWE-1",
        cwe_name="weakness",
        severity="HIGH",
        language="Python",
        repo_url="https://example.test/repo",
        commit_hash="abc",
        instruction="Fix it",
        input="bad code",
        output="fixed code",
        diff="",
    )


def test_tokenize_training_example_masks_only_prompt():
    outcome = tokenize_training_example(example(), FakeTokenizer(), max_seq_len=6)
    assert outcome.kept
    assert outcome.drop_reason is None
    assert outcome.token_count == 6
    assert outcome.record == {
        "input_ids": [10, 11, 12, 20, 21, 99],
        "attention_mask": [1, 1, 1, 1, 1, 1],
        "labels": [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 20, 21, 99],
    }


def test_tokenize_training_example_drops_over_budget_without_truncating():
    outcome = tokenize_training_example(example(), FakeTokenizer(), max_seq_len=5)
    assert outcome.record is None
    assert outcome.drop_reason == TokenizationDropReason.OVER_BUDGET
    assert outcome.token_count == 6


def test_tokenize_training_example_keeps_exact_budget_boundary():
    outcome = tokenize_training_example(example(), FakeTokenizer(), max_seq_len=6)
    assert outcome.kept
    assert len(outcome.record["input_ids"]) == 6


def test_tokenize_training_example_rejects_boundary_mismatch_loudly():
    tokenizer = FakeTokenizer(prompt_ids=[1, 2], full_ids=[1, 3, 20, 99])
    with pytest.raises(ValueError, match="assistant-label boundary"):
        tokenize_training_example(example(), tokenizer, max_seq_len=10)


def test_tokenize_training_example_drops_missing_eos():
    tokenizer = FakeTokenizer(full_ids=[10, 11, 12, 20])
    outcome = tokenize_training_example(example(), tokenizer, max_seq_len=10)
    assert outcome.drop_reason == TokenizationDropReason.MISSING_EOS


def test_tokenize_training_example_drops_empty_assistant_turn():
    tokenizer = FakeTokenizer(full_ids=[10, 11, 12], prompt_ids=[10, 11, 12])
    outcome = tokenize_training_example(example(), tokenizer, max_seq_len=10)
    assert outcome.drop_reason == TokenizationDropReason.NO_ASSISTANT_TOKENS


def test_completion_only_collator_preserves_labels_and_masks_padding():
    collator = CompletionOnlyDataCollator(
        tokenizer=FakeTokenizer(),
        pad_to_multiple_of=4,
        as_tensors=False,
    )
    batch = collator(
        [
            {
                "input_ids": [1, 2, 3],
                "attention_mask": [1, 1, 1],
                "labels": [IGNORE_INDEX, 2, 3],
            },
            {
                "input_ids": [4, 5],
                "attention_mask": [1, 1],
                "labels": [IGNORE_INDEX, 5],
            },
        ]
    )
    assert batch["input_ids"] == [[1, 2, 3, 0], [4, 5, 0, 0]]
    assert batch["attention_mask"] == [[1, 1, 1, 0], [1, 1, 0, 0]]
    assert batch["labels"] == [
        [IGNORE_INDEX, 2, 3, IGNORE_INDEX],
        [IGNORE_INDEX, 5, IGNORE_INDEX, IGNORE_INDEX],
    ]


def test_completion_only_collator_rejects_feature_without_assistant_labels():
    collator = CompletionOnlyDataCollator(FakeTokenizer(), as_tensors=False)
    with pytest.raises(ValueError, match="assistant label"):
        collator(
            [
                {
                    "input_ids": [1],
                    "attention_mask": [1],
                    "labels": [IGNORE_INDEX],
                }
            ]
        )
