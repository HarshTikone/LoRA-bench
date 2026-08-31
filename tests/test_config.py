import pytest

from lora_bench.config import (
    CHARS_PER_TOKEN_ESTIMATE,
    INSTRUCTION_OVERHEAD_CHARS_ESTIMATE,
    Config,
    DataConfig,
    LoRAConfig,
    ModelConfig,
    load_config,
)


def test_load_default_config_matches_dataclass_defaults():
    cfg = load_config("configs/default.yaml")
    assert cfg == Config()


def test_load_config_unknown_top_level_section_raises(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("bogus_section:\n  x: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown config section"):
        load_config(p)


def test_load_config_unknown_field_in_section_raises(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("data:\n  not_a_real_field: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown DataConfig field"):
        load_config(p)


def test_load_config_empty_file_uses_all_defaults(tmp_path):
    p = tmp_path / "empty.yaml"
    p.write_text("", encoding="utf-8")
    assert load_config(p) == Config()


@pytest.mark.parametrize("contents", ["[]\n", "42\n", "plain-string\n"])
def test_load_config_rejects_non_mapping_root(tmp_path, contents):
    p = tmp_path / "bad.yaml"
    p.write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError, match="Config root must be a mapping"):
        load_config(p)


@pytest.mark.parametrize("contents", ["data: []\n", "model: value\n", "lora: 3\n"])
def test_load_config_rejects_non_mapping_section(tmp_path, contents):
    p = tmp_path / "bad.yaml"
    p.write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError, match="section must be a mapping"):
        load_config(p)


@pytest.mark.parametrize(
    "contents,match",
    [
        ("data:\n  languages: Python\n", "languages"),
        ("data:\n  drop_noop_pairs: 1\n", "drop_noop_pairs"),
        ("data:\n  max_chars: '4000'\n", "max_chars"),
        ("data:\n  train_ratio: .nan\n", "train_ratio"),
        ("model:\n  max_seq_len: 0\n", "max_seq_len"),
        ("model:\n  base_model: ''\n", "base_model"),
        ("lora:\n  target_modules: q_proj\n", "target_modules"),
        ("lora:\n  dropout: .inf\n", "dropout"),
    ],
)
def test_load_config_rejects_malformed_field_types(tmp_path, contents, match):
    p = tmp_path / "bad.yaml"
    p.write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        load_config(p)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"train_ratio": 0.5, "val_ratio": 0.3, "test_ratio": 0.3},  # sums to 1.1
        {"min_chars": 100, "max_chars": 50},  # max <= min
        {"max_examples": 0},
        {"languages": []},
        {"min_chars": 100, "max_combined_chars": 100},  # combined < 2 * min_chars
    ],
)
def test_data_config_validation_rejects_bad_values(kwargs):
    with pytest.raises(ValueError):
        DataConfig(**kwargs)


# --- Config cross-section validation: max_combined_chars vs. max_seq_len ---


def test_config_rejects_combined_chars_that_cannot_fit_seq_len():
    with pytest.raises(ValueError, match="max_seq_len"):
        Config(
            data=DataConfig(max_combined_chars=100_000),
            model=ModelConfig(max_seq_len=1024),
        )


def test_config_accepts_combined_chars_at_the_budget_boundary():
    max_seq_len = 1024
    # Exactly at the boundary: worst_case_tokens == max_seq_len.
    max_combined_chars = (
        round(max_seq_len * CHARS_PER_TOKEN_ESTIMATE) - INSTRUCTION_OVERHEAD_CHARS_ESTIMATE
    )
    cfg = Config(
        data=DataConfig(max_combined_chars=max_combined_chars),
        model=ModelConfig(max_seq_len=max_seq_len),
    )
    assert cfg.data.max_combined_chars == max_combined_chars

    with pytest.raises(ValueError, match="max_seq_len"):
        Config(
            data=DataConfig(max_combined_chars=max_combined_chars + 1),
            model=ModelConfig(max_seq_len=max_seq_len),
        )


def test_default_config_satisfies_its_own_token_budget():
    # The shipped defaults must not violate the invariant they're supposed
    # to demonstrate — this would have caught the original bug directly.
    Config()  # must not raise


@pytest.mark.parametrize(
    "kwargs",
    [
        {"r": 0},
        {"alpha": -1},
        {"dropout": 1.0},
        {"target_modules": []},
    ],
)
def test_lora_config_validation_rejects_bad_values(kwargs):
    with pytest.raises(ValueError):
        LoRAConfig(**kwargs)


def test_config_sections_are_independently_overridable(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text(
        "lora:\n  r: 8\n  alpha: 16\nmodel:\n  base_model: some/other-model\n",
        encoding="utf-8",
    )
    cfg = load_config(p)
    assert cfg.lora.r == 8
    assert cfg.lora.alpha == 16
    assert cfg.model.base_model == "some/other-model"
    assert cfg.data == DataConfig()  # untouched section keeps its defaults
    assert cfg.model.max_seq_len == ModelConfig().max_seq_len
