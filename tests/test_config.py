import pytest

from lora_bench.config import Config, DataConfig, LoRAConfig, ModelConfig, load_config


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


@pytest.mark.parametrize(
    "kwargs",
    [
        {"train_ratio": 0.5, "val_ratio": 0.3, "test_ratio": 0.3},  # sums to 1.1
        {"min_chars": 100, "max_chars": 50},  # max <= min
        {"max_examples": 0},
        {"languages": []},
    ],
)
def test_data_config_validation_rejects_bad_values(kwargs):
    with pytest.raises(ValueError):
        DataConfig(**kwargs)


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
