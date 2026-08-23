"""Unit tests for the CVEfixes data-prep pipeline.

All tests run against the bundled sample (src/lora_bench/data/sample_records.json,
loaded via _load_fixture_records) or synthetic in-memory records — no network
access, per pyproject.toml's `-m "not network"` default (the `network` marker
is reserved for a live-HF-Hub test we don't have; manual verification against
the real dataset is documented in ADR.md instead).
"""

import json
import os
import random

import pytest

from lora_bench.config import Config, DataConfig
from lora_bench.data.cvefixes import (
    DropReason,
    _load_env_file,
    _load_fixture_records,
    build_dataset,
    clean_record,
    filter_records,
    load_raw_dataset,
    main,
    normalize_severity,
    parse_cve_description,
    read_jsonl,
    run_pipeline,
    split_examples,
    to_example,
    write_jsonl,
)
from lora_bench.data.schema import FixDiffExample


def load_fixture() -> list[dict]:
    # Reads the same packaged sample_records.json the CLI's --dry-run path
    # reads (via _load_fixture_records) -- one copy of the sample data, not
    # a separate tests/fixtures/ copy that could silently drift from it.
    return _load_fixture_records()


def make_raw(**overrides) -> dict:
    base = {
        "cve_id": "CVE-2099-0000",
        "hash": "deadbeef",
        "repo_url": "https://github.com/example/repo",
        "cve_description": "[{'lang': 'en', 'value': 'desc'}]",
        "severity": "HIGH",
        "cwe_id": "CWE-1",
        "cwe_name": "Some Weakness",
        "language": "Python",
        "vulnerable_code": "a" * 30,
        "fixed_code": "b" * 30,
        "diff_with_context": "diff --git a b",
    }
    base.update(overrides)
    return base


# --- parse_cve_description ---------------------------------------------


def test_parse_cve_description_python_repr_list():
    raw = "[{'lang': 'en', 'value': 'Some CVE text'}]"
    assert parse_cve_description(raw) == "Some CVE text"


def test_parse_cve_description_prefers_english_entry():
    raw = "[{'lang': 'fr', 'value': 'texte'}, {'lang': 'en', 'value': 'text'}]"
    assert parse_cve_description(raw) == "text"


def test_parse_cve_description_plain_string_fallback():
    raw = "just a plain description, not a list repr"
    assert parse_cve_description(raw) == raw


def test_parse_cve_description_empty():
    assert parse_cve_description("") == ""
    assert parse_cve_description(None) == ""


# --- normalize_severity --------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("HIGH", "HIGH"),
        ("medium", "MEDIUM"),
        ("nan", "UNKNOWN"),
        (None, "UNKNOWN"),
        ("", "UNKNOWN"),
        ("CRITICAL", "CRITICAL"),
        ("bogus", "UNKNOWN"),
    ],
)
def test_normalize_severity(raw, expected):
    assert normalize_severity(raw) == expected


# --- clean_record ---------------------------------------------------------


def test_clean_record_valid_passes():
    cfg = DataConfig()
    rec, reason = clean_record(make_raw(), cfg)
    assert reason is None
    assert rec is not None
    assert rec["cve_id"] == "CVE-2099-0000"
    assert rec["severity"] == "HIGH"


def test_clean_record_drops_empty_code():
    cfg = DataConfig()
    assert clean_record(make_raw(vulnerable_code=""), cfg) == (None, DropReason.EMPTY_CODE)
    assert clean_record(make_raw(fixed_code="   "), cfg) == (None, DropReason.EMPTY_CODE)


def test_clean_record_drops_disallowed_language():
    cfg = DataConfig()
    assert clean_record(make_raw(language="Ruby"), cfg) == (None, DropReason.DISALLOWED_LANGUAGE)


def test_clean_record_drops_too_long():
    cfg = DataConfig(max_chars=100)
    assert clean_record(make_raw(vulnerable_code="a" * 200), cfg) == (None, DropReason.TOO_LONG)


def test_clean_record_drops_combined_too_long_even_when_each_field_is_individually_fine():
    # Regression for the max_chars/max_seq_len mismatch in ADR-0003: two
    # fields each comfortably under max_chars can still combine into more
    # code than the model's sequence budget allows.
    cfg = DataConfig(max_chars=2000, max_combined_chars=3000, min_chars=20)
    # each field < 2000 (max_chars), but their sum 3200 > 3000 (max_combined_chars)
    rec = make_raw(vulnerable_code="a" * 1600, fixed_code="b" * 1600)
    assert clean_record(rec, cfg) == (None, DropReason.COMBINED_TOO_LONG)


def test_clean_record_per_field_max_chars_still_binds_independently():
    # max_combined_chars doesn't replace max_chars -- a single oversized
    # field is still dropped even if a generous combined budget would allow it.
    cfg = DataConfig(max_chars=100, max_combined_chars=1_000_000, min_chars=20)
    rec = make_raw(vulnerable_code="a" * 150, fixed_code="b" * 30)
    assert clean_record(rec, cfg) == (None, DropReason.TOO_LONG)


def test_clean_record_drops_too_short():
    cfg = DataConfig(min_chars=50)
    rec, reason = clean_record(make_raw(vulnerable_code="a" * 10, fixed_code="b" * 10), cfg)
    assert (rec, reason) == (None, DropReason.TOO_SHORT)


def test_clean_record_drops_noop_pair_by_default():
    cfg = DataConfig()
    same = "identical code\n" * 3
    assert clean_record(make_raw(vulnerable_code=same, fixed_code=same), cfg) == (
        None,
        DropReason.NOOP_PAIR,
    )


def test_clean_record_keeps_noop_pair_when_disabled():
    cfg = DataConfig(drop_noop_pairs=False)
    same = "identical code\n" * 3
    rec, reason = clean_record(make_raw(vulnerable_code=same, fixed_code=same), cfg)
    assert reason is None
    assert rec is not None


def test_clean_record_defaults_missing_cwe():
    cfg = DataConfig()
    rec, reason = clean_record(make_raw(cwe_id=None, cwe_name=None), cfg)
    assert reason is None
    assert rec is not None
    assert rec["cwe_id"] == "UNKNOWN"
    assert rec["cwe_name"] == "unspecified weakness"


def test_clean_record_drops_missing_identifiers():
    cfg = DataConfig()
    assert clean_record(make_raw(cve_id=None), cfg) == (None, DropReason.MISSING_IDENTIFIER)
    assert clean_record(make_raw(hash=None), cfg) == (None, DropReason.MISSING_IDENTIFIER)
    assert clean_record(make_raw(repo_url=None), cfg) == (None, DropReason.MISSING_IDENTIFIER)


_TEXT_FIELDS = ["vulnerable_code", "fixed_code", "language", "cve_id", "hash", "repo_url"]


@pytest.mark.parametrize("bad_value", [float("nan"), 3.14, 42, ["a", "b"], {"x": 1}])
@pytest.mark.parametrize("field", _TEXT_FIELDS)
def test_clean_record_drops_non_string_typed_fields_instead_of_crashing(field, bad_value):
    # Reproduces the pre-fix crash: a non-string value in a field the
    # pipeline treats as text (e.g. a float from an upstream column that
    # wasn't stringified, the same way `severity` sometimes arrives as the
    # literal string "nan" instead of a real NaN) must not raise.
    rec, reason = clean_record(make_raw(**{field: bad_value}), DataConfig())
    assert rec is None
    assert reason == DropReason.INVALID_FIELD_TYPE


def test_clean_record_coerces_non_string_descriptive_fields_instead_of_dropping():
    # cwe_id/cwe_name/diff are metadata, not join keys or training-critical
    # content, so a wrong type there is coerced with str() rather than
    # treated as a broken row.
    raw = make_raw(cwe_id=120, cwe_name=None, diff_with_context=["a", "b"])
    rec, reason = clean_record(raw, DataConfig())
    assert reason is None
    assert rec is not None
    assert rec["cwe_id"] == "120"
    assert rec["cwe_name"] == "unspecified weakness"
    assert rec["diff"] == "['a', 'b']"


# --- filter_records (against the bundled fixture) -------------------------


def test_filter_records_against_fixture_default_config():
    cfg = DataConfig()
    raw = load_fixture()
    cleaned, drop_counts = filter_records(raw, cfg)
    kept_ids = {r["cve_id"] for r in cleaned}

    # Fixture is built so exactly these 10 survive: a clean Python, C, and
    # JavaScript row, a Java row with missing cwe/severity metadata
    # (exercises the UNKNOWN-defaulting path), and six more valid rows (one
    # per default-allowlisted language) added so the bundled fixture is
    # large enough to produce non-empty val/test splits under the default
    # ratios -- see split_examples' empty-split guard. The rest are
    # deliberately invalid: an exact duplicate, a no-op diff, an empty-code
    # row, a disallowed language (Ruby), an oversized blob, and an
    # undersized pair.
    assert kept_ids == {
        "CVE-2023-0001",
        "CVE-2023-0002",
        "CVE-2023-0003",
        "CVE-2023-0009",
        "CVE-2024-1001",
        "CVE-2024-1002",
        "CVE-2024-1003",
        "CVE-2024-1004",
        "CVE-2024-1005",
        "CVE-2024-1006",
    }
    assert len(cleaned) == 10  # confirms the exact duplicate was deduped, not just filtered
    assert drop_counts == {
        DropReason.DUPLICATE.value: 1,
        DropReason.NOOP_PAIR.value: 1,
        DropReason.EMPTY_CODE.value: 1,
        DropReason.DISALLOWED_LANGUAGE.value: 1,
        DropReason.TOO_LONG.value: 1,
        DropReason.TOO_SHORT.value: 1,
    }


def test_filter_records_normalizes_unknown_metadata():
    cfg = DataConfig()
    cleaned, _ = filter_records(load_fixture(), cfg)
    by_id = {r["cve_id"]: r for r in cleaned}

    assert by_id["CVE-2023-0002"]["severity"] == "UNKNOWN"  # source had literal "nan"
    rec9 = by_id["CVE-2023-0009"]
    assert rec9["severity"] == "UNKNOWN"
    assert rec9["cwe_id"] == "UNKNOWN"
    assert rec9["cwe_name"] == "unspecified weakness"
    # non-list-repr description (fixture's CVE-2023-0009) falls back to the
    # raw string unchanged rather than raising or mangling it
    assert rec9["cve_description"] == (
        "not a python-repr list, just a plain string, "
        "to test the parse_cve_description fallback path"
    )


def test_filter_records_respects_max_examples_deterministically():
    cfg = DataConfig(max_examples=2, seed=7)
    raw = load_fixture()
    first, _ = filter_records(raw, cfg)
    second, _ = filter_records(raw, cfg)
    assert len(first) == 2
    assert [r["cve_id"] for r in first] == [r["cve_id"] for r in second]


def test_filter_records_cap_does_not_inflate_drop_counts():
    # Rows removed by the max_examples cap are a deliberate size-budget
    # choice, not a data-quality drop, so they must not show up in
    # drop_counts (which would otherwise conflate the two).
    cfg = DataConfig(max_examples=1, seed=1)
    _, drop_counts = filter_records(load_fixture(), cfg)
    assert sum(drop_counts.values()) == 6  # same 6 quality-drops as the uncapped run above


# --- to_example / build_dataset -------------------------------------------


def test_to_example_builds_instruction_and_fields():
    cfg = DataConfig()
    cleaned, reason = clean_record(
        make_raw(language="Python", cve_id="CVE-1-1", cwe_id="CWE-9", cwe_name="Foo"), cfg
    )
    assert reason is None
    ex = to_example(cleaned)
    assert isinstance(ex, FixDiffExample)
    # cve_id is kept as metadata (used for the group-aware split and
    # failure-case analysis) but deliberately NOT embedded in the prompt
    # text itself -- see ADR-0005.
    assert ex.cve_id == "CVE-1-1"
    assert "CVE-1-1" not in ex.instruction
    assert "Python" in ex.instruction
    assert "CWE-9" in ex.instruction
    assert ex.input == cleaned["vulnerable_code"]
    assert ex.output == cleaned["fixed_code"]


def test_build_dataset_end_to_end_on_fixture():
    cfg = DataConfig()
    examples, drop_counts = build_dataset(load_fixture(), cfg)
    assert len(examples) == 10
    assert all(isinstance(e, FixDiffExample) for e in examples)
    assert sum(drop_counts.values()) == 6


# --- split_examples ---------------------------------------------------------


def _dummy_examples(n: int) -> list[FixDiffExample]:
    return [
        FixDiffExample(
            cve_id=f"CVE-X-{i}",
            cwe_id="CWE-0",
            cwe_name="n/a",
            severity="UNKNOWN",
            language="Python",
            repo_url="https://example.com/r",
            commit_hash=f"hash{i}",
            instruction="do it",
            input=f"in{i}",
            output=f"out{i}",
            diff="",
        )
        for i in range(n)
    ]


def test_split_examples_sizes_and_full_coverage():
    cfg = DataConfig(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=1)
    examples = _dummy_examples(20)
    train, val, test = split_examples(examples, cfg)
    assert len(train) == 16
    assert len(val) == 2
    assert len(test) == 2
    all_ids = [e.cve_id for e in train + val + test]
    assert sorted(all_ids) == sorted(e.cve_id for e in examples)
    assert len(set(all_ids)) == 20  # no example duplicated across splits


def test_split_examples_deterministic_for_same_seed():
    cfg = DataConfig(seed=123)
    examples = _dummy_examples(20)
    a = split_examples(examples, cfg)
    b = split_examples(examples, cfg)
    assert [e.cve_id for e in a[0]] == [e.cve_id for e in b[0]]
    assert [e.cve_id for e in a[1]] == [e.cve_id for e in b[1]]
    assert [e.cve_id for e in a[2]] == [e.cve_id for e in b[2]]


def _grouped_examples(n_groups: int, seed: int) -> list[FixDiffExample]:
    """Synthetic examples with varied group sizes (some CVEs span several
    rows, most don't) -- unlike _dummy_examples' all-unique cve_ids, this
    is what actually exercises the group-aware split's group-packing logic.
    """
    rng = random.Random(seed)
    examples = []
    for g in range(n_groups):
        size = rng.choice([1, 1, 1, 1, 2, 2, 3])
        for i in range(size):
            examples.append(
                FixDiffExample(
                    cve_id=f"CVE-{g}",
                    cwe_id="CWE-0",
                    cwe_name="n/a",
                    severity="UNKNOWN",
                    language="Python",
                    repo_url=f"https://example.com/r{g}",
                    commit_hash=f"h{g}-{i}",
                    instruction="x",
                    input=f"in{g}-{i}",
                    output=f"out{g}-{i}",
                    diff="",
                )
            )
    return examples


def test_split_examples_never_splits_a_group_across_sets():
    examples = _grouped_examples(n_groups=150, seed=1)
    cfg = DataConfig(seed=1)
    train, val, test = split_examples(examples, cfg)

    train_ids = {e.cve_id for e in train}
    val_ids = {e.cve_id for e in val}
    test_ids = {e.cve_id for e in test}
    assert not (train_ids & val_ids)
    assert not (train_ids & test_ids)
    assert not (val_ids & test_ids)
    # every example still lands somewhere, exactly once
    assert len(train) + len(val) + len(test) == len(examples)


def test_split_examples_raises_on_empty_split_with_positive_ratio():
    # A handful of examples with round() collapsing val/test to zero must
    # not silently succeed -- that's a fine-tune with no validation signal
    # and no error to say why.
    cfg = DataConfig(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=1)
    examples = _dummy_examples(4)
    with pytest.raises(ValueError, match="came out empty"):
        split_examples(examples, cfg)


def test_split_examples_allows_empty_split_when_its_ratio_is_zero():
    cfg = DataConfig(train_ratio=0.5, val_ratio=0.5, test_ratio=0.0, seed=1)
    examples = _dummy_examples(4)
    train, val, test = split_examples(examples, cfg)
    assert test == []  # empty, but ratio was 0 -- not an error
    assert train and val  # sanity: the other two splits did get examples


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_split_examples_ratios_stay_within_tolerance(seed):
    # Group sizes vary, so exact target ratios aren't achievable -- measured
    # empirically to stay within ~1 percentage point at this scale; 0.05
    # gives comfortable margin without being a vacuous assertion.
    examples = _grouped_examples(n_groups=200, seed=seed)
    cfg = DataConfig(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=seed)
    train, val, test = split_examples(examples, cfg)
    total = len(examples)

    assert abs(len(train) / total - 0.8) <= 0.05
    assert abs(len(val) / total - 0.1) <= 0.05
    assert abs(len(test) / total - 0.1) <= 0.05


# --- load_raw_dataset (network call itself is mocked out) -----------------


def test_load_raw_dataset_passes_through_revision(monkeypatch):
    captured = {}
    fake_rows = [make_raw(cve_id="CVE-A"), make_raw(cve_id="CVE-B")]

    def fake_load_dataset(name, split, revision):
        captured["name"] = name
        captured["split"] = split
        captured["revision"] = revision
        return fake_rows

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)

    rows = list(load_raw_dataset("some/dataset", "train", revision="deadbeef123"))
    assert rows == fake_rows
    assert captured == {"name": "some/dataset", "split": "train", "revision": "deadbeef123"}


def test_load_raw_dataset_raises_on_missing_schema_column(monkeypatch):
    bad_row = make_raw()
    del bad_row["vulnerable_code"]  # simulate an upstream column rename/removal

    monkeypatch.setattr("datasets.load_dataset", lambda name, split, revision: [bad_row])

    with pytest.raises(ValueError, match="vulnerable_code"):
        list(load_raw_dataset("some/dataset", "train"))


def test_load_raw_dataset_empty_source_yields_nothing(monkeypatch):
    monkeypatch.setattr("datasets.load_dataset", lambda name, split, revision: [])
    assert list(load_raw_dataset("some/dataset", "train")) == []


def test_default_data_config_pins_a_real_looking_revision():
    # Guards against "revision: None" (follows branch head, defeating the
    # point) or an invented placeholder slipping back in.
    revision = DataConfig().revision
    assert isinstance(revision, str)
    assert len(revision) == 40  # a full git commit SHA, not a branch name
    assert all(c in "0123456789abcdef" for c in revision)


# --- manifest ---------------------------------------------------------------


def test_run_pipeline_writes_a_manifest_with_expected_shape(tmp_path):
    cfg = Config()
    stats = run_pipeline(cfg, load_fixture(), tmp_path)

    manifest_path = tmp_path / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert "generated_at" in manifest
    assert manifest["lora_bench_version"]
    assert "git_sha" in manifest  # value may legitimately be None outside a git checkout

    assert manifest["config"]["data"]["dataset_name"] == cfg.data.dataset_name
    assert manifest["config"]["data"]["revision"] == cfg.data.revision
    assert manifest["config"]["model"]["base_model"] == cfg.model.base_model
    assert manifest["config"]["lora"]["r"] == cfg.lora.r

    counts = manifest["counts"]
    assert counts == {
        "total": stats["total"],
        "train": stats["train"],
        "val": stats["val"],
        "test": stats["test"],
    }
    assert counts["total"] == counts["train"] + counts["val"] + counts["test"]
    assert manifest["drop_counts"] == stats["drop_counts"]


# --- .env loading ------------------------------------------------------------


def test_load_env_file_populates_hf_token_from_dotenv(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("HF_TOKEN=test-token-value\n", encoding="utf-8")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.chdir(tmp_path)
    try:
        _load_env_file()
        assert os.environ.get("HF_TOKEN") == "test-token-value"
    finally:
        monkeypatch.delenv("HF_TOKEN", raising=False)


def test_load_env_file_is_a_noop_without_a_dotenv_file(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.chdir(tmp_path)
    _load_env_file()  # must not raise just because no .env exists here
    assert "HF_TOKEN" not in os.environ


# --- main() / CLI -----------------------------------------------------------


def test_main_dry_run_succeeds_and_returns_zero(tmp_path, capsys):
    exit_code = main(["--dry-run", "--out-dir", str(tmp_path)])
    assert exit_code == 0
    assert (tmp_path / "train.jsonl").exists()
    assert (tmp_path / "val.jsonl").exists()
    assert (tmp_path / "test.jsonl").exists()
    assert (tmp_path / "manifest.json").exists()

    out = capsys.readouterr().out
    assert "Wrote 10 examples" in out
    assert "manifest.json" in out


def test_main_returns_nonzero_below_min_examples(tmp_path, capsys):
    # The fixture yields exactly 10 examples under default filtering (see
    # test_filter_records_against_fixture_default_config) -- 11 is
    # unreachable, so this exercises the --min-examples floor specifically,
    # distinct from the true-zero-yield path exercised below.
    exit_code = main(["--dry-run", "--min-examples", "11", "--out-dir", str(tmp_path)])
    assert exit_code == 1
    assert "ERROR" in capsys.readouterr().err


def test_main_returns_nonzero_cleanly_on_total_filtering_collapse(tmp_path, capsys):
    # Every fixture row gets dropped (no allowlisted language matches),
    # which makes split_examples' empty-split guard raise ValueError before
    # main() ever reaches its --min-examples check. main() must still exit
    # 1 with a clean stderr message, not propagate a raw traceback -- the
    # two "pipeline produced nothing" failure modes should look the same to
    # an operator, e.g. someone watching a Colab run's output.
    config_path = tmp_path / "impossible.yaml"
    config_path.write_text("data:\n  languages: [Rust]\n", encoding="utf-8")

    exit_code = main(
        ["--dry-run", "--config", str(config_path), "--out-dir", str(tmp_path / "out")]
    )
    assert exit_code == 1
    assert "ERROR" in capsys.readouterr().err


# --- write_jsonl / read_jsonl ------------------------------------------------


def test_jsonl_round_trip(tmp_path):
    examples = _dummy_examples(3)
    path = tmp_path / "out.jsonl"
    write_jsonl(examples, path)

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3

    loaded = read_jsonl(path)
    assert [e.to_dict() for e in loaded] == [e.to_dict() for e in examples]
