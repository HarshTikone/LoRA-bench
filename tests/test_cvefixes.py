"""Unit tests for the CVEfixes data-prep pipeline.

All tests run against the bundled fixture (tests/fixtures/sample_raw_records.json)
or synthetic in-memory records — no network access, per pyproject.toml's
`-m "not network"` default (see also the `network` marker for the one live
smoke check in test_cvefixes_network.py-equivalent, which we don't add here;
manual verification against the real HF Hub is documented in ADR.md instead).
"""

import json

import pytest

from lora_bench.config import DataConfig
from lora_bench.data.cvefixes import (
    FIXTURE_PATH,
    DropReason,
    build_dataset,
    clean_record,
    filter_records,
    load_raw_dataset,
    normalize_severity,
    parse_cve_description,
    read_jsonl,
    split_examples,
    to_example,
    write_jsonl,
)
from lora_bench.data.schema import FixDiffExample


def load_fixture() -> list[dict]:
    with FIXTURE_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


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
    rec = make_raw(vulnerable_code="a" * 1600, fixed_code="b" * 1600)  # each < 2000, sum 3200 > 3000
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


@pytest.mark.parametrize("bad_value", [float("nan"), 3.14, 42, ["a", "b"], {"x": 1}])
@pytest.mark.parametrize("field", ["vulnerable_code", "fixed_code", "language", "cve_id", "hash", "repo_url"])
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
    rec, reason = clean_record(make_raw(cwe_id=120, cwe_name=None, diff_with_context=["a", "b"]), DataConfig())
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

    # Fixture is built so exactly these 4 survive: a clean Python, C, and
    # JavaScript row, plus a Java row with missing cwe/severity metadata
    # (exercises the UNKNOWN-defaulting path). The rest are deliberately
    # invalid: an exact duplicate, a no-op diff, an empty-code row, a
    # disallowed language (Ruby), an oversized blob, and an undersized pair.
    assert kept_ids == {"CVE-2023-0001", "CVE-2023-0002", "CVE-2023-0003", "CVE-2023-0009"}
    assert len(cleaned) == 4  # confirms the exact duplicate was deduped, not just filtered
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
    assert len(examples) == 4
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


# --- load_raw_dataset (network call itself is mocked out) -----------------


def test_load_raw_dataset_passes_through_revision(monkeypatch):
    captured = {}

    def fake_load_dataset(name, split, revision):
        captured["name"] = name
        captured["split"] = split
        captured["revision"] = revision
        return [{"a": 1}, {"a": 2}]

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)

    rows = list(load_raw_dataset("some/dataset", "train", revision="deadbeef123"))
    assert rows == [{"a": 1}, {"a": 2}]
    assert captured == {"name": "some/dataset", "split": "train", "revision": "deadbeef123"}


def test_default_data_config_pins_a_real_looking_revision():
    # Guards against "revision: None" (follows branch head, defeating the
    # point) or an invented placeholder slipping back in.
    revision = DataConfig().revision
    assert isinstance(revision, str)
    assert len(revision) == 40  # a full git commit SHA, not a branch name
    assert all(c in "0123456789abcdef" for c in revision)


# --- write_jsonl / read_jsonl ------------------------------------------------


def test_jsonl_round_trip(tmp_path):
    examples = _dummy_examples(3)
    path = tmp_path / "out.jsonl"
    write_jsonl(examples, path)

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3

    loaded = read_jsonl(path)
    assert [e.to_dict() for e in loaded] == [e.to_dict() for e in examples]
