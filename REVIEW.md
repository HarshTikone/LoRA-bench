# Review — Day 1 hardening pass (98b0388..53af83d) — 2026-08-22

Scope reviewed: the Day 1 hardening pass — 15 commits (`3073923`..`53af83d`,
range `98b0388..53af83d`) applied after a first self-review (recorded
previously in this file) closed Day 1 with "no blocking items," and a
separate, deeper review then found real defects that first pass missed.
This review overwrites that earlier Day 1 review per this agent's standing
instruction that `REVIEW.md` reflects only the latest review, not a running
log — see the note at the end of this file for what that supersession
means concretely here.

All 13 claimed fixes were checked against the actual diff (`git show
<sha> -p` per commit), not just the hardening-pass summary. Two of the
highest-severity claims (crash bug, data-corruption bug) plus the
group-aware split were read line-by-line; the remainder were verified by
reading the real diff and, where behavior was non-obvious, exercising it
directly.

## Passed

- **Claim 1 — `clean_record` crash fixed (`3073923`).** Confirmed:
  `clean_record` (`src/lora_bench/data/cvefixes.py:116-182`) now
  type-checks every identifier/language/code field and returns
  `DropReason.INVALID_FIELD_TYPE` instead of calling `.strip()`/`.lower()`
  on a non-string value; descriptive-only fields (`cwe_id`, `cwe_name`,
  `diff_with_context`) are coerced with `str()` instead, matching the
  stated rationale (coercing a float into a code field would train on
  garbage). `test_clean_record_drops_non_string_typed_fields_instead_of_crashing`
  (`tests/test_cvefixes.py`) is parametrized over 5 bad-type values x 6
  fields (30 cases) and reproduces the pre-fix crash inputs directly.
  `filter_records`/`build_dataset`/`run_pipeline` now thread a
  `Counter[DropReason]` end to end into `main()`'s printed summary — real
  logic change, not just a signature change (verified the counter
  excludes cap-driven drops, per `test_filter_records_cap_does_not_inflate_drop_counts`).

- **Claim 2 — combined character budget vs. `max_seq_len` (`4d28bf8`).**
  Confirmed: `DataConfig.max_combined_chars` (default 2600, checked in
  `clean_record` at `cvefixes.py:157` alongside the existing per-field
  `max_chars`) plus a new `Config.__post_init__` cross-section validator
  (`src/lora_bench/config.py:174-198`) that rejects any config where
  `max_combined_chars` + a documented instruction-overhead estimate,
  divided by a documented chars/token estimate, exceeds `max_seq_len`.
  Verified this validator actually fires on real config construction, not
  just in isolation: `load_config` (`config.py:210-229`) builds each
  section then calls `Config(data=..., model=..., lora=...)`, which
  invokes `__post_init__` — confirmed by `test_default_config_satisfies_its_own_token_budget`
  passing and `test_config_rejects_combined_chars_that_cannot_fit_seq_len`
  raising. ADR-0003 documents real measured numbers from
  `scripts/token_budget.py` (7.5% -> 0.2% truncation-risk rate,
  ~1,412 -> ~448 avg chars lost) and is honest about the estimate's
  limitations (conservative chars/token constant, not a hard guarantee,
  explicitly flags a residual 0.2% as a Day-2 follow-up rather than
  claiming full closure).

- **Claim 3 — dataset revision pinned (`32c55be`).** Confirmed:
  `DataConfig.revision` defaults to a real 40-hex-char SHA
  (`d4f5c4ea65329d9ccbb8a3b3149e5d06eda5edb2`), threaded through
  `load_raw_dataset(dataset_name, split, revision)` into
  `datasets.load_dataset(..., revision=revision)`, and into
  `configs/default.yaml`. `test_default_data_config_pins_a_real_looking_revision`
  checks it's a genuine 40-char hex string, guarding against a
  reintroduced `None`/placeholder. `scripts/scan_dataset.py` deliberately
  keeps its own `--revision` default of `None` (branch head) — documented
  in its `--help` text as intentional for a diagnostic tool, distinct from
  the pipeline's pinned default.

- **Claim 4 — group-aware train/val/test split by `cve_id` (`8e5d8f9`,
  `17e85ea`).** Confirmed: `split_examples` (`cvefixes.py:256-314`) now
  groups examples by `cve_id`, shuffles group order by seed, and assigns
  each group wholly via a largest-remaining-need greedy heuristic —
  verified no group is ever split across sets
  (`test_split_examples_never_splits_a_group_across_sets`) and ratios stay
  within a measured tolerance across 5 seeds. I additionally reasoned
  through and confirmed (not just trusted the test) that the zero-ratio
  case is correct by construction, not by luck: whenever a split's
  configured ratio is 0, the invariant `train_diff + val_diff =
  (train_target + val_target) - (train_count + val_count) >= 0` (given
  ratios sum to 1) guarantees at least one of the two positive-ratio
  splits always has a non-negative remaining-need at every assignment
  step, so the zero-ratio split's diff of exactly 0 never wins the
  tie-broken `max()` — `test_split_examples_allows_empty_split_when_its_ratio_is_zero`
  passes for a structural reason, not coincidentally.

- **Claim 5 — CVE ID dropped from the training prompt (`f4f5ee8`).**
  Confirmed: `INSTRUCTION_TEMPLATE` (`cvefixes.py:67-71`) no longer
  contains `{cve_id}`; `to_example` (`cvefixes.py:227-233`) no longer
  passes it in. `cve_id` remains a `FixDiffExample` field (used by the
  group-aware split and left available for failure-case analysis).
  `test_to_example_builds_instruction_and_fields` explicitly asserts
  `"CVE-1-1" not in ex.instruction`. ADR-0005's rationale (memorization
  risk vs. CWE as a coarse, non-memorizable, realistic signal) is
  substantive, not just asserted.

- **Claim 6 — `RAW_FIELDS` enforced, non-zero exit on low yield
  (`21810cf`).** Confirmed: `_validate_raw_schema`
  (`cvefixes.py:337-356`) checks the first row against `RAW_FIELDS` inside
  `load_raw_dataset` and raises `ValueError` naming missing columns;
  `main()` now returns an int exit code and exits 1 when
  `stats["total"] < args.min_examples` (default 1). See **Questionable**
  below for a real gap in how this composes with Claim 7 on the actual
  worst case (total collapse to zero).

- **Claim 7 — raise on silently empty split (`17e85ea`).** Confirmed:
  `split_examples` (`cvefixes.py:302-312`) raises `ValueError` naming the
  split and counts whenever a split with `ratio > 0` comes out empty, and
  allows it when the configured ratio is genuinely 0
  (`test_split_examples_allows_empty_split_when_its_ratio_is_zero`). The
  bundled fixture was grown from 4 to 10 records specifically because this
  guard immediately fired against the old 4-record `--dry-run` fixture
  under default ratios — a real, not hypothetical, catch.

- **Claim 8 — `--dry-run` sample moved to package data (`c54b26d`).**
  Confirmed: `sample_records.json` now lives at
  `src/lora_bench/data/sample_records.json`, declared in
  `pyproject.toml`'s `[tool.setuptools.package-data]`
  (`"lora_bench.data" = ["sample_records.json"]`), loaded via
  `importlib.resources.files("lora_bench.data").joinpath(...)`
  (`cvefixes.py:395-410`) instead of `Path(__file__).resolve().parents[3]`.
  Tests now load the same file through the same `_load_fixture_records()`
  function (no second copy to drift). This is a correctness fix that
  matters: the old scheme would have broken `--dry-run` — the first
  command a new reader runs — under a real `pip install .` wheel.

- **Claim 9 — `scan_dataset.py` calls the real `filter_records`
  (`0626436`).** Confirmed: `scan()` (`scripts/scan_dataset.py`) now
  imports and calls `lora_bench.data.cvefixes.filter_records` directly
  instead of re-implementing the empty-code/no-op checks inline, and
  reports the real `DropReason` counter. The `--limit 0` divide-by-zero is
  fixed by an early return (`if n == 0: print(...); return`) before any
  division — read directly, this guard sits before every place `n` is
  used as a divisor. ADR-0002 was updated with a fresh, script-reproduced
  drop-reason breakdown and honestly explains why it isn't numerically
  comparable to the old ad-hoc "~7%" figure (a conditional vs. marginal
  rate) rather than silently presenting a different number as if it
  corrected the old one.

- **Claim 10 — `manifest.json` (`87db850`).** Confirmed:
  `build_manifest`/`run_pipeline` (`cvefixes.py:434-483`) write
  `manifest.json` alongside the splits with `generated_at`,
  `lora_bench_version` (from `lora_bench.__version__`), `git_sha`
  (best-effort via `git rev-parse HEAD`, `None` — not a placeholder — on
  failure/non-git-checkout, wrapped in `try/except Exception`), the full
  resolved config via `dataclasses.asdict(cfg)`, per-split counts, and
  `drop_counts`. `test_run_pipeline_writes_a_manifest_with_expected_shape`
  and `test_main_dry_run_succeeds_and_returns_zero` both check its
  presence and shape.

- **Claim 11 — `.env` actually loaded (`627fb49`).** Confirmed:
  `_load_env_file()` (`cvefixes.py:486-509`) calls
  `load_dotenv(find_dotenv(usecwd=True))`, explicitly not the bare
  `load_dotenv()` default — the diff and its comment correctly identify
  that the default searches from the calling module's file location
  (which would always resolve to this repo's own `.env` in dev, and find
  nothing under an installed wheel), not the process's cwd.
  `test_load_env_file_populates_hf_token_from_dotenv` verifies this with a
  real temp `.env` + `monkeypatch.chdir`, not a mock of `dotenv` itself.

- **Claim 12 — deps in `pyproject.toml`, ruff CI gate (`12765ca`).**
  Confirmed: `[project.dependencies]` in `pyproject.toml` now declares
  `datasets`, `huggingface_hub`, `python-dotenv`, `PyYAML`;
  `requirements-repo.txt` is now literally `-e .[dev]` plus comments — a
  single source of truth. `.github/workflows/tests.yml` gained a separate
  `lint` job running `ruff check .` and `ruff format --check .`. **I ran
  this myself**: `.venv/Scripts/python.exe -m ruff check .` ->
  `All checks passed!`; `.venv/Scripts/python.exe -m ruff format --check
  src tests scripts` -> `9 files already formatted`. Both match the
  claimed clean state.

- **Claim 13 — LICENSE + stale `.gitignore` comment (`aaf3421`,
  `d85266d`).** Confirmed: root `LICENSE` is a standard MIT license,
  README updated to reference it. `.gitignore`'s comment above `/data/`
  no longer references the never-existing `scripts/prepare_data.py` or
  the now-removed `tests/fixtures/` path; it correctly points at
  `python -m lora_bench.data.cvefixes` and
  `src/lora_bench/data/sample_records.json`.

- **Tests actually exercise the logic, and pass.** Ran
  `.venv/Scripts/python.exe -m pytest -q`: **96 passed in 1.47s**, matching
  the claimed count. Spot-checked (not just counted) that the new/changed
  tests assert real behavior, not smoke checks: exact `DropReason` sets,
  exact surviving `cve_id` sets, boundary-exact config-validator tests
  (`test_config_accepts_combined_chars_at_the_budget_boundary` checks both
  sides of the boundary), and a genuine crash-reproduction parametrization
  for Claim 1.

- **Free-tier / no hard GPU dependency.** `requirements-colab.txt` is
  untouched by this diff (`git diff 98b0388..53af83d --
  requirements-colab.txt` is empty). `scripts/token_budget.py` (new)
  needs `transformers` + network but is explicitly documented as NOT part
  of `requirements-repo.txt`/CI, NOT part of the pytest suite, and to be
  run manually in a throwaway env or Colab — the numbers it produced are
  reported as real output ("was run... measured"), not estimated and
  presented as measured.

- **Secrets.** Grepped the full hardening-pass diff (`git diff
  98b0388..53af83d`) for `hf_[A-Za-z0-9]{10,}` and
  `HF_TOKEN\s*=\s*['"a-zA-Z0-9]` token patterns — the only hit is a
  synthetic `HF_TOKEN=test-token-value` string inside a unit test
  (`tests/test_cvefixes.py`, `test_load_env_file_populates_hf_token_from_dotenv`),
  not a real credential. `.env.example` is unchanged and still contains
  only variable names, no values. `git check-ignore -v .env` confirms
  `.gitignore:1:.env` still matches; `.env` remains untracked
  (`git status` shows it under `!!`, ignored).

- **Scope discipline.** No GPU/notebook/PEFT/bitsandbytes/torch imports
  anywhere in the diff. `ROADMAP.md` was updated to document the
  hardening pass explicitly under Day 1's existing checklist item, not as
  a new day, and its description accurately summarizes what changed.

- **Atomic, real Conventional Commits.** All 15 commits in the range
  (`98b0388..53af83d`; note this is 15, not 14 — `git rev-list --count`
  confirms — a minor discrepancy in the invoking prompt's count, not a
  finding) are correctly typed (`fix:`, `feat:`, `refactor:`, `chore:`,
  `docs:`) and each is a single coherent change with its own tests/docs
  updated in the same commit. None bundle unrelated work.

- **MVP non-negotiables 1-3 / past "just a demo" checklist:** still not
  applicable — this diff is entirely repo-side data-prep hardening, no
  fine-tune, comparison, or report exists yet.

## Questionable

- **A true zero-yield run does not go through the graceful
  `--min-examples` path; it crashes with an uncaught `ValueError`
  instead.** Claim 6 and Claim 7 were each verified correct in isolation,
  but their composition on the actual worst case they're jointly meant to
  guard against — every row getting filtered out — has a gap. I
  reproduced this directly:

  ```
  RAISED: ValueError train split came out empty even though train_ratio=0.8 > 0
  (0 examples across 0 CVE groups). ...
  ```

  (via `main(["--dry-run", "--config", <cfg with languages: [Rust]>, ...])`,
  which drops every fixture row). Because `split_examples` raises
  *before* `main()` ever reaches its `stats["total"] < args.min_examples`
  check, the polished "ERROR: only N example(s) survived filtering..."
  stderr message and clean exit code 1 (`cvefixes.py:560-567`) never
  fires on this input — instead an unhandled exception propagates out of
  `run_pipeline`/`main()` as a raw Python traceback. The process does
  still terminate with a non-zero exit code (Python's default behavior
  for an uncaught exception), so this isn't a silent-success regression
  and isn't data corruption — but it means the two purpose-built guard
  rails produce inconsistent, and in this case worse, operator-facing
  failure modes: a clean one-line message in the "some rows survived but
  too few" case, vs. a traceback several stack frames removed from the
  actual root cause ("every row was filtered out") in the strictly worse
  "zero rows survived" case. Notably, the hardening pass's own test suite
  is aware of exactly this: the comment in
  `test_main_returns_nonzero_below_min_examples`
  (`tests/test_cvefixes.py`) reads "11 is unreachable, so this exercises
  the floor without also tripping split_examples' separate empty-split
  guard (which fires first on a truly empty result and would raise
  instead of returning cleanly)." This was a known, accepted trade-off,
  not an oversight — but it's exactly the kind of gap worth a human
  decision: either catch the `ValueError` from `run_pipeline` in `main()`
  and route it through the same clean stderr-message-and-exit-1 path (so
  an automated Colab/CI log always sees the same shape of failure
  message), or explicitly document in `main()`'s docstring/README that a
  full-collapse run fails via traceback rather than the `--min-examples`
  message, so a future reader doesn't assume the two failure modes are
  equivalent.

- **`--min-examples` default of 1 only catches literal zero-survivor
  runs, not a "near-zero" partial collapse.** The commit message and this
  task's framing describe the fix as guarding against a "near-zero-yield
  run," but the shipped default (`--min-examples 1`) only fails a run
  that produces *zero* examples; a schema/language-distribution shift
  that drops yield from ~2,400 to, say, 5 examples would still exit 0
  unless an operator explicitly raises `--min-examples` for a given run.
  This is minor and easily addressed by the operator (the flag exists and
  is documented), but it's worth noting the default doesn't by itself
  change behavior for a partial collapse — only a total one (and even
  then, see the finding above).

## Blocking

None.

## Stray file (not a code finding)

`D:\LoRA-bench\DAY1_FIX_PROMPT.md` is present in the working tree,
untracked (`git status` shows it as `??`, not ignored), and is not part of
any commit in the reviewed range or created by the hardening-pass work
itself — it reads as the prompt used to kick off this hardening session,
apparently saved to the repo root. It contains no secrets. It's not a
defect and requires no code change, but it's clutter in the repo root the
user likely wants to delete, move outside the repo, or explicitly
`.gitignore` before it gets accidentally committed. Left untouched here.

## Note on this file superseding the prior review

Per this agent's standing instruction, `REVIEW.md` is overwritten
in full on every run — it reflects only the latest review, not a running
log. This file replaces the previous Day 1 review (which covered
`351e77f..1585b68` and closed with "no blocking items," later shown by a
deeper review to have missed the crash bug and the data-corruption bug
this hardening pass fixes). That prior review's content no longer appears
anywhere in this file, as intended — the durable record of what the first
pass missed and why now lives in `ROADMAP.md`'s Day 1 entry and in
ADR-0003/ADR-0004/ADR-0005, not here.

## Resolved after review

The first questionable item (total-filtering-collapse raising an uncaught
`ValueError` instead of routing through the clean `--min-examples` exit
path) was fixed in commit `343d0c6`, chosen over documenting the
discrepancy since routing both failure modes through the same shape is
strictly better UX for an unattended run: `main()` now catches `ValueError`
around `run_pipeline()` and reports it via the same
stderr-message-and-exit-1 path. Added
`test_main_returns_nonzero_cleanly_on_total_filtering_collapse`,
reproducing the reviewer's exact repro case (a config with a disallowed-
only language). Full suite: 97 passed (was 96); `ruff check`/`ruff format
--check` still clean.

The second questionable item (`--min-examples` default of 1 only catches
literal-zero collapse, not a partial one) is left as-is: the flag exists,
is documented, and requires an operator to set an appropriate floor for
their own run size — treated as expected, not a gap to silently paper
over with an arbitrary default.

## Verdict

No blocking items — day may be closed out.
