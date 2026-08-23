# Review — Day 1 (351e77f..1585b68, full project history through HEAD) — 2026-08-22

Scope reviewed: the repo's entire commit history so far — 9 commits from the
root commit `351e77f` (scaffold) through `1585b68` (add reviewer subagent).
Per `ROADMAP.md`, this is all Day 1 work: scaffolding, config, the CVEfixes
data-prep pipeline + tests, ADRs, README/ROADMAP, and CI.

## Passed

- **Scope discipline.** No GPU code, no notebook content, no fine-tuning,
  no quantization, no benchmark harness, and no comparison numbers exist
  anywhere in the diff. `notebooks/README.md` explicitly states it's empty
  by design pending Day 2. `ROADMAP.md` only checks off Day 1. Everything
  added matches the Day 1 bullet in `ROADMAP.md` line-for-line (src layout,
  pytest config, requirements split, ADR-0001/0002, CVEfixes pipeline +
  tests + fixture, typed config, README, CI).
- **Free-tier only.** No paid API keys or paid-tier assumptions in code,
  config, or docs. `.env.example` documents `HF_TOKEN` (free tier) and an
  optional `WANDB_API_KEY` (free tier, explicitly marked optional/not
  required by MVP). `ADR.md` ADR-0001 explicitly picks Qwen2.5-Coder over
  gated-license alternatives (Llama-3.2) specifically to avoid a
  click-through wall breaking unattended Colab runs.
- **Atomic, real Conventional Commits.** All 9 commits are single-purpose
  and correctly typed: `chore` (scaffold, reviewer subagent), `feat`
  (config, data pipeline), `test` (config tests, pipeline tests+fixture),
  `docs` (ADR, README/ROADMAP), `ci` (test workflow). None bundle unrelated
  work; none are placeholder/padding commits.
- **Secrets.** Grepped the full history (`git log -p --all`) for
  `HF_TOKEN\s*=\s*<value>` and `hf_[A-Za-z0-9]{10,}` token patterns — no
  matches. `.env` is untracked (`git ls-files` shows only `.env.example`);
  `git check-ignore -v .env` confirms it's matched by `.gitignore:1:.env`.
  The real `.env` on disk (with a live `HF_TOKEN`) was never staged or
  committed. Every mention of `HF_TOKEN` in tracked files (`.env.example`,
  `README.md`, `.claude/agents/reviewer.md`) is the variable name only,
  never a value.
- **No fabricated "final" results.** `README.md` states plainly "no
  comparison numbers yet" and commits to never reporting a result without a
  real Colab run behind it. No comparison table, benchmark numbers, or
  round/placeholder-looking metrics exist anywhere in this diff.
- **Tests actually exercise the logic, and they pass.** Ran
  `.venv/Scripts/python.exe -m pytest -q`: **41 passed in 0.75s.**
  `tests/test_cvefixes.py` exercises real branches of `clean_record`/
  `filter_records` (empty code, disallowed language, too long/short, no-op
  dedup, missing-CWE defaulting, dedup-by-key) against a fixture
  (`tests/fixtures/sample_raw_records.json`) built with deliberately messy
  rows (literal `"nan"` severity, Python-repr description, duplicate row,
  oversized blob, undersized pair) — not just import/smoke checks — and
  asserts an exact expected surviving set
  (`test_filter_records_against_fixture_default_config`,
  `tests/test_cvefixes.py:844-856`). `tests/test_config.py` exercises
  validation failure paths (bad ratios, non-positive LoRA rank, empty
  language list, unknown section/field) via `pytest.raises`.
- **Repo-side code has no hard GPU/Colab dependency.** `requirements-repo.txt`
  is CPU-only (`datasets`, `huggingface_hub`, `pandas`, `PyYAML`, `pytest`,
  `tqdm`); `src/lora_bench/data/cvefixes.py`'s only non-stdlib imports are
  `yaml` (via config) and a *local* `from datasets import load_dataset`
  inside `load_raw_dataset()` (`cvefixes.py:1224`), which only executes on
  the live (non-`--dry-run`) path — nothing GPU-bound is imported at module
  load time. `.github/workflows/tests.yml` installs only
  `requirements-repo.txt` + `pip install -e .` and runs `pytest -q`,
  matching the local setup exactly.
- **`.gitignore` correctness.** `/data/` (anchored, root-only) does not
  collide with the tracked package directory `src/lora_bench/data/` —
  verified with `git check-ignore -v src/lora_bench/data/cvefixes.py`
  (exit 1, not ignored) and `git ls-files src/lora_bench/data/` (all three
  files tracked as expected).
- **MVP non-negotiables 1–3 (fine-tune runnability, 4-axis comparison,
  full-pipeline no-babysitting):** not yet applicable — Day 1 correctly
  doesn't touch any of these, per the reviewer's own instruction not to
  fault Day 1 for lacking a fine-tune.
- **Past "just a demo" checklist:** not yet applicable — no report/
  comparison exists yet.

## Questionable

- **ADR-0002's live-scan statistics have no committed, reproducible
  artifact.** `ADR.md:77-93` (and the near-identical comment block in
  `src/lora_bench/config.py:1510-1533`) states specific findings — "~7% of
  scanned rows" are no-op diffs, "at least one row's code field is ~55MB,"
  a described language distribution — from "streaming and scanning ~3,000
  live rows on 2026-08-22." These numbers directly justify shipped defaults
  (`max_chars=4000`, `drop_noop_pairs=True`, the six-language allowlist), so
  they're doing real design work, not just color commentary. But there's no
  scan script, notebook cell, or log file committed anywhere in the repo
  that reproduces or backs these figures — they're asserted in prose only.
  This isn't the "comparison table with fabricated numbers" pattern the
  hard-constraints checklist calls blocking (it's a data-quality rationale,
  not a claimed benchmark result, and the pipeline's correctness doesn't
  depend on the exact percentages), but it's the same category of risk on a
  smaller scale. Would resolve with either a short committed script (even
  `scripts/scan_dataset.py`, not unit-tested, just documented as a one-off)
  that reproduces the scan, or softened language ("based on manual
  inspection of a sample" rather than a precise "~7%"/"~55MB").
- **Unused dependencies in `requirements-repo.txt`.** `pandas>=2.0`
  (`requirements-repo.txt:7`) is not imported anywhere in `src/` or
  `tests/` (confirmed via grep) — the pipeline uses plain dicts/dataclasses
  and the stdlib `json`/`csv`-free path throughout. `tqdm` and
  `huggingface_hub` are also not directly imported, though both are
  plausible transitive dependencies of `datasets` (progress bars, Hub
  auth) so are lower-risk. `pandas` in particular looks like a leftover
  from the (uncommitted) ad-hoc scan mentioned in ADR-0002 above — same
  root cause as that finding. Minor; worth trimming or explaining in a
  comment if intentionally kept for the Day 2+ scan tooling.

## Blocking

None.

## Resolved after review

Both questionable items were addressed in commit `a4b9e7f` (fix:) after
this review ran:

- Committed `scripts/scan_dataset.py`, a one-off network-requiring script
  that reproduces the language/severity/no-op/outlier statistics ADR-0002
  cites. Ran it manually (`--limit 500`) to confirm it reproduces the same
  pattern (7.0% no-op pairs, a multi-MB outlier row, the same language
  ordering) as the original ad-hoc scan. ADR-0002 now points to it.
- Removed the unused `pandas>=2.0` entry from `requirements-repo.txt`
  (confirmed nothing in `src/` or `tests/` imports it); full suite still
  passes (`41 passed`).

## Verdict
No blocking items — day may be closed out.
