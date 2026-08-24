# Review — Day 1 second hardening pass (53af83d..HEAD) — 2026-08-23

Scope reviewed: 8 commits, `343d0c6` through the current `HEAD`
(`2492523`), range `53af83d..HEAD`. This is the second hardening pass on
Day 1's data-prep pipeline. Pass 1 (self-review) closed Day 1 clean but
missed a crash bug and a data-corruption bug. Pass 2 (deeper review)
caught those plus 11 more; a first hardening pass fixed all 13, plus one
questionable item found during that review and fixed immediately in
`343d0c6`. Pass 3 (an even deeper review) then found 5 issues in that
first hardening pass's own new code/docs — including one claimed-blocking
defect — plus one piece of repo housekeeping. This review independently
verifies the second hardening pass that claims to fix all of those.

All 6 claimed fixes were checked against the actual diff (`git show <sha>
-p`), not the round's own summary. Item 1 (the blocking-severity
group-size-skew fix) was read line-by-line, independently re-derived with
a standalone script against both the fixed and reconstructed pre-fix
`split_examples`, and cross-checked against the shipped tests. Item 2
(ADR-0003 causal correction) was checked for internal consistency across
every file the old "conservative" framing touched. Item 6 (`.gitignore`
housekeeping) was checked empirically, not just read — confirmed the
files actually show as ignored and confirmed ruff's behavior differs with
and without the ignore rule.

## Passed

- **Item 1 (BLOCKING per pass 3) — `split_examples` largest-first sort
  (`02fc4af`).** Confirmed the fix is real: `cvefixes.py:299-306` now
  stable-sorts `group_ids` descending by size (`group_ids.sort(key=lambda
  gid: -len(groups[gid]))`) immediately after the seeded shuffle and
  before the greedy assignment loop — exactly the claimed
  longest-processing-time-first bin-packing order, applied before, not
  after, the per-group `max(split_order, key=...)` assignment.

  Independently re-derived the numbers rather than trusting the commit
  message. Using the exact shipped `_heavy_tailed_grouped_examples`
  fixture against the **current** `split_examples`, 10 seeds each:
  worst deviation 0.0007 (6 groups of 60-200 rows) and 0.0000 (1 dominant
  700-row group) — matches the ADR/commit-message post-fix numbers
  exactly. Then reconstructed the **pre-fix** function (identical except
  omitting the sort line) and ran the same fixture through it: worst
  deviation 0.0837 and 0.1920 respectively. The dominant-group number
  (0.1920) matches the claimed 0.192 exactly; the six-groups number
  (0.0837 vs. the claimed 0.102) is in the same order of magnitude and
  supports the same qualitative conclusion (severe, seed-dependent skew
  pre-fix, ~0 post-fix) but isn't bit-identical — plausibly because the
  "before" figures came from an ephemeral, non-committed verification
  script during development rather than the exact fixture that ended up
  shipped. This is a minor precision gap in the prose, not a
  misrepresentation: the direction, order of magnitude, and the fully
  reproducible post-fix numbers all check out. See Questionable below.

  Checked the new tests actually would have caught the old bug, not just
  assumed it: ran the shipped
  `test_split_examples_ratios_stay_within_tolerance_under_group_size_skew`
  and `..._with_one_dominant_group` fixtures against the reconstructed
  pre-fix logic — both fail hard against the pre-fix code (0.0837 and
  0.1920 vs. an asserted 0.01 band) and pass against the current code.
  `_grouped_examples` (max size 3) genuinely could not have caught this;
  confirmed by inspection that its size range is `sizes = [1,1,1,1,2,2,3]`
  cyclic, nowhere near a target-busting size at the tested scale.

  Checked the leakage-guarantee test
  (`test_split_examples_never_splits_a_group_across_sets`,
  `tests/test_cvefixes.py:70-72` in the current file) really is untouched:
  `git diff 53af83d..HEAD -- tests/test_cvefixes.py` shows only unchanged
  context lines around it, no `+`/`-` inside the function body.

  Checked the residual-honesty test
  (`test_split_examples_cannot_balance_a_group_larger_than_a_splits_target`)
  is meaningful, not decorative: it builds a 950-row dominant group + 50
  singletons, then actually asserts (not just prints) `sorted(
  huge_split_sizes) == [0, 0, 950]` (the group lands wholly in one split —
  leakage guarantee holds) and `worst > 0.1` (that split's ratio blows
  through the tolerance band). Both assertions are load-bearing and would
  fail if the leakage guarantee broke or if the residual were somehow
  papered over. This test documents an inherent limit of group-level
  splitting (a group larger than a split's target must overshoot that
  split), not a regression guard for the largest-first fix — correctly
  distinguished as such in both the commit message and ADR-0004.

  Checked the tightened tolerance test
  (`test_split_examples_ratios_stay_within_tolerance`, 0.05 -> 0.01): ran
  it against reconstructed pre-fix logic across seeds 1-5 — worst observed
  deviation 0.008, under the new 0.01 band (this fixture's group sizes are
  small, 1-3, so it was never the regression case; the heavy-tailed tests
  above are what actually exercise the fix). Confirms the tightened band
  is real (earned by the current small-group behavior, not vacuous) and
  that reusing the old fixture alone would not have exposed the bug,
  matching the commit message's own claim about this fixture's
  limitations.

  ADR-0004 (`ADR.md:288-408`) was updated with a full "Amendment
  (2026-08-23, second hardening pass)" section: the bug, the two
  measured-before numbers, the fix, the two measured-after numbers, and an
  explicit "residual this doesn't close" section. Cross-checked every
  number quoted there against the actual test run — all consistent.

- **Item 2 — ADR-0003 causal-explanation fix (`86a6e4b`).** Confirmed the
  correction is real and directionally right: the old text ("median...
  notably better than the conservative 3.0 estimate assumed") is replaced
  with an explanation attributing the 4/2400 residual to the *tail*
  running denser than assumed, with the derived ~2.37 chars/token figure
  (2,900 nominal cap / 1,223 measured max tokens ≈ 2.371) correctly framed
  as an *implied worst-case ratio* ("implies... around 2.37"), not
  presented as a directly measured per-example ratio — this is an honest
  hedge, since the 2,900 figure is a nominal cap (`max_combined_chars`
  2600 + unmeasured instruction-overhead estimate 300), not that specific
  example's actual character count. No new number is presented as
  measured that wasn't actually measured; the arithmetic is transparent
  and reproducible from the ADR's own already-measured/stated inputs.

  Checked internal consistency across every location the old "conservative"
  framing touched, not just the one paragraph pass 3 flagged: grepped the
  whole repo for "conservative" post-fix — the five remaining hits
  (`ADR.md:215,276`; `config.py:21,29,212`) are all now framed as *denying*
  the old "hard conservative bound" reading (e.g. "not a hard conservative
  bound", "a genuinely conservative bound would LOWER this constant"), not
  reintroducing the backwards claim. `CHARS_PER_TOKEN_ESTIMATE`'s comment
  (`config.py:18-40`), `Config.__post_init__`'s docstring
  (`config.py:189-194`), and its `ValueError` message (`config.py:209-215`)
  all consistently describe the estimate as a typical-case heuristic the
  measured tail can and does breach, not a guaranteed bound — verified
  each of the three sites individually, not just the first one found.
  `CHARS_PER_TOKEN_ESTIMATE` itself is confirmed unchanged at `3.0`
  (`config.py:41`), and the ADR/comment both explicitly and honestly state
  why it wasn't re-derived (would require a fresh `token_budget.py` run
  against a live tokenizer + dataset pull, not done here) rather than
  quietly picking a new number.

- **Item 3 — `scan_dataset.py` keep-rate cap fix (`5e3f03e`).** Confirmed:
  `scan()` (`scripts/scan_dataset.py`) now constructs
  `DataConfig(max_examples=None)` instead of `DataConfig()`, with a
  comment correctly explaining why the pipeline's real default
  (`max_examples=3000`) would otherwise turn `len(cleaned)` into a sample
  size rather than a keep count once more than 3,000 rows survive
  filtering at a large `--limit`.

- **Item 4 — `scan_dataset.py` `--revision` default (`52545a6`).**
  Confirmed: `--revision` now defaults to `DataConfig().revision` (the
  pipeline's pinned SHA) instead of `None` (branch head), with updated
  `--help` text explaining the rationale and how to opt back into a
  branch-head scan (`--revision main`). This is exactly backwards-to-right
  as claimed — a diagnostic whose whole job is producing ADR-0002's cited
  numbers now scans the same snapshot the real pipeline reads by default.

- **Item 5 — stale `--dry-run` help text (`947e211`).** Confirmed:
  `cvefixes.py`'s `--dry-run` help text now reads "Use the bundled
  sample_records.json package-data sample..." instead of the stale
  "tests/fixtures sample" reference. Grepped the whole repo for remaining
  `tests/fixtures` mentions: exactly two remain
  (`REVIEW.md:173` in the pre-overwrite version, and
  `tests/test_cvefixes.py:40`'s comment on `load_fixture()`), both
  deliberately describing the historical move (per the commit message),
  not stale pointers — confirmed by reading both in context.

- **Item 6 — `.gitignore` housekeeping (`2492523`).** Confirmed
  empirically, not just by reading the diff. `git status --porcelain`
  shows a clean tree (the two files don't appear); `git status
  --ignored --porcelain` shows them as `!! DAY1_FIX_PROMPT.md` / `!!
  DAY1_FIX_PROMPT_2.md` (ignored, not merely untracked); `git check-ignore
  -v` confirms both match the new `.gitignore:29: /DAY1_FIX_PROMPT*.md`
  rule specifically. Also verified the ruff claim empirically rather than
  taking it on faith: both files do contain embedded ```python fences
  (confirmed by grep); with `--no-respect-gitignore`, `ruff format --check
  .` fails (exit 1, "2 files would be reformatted") specifically on these
  two `.md` files' embedded Python blocks, while `ruff check .` (lint)
  passes regardless — so the `.gitignore` fix is specifically what makes
  `ruff format --check .` (not `ruff check .`) clean against the whole
  repo, exactly as the commit message states. Neither file was deleted or
  moved, as claimed (non-destructive choice, explicitly justified).

- **Full test suite.** Ran `.venv/Scripts/python.exe -m pytest -q`:
  **118 passed** (matches the claimed count; confirmed the +22 over the
  prior round's 96 comes from 10+10 new parametrized heavy-tailed split
  tests plus the 1 total-collapse regression test (`343d0c6`) plus the 1
  residual-documentation test — `96 + 22 = 118`, verified by counting new
  `test_`/`@pytest.mark.parametrize` lines added in the range).

- **Lint/format, whole repo.** Ran `.venv/Scripts/python.exe -m ruff check
  .` -> `All checks passed!`; `.venv/Scripts/python.exe -m ruff format
  --check .` -> `15 files already formatted`. Both against the bare `.`
  path (not `src tests scripts`), which only works cleanly because of the
  item-6 `.gitignore` fix (see above).

- **`343d0c6` (resolved-after-review item from the prior round, carried
  into this range).** Confirmed real: `main()` (`cvefixes.py:549-559`) now
  wraps `run_pipeline(...)` in `try/except ValueError`, printing the same
  `ERROR: data-prep pipeline failed: {e}` / exit-1 shape used by the
  `--min-examples` path.
  `test_main_returns_nonzero_cleanly_on_total_filtering_collapse`
  reproduces the exact repro case (a config allowing only `languages:
  [Rust]` against the fixture, which contains none) and asserts exit 1 +
  `"ERROR"` in stderr. Ran this test in isolation — passes.

- **Secrets.** Grepped `git diff 53af83d..HEAD` for `hf_[A-Za-z0-9]{10,}`
  and `token\s*=\s*['"][a-z0-9]{15,}` patterns — no hits. `.env` remains
  gitignored (`git check-ignore -v .env` -> `.gitignore:1:.env`) and shows
  as `!!` (ignored) in `git status --ignored`, not tracked.

- **Scope discipline / Conventional Commits.** All 8 commits in range are
  correctly typed (`fix:` x6, `docs:` x1, `chore:` x1), each a single
  coherent change with its own tests/docs in the same commit — no bundling
  of unrelated work. `git diff 53af83d..HEAD --stat` touches exactly the
  files the 6 claims describe (`.gitignore`, `ADR.md`, `REVIEW.md`,
  `scripts/scan_dataset.py`, `src/lora_bench/config.py`,
  `src/lora_bench/data/cvefixes.py`, `tests/test_cvefixes.py`) plus
  nothing else. No GPU/notebook/PEFT/bitsandbytes/torch code anywhere in
  the diff. `requirements-colab.txt` untouched.

- **MVP non-negotiables 1-3 / past "just a demo" checklist:** still not
  applicable — this range is entirely repo-side data-prep bug fixes and
  doc corrections; no fine-tune, comparison, or report exists yet.

## Questionable

- **ADR-0004's quoted pre-fix "0.102" figure for the six-heavy-groups
  scenario doesn't independently reproduce exactly.** Re-deriving the
  pre-fix (shuffle-order-only) behavior against the exact shipped
  `_heavy_tailed_grouped_examples(seed)` fixture (default args: 280
  singletons + 6 groups sized `randint(60,200)` per seed) across seeds
  1-10 gives a worst deviation of 0.0837, not the claimed 0.102. The
  companion one-dominant-group figure (0.192) reproduces exactly, and the
  qualitative story (severe, seed-dependent skew pre-fix; ~0 post-fix) is
  fully supported either way — this doesn't change the verdict on the fix
  itself. Most likely explanation: the "before" numbers were captured by a
  throwaway verification script during development (as pass 3's own
  review prescribed) that wasn't necessarily using byte-identical
  parameters to the fixture that ultimately shipped, and the after-the-
  fact prose rounds/restates a number from that ephemeral run rather than
  the shipped fixture. Low stakes since the *shipped, re-runnable*
  evidence (all four post-fix numbers, the dominant-group pre-fix number,
  and the tests' own asserted 0.01 band) is fully reproducible and
  consistent — but if this ADR is ever cited as a precise historical
  measurement rather than an illustrative one, it's worth a one-line
  amendment noting the six-groups pre-fix figure is approximate/from a
  non-committed script, or re-running and updating it to match exactly.

- **`ROADMAP.md`'s Day 1 entry doesn't mention this second hardening
  pass.** `ROADMAP.md:16-26` documents the *first* hardening pass in
  detail (crash bug, data-corruption bug, unpinned revision, etc.) and
  says "none of this was caught by the first self-review pass, which is
  itself worth remembering going into Day 2's review" — but doesn't
  mention that a *third* review then found a blocking-severity defect
  inside that hardening pass's own new code (the split-skew bug), which
  this range fixes. Given the file's own stated purpose (a concrete record
  of what's in scope and, per its Day 1 entry, a place to record what
  review passes catch), a reader skimming `ROADMAP.md` alone would not
  learn that the hardening pass itself needed hardening. Not a code defect
  and doesn't block Day 1's closure, but worth a short addendum alongside
  the existing note, given the project's own precedent of recording this
  kind of self-review miss there.

## Blocking

None. All 6 claimed fixes were verified against the actual diff (not just
the round's summary), the specific blocking-severity item (split-group
skew) was independently re-derived and its fix confirmed to genuinely
close the measured gap (worst deviation 0.192 -> 0.0000 on the dominant-
group scenario, reproduced exactly), and the full test suite (118 tests),
`ruff check .`, and `ruff format --check .` all pass cleanly against the
current `HEAD`.

## Resolved after review

Both questionable items were addressed in commit `a5d44f0`:

- ADR-0004's six-heavy-groups pre-fix figure was independently re-derived
  against the exact shipped `_heavy_tailed_grouped_examples` fixture
  (0.0837, matching this review's own finding, not the previously-stated
  0.102) and corrected in both places it appeared.
- `ROADMAP.md`'s Day 1 entry now records that a third review found a
  blocking-severity defect inside the first hardening pass's own new code.

Full suite re-verified after both fixes: 118 passed; `ruff check .` and
`ruff format --check .` both clean.

## Verdict

No blocking items — day may be closed out.
