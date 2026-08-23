# Architecture Decision Records

Dated entries for decisions that deviate from, or materially specialize,
the project's default stack (PyTorch, PEFT/QLoRA, bitsandbytes, llama.cpp
or vLLM, Colab free T4 — see the project brief). Each entry: date, decision,
why, and what alternatives were considered and rejected.

## ADR-0001 — Base model: Qwen2.5-Coder-1.5B-Instruct

**Date:** 2026-08-22 (Day 1)

**Decision:** Use `Qwen/Qwen2.5-Coder-1.5B-Instruct` as the base model for
the Day 2 LoRA/QLoRA fine-tune.

**Why:**
- **License friction.** Apache-2.0, no gated-license click-through on
  Hugging Face. A one-shot Colab notebook that stops mid-run on an
  "accept this model's license" wall is a real failure mode for the
  no-babysitting requirement; Qwen2.5-Coder sidesteps it entirely.
- **Domain fit.** It's code-pretrained, which matters directly here: the
  task is reading a vulnerable code snippet and producing a fixed one, not
  general natural-language instruction-following.
- **Size vs. T4 budget.** At 1.5B params, QLoRA fine-tuning, later GGUF
  conversion, and running base/fine-tuned/quantized inference side by side
  all comfortably fit a free-tier T4's ~15GB VRAM with room to spare — this
  buys headroom for the comparison step (loading multiple model variants
  during benchmarking) without a second thought about OOM. A 7B model would
  fit too under 4-bit QLoRA, but leaves much less slack and a longer,
  less-reliable single-session run.

**Alternatives considered:**
- *CodeLlama-7b / StarCoder2-3b* — solid code models, but larger and (for
  StarCoder2) under BigCode OpenRAIL-M, which carries use restrictions;
  rejected mainly on session time/memory headroom, not capability.
- *Phi-3-mini-4k-instruct (3.8B)* — strong general instruct model, MIT
  license, but not code-specialized; rejected in favor of a code-pretrained
  model for this task.
- *Llama-3.2-1B/3B-Instruct* — gated on Hugging Face (requires accepting
  Meta's license and waiting for/using per-account approval), which
  conflicts with a script that should run unattended once started.

**Revisit if:** Day 2's actual fine-tuning run shows this model saturates
too quickly to show a meaningful quality delta, or GGUF conversion (Day 3)
turns out not to support the Qwen2 architecture cleanly in the llama.cpp
version pinned then.

## ADR-0002 — Dataset: `hitoshura25/cvefixes` (Hugging Face mirror of CVEfixes)

**Date:** 2026-08-22 (Day 1)

**Decision:** Source the CVE fix-diff data from the Hugging Face dataset
`hitoshura25/cvefixes`, loaded via `datasets.load_dataset(...)`, rather than
the original CVEfixes SQL dump on Zenodo.

**Why:**
- **Same underlying data, far less plumbing.** `hitoshura25/cvefixes` is a
  pre-flattened, per-row export of the CVEfixes research dataset (Bhandari
  et al., "CVEfixes: Automated Collection of Vulnerabilities and Their
  Fixes from Open-Source Software", PROMISE 2021) — CVE metadata, CWE
  classification, commit info, and `vulnerable_code`/`fixed_code`/
  `diff_with_context` per row — as a single `load_dataset()` call instead of
  downloading a multi-GB SQL dump and writing our own schema/joins to get
  from "database of commits" to "instruction-tuning rows." That plumbing
  would be pure overhead for this project's goal.
- **Colab-friendly.** One dependency (`datasets`), no SQLite wrangling
  inside a notebook cell, no separate multi-GB download step competing with
  the free tier's session/disk limits.
- **License.** Apache-2.0 on the HF dataset card, consistent with reusing
  CVEfixes' own research-use framing. Understand that the *code snippets
  themselves* originate from many different upstream GitHub repos under
  their own individual licenses (see README's dataset attribution note) —
  the dataset wrapper's Apache-2.0 license doesn't relicense the underlying
  source code, only this project's redistribution of the extracted
  fix-commit rows for research/fine-tuning purposes.

**Data-quality findings that directly shaped `DataConfig` defaults**
(qualitative findings from an initial ad-hoc scan of ~3,000 live rows on
2026-08-22; reproduce with `python scripts/scan_dataset.py --limit 3000` —
see also `src/lora_bench/config.py` and `configs/default.yaml`):
- `severity` is frequently the literal string `"nan"` rather than missing —
  normalized to `"UNKNOWN"` (`normalize_severity`), not left as a stray
  string a downstream branch could mis-handle.
- A meaningful fraction of rows have an empty `vulnerable_code` or
  `fixed_code`, or `vulnerable_code == fixed_code` (a no-op diff, e.g. a
  doc-only commit swept in by the source mining) — both dropped
  (`clean_record`, `drop_noop_pairs=True`).
- At least one row's code field is ~55MB — `max_chars` (default 4000) is
  load-bearing, not cosmetic; without it, a single row can dominate a T4
  session's token/time budget.
- Language distribution (3,000-row scan): PHP and C dominate, followed by
  Python/JavaScript/Go/C++/Java in the low hundreds each, with "Other"/
  "Unknown"/"JSON"/"Markdown" buckets that aren't really source code. The
  default `languages` allowlist (`Python, C, C++, JavaScript, Java, Go`)
  keeps the latter out while covering most of the signal.

**Exact drop-reason breakdown (Day 1 hardening pass, 2026-08-22)** — after
`scripts/scan_dataset.py` was rewritten to call the pipeline's own
`clean_record`/`filter_records` instead of keeping its own inline copy of
those checks, a fresh 3,000-row scan through the *current* filtering
pipeline (including `max_combined_chars` from ADR-0003) measured:

```
scanned: 3000
kept by current DataConfig defaults: 746 (24.9%)
dropped by reason:
  disallowed_language: 1240 (41.3%)
  empty_code: 622 (20.7%)
  too_long: 187 (6.2%)
  combined_too_long: 154 (5.1%)
  too_short: 22 (0.7%)
  duplicate: 20 (0.7%)
  noop_pair: 9 (0.3%)
```

Note this isn't directly comparable to the informal "~7% no-op" figure an
earlier (pre-pipeline, ad-hoc) version of this scan reported: that number
measured no-op pairs across *all* scanned rows independently, while this
one measures them *after* a row has already survived every earlier check
in `clean_record`'s sequence (identifiers present, code non-empty,
language allowed, length bounds) — a conditional, not marginal, rate.
Funnel effects like this are exactly why this script was rewritten to call
the pipeline's real `clean_record`/`filter_records` instead of keeping its
own copy of the checks: the two had already silently diverged once.

**Alternatives considered:**
- *Raw CVEfixes Zenodo SQL dump* — the authoritative source, and the
  fallback if the HF mirror ever disappears or diverges, but heavier to
  integrate for no clear benefit at this project's scale (~3k training
  examples, not the full ~13k-row corpus).
- *MoreFixes* (larger, ~29k CVEs mined with an enhanced pipeline) —
  bigger corpus, but no ready-made HF `datasets` loader found at review
  time; would reintroduce the SQL-plumbing cost this ADR is avoiding.
- *BigVul / Devign* — function-level vulnerable/fixed pairs, but not CVE-
  linked fix-diffs specifically, and the mission asks for a CVE fix-diff
  dataset by name.

**Revisit if:** the HF mirror is taken down or found to silently diverge
from the CVEfixes source, or the Day 2 fine-tune needs more than ~13k
available rows can supply after filtering.

**Amendment (2026-08-22, Day 1 hardening pass): pin the dataset revision.**
`load_raw_dataset` originally called `load_dataset(dataset_name,
split=split)` with no `revision`, so it followed the HF repo's branch
head — meaning the training data behind any reported comparison number
could silently change out from under it, which directly contradicts this
ADR's own "revisit if... found to silently diverge" condition by not even
detecting a divergence, let alone reacting to one. `DataConfig.revision`
now defaults to `d4f5c4ea65329d9ccbb8a3b3149e5d06eda5edb2` — the dataset
repo's actual current commit SHA, resolved via
`HfApi().dataset_info("hitoshura25/cvefixes").sha` on 2026-08-22, not an
invented placeholder — and `load_raw_dataset`/`configs/default.yaml`
thread it through. Revisit if the dataset repo is updated with fixes worth
pulling in (bump the pin deliberately, don't just drop it).

## ADR-0003 — Combined character budget instead of raising `max_seq_len`

**Date:** 2026-08-22 (Day 1 hardening pass)

**The bug:** `DataConfig.max_chars` (4000) was applied independently to
`vulnerable_code` and `fixed_code`, so a single surviving example could
carry up to ~8,000 characters of code plus the instruction text, while
`ModelConfig.max_seq_len` (1024 tokens) is the actual training sequence
budget. At a conservative ~3 chars/token for source code, a worst-case
pair needs on the order of 2,400-2,700+ tokens — roughly 2.5x the budget.
Tokenizers truncate from the right, so the part that gets cut in a
teacher-forced training example is the end of the sequence — the `output`
(the fixed code, i.e. the training label). A meaningful share of examples
would have silently taught the model to emit a fix that stops mid-token,
and the Day 4 quality comparison would then be measuring a model trained
on mutilated targets, not a real capability difference. Nothing validated
this relationship and nothing caught it — see REVIEW.md's hardening-pass
notes for how it surfaced.

**Decision:** Add `DataConfig.max_combined_chars` (default 2600) as an
explicit, independently-enforced cap on `len(vulnerable_code) +
len(fixed_code)`, checked in `clean_record` alongside (not instead of) the
existing per-field `max_chars`. `Config.__post_init__` (new — this
invariant spans `data` and `model`, so it can't live in either section's
own validator) rejects any config where `max_combined_chars` plus a fixed
instruction-overhead estimate, divided by a conservative chars-per-token
estimate, exceeds `max_seq_len`. `max_seq_len` itself stays at 1024.

**Why this option over the others:**
- **Vs. raising `max_seq_len`:** covering the *full* per-field worst case
  (2 x 4000 = 8000 chars) would need `max_seq_len` around 2800-3100, roughly
  3x today's value. Sequence-length increases cost T4 memory and step time
  during Day 2's actual fine-tune in a way this repo-side environment has
  no GPU to verify — deciding "it'll still fit" here would be exactly the
  kind of unverified claim the project's hard constraints warn against.
  Leaving `max_seq_len` untouched keeps this a Day-1, non-GPU decision.
- **Vs. uniformly lowering `max_chars`:** to make `2 x max_chars` fit the
  budget without a combined field would mean roughly halving max_chars to
  ~1400 for *both* fields, even in the common case where one field is
  short and the other does the real work (e.g. a one-line vulnerable
  snippet with a more involved fix, or vice versa). That discards
  legitimate, well-balanced examples to guard against a worst case that a
  combined cap already prevents more precisely.
- **Vs. truncating instead of dropping:** truncation reintroduces exactly
  the mid-token-cutoff risk this ADR exists to close, just moved earlier
  in the pipeline instead of left to the tokenizer. Dropping is a smaller
  dataset; truncating is silently corrupted labels. Given the choice, drop.

**The numbers behind `max_combined_chars = 2600`:** solved so that
`(max_combined_chars + INSTRUCTION_OVERHEAD_CHARS_ESTIMATE) /
CHARS_PER_TOKEN_ESTIMATE <= max_seq_len` with margin: at
`CHARS_PER_TOKEN_ESTIMATE = 3.0` (chars/token; conservative because real
code tokenizers typically run higher, ~3.5-4.5, so this demands *more*
token budget than likely needed) and `INSTRUCTION_OVERHEAD_CHARS_ESTIMATE
= 300` (rough pad for `INSTRUCTION_TEMPLATE`'s text plus chat-template
role markers/special tokens, not measured), `(2600 + 300) / 3.0 ≈ 967`
tokens worst case against a 1024 budget — about 5.6% margin. Both
constants are named, documented as heuristics, and explicitly subordinate
to real measurement.

**Measured, not just estimated:** `scripts/token_budget.py` was run (in a
throwaway environment with `transformers` + `jinja2` installed — not the
repo-side venv, and not committed as a dependency) against a live
2,400-example train split generated by the *current* (fixed) config, using
the real Qwen2.5-Coder-1.5B-Instruct tokenizer and chat template:

```
examples: 2400
tokens -- min: 86  median: 229  p90: 584  p95: 679  p99: 834  max: 1223
exceed max_seq_len (1024): 4 (0.2%)
avg characters lost to right-truncation among those examples: 448
```

For comparison, the *same* 2,400 examples regenerated with the pre-fix
behavior (`max_chars=4000` per field, no combined cap — i.e. simulating
the bug this ADR fixes) measured:

```
examples: 2400
tokens -- min: 90  median: 267  p90: 922  p95: 1227  p99: 1809  max: 4209
exceed max_seq_len (1024): 180 (7.5%)
avg characters lost to right-truncation among those examples: 1412
```

So the fix took truncation-risking examples from 7.5% to 0.2% of the
training set, and cut the average damage per still-affected example from
~1,412 to ~448 characters. It is not a hard 100% guarantee — real
token/char ratios vary per example (median tokens/char in this run was
notably better than the conservative 3.0 estimate assumed, which is why
the heuristic bound left a small residual rather than zero), and this
repo-side stage deliberately doesn't depend on the real tokenizer to stay
GPU/Colab-independent. The residual 0.2% is a reasonable target for Day 2
to close directly: once the real tokenizer is available there anyway, an
additional exact filter (`len(tokenizer(...).input_ids) <= max_seq_len`,
not just the char heuristic) is a natural refinement, not a Day-1 gap.

**Revisit if:** Day 2's actual fine-tune shows 1024 tokens is too short
for the fixes worth learning from regardless of the char budget, or the
real tokenizer filter mentioned above turns out to remove enough examples
to matter.

## ADR-0004 — Group-aware train/val/test split, grouped by CVE ID

**Date:** 2026-08-22 (Day 1 hardening pass)

**The bug:** the dedup key in `filter_records` is `(commit_hash, repo_url,
cve_id)`, and `split_examples` then shuffled and sliced at the *row*
level. But a single CVE routinely spans several files fixed in one commit,
or gets a follow-up fix commit later — both produce multiple rows sharing
a `cve_id` but not deduped away by that key. Row-level random splitting
therefore could put sibling rows of the same CVE — often the same fix,
lightly varied — on both sides of the train/test boundary. Every quality
number Day 4 would report off that test split was inflated by an unknown
amount, and "the fine-tune improved exact-match by X" is the single claim
this whole project exists to make defensibly.

**Decision:** `split_examples` now groups examples by `cve_id` before
splitting and assigns each group wholly to one split, via a deterministic
largest-remaining-need greedy heuristic: shuffle group order (seeded),
then repeatedly hand the next group to whichever split is furthest below
its target count (ties broken train > val > test).

**Why `cve_id` and not `repo_url`:** grouping by `repo_url` instead (or as
well) would additionally guard against a *different*, smaller leak: a repo
with a recurring bug pattern fixed across several distinct CVEs. But at
this project's scale (~3k examples after filtering), some repos
contribute disproportionately many rows — grouping by repo could force
one or two repos' entire contribution into a single split, both distorting
the realized ratios far more than CVE-grouping does and reducing
language/pattern diversity in whichever split loses that repo entirely.
CVE-level grouping directly closes the leak this ADR was written to fix
(rows that are near-duplicates of the *same* fix); residual repo-level
similarity across *different* CVEs is a real but smaller risk, deliberately
left open rather than traded for a bigger diversity/balance cost today.

**Measured ratio tolerance:** exact target ratios aren't achievable once
grouping is in play (group sizes vary), so this is a heuristic, not a
guarantee. Measured across 5 seeds on ~300-example synthetic data with
mixed group sizes (see `tests/test_cvefixes.py`'s
`test_split_examples_ratios_stay_within_tolerance`): realized train ratio
stayed within 0.80 ± 0.01 of the 0.8 target, val/test within 0.10 ± 0.005
of their 0.1 targets — well inside a 0.05 tolerance band the tests assert
generously to avoid a flaky, over-tight bound.

**Alternatives considered:**
- *Keep row-level splitting, dedup harder instead*: doesn't work — the
  existing dedup key already treats different files/commits of the same
  CVE as distinct rows on purpose (they're genuinely different training
  examples), so deduping them away would just lose data rather than fix
  the leak.
- *Group by (repo_url, cve_id) instead of cve_id alone*: rejected as
  redundant — the leak mechanism described above is CVE-level, and this
  wouldn't catch anything CVE-level grouping doesn't already catch, since
  a CVE's sibling rows already share both fields in this dataset.

**Revisit if:** Day 3/4's eval shows evidence of repo-level leakage
mattering in practice (see ADR-0005's note on what that eval should check
for), or a future, much larger dataset changes the diversity/balance
trade-off against repo-grouping.

## ADR-0005 — Drop the CVE ID from the training prompt; keep CWE

**Date:** 2026-08-22 (Day 1 hardening pass)

**Decision:** `INSTRUCTION_TEMPLATE` no longer embeds `{cve_id}`. The
instruction now names only the language and CWE (e.g. "contains a known
security vulnerability (CWE-79: Cross-site Scripting)"), not the specific
CVE identifier. `cve_id` stays a field on `FixDiffExample` — it's still
used for the group-aware split (ADR-0004) and is available for
failure-case analysis in the Day 4 report — it's just not in the text the
model is trained to condition on.

**Why:**
- **Distribution mismatch with the actual use case.** At real inference
  time on an unknown vulnerability, there is no CVE ID to hand the model —
  that's the entire premise of needing a fix in the first place. Training
  the model to expect one in the prompt teaches a shortcut that won't
  exist at the point this project claims to be useful.
- **Memorization risk.** A CVE ID is a near-unique identifier per training
  example — almost the definition of a key a model can memorize an output
  against instead of learning the general skill of reading vulnerable code
  and fixing it. That's a direct threat to the Day 4 quality claim: an
  inflated exact-match number driven by ID memorization would look
  identical to a genuine capability improvement without careful eval
  design, and this project doesn't have a mechanism today to tell the two
  apart.
- **Why CWE stays, not just gets dropped too:** unlike a CVE ID, a CWE
  (e.g. CWE-79, "Cross-site Scripting") is a coarse category shared by
  hundreds of examples across the dataset — it can't function as a
  memorization key the way a near-unique CVE ID can. It's also a
  realistic signal: a real deployment of this kind of tool would plausibly
  pair with a SAST/vulnerability scanner that flags *what kind* of
  weakness it found, even without knowing it's specifically "CVE-2023-
  0001." Keeping CWE frames the task as "given a flagged weakness
  category, fix it" rather than either the unrealistic "given the exact
  CVE" or the much vaguer, likely harder-to-learn "find and fix anything
  wrong with this code, no hints."

**How Day 3's eval should show this isn't just memorization:** this ADR
is a design decision, not proof the risk is fully closed — closing it is
eval work, not a data-prep-stage claim. The test split is disjoint at the
CVE-group level (ADR-0004), which already prevents literal train/test
duplication of the same CVE's rows. Day 3/4 should go further and report
performance broken out at minimum by whether the exact CWE category
appeared in training (it will, broadly, since CWEs repeat across many
CVEs) versus a held-out sanity check on a handful of examples whose
specific *repo* never appeared in training — the closest available proxy
here for "does this generalize past a memorized specific case" given the
dataset's structure. Flagged here now so it doesn't get improvised later.

**Alternatives considered:**
- *Keep the CVE ID*: rejected for the memorization/distribution-mismatch
  reasons above.
- *Drop CWE too, describe the task generically ("this code has a
  vulnerability, fix it")*: rejected as a harder, vaguer training signal
  with no realistic offsetting benefit — CWE isn't a memorization risk the
  way CVE ID is, so dropping it buys safety it doesn't need to buy at a
  real cost to how learnable the task is.

**Revisit if:** Day 3/4's eval (see above) finds evidence of CWE-level (not
just CVE-level) memorization, or finds the CWE-conditioning framing itself
doesn't match how the eventual benchmark prompts are posed.
