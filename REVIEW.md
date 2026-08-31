# Review — Day 2 (`edccc10..HEAD`, 3 commits) — 2026-08-24

Scope reviewed: `120d3b6` (extract `to_chat_messages`), `f689fe5` (Day 2
fine-tuning notebook), `6908a54` (docs update). Per `ROADMAP.md`, Day 2 is
"build the Colab fine-tuning notebook (data prep -> QLoRA fine-tune ->
LoRA rank sweep -> save adapter)," explicitly not-yet-run since this
repo-side environment has no GPU.

## Passed

- **Notebook stays honest about not being run.** Inspected the raw
  `notebooks/finetune.ipynb` JSON directly: all 23 cells have
  `outputs: []` and no `execution_count` set — nothing was executed and
  then committed with baked-in output. `git diff edccc10..HEAD` and a
  targeted grep of `README.md`/`ROADMAP.md`/the notebook for
  `val_loss|eval_loss|BLEU|perplexity|latency|tokens/sec|\$|% accuracy`
  turned up zero embedded numbers — every mention of sweep results,
  winning config, training loss, or adapter quality is phrased as
  pending a real run (notebook cell-16's printed instructions, the
  "Next" section, `README.md`'s new "Running the fine-tuning notebook"
  section, and `ROADMAP.md`'s "What's still open"). The one numeric
  claim in the notebook (`"well under an hour"` runtime estimate, cell-0)
  is explicitly flagged as "an estimate, not a measurement."
- **`to_chat_messages()` (120d3b6) is a real, tested extraction, not a
  cosmetic move.** `src/lora_bench/data/cvefixes.py:249-262` adds the
  function; `tests/test_cvefixes.py:325-341`
  (`test_to_chat_messages_shape_and_content`) builds a real cleaned
  record through `clean_record`/`to_example` (not a synthetic dict) and
  asserts both the exact message shape and that the fix text lands only
  in the assistant turn, never the prompt — this exercises real logic,
  not just an import. `scripts/token_budget.py`'s `render_example`
  (lines 30-40) now calls it via `FixDiffExample.from_dict(rec)`,
  removing the previous hand-rolled duplicate dict construction; the
  module docstring was updated to reflect the new `lora_bench`
  import dependency.
- **The commit's verification claims are real and independently
  reproducible.** Built a throwaway venv and installed current PyPI
  `transformers` (5.15.1): confirmed `TrainingArguments.warmup_ratio`
  genuinely no longer exists (only `warmup_steps`, which now accepts a
  float treated as a ratio when `< 1` — see
  `transformers/training_args.py:788`, `2108-2115`) — exactly the API
  break the commit message describes, and the notebook correctly uses
  `warmup_steps=max(1, int(0.03 * total_steps))` everywhere (cells 14,
  18), never the removed kwarg. Also checked every other library call
  the notebook makes against the current installed API surface:
  `BitsAndBytesConfig(load_in_4bit=, bnb_4bit_quant_type=,
  bnb_4bit_use_double_quant=, bnb_4bit_compute_dtype=)`,
  `LoraConfig(r=, lora_alpha=, lora_dropout=, target_modules=,
  task_type=, bias=)` + `TaskType.CAUSAL_LM`,
  `DataCollatorForLanguageModeling(tokenizer=, mlm=)`, and
  `TrainingArguments`'s other kwargs (`eval_strategy`, `save_strategy`,
  `optim="paged_adamw_8bit"`, `load_best_model_at_end`,
  `metric_for_best_model`, etc.) — all present and named exactly as the
  notebook uses them. `save_strategy`/`eval_strategy` are both `"epoch"`
  in the full fine-tune (cell-18), satisfying `load_best_model_at_end`'s
  requirement that the two match.
- **`requirements-colab.txt` matches what the notebook installs.** Cell-3
  runs `pip install -q -U transformers peft bitsandbytes accelerate`;
  the file lists exactly those four under "Day 2." `torch` (Colab
  preinstalled), `datasets`/`huggingface_hub` (repo-side deps pulled by
  `pip install -e .`), `trl`, and `sentencepiece` are correctly absent,
  with reasoning documented in both the file's comment block and the
  commit message.
- **Scope discipline held.** No quantization or benchmark-harness cells
  exist in the notebook — Day 3 work is explicitly deferred to the
  "Next" markdown cell and `notebooks/README.md`. `git diff
  edccc10..HEAD` touches nothing in `pyproject.toml`,
  `requirements-repo.txt`, or `.github/workflows/tests.yml` — confirmed
  with an empty diff on all three — so repo-side deps stay CPU-only and
  CI is unaffected by this range.
- **Tests and lint, run against current `HEAD`:**
  `.venv/Scripts/python.exe -m pytest -q` → `119 passed` (up from 118;
  confirmed `test_to_chat_messages_shape_and_content` is the new one via
  `-k to_chat_messages`). `.venv/Scripts/python.exe -m ruff check .` →
  `All checks passed!`. `.venv/Scripts/python.exe -m ruff format --check
  .` → `16 files already formatted`; confirmed this includes the
  notebook itself by running `ruff format --check --diff
  notebooks/finetune.ipynb` and `ruff check notebooks/finetune.ipynb -v`
  directly (both processed the `.ipynb` file, not skipped it).
- **Secrets.** `HF_TOKEN` is read from Colab's Secrets panel
  (`google.colab.userdata`) with a fallback to `os.environ` (cell-4),
  never hardcoded, and the only related print statement
  (`"Logged in to Hugging Face Hub."`) doesn't log the token value.
  `.env.example` (untouched by this diff) still ships with a blank
  `HF_TOKEN=`. `.gitignore` covers `.env`. Grepped the diff itself for
  token-like patterns (`hf_[A-Za-z0-9]{20,}`, `sk-...`,
  `token\s*=\s*['"]...`, `api[_-]?key`) — no matches. `REPO_URL` in
  cell-3 is the project's own public GitHub URL (matches `git remote -v`
  exactly), not a credential; the notebook's own "Before you start"
  explicitly warns against hardcoding a token into a private-repo clone
  URL.
- **Commits are real, atomic Conventional Commits** — `120d3b6` (feat:
  one coherent extraction across 3 related files), `f689fe5` (feat: the
  notebook + its two directly-supporting doc/requirements files),
  `6908a54` (docs: README/ROADMAP only). None mixes unrelated concerns.

## Questionable

- **Training loss isn't masked to the response.** Cell-10's
  `render_and_tokenize` and the `DataCollatorForLanguageModeling(
  tokenizer=tokenizer, mlm=False)` used in both the sweep (cell-14) and
  full fine-tune (cell-18) produce `labels = input_ids` over the *entire*
  rendered sequence — the vulnerable-code prompt as well as the
  fixed-code response — rather than masking the user turn out of the
  loss (e.g. with `-100`). This is a common simplification, not a crash
  or data-corruption bug, but it means part of every gradient step's
  signal goes toward the model reproducing its own prompt rather than
  learning the fix, and it's not called out anywhere in the notebook's
  markdown as a deliberate choice with a rationale (the kind of thing
  ADR-0006 or a notebook markdown cell should probably address once real
  loss curves exist). Worth a conscious decision (mask or don't, and
  say why) rather than an implicit default.
- **`requirements-colab.txt` doesn't list `jinja2`, which the notebook
  actually needs at runtime.** `tokenizer.apply_chat_template` (cells 10,
  21) and `scripts/token_budget.py` both go through
  `transformers.utils.chat_template_utils`, which does `import jinja2`
  directly — confirmed by grepping installed `transformers` source. It
  works today only because Colab's preinstalled `torch` build pulls in
  Jinja2 as one of *its own* transitive dependencies (confirmed:
  `pip show jinja2` → `Required-by: torch` in a clean venv); `transformers`
  itself does not declare `jinja2` as a hard dependency
  (`pip show transformers` lists no `jinja2`). `requirements-colab.txt`'s
  stated purpose is to make "the intended stack... reviewable as
  text/diff," but this implicit, torch-transitive dependency isn't in
  it — inconsistent with how `scripts/token_budget.py`'s own docstring
  already lists `jinja2` explicitly as a needed package. Low practical
  risk (Colab's torch build has carried Jinja2 for years), but the file
  should either list it or note explicitly that it's expected via
  Colab's preinstalled torch, so the "what this notebook actually
  installs" contract stays exact.
- **MVP non-negotiable #1 (fits a free T4 session) is plausible but
  unverified.** 1.5B params in 4-bit NF4 (~1-1.5GB) + small-rank LoRA
  adapters on `q/k/v/o_proj` only, batch size 4 x grad-accum 4, with
  `del model, trainer; gc.collect(); torch.cuda.empty_cache()` between
  sweep candidates (cell-14) so memory doesn't accumulate across the
  3-candidate sweep — all reads as reasonable for a T4's ~15GB VRAM on
  paper. But per the notebook's own framing and `ROADMAP.md`'s "What's
  still open," this — along with bitsandbytes' actual 4-bit quantized
  loading and real training throughput — is genuinely unverifiable
  without a GPU and remains an open assumption until a live Colab run
  confirms it. Flagging this explicitly per the review brief, not as a
  new finding beyond what the project has already disclosed.

## Blocking

None.

## Resolved after review

Both actionable questionable items were addressed in commit `8eb0622`:

- Added a markdown paragraph (cell-8) documenting the full-sequence-loss
  choice as deliberate, with the tradeoff and what would justify
  revisiting it.
- Added `jinja2` explicitly to `requirements-colab.txt` and the notebook's
  install cell, rather than relying on Colab's torch build transitively
  providing it.

The third item (MVP non-negotiable #1's T4-fit being plausible but
unverified without a GPU) isn't something a repo-side fix can close — it
stays an open item until an actual Colab run confirms it, consistent with
`ROADMAP.md`'s existing "What's still open" framing.

Full suite re-verified after both fixes: 119 passed; `ruff check .` and
`ruff format --check .` both clean (including the notebook itself).

## Verdict

No blocking items — day may be closed out.

---

# Review — Next-phase local hardening — 2026-08-30

Scope: preserve distinct multi-file fixes, enforce config field types,
replace truncating/full-sequence preprocessing with exact-length,
completion-only package code, and add a bounded Colab smoke mode.

## Passed locally

- Exact duplicate records are removed by a stable content fingerprint;
  distinct code/diff rows sharing a repository, commit, and CVE survive.
- YAML roots, sections, and typed fields reject malformed values with
  actionable errors.
- `tokenize_training_example` never truncates: it validates the prompt
  prefix, drops over-budget examples, requires terminal EOS, and masks all
  prompt labels with `-100`.
- `CompletionOnlyDataCollator` preserves assistant labels and masks padding;
  torch remains a lazy Colab-only import.
- Smoke mode is bounded to 64 retained training examples, 16 validation
  examples, one rank-8 candidate, and two optimizer steps. The final cell
  packages manifests, counters, logs, adapter files, environment data,
  peak VRAM, and a reloaded-adapter generation into one ZIP.
- Python 3.11 suite: 148 passed; Ruff lint and format checks pass; all
  notebook Python cells parse after accounting for Colab magics.

## Open GPU gate

The bitsandbytes 4-bit load, two optimizer steps, finite validation loss,
peak T4 memory, adapter reload, and generation remain unverified until the
user runs the default smoke mode in a live Colab T4 session and returns
`lora_bench_smoke_artifacts.zip`. The full sweep must remain disabled until
that review succeeds.

## Verdict

Local hardening passes. T4 smoke validation is the only blocking gate for
enabling the full Day 2 run.

---

# Review — Colab T4 smoke validation — 2026-08-31

Artifact: private `lora_bench_smoke_artifacts.zip` returned from a fresh
Colab Tesla T4 run of commit `8722bc5bb18499ca2ab70291bb7cd88c55fefc8a`.
ZIP SHA-256:
`5d68bd9f403b67437dacdec5b614aebb7ff719d0cb506d78c785b1d830061d8e`.

## Passed

- Dataset revision `d4f5c4ea65329d9ccbb8a3b3149e5d06eda5edb2` and git
  provenance match the committed configuration and run metadata.
- Smoke mode retained 64 training and 16 validation examples. Three
  over-budget training examples were dropped without truncation; all 300
  validation examples were eligible before the smoke limit.
- Exactly one rank-8/alpha-16 candidate ran for two optimizer steps with
  finite losses: training loss `0.7287055`, completion-only validation loss
  `1.2186365`, and finite gradient norms.
- Four-bit QLoRA training and fresh-base adapter reload completed on a
  Tesla T4. Peak allocated VRAM was `9,274,672,640` bytes, below the
  reported `15,360 MiB` device capacity. Total notebook elapsed time was
  `221.65` seconds.
- The reloaded adapter produced a non-empty deterministic generation.
- The ZIP contains adapter configuration, tokenizer files, an 8.75 MB
  safetensors weight file, manifests, preprocessing counters, trainer
  history, dependency/GPU metadata, and generation evidence. The weight
  header contains 224 F32 LoRA tensors with shapes consistent with rank 8;
  every archive entry decompressed and hashed successfully.
- Local regression suite after the two Colab-only repairs: 151 passed;
  Ruff lint and format checks pass.

## Verdict

Sprint 5 passes. The smoke figures validate the stack only and are not a
rank decision or benchmark. The full rank sweep may now be implemented and
run; adapter weights remain private.
