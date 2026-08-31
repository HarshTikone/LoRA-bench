# Roadmap

Day-by-day scope for LoRA Bench. Each day's check-in updates the status
here. The rule from the project brief: stay inside a day's scope, don't
pull the next day's work forward — this file is what "in scope" means in
practice, so a later self-review can check against something concrete
instead of an implicit boundary.

- [x] **Day 1 — Data prep pipeline (repo-side, non-GPU).**
  Project scaffolding (src layout, pytest config, requirements split);
  dataset + base-model choice (ADR-0001, ADR-0002); the CVEfixes ->
  instruction-tuning-JSONL pipeline (filter, clean, split, write) with unit
  tests against a bundled fixture and a `--dry-run` no-network smoke test;
  typed/validated YAML config. No GPU code, no notebook yet.

  **Hardening pass (same day, after a deeper second review):** a real
  crash bug (`clean_record` on a non-string code field), a real data-
  corruption bug (`max_chars`/`max_seq_len` mismatch silently truncating
  training labels), and an unpinned/unreproducible dataset revision were
  all found and fixed, alongside a group-aware train/val/test split
  (previous row-level split leaked near-duplicate CVE rows across the
  boundary), a manifest.json for provenance, per-drop-reason accounting,
  and several smaller correctness/packaging fixes. See ADR-0003 through
  ADR-0005 and REVIEW.md for the full account — none of this was caught
  by the first self-review pass, which is itself worth remembering going
  into Day 2's review.

  **Second hardening pass (same day, after a third, deeper-still review):**
  the *first* hardening pass's own new code had a real blocking-severity
  bug of its own — the group-aware split's largest-remaining-need
  heuristic assigned CVE-groups in shuffled order only, which degrades
  badly under realistic heavy-tailed group sizes (a CVE fixed across many
  files is one large group; measured worst-case realized-ratio deviation
  up to 0.192 against a 0.1 target). Fixed by sorting groups largest-first
  before the greedy pass (ADR-0004's amendment). The same review also
  caught a backwards causal explanation in ADR-0003 (the truncation-risk
  residual comes from the token-length *tail* being denser than assumed,
  not the median being sparser — an earlier version of that ADR said the
  opposite) and two smaller `scan_dataset.py` diagnostic bugs. Two hardening
  passes catching real defects in a single day, including one pass
  introducing a genuine bug while fixing others, is itself the strongest
  argument in this repo for never skipping the review step — noted here
  explicitly so it isn't lost to a git log nobody rereads.

- [x] **Day 2 — LoRA/QLoRA fine-tuning (Colab, GPU): smoke validated, full run pending.**
  `notebooks/finetune.ipynb`: runs Day 1's data prep inside Colab, QLoRA
  fine-tunes Qwen2.5-Coder-1.5B-Instruct (bitsandbytes 4-bit + PEFT), runs
  a 3-candidate LoRA rank sweep (r=8/16/32, compared by validation loss)
  per the past-"just a demo" checklist, then does the full fine-tune with
  the winning config and saves the adapter. Added
  `to_chat_messages(FixDiffExample)` to the repo as the single tested
  source of truth for prompt rendering, shared by `scripts/token_budget.py`
  and the notebook, so they can't silently diverge.

  **What's still open:** the bounded smoke run passed on a real Colab T4 at
  commit `8722bc5`, but the three-candidate sweep and three-epoch training
  have not run. No winning hyperparameters, production adapter, or ADR-0006
  exists yet. Those require the reviewed full-mode notebook to run in a
  fresh Colab T4 session and return its private artifact ZIP.

  **Next-phase hardening (2026-08-30):** fixed identifier-only
  deduplication that discarded distinct files from the same fix, added
  runtime config-type validation, and moved exact token-length filtering,
  completion-only labels, and padding into tested package code. The
  notebook now defaults to a bounded rank-8/two-step smoke mode and packages
  all evidence into one ZIP. The returned smoke artifact passed review on
  2026-08-31: rank 8 trained for exactly two steps, validation loss was
  finite, peak allocated VRAM was 9.27 GB, and the adapter reloaded into a
  fresh 4-bit base model and generated non-empty output. Day 2 remains
  operationally open until the full sweep/training artifact is reviewed.

  **Full-mode capture (2026-08-31):** the notebook now exposes only the
  validated `RUN_MODE = "smoke" | "full"` control, keeps smoke committed as
  the default, and records all three fixed-seed probe results, the explicit
  selection rule, three-epoch metrics/history, fresh 4-bit reload evidence,
  and per-payload SHA-256 checksums in the private full-training ZIP.

- [ ] **Day 3 — Quantization + benchmark harness.**
  Quantize the fine-tuned model to GGUF or AWQ (ADR entry for whichever is
  picked); repo-side (non-GPU, testable) eval/benchmark harness code:
  quality metric, latency, memory footprint, cost-per-1K-tokens, run against
  base/fine-tuned/quantized side by side. Harness code and its unit tests
  run here; the actual quantized-model benchmarking run happens in Colab.

- [ ] **Day 4 — Wire the full pipeline + report.**
  One Colab notebook that runs data prep -> fine-tune -> quantize ->
  benchmark -> report start to finish, cells in order, no manual
  babysitting beyond starting them. README's comparison table filled in
  with real numbers from an actual Colab run (never fabricated). Failure
  cases included, not just wins.

- [ ] **Day 5 — Polish + final self-review.**
  Close out anything REVIEW.md still flags as questionable/blocking across
  prior days. Final read-through against all three MVP non-negotiables and
  the past-"just a demo" checklist before calling the project done.

This breakdown is a plan, not a contract — a day's actual check-in is the
source of truth for what shipped, and this file gets checked off/adjusted
to match reality as each day closes.
