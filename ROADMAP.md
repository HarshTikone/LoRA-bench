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

- [ ] **Day 2 — LoRA/QLoRA fine-tuning (Colab, GPU).**
  The Colab notebook: load the Day 1 train/val JSONL, QLoRA fine-tune
  Qwen2.5-Coder-1.5B-Instruct (bitsandbytes 4-bit + PEFT), run the small
  LoRA rank/hyperparameter sweep the past-"just a demo" checklist requires,
  write the winner + reasoning into ADR.md, save adapter weights. Runs only
  in Colab — this repo's non-GPU environment can't execute it, only build
  and review the notebook/config for correctness.

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
