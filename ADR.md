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
(from streaming and scanning ~3,000 live rows on 2026-08-22, see
`src/lora_bench/config.py` and `configs/default.yaml`):
- `severity` is frequently the literal string `"nan"` rather than missing —
  normalized to `"UNKNOWN"` (`normalize_severity`), not left as a stray
  string a downstream branch could mis-handle.
- ~7% of scanned rows have `vulnerable_code == fixed_code` (no-op diffs,
  e.g. doc-only commits swept in by the source mining) — dropped by
  default (`drop_noop_pairs=True`); they teach the model nothing about
  fixing anything.
- A meaningful fraction of rows have an empty `vulnerable_code` or
  `fixed_code` — dropped (`clean_record`).
- At least one row's code field is ~55MB — `max_chars` (default 4000) is
  load-bearing, not cosmetic; without it, a single row can dominate a T4
  session's token/time budget.
- Language distribution (3,000-row scan): PHP and C dominate, followed by
  Python/JavaScript/Go/C++/Java in the low hundreds each, with "Other"/
  "Unknown"/"JSON"/"Markdown" buckets that aren't really source code. The
  default `languages` allowlist (`Python, C, C++, JavaScript, Java, Go`)
  keeps the latter out while covering most of the signal.

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
