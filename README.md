# LoRA Bench

A fine-tuning and inference-cost study: LoRA/QLoRA-tune a small open LLM on
a CVE fix-diff dataset, then compare **base vs. fine-tuned vs. quantized**
versions on quality, latency, memory, and cost per 1K tokens. Everything
GPU-bound runs in a single free-tier Google Colab T4 session; everything
else (data prep, config, the eval/benchmark harness) is plain, unit-tested
Python that runs here, with no GPU.

**Status:** Day 1 and the local Day 2 hardening are done. The fine-tuning
notebook (`notebooks/finetune.ipynb`) completed its bounded two-step smoke
test on a real Colab T4 at commit `8722bc5`, including 4-bit load, training,
finite completion-only validation loss, adapter reload, and generation.
The three-rank sweep and full training run are still pending, so there are
no comparison numbers or production adapter weights yet.
See [ROADMAP.md](ROADMAP.md) for what's next and [ADR.md](ADR.md) for why
the model/dataset were chosen.

This repo will not claim a comparison result until an actual Colab T4 run
has produced it — anything reported as measured always came from a real
run, never a fabricated placeholder.

## What's here vs. what's coming

| Piece | Runs where | Status |
|---|---|---|
| Data prep (CVEfixes -> instruction JSONL) | here (CPU, tested) | done (Day 1) |
| LoRA/QLoRA fine-tune + rank sweep | Colab notebook (T4) | smoke validated; full sweep/training pending (Day 2) |
| GGUF/AWQ quantization | Colab notebook (T4) | not started (Day 3) |
| Eval/benchmark harness (quality/latency/memory/cost) | here (CPU, tested) + Colab (GPU run) | not started (Day 3) |
| End-to-end notebook + report | Colab notebook (T4) | not started (Day 4) |

## Repo layout

```
src/lora_bench/
  config.py          # typed, validated YAML config (data/model/lora sections)
  data/
    schema.py         # FixDiffExample record shape
    cvefixes.py        # CVEfixes -> instruction-tuning JSONL pipeline
    sample_records.json  # bundled --dry-run/test sample (real package data, not a tests/ path)
configs/default.yaml  # the config values actually used by a run
tests/                 # unit tests (no network — see "Tests" below)
notebooks/finetune.ipynb  # Day 2: data prep + LoRA/QLoRA fine-tune + rank sweep (Colab T4)
ADR.md                 # dated decisions that specialize the default stack, and why
ROADMAP.md             # day-by-day scope, so later days don't creep into each other
```

## Setup (local, repo-side only)

This installs only the CPU-side tooling (data prep, config, tests) — not
the GPU stack, which lives in the Colab notebook instead
(`requirements-colab.txt` documents it but is not meant to be pip-installed
here).

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements-repo.txt
.venv/Scripts/pip install -e .
```

(On macOS/Linux use `.venv/bin/pip` instead of `.venv/Scripts/pip`.)

Copy `.env.example` to `.env` and fill in `HF_TOKEN` (a free, read-scope
Hugging Face token) if you want to run data prep against the live dataset.
The CLI loads `.env` automatically (via `python-dotenv`, searching from
wherever you run the command upward — same convention as `.git`) before
touching the network, so nothing else needs to reference the token
explicitly: `huggingface_hub` picks `HF_TOKEN` up from the environment on
its own. `.env` is git-ignored and is for this repo-side tooling only —
inside the Colab notebook, set the same token via Colab's Secrets panel
instead; never hardcode a token in a notebook cell or commit it.

## Running data prep

Against the live Hugging Face dataset (needs network; `HF_TOKEN` in `.env`
raises the anonymous rate limit but isn't strictly required for this
public dataset):

```bash
.venv/Scripts/python -m lora_bench.data.cvefixes --config configs/default.yaml --out-dir data/processed
```

Writes `data/processed/{train,val,test}.jsonl` plus `manifest.json`
alongside them — the config used (including the pinned dataset revision),
this package's version, the git commit that produced the run (if
available), per-split counts, and a breakdown of *why* every dropped row
was dropped. `data/` is git-ignored — it's regenerated from the script,
not checked in; the manifest is what lets a later run (or a human) tell
what a given `train.jsonl` actually came from. The command also prints
that drop-reason breakdown and exits non-zero (`--min-examples`, default
1) if too few examples survive filtering, instead of silently succeeding
on an empty dataset.

Without network, to sanity-check the pipeline wiring against the bundled
fixture sample instead of the real dataset:

```bash
.venv/Scripts/python -m lora_bench.data.cvefixes --dry-run --out-dir /tmp/lora_bench_dryrun
```

## Running the fine-tuning notebook (Day 2, Colab, GPU)

`notebooks/finetune.ipynb` runs data prep + LoRA/QLoRA fine-tuning + the
rank sweep, entirely inside Colab on the free T4 tier:

1. Push this repo to GitHub if you haven't (the notebook's first code
   cell clones it — see the notebook's own "Before you start" for the
   private-repo case).
2. Open the notebook in Colab (upload it, or open directly from GitHub).
3. **Runtime > Change runtime type > T4 GPU.**
4. Optional: add an `HF_TOKEN` secret (Colab's key-icon sidebar panel) —
   raises Hub rate limits, not required since the dataset/model are public.
5. The committed default is `RUN_MODE = "smoke"`. It trains only one
   rank-8 adapter for two optimizer steps and cannot silently run the full
   sweep. This path passed on a real T4 at commit `8722bc5`.
6. For the approved full Day 2 run, use a fresh T4 runtime and change only
   `RUN_MODE` to `"full"`. This runs ranks 8/16/32 for 50 steps each with
   seed 42, selects the lowest finite completion-only validation loss, and
   trains that configuration for three epochs.
7. Download and privately preserve `lora_bench_training_artifacts.zip`.
   Return the ZIP for checksum/result review; do not commit adapter weights.

The first returned smoke ZIP verified this path on a Tesla T4 at commit
`8722bc5`: two rank-8 steps completed, completion-only validation loss was
finite (`1.2186365`), peak allocated VRAM was `9,274,672,640` bytes, and a
freshly reloaded adapter generated a non-empty deterministic response. The
private ZIP's SHA-256 is recorded in `REVIEW.md`; these smoke figures are
stack-validation evidence, not a rank decision or a quality benchmark.
Full-mode artifacts add all three probe results, the selected winner,
three-epoch trainer history, fresh-base reload evidence, and SHA-256 hashes
for every staged payload file (excluding the checksum manifest itself).

## Tests

```bash
.venv/Scripts/python -m pytest
```

All tests run against fixtures or synthetic data — none hit the network,
so they run the same here, in CI, or on a laptop with no internet. (A
`network` pytest marker is reserved for any future test that deliberately
hits a live service; the default `-m "not network"` in `pyproject.toml`
skips those.)

## Dataset

[`hitoshura25/cvefixes`](https://huggingface.co/datasets/hitoshura25/cvefixes)
on Hugging Face — a flattened, per-row export of the CVEfixes research
dataset (Bhandari et al., *CVEfixes: Automated Collection of
Vulnerabilities and Their Fixes from Open-Source Software*, PROMISE 2021):
CVE/CWE metadata plus `vulnerable_code`/`fixed_code`/diff per fix commit,
mined from real open-source GitHub repositories. See
[ADR-0002](ADR.md#adr-0002--dataset-hitoshura25cvefixes-hugging-face-mirror-of-cvefixes)
for why this source over the raw Zenodo dump.

**Attribution/license note:** the dataset wrapper is Apache-2.0, but the
individual code snippets it contains originate from many different
upstream GitHub projects, each under its own license. This project uses
them for research/fine-tuning in the same spirit as the original CVEfixes
paper; it does not relicense or redistribute those projects' source code
beyond the short fix-diff excerpts the dataset already provides.

## Base model

`Qwen/Qwen2.5-Coder-1.5B-Instruct` (Apache-2.0). See
[ADR-0001](ADR.md#adr-0001--base-model-qwen25-coder-15b-instruct) for why.

## License

This repo's own code is [MIT-licensed](LICENSE). That covers the code in
this repository only — it doesn't relicense the CVEfixes-derived dataset
or the base model, each under its own terms noted above.

## Hard constraints this project holds itself to

- Free tier only — no paid API keys, no paid Colab tier assumed.
- Secrets never committed: `HF_TOKEN` lives in a git-ignored `.env` here
  and in Colab's Secrets panel there, never in a notebook cell or history.
- Every day's diff gets a self-review pass (`.claude/agents/reviewer.md`)
  before being called done; open blocking items in `REVIEW.md` block
  closing out the day.
- The eventual report has to show a quality metric that could regress (not
  just the wins), document the LoRA hyperparameter choice as a defended
  decision, and include real failure cases — a table with zero failures
  reads as cherry-picked.
