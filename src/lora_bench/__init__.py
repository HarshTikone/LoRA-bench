"""LoRA Bench: fine-tuning and inference-cost study for CVE fix-diffs.

Repo-side package. Everything importable from here is plain CPU Python —
data prep, config, and (later) the eval/benchmark harness — so it can be
unit tested in environments without a GPU. The GPU steps (fine-tuning,
quantization, quantized-model serving) live in the Colab notebook, not here.
"""

__version__ = "0.1.0"
