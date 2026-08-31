# Notebooks

`finetune.ipynb` — Day 2: data prep (re-running Day 1's pipeline inside
Colab) through LoRA/QLoRA fine-tuning, including the rank sweep and
adapter save. Built to run standalone on a free-tier T4. Its bounded smoke
mode passed on a real Tesla T4 at commit `8722bc5`; the full three-rank
sweep and three-epoch training run are still pending. The committed
`RUN_MODE = "smoke"` default prevents an accidental expensive run; change
only that value to `"full"` in a fresh runtime for the reviewed Day 2 run
(see the notebook's own "Before you start"/"Next" sections).

Day 3 will extend this same notebook with quantization + benchmark cells
rather than adding a second notebook, so the eventual deliverable is one
notebook running the full `data prep -> fine-tune -> quantize -> benchmark
-> report` pipeline per the project brief, not several disconnected ones.
