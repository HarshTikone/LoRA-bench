# Notebooks

`finetune.ipynb` — Day 2: data prep (re-running Day 1's pipeline inside
Colab) through LoRA/QLoRA fine-tuning, including the rank sweep and
adapter save. Built to run standalone on a free-tier T4; **not yet run**
— it needs an actual Colab GPU session this environment doesn't have, so
no result from it should be treated as measured until a human has
actually run it and brought back real numbers (see the notebook's own
"Before you start"/"Next" sections).

Day 3 will extend this same notebook with quantization + benchmark cells
rather than adding a second notebook, so the eventual deliverable is one
notebook running the full `data prep -> fine-tune -> quantize -> benchmark
-> report` pipeline per the project brief, not several disconnected ones.
