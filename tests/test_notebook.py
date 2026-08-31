"""Static contracts for the GPU-only Colab notebook."""

import ast
import json
from pathlib import Path

NOTEBOOK_PATH = Path("notebooks/finetune.ipynb")


def load_notebook() -> dict:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def python_only(source: str) -> str:
    """Replace IPython shell/magic lines with same-indent Python no-ops."""
    lines = []
    for line in source.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped.startswith(("!", "%")):
            indent = line[: len(line) - len(stripped)]
            lines.append(f"{indent}pass\n")
        else:
            lines.append(line)
    return "".join(lines)


def test_every_notebook_code_cell_parses_as_python_after_magics_are_removed():
    notebook = load_notebook()
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code_cells
    for cell in code_cells:
        ast.parse(python_only("".join(cell["source"])))


def test_notebook_is_committed_without_execution_outputs():
    notebook = load_notebook()
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []


def test_run_mode_is_validated_with_smoke_as_the_bounded_default():
    notebook = load_notebook()
    source = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
    )
    assert 'RUN_MODE = "smoke"' in source
    assert 'RUN_MODE not in {"smoke", "full"}' in source
    assert 'SMOKE_TEST = RUN_MODE == "smoke"' in source
    assert "SMOKE_TEST = True" not in source
    assert "64 if SMOKE_TEST else None" in source
    assert "16 if SMOKE_TEST else None" in source
    assert "PROBE_MAX_STEPS = 2 if SMOKE_TEST else 50" in source
    assert "PROBE_EVAL_BATCH_SIZE = 4" in source
    assert "ALL_SWEEP_CANDIDATES[:1] if SMOKE_TEST else ALL_SWEEP_CANDIDATES" in source
    assert "expected_candidates = 1 if SMOKE_TEST else 3" in source
    assert "[8] if SMOKE_TEST else [8, 16, 32]" in source


def test_full_mode_records_reproducible_sweep_and_training_evidence():
    notebook = load_notebook()
    source = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
    )
    assert "SEED = 42" in source
    assert source.count('"r": 8, "lora_alpha": 16') == 1
    assert source.count('"r": 16, "lora_alpha": 32') == 1
    assert source.count('"r": 32, "lora_alpha": 64') == 1
    assert '"selection_rule": "lowest finite completion-only validation loss"' in source
    assert 'os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")' in source
    assert source.index("PYTORCH_CUDA_ALLOC_CONF") < source.index("import torch")
    assert source.count("prediction_loss_only=True") == 2
    assert "per_device_eval_batch_size=PROBE_EVAL_BATCH_SIZE" in source
    assert "per_device_eval_batch_size=FULL_EVAL_BATCH_SIZE" in source
    assert "FULL_EPOCHS = 3" in source
    assert '"completed_optimizer_steps": trainer.state.global_step' in source
    assert '"log_history": trainer.state.log_history' in source
    assert '"peak_vram_bytes": torch.cuda.max_memory_allocated()' in source
    assert 'with open("/content/full_training_result.json", "w")' in source
    assert source.count('"run_mode": RUN_MODE') >= 6


def test_saved_adapter_is_reloaded_into_a_fresh_four_bit_base_for_generation():
    notebook = load_notebook()
    source = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
    )
    save_index = source.index("model.save_pretrained(ADAPTER_DIR)")
    delete_index = source.index("del model, trainer", save_index)
    reload_index = source.index(
        "PeftModel.from_pretrained(load_base_model(for_training=False), ADAPTER_DIR)",
        delete_index,
    )
    generation_index = source.index("reload_output = generate(model", reload_index)
    assert save_index < delete_index < reload_index < generation_index
    assert 'getattr(model, "is_loaded_in_4bit", False)' in source
    assert '"fresh_base_model": True' in source
    assert '"base_loaded_in_4bit"' in source
    assert '"do_sample": False' in source
    assert 'return_tensors="pt", add_special_tokens=False' in source


def test_notebook_installs_and_runs_repo_code_with_the_active_kernel():
    notebook = load_notebook()
    source = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
    )
    assert "%pip install -q -e ." in source
    assert "%pip install -q -U transformers" in source
    assert "!pip install" not in source
    assert "importlib.invalidate_caches()" in source
    assert 'source_path = (Path(REPO_DIR) / "src").resolve()' in source
    assert "sys.path.insert(0, str(source_path))" in source
    assert "package_path = Path(lora_bench.__file__).resolve()" in source
    assert "package_path.is_relative_to(Path(REPO_DIR).resolve())" in source
    assert "!{sys.executable} -m lora_bench.data.cvefixes" in source


def test_artifact_download_is_the_final_notebook_action():
    notebook = load_notebook()
    final_cell = notebook["cells"][-1]
    assert final_cell["cell_type"] == "code"
    final_source = "".join(final_cell["source"])
    assert "lora_bench_smoke_artifacts" in final_source
    assert "lora_bench_training_artifacts" in final_source
    assert "full_training_result.json" in final_source
    assert "run_metadata.json" in final_source
    assert "reloaded_generation.json" in final_source
    assert "lora_sweep_results.json" in final_source
    assert "preprocessing_counters.json" in final_source
    assert "write_checksums(artifact_dir, RUN_MODE)" in final_source
    assert "sha256_file(artifact_dir / relative_path) == digest" in final_source
    assert final_source.rstrip().endswith("files.download(archive_path)")

    earlier_source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"][:-1])
    assert "files.download(" not in earlier_source
