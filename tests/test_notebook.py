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


def test_smoke_mode_is_default_and_bounded():
    notebook = load_notebook()
    source = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
    )
    assert "SMOKE_TEST = True" in source
    assert "64 if SMOKE_TEST else None" in source
    assert "16 if SMOKE_TEST else None" in source
    assert "PROBE_MAX_STEPS = 2 if SMOKE_TEST else 50" in source
    assert "SWEEP_CANDIDATES = SWEEP_CANDIDATES[:1]" in source
    assert 'sweep_results[0]["r"] == 8' in source


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
    assert "run_metadata.json" in final_source
    assert "reloaded_generation.json" in final_source
    assert final_source.rstrip().endswith("files.download(archive_path)")

    earlier_source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"][:-1])
    assert "files.download(" not in earlier_source
