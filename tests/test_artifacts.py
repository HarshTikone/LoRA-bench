import hashlib
import json

import pytest

from lora_bench.artifacts import sha256_file, write_checksums


def test_write_checksums_covers_nested_payloads_and_excludes_itself(tmp_path):
    (tmp_path / "adapter").mkdir()
    (tmp_path / "manifest.json").write_text("{}\n", encoding="utf-8")
    weights = tmp_path / "adapter" / "adapter.safetensors"
    weights.write_bytes(b"private adapter bytes")

    manifest = write_checksums(tmp_path, "full")

    assert manifest["run_mode"] == "full"
    assert manifest["self_excluded"] == "checksums.json"
    assert set(manifest["files"]) == {
        "adapter/adapter.safetensors",
        "manifest.json",
    }
    assert (
        manifest["files"]["adapter/adapter.safetensors"]
        == hashlib.sha256(b"private adapter bytes").hexdigest()
    )
    assert json.loads((tmp_path / "checksums.json").read_text(encoding="utf-8")) == manifest
    assert sha256_file(weights) == manifest["files"]["adapter/adapter.safetensors"]


@pytest.mark.parametrize("run_mode", ["", "SMOKE", "training", None])
def test_write_checksums_rejects_invalid_run_mode(tmp_path, run_mode):
    with pytest.raises(ValueError, match="run_mode"):
        write_checksums(tmp_path, run_mode)


def test_write_checksums_requires_existing_directory(tmp_path):
    with pytest.raises(ValueError, match="existing directory"):
        write_checksums(tmp_path / "missing", "smoke")
