import json

import pytest

from flashns.provenance import sha256, verify_sources, write_report


def test_changed_artifact_is_rejected(tmp_path):
    directory = tmp_path / "sources"
    directory.mkdir()
    file = directory / "source.txt"
    file.write_text("frozen input")
    (directory / "registry.json").write_text(
        json.dumps(
            {"artifacts": {"x": {"path": "sources/source.txt", "sha256": sha256(file)}}}
        )
    )
    assert verify_sources(tmp_path)["x"]["bytes"] == 12
    file.write_text("changed input")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_sources(tmp_path)


def test_report_cannot_emit_nan_as_success(tmp_path):
    with pytest.raises(ValueError):
        write_report(tmp_path / "bad.json", {"residual": float("nan")})
    assert not (tmp_path / "bad.json").exists()
