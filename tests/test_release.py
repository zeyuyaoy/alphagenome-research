"""Tests release guarantees for imports, public exports, and README examples."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import requests

import ag2kit
from ag2kit import variants as va

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_import_has_no_side_effects(tmp_path):
    env = {**os.environ, "HOME": str(tmp_path), "AG2KIT_CACHE": str(tmp_path / "c")}
    env.pop("XDG_CACHE_HOME", None)
    code = "import ag2kit; c = ag2kit.AG2Client(); print(c.cache_dir)"
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == str(tmp_path / "c" / "ag2")
    assert list(tmp_path.iterdir()) == []


def test_public_api_is_exported():
    for name in ag2kit.__all__:
        assert hasattr(ag2kit, name), name
    assert ag2kit.__version__


class FakeResponse:
    def __init__(self, status, text=""):
        self.status_code, self.text, self.ok = status, text, 200 <= status < 300


def test_failed_reference_lookup_is_not_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(va.time, "sleep", lambda s: None)
    monkeypatch.setattr(va.requests, "get", lambda *a, **k: FakeResponse(503))
    ref = va.ReferenceBase(cache_path=tmp_path / "b.json")
    assert ref.get("chr1", 100) is None
    monkeypatch.setattr(va.requests, "get", lambda *a, **k: FakeResponse(200, "g\n"))
    assert ref.get("1", 100) == "G"
    assert va.ReferenceBase(cache_path=tmp_path / "b.json").get("1", 100, offline=True) == "G"


def test_network_errors_in_reference_lookup_return_none(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise requests.ConnectionError("down")

    monkeypatch.setattr(va.time, "sleep", lambda s: None)
    monkeypatch.setattr(va.requests, "get", boom)
    assert va.ReferenceBase(cache_path=tmp_path / "b.json").get("1", 5) is None
    assert not (tmp_path / "b.json").exists()


def test_cluster_loci_ignores_chr_prefix():
    import pandas as pd

    df = pd.DataFrame(
        {
            "variant_id": ["a", "b"],
            "chrom": ["chr7", "7"],
            "pos": [100, 200],
            "gene": ["G1", "G2"],
            "pval": [1e-3, 1e-5],
        }
    )
    out = ag2kit.cluster_loci(df)
    assert out["is_lead"].tolist() == [False, True]
    assert out["locus_group"].tolist() == ["b", "b"]


def test_empty_null_set_is_an_error_not_a_zero_threshold():
    with pytest.raises(ValueError):
        ag2kit.no_call_threshold([float("nan")])


def _readme_blocks() -> list[str]:
    readme = REPO_ROOT / "README.md"
    text = readme.read_text() if readme.exists() else ""
    return re.findall(r"```python\n(.*?)```", text, flags=re.S)


@pytest.mark.parametrize("block", range(len(_readme_blocks())))
def test_readme_examples_run(block, tmp_path, monkeypatch):
    source = _readme_blocks()[block]
    if "dry_run=False" in source:
        pytest.skip("live example; exercised by the live test")
    monkeypatch.chdir(tmp_path)
    exec(compile(source, f"README.md[python block {block}]", "exec"), {})


def test_sign_metrics_match_hand_calculations():
    deltas = [0.9, 0.5, -0.2, -0.8, 0.1, float("nan")]
    betas = [1.0, -0.5, -1.0, -2.0, 0.3, 1.0]
    # The two positive effects outrank five of the six positive-negative pairs.
    assert ag2kit.sign_auroc(deltas, betas) == pytest.approx(5 / 6)
    assert np.isnan(ag2kit.sign_auroc([0.1, 0.2], [1.0, 2.0]))
    # Running accuracy in descending score magnitude is 1, 1, 2/3, 3/4, 4/5.
    out = ag2kit.recall_at_sign_accuracy(deltas, betas, accuracy=0.75)
    assert (out["n_called"], out["n"], out["recall"]) == (5, 5, 1.0)
    out = ag2kit.recall_at_sign_accuracy(deltas, betas, accuracy=0.9)
    assert (out["n_called"], out["min_abs_delta"]) == (2, 0.8)
    assert ag2kit.spearman(deltas, betas)["n"] == 5


def test_zero_predictions_never_count_toward_sign_accuracy():
    out = ag2kit.recall_at_sign_accuracy([0.0, 0.0], [1.0, -1.0], accuracy=0.5)
    assert out["n_called"] == 0 and out["recall"] == 0.0


def test_group_weighting_skips_small_and_undefined_groups():
    deltas = [0.9, -0.1, 0.4, -0.3, 0.5, 0.2, 0.7]
    betas = [1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0]
    groups = ["lung", "lung", "blood", "blood", "blood", "liver", "skin"]
    weighted, table = ag2kit.by_group(ag2kit.sign_auroc, deltas, betas, groups)
    blood = ag2kit.sign_auroc([0.4, -0.3, 0.5], [1.0, 1.0, -1.0])
    assert weighted == pytest.approx((2 * 1.0 + 3 * blood) / 5)
    assert table.set_index("group").loc["skin", "n"] == 1
