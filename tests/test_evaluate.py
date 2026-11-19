"""Tests evaluation metrics and the GTEx adapter using pinned Portal responses."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import ag2kit
from ag2kit import gtex

FIXTURE = Path(__file__).parent / "data" / "gtex_v10_mx1_whole_blood.json"


@pytest.fixture
def gtex_offline(monkeypatch):
    """Serves the pinned GTEx v10 response for the first 12 MX1 pairs in Whole Blood.

    Args:
        monkeypatch: Fixture used to replace the Portal transport.
    """
    saved = json.loads(FIXTURE.read_text())

    def fake_get(path, params, refresh=False):
        if path == "reference/gene":
            assert params["gencodeVersion"] == "v39"
            return saved["gene"]["response"]
        assert params["gencodeId"] == "ENSG00000157601.15"
        return saved["eqtl"]["response"]

    monkeypatch.setattr(gtex, "_get", fake_get)


def test_gtex_eqtls_schema_from_real_response(gtex_offline):
    df = gtex.eqtls(["ENSG00000157601"], "Whole_Blood", "gtex_v10")
    assert list(df.columns) == list(gtex.COLUMNS) and len(df) == 12
    first = df.iloc[0]
    raw = json.loads(FIXTURE.read_text())["eqtl"]["response"]["data"][0]
    chrom, pos, ref, alt, build = raw["variantId"].split("_")
    assert (first["chrom"], first["pos"], first["ref"], first["alt"]) == (chrom, int(pos), ref, alt)
    assert first["beta"] == raw["nes"] and first["effect_allele"] == alt and build == "b38"
    assert first["gene_id"] == "ENSG00000157601" and first["ontology_id"] == "UBERON:0013756"
    assert df["pos"].dtype == "int64"


def test_gtex_rejects_grch37_and_bad_ids():
    with pytest.raises(ValueError, match="GRCh38"):
        gtex.eqtls(["MX1"], "Whole_Blood", "gtex_v7")
    with pytest.raises(ValueError, match="b38"):
        gtex._split_variant(pd.Series(["chr1_100_A_G_b37"]))


def test_read_signif_pairs_v10_parquet(tmp_path):
    raw = pd.DataFrame(
        {
            "phenotype_id": ["ENSG00000157601.15"] * 2,
            "gene_id": ["ENSG00000157601.15"] * 2,
            "variant_id": ["chr21_41394865_G_A_b38", "chr21_41399435_T_C_b38"],
            "slope": [-0.13, 0.2],
            "slope_se": [0.03, 0.05],
            "pval_nominal": [5e-5, 3e-5],
        }
    )
    raw.to_parquet(tmp_path / "Whole_Blood.v10.signif_pairs.parquet")
    df = gtex.read_signif_pairs(
        tmp_path / "Whole_Blood.v10.signif_pairs.parquet", "Whole_Blood", "gtex_v10"
    )
    assert df["se"].tolist() == [0.03, 0.05] and df["variant_id"][1] == "chr21:41399435:T>C"


def preds(rows):
    return pd.DataFrame(
        [
            {"chrom": c, "pos": p, "ref": r, "alt": a, "gene_id": g, "raw_effect": e, "status": s}
            for c, p, r, a, g, e, s in rows
        ]
    )


def obs(rows):
    return pd.DataFrame(
        [
            {"chrom": c, "pos": p, "ref": r, "alt": a, "gene_id": g, "beta": b, "effect_allele": ea}
            for c, p, r, a, g, b, ea in rows
        ]
    )


def test_join_orients_to_the_predicted_alt_and_accounts_for_every_record():
    p = preds(
        [
            ("chr1", 100, "A", "G", "G1", 0.5, "completed"),
            ("chr1", 200, "C", "T", "G1", -0.2, "completed"),
            ("chr1", 300, "A", "T", "G2", 0.1, "completed"),
            ("chr1", 400, "G", "A", "G2", np.nan, "failed"),
            ("chr1", 500, "G", "A", "G3", 0.3, "completed"),
            ("chr1", 500, "G", "A", "G3", 0.4, "completed"),
        ]
    )
    o = obs(
        [
            ("1", 100, "A", "G", "G1.5", 1.0, "G"),
            ("chr1", 200, "T", "C", "G1", 1.0, "C"),
            ("chr1", 300, "A", "T", "G2", 1.0, "T"),
            ("chr1", 400, "G", "A", "G2", 1.0, "A"),
            ("chr1", 500, "G", "A", "G3", 1.0, "A"),
            ("chr1", 600, "G", "A", "G3", 1.0, "A"),
            ("chr1", 100, "A", "G", "G1", 1.0, "C"),
        ]
    )
    from ag2kit.evaluate import join

    pairs = join(p, o)
    assert len(pairs) == len(o)
    assert pairs["pair_status"].tolist() == [
        "ok",
        "ref_mismatch",
        "ok",
        "prediction_failed",
        "ambiguous_prediction",
        "no_prediction",
        "effect_allele_not_in_ref_alt",
    ]
    assert pairs.loc[0, "beta_alt"] == 1.0 and pairs.loc[2, "beta_alt"] == 1.0
    # A strand-unknown source can use swapped labels if its effect allele matches.
    strict = join(p, o, forward_strand=False)
    assert strict.loc[1, "beta_alt"] == -1.0 and strict.loc[1, "allele_flipped"]
    assert strict.loc[2, "pair_status"] == "palindromic"


def test_match_columns_keep_tissues_apart():
    p = preds([("chr1", 100, "A", "G", "G1", 0.5, "completed")] * 2)
    p["condition"] = ["UBERON:0013756", "UBERON:0002048"]
    p.loc[1, "raw_effect"] = -0.5
    o = obs([("chr1", 100, "A", "G", "G1", 1.0, "G")] * 2)
    o["ontology_id"] = ["UBERON:0002048", "UBERON:0013756"]
    from ag2kit.evaluate import join

    pairs = join(p, o, match={"ontology_id": "condition"})
    assert pairs["predicted"].tolist() == [-0.5, 0.5]


def synthetic(n_genes=30, per_gene=4, seed=1):
    rng = np.random.default_rng(seed)
    rows_p, rows_o = [], []
    for g in range(n_genes):
        shared = rng.normal()
        for k in range(per_gene):
            pos = 1000 * g + k + 1
            beta = shared + 0.3 * rng.normal()
            pred = 0.6 * beta + rng.normal()
            rows_p.append(("chr2", pos, "C", "T", f"G{g}", pred, "completed"))
            rows_o.append(("chr2", pos, "C", "T", f"G{g}", beta, "T"))
    p, o = preds(rows_p), obs(rows_o)
    o["tissue"] = np.where(np.arange(len(o)) % 3 == 0, "lung", "blood")
    return p, o


def test_metrics_match_independent_reference_implementations():
    p, o = synthetic()
    ev = ag2kit.evaluate(p, o, n_boot=0)
    m = ev.metrics.set_index("metric")["estimate"]
    d, b = p["raw_effect"].to_numpy(), o["beta"].to_numpy()
    # Count positive-negative pairs directly to check the rank-based AUROC.
    pos, neg = d[b > 0], d[b < 0]
    wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    assert m["sign_auroc"] == pytest.approx(wins / (pos.size * neg.size))
    assert m["spearman_rho"] == pytest.approx(stats.spearmanr(d, b).statistic)
    assert m["sign_accuracy"] == pytest.approx(np.mean(np.sign(d) == np.sign(b)))


def test_grouped_metrics_are_size_weighted():
    p, o = synthetic()
    ev = ag2kit.evaluate(p, o, group="tissue", n_boot=0)
    t = ev.per_group[ev.per_group["metric"] == "sign_auroc"].set_index("group")
    expected = np.average(t["value"], weights=t["n"])
    assert ev.metrics.set_index("metric").loc["sign_auroc", "estimate"] == pytest.approx(expected)


def test_cluster_bootstrap_is_reproducible_and_wider_than_pair_bootstrap():
    p, o = synthetic(n_genes=40, per_gene=8)
    a = ag2kit.evaluate(p, o, n_boot=400, seed=3).metrics.set_index("metric")
    b = ag2kit.evaluate(p, o, n_boot=400, seed=3).metrics.set_index("metric")
    pd.testing.assert_frame_equal(a, b)
    naive = ag2kit.evaluate(p, o, cluster=None, n_boot=400, seed=3).metrics.set_index("metric")
    width = (a["ci95_high"] - a["ci95_low"])["spearman_rho"]
    naive_width = (naive["ci95_high"] - naive["ci95_low"])["spearman_rho"]
    assert width > naive_width
    assert a.loc["spearman_rho", "n_clusters"] == 40
    assert a.loc["sign_auroc", "ci95_low"] <= a.loc["sign_auroc", "estimate"]
    assert a.loc["sign_auroc", "estimate"] <= a.loc["sign_auroc", "ci95_high"]


def test_zero_effects_and_zero_predictions():
    p = preds(
        [("chr1", i, "A", "G", "G", e, "completed") for i, e in [(1, 0.0), (2, 0.5), (3, -1.0)]]
    )
    o = obs([("chr1", i, "A", "G", "G", b, "G") for i, b in [(1, 1.0), (2, 0.0), (3, -1.0)]])
    m = ag2kit.evaluate(p, o, n_boot=0).metrics.set_index("metric")
    assert m.loc["sign_accuracy", "n_pairs"] == 2
    assert m.loc["sign_accuracy", "estimate"] == 0.5


def test_save_round_trip(tmp_path):
    p, o = synthetic(n_genes=5)
    ev = ag2kit.evaluate(p, o, n_boot=10)
    out = ev.save(tmp_path / "eval")
    back = pd.read_parquet(out / "pairs.parquet")
    assert back["beta_alt"].tolist() == ev.pairs["beta_alt"].tolist()
    meta = json.loads((out / "evaluation.json").read_text())
    assert meta["cluster"] == "gene_id" and "ag2kit" in meta["software"]


def test_missing_columns_are_reported():
    p, o = synthetic(n_genes=2)
    with pytest.raises(ValueError, match="effect_allele"):
        ag2kit.evaluate(p, o.drop(columns="effect_allele"))


def test_identical_duplicate_predictions_are_not_ambiguous():
    p = preds([("chr1", 100, "A", "G", "G1", 0.5, "completed")] * 2)
    from ag2kit.evaluate import join

    assert join(p, obs([("chr1", 100, "A", "G", "G1", 1.0, "G")]))["pair_status"][0] == "ok"


def test_failed_variant_level_predictions_are_prediction_failed():
    p = pd.DataFrame(
        [
            {
                "chrom": "chr1",
                "pos": 100,
                "ref": "A",
                "alt": "G",
                "gene_id": "",
                "raw_effect": np.nan,
                "status": "failed",
            }
        ]
    )
    from ag2kit.evaluate import join

    out = join(p, obs([("chr1", 100, "A", "G", "G1", 1.0, "G")]))
    assert out["pair_status"][0] == "prediction_failed"


def test_too_few_clusters_give_no_interval():
    p, o = synthetic(n_genes=3, per_gene=10)
    m = ag2kit.evaluate(p, o, n_boot=50).metrics
    assert m["ci95_low"].isna().all() and m["estimate"].notna().all()
