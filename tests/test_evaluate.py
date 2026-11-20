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


def test_multiple_observed_keys_can_match_one_prediction_column():
    p = preds([("chr1", 100, "A", "G", "G1", 0.5, "completed")])
    p["condition"] = "blood"
    o = obs([("chr1", 100, "A", "G", "G1", 1.0, "G")] * 2)
    o["tissue"] = "blood"
    o["mapped_tissue"] = ["blood", "lung"]
    from ag2kit.evaluate import join

    pairs = join(p, o, match={"tissue": "condition", "mapped_tissue": "condition"})
    assert pairs["pair_status"].tolist() == ["ok", "no_prediction"]


@pytest.mark.parametrize("column", ["chrom", "pos", "ref", "alt", "gene_id", "condition"])
def test_missing_join_keys_do_not_match(column):
    p = preds([("chr1", 100, "A", "G", "G1", 0.5, "completed")])
    o = obs([("chr1", 100, "A", "G", "G1", 1.0, "G")])
    p[column] = o[column] = pd.NA
    from ag2kit.evaluate import join

    pairs = join(p, o, match={"condition": "condition"} if column == "condition" else None)
    assert pairs["pair_status"].tolist() == ["no_prediction"]


@pytest.mark.parametrize("column", ["status", "raw_effect"])
def test_missing_prediction_values_are_failed(column):
    p = preds([("chr1", 100, "A", "G", "G1", 0.5, "completed")])
    o = obs([("chr1", 100, "A", "G", "G1", 1.0, "G")])
    p[column] = pd.NA
    from ag2kit.evaluate import join

    assert join(p, o)["pair_status"].tolist() == ["prediction_failed"]


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


def test_sign_accuracy_rejects_unaligned_pairs():
    with pytest.raises(ValueError, match="same length"):
        ag2kit.sign_accuracy([1.0], [1.0, -1.0])


def test_grouped_metrics_are_size_weighted():
    p, o = synthetic()
    ev = ag2kit.evaluate(p, o, group="tissue", n_boot=0)
    t = ev.per_group[ev.per_group["metric"] == "sign_auroc"].set_index("group")
    expected = np.average(t["value"], weights=t["n"])
    assert ev.metrics.set_index("metric").loc["sign_auroc", "estimate"] == pytest.approx(expected)


def test_group_weighting_retains_empty_categorical_groups():
    groups = pd.Categorical(["blood", "blood"], categories=["blood", "lung"])
    estimate, table = ag2kit.by_group(ag2kit.sign_accuracy, [0.5, -0.5], [1.0, -1.0], groups)
    assert estimate == 1.0
    assert table["group"].tolist() == ["blood", "lung"]
    assert table["n"].tolist() == [2, 0]
    assert np.isnan(table["value"].iloc[1])


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


def test_nonfinite_observed_effects_do_not_weight_groups_or_count_as_pairs():
    p = preds([("chr1", i, "A", "G", f"G{i}", 0.5, "completed") for i in range(1, 4)])
    o = obs([("chr1", i, "A", "G", f"G{i}", b, "G") for i, b in enumerate([1.0, -1.0, np.inf], 1)])
    o["tissue"] = ["blood", "lung", "blood"]
    ev = ag2kit.evaluate(p, o, group="tissue", min_group=1, n_boot=0)
    acc = ev.metrics.set_index("metric").loc["sign_accuracy"]
    assert ev.pairs["pair_status"].tolist() == ["ok", "ok", "missing_statistics"]
    assert acc["estimate"] == 0.5
    assert acc["n_pairs"] == acc["n_pairs_used"] == acc["n_clusters"] == 2


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


def test_predictions_with_swapped_reference_are_distinct_even_if_scores_match():
    p = preds(
        [
            ("chr1", 100, "A", "G", "G1", 0.5, "completed"),
            ("chr1", 100, "G", "A", "G1", 0.5, "completed"),
        ]
    )
    o = obs([("chr1", 100, "A", "G", "G1", 1.0, "G")])
    from ag2kit.evaluate import join

    assert join(p, o)["pair_status"].tolist() == ["ambiguous_prediction"]
    assert join(p.iloc[::-1], o)["pair_status"].tolist() == ["ambiguous_prediction"]


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


def test_evaluate_equals_join_then_evaluate_pairs():
    p, o = synthetic()
    ev = ag2kit.evaluate(p, o, group="tissue", n_boot=50, seed=2)
    again = ag2kit.evaluate_pairs(ag2kit.join(p, o), group="tissue", n_boot=50, seed=2)
    pd.testing.assert_frame_equal(ev.metrics, again.metrics)
    pd.testing.assert_frame_equal(ev.per_group, again.per_group)


def test_intervals_share_resamples_so_metric_order_does_not_matter(monkeypatch):
    import sys

    ev_mod = sys.modules["ag2kit.evaluate"]

    p, o = synthetic(n_genes=20)
    pairs = ag2kit.join(p, o)
    first = ag2kit.evaluate_pairs(pairs, n_boot=100, seed=5).metrics.set_index("metric")
    reversed_metrics = dict(reversed(list(ev_mod.METRICS.items())))
    monkeypatch.setattr(ev_mod, "METRICS", reversed_metrics)
    second = ag2kit.evaluate_pairs(pairs, n_boot=100, seed=5).metrics.set_index("metric")
    pd.testing.assert_frame_equal(first, second.loc[first.index])


def test_cluster_sampler_draws_whole_clusters():
    from ag2kit.evaluate import _ClusterSampler

    clusters = np.array(["b", "a", "b", "c", "a", "b"])
    sampler = _ClusterSampler(clusters, np.random.default_rng(0))
    for _ in range(50):
        idx = sampler.draw()
        drawn = pd.Series(clusters[idx]).value_counts()
        sizes = pd.Series(clusters).value_counts()
        assert all(drawn[c] % sizes[c] == 0 for c in drawn.index)
        assert len(set(idx)) == sum(sizes[c] for c in drawn.index)


def test_per_group_intervals_cover_group_estimates():
    p, o = synthetic(n_genes=40, per_gene=6)
    table = ag2kit.evaluate(p, o, group="tissue", n_boot=200).per_group
    finite = table.dropna(subset=["ci95_low"])
    assert len(finite) == 6
    assert (finite["ci95_low"] <= finite["value"]).all()
    assert (finite["value"] <= finite["ci95_high"]).all()


def test_stratifying_the_ledger_by_prediction_magnitude():
    p, o = synthetic(n_genes=40, per_gene=6)
    pairs = ag2kit.join(p, o)
    pairs["magnitude"] = pd.qcut(pairs["predicted"].abs(), 3, labels=["low", "mid", "high"])
    table = ag2kit.evaluate_pairs(pairs, group="magnitude", n_boot=0).per_group
    acc = table[table["metric"] == "sign_accuracy"].set_index("group")["value"]
    assert acc["high"] > acc["low"]


def test_compare_with_itself_is_exactly_zero():
    p, o = synthetic()
    pairs = ag2kit.join(p, o)
    table = ag2kit.compare(pairs, pairs, n_boot=100).set_index("metric")
    assert (table["difference"] == 0).all()
    assert (table["ci95_low"] == 0).all() and (table["ci95_high"] == 0).all()


def test_compare_detects_a_better_predictor_on_common_pairs():
    p, o = synthetic(n_genes=60, per_gene=5)
    good = ag2kit.join(p, o)
    noisy = p.copy()
    noisy["raw_effect"] = np.random.default_rng(9).normal(size=len(p))
    noisy.loc[0, "status"] = "failed"
    bad = ag2kit.join(noisy, o)
    table = ag2kit.compare(good, bad, n_boot=300).set_index("metric")
    assert table.loc["sign_auroc", "n_pairs"] == len(o) - 1
    assert table.loc["sign_auroc", "ci95_low"] > 0
    expected = ag2kit.evaluate_pairs(good.drop(index=0), n_boot=0).metrics.set_index("metric")
    assert table.loc["sign_auroc", "estimate_a"] == expected.loc["sign_auroc", "estimate"]


def test_compare_refuses_ledgers_from_different_observations():
    p, o = synthetic()
    a = ag2kit.join(p, o)
    with pytest.raises(ValueError, match="same observed table"):
        ag2kit.compare(a, a.iloc[::-1], n_boot=0)


def test_evaluate_pairs_requires_a_ledger():
    with pytest.raises(ValueError, match="ledger"):
        ag2kit.evaluate_pairs(pd.DataFrame({"x": [1]}))


def test_mitochondrial_names_match_across_conventions():
    p = preds([("chrMT", 10, "A", "G", "G1", 0.5, "completed")])
    o = obs([("M", 10, "A", "G", "G1", 0.2, "G")])
    assert ag2kit.join(p, o)["pair_status"].tolist() == ["ok"]


def test_compare_scores_each_ledger_against_its_own_orientation():
    p, o = synthetic(n_genes=30)
    flipped = p.assign(ref=p["alt"], alt=p["ref"], raw_effect=-p["raw_effect"])
    a = ag2kit.join(p, o, forward_strand=False)
    b = ag2kit.join(flipped, o, forward_strand=False)
    table = ag2kit.compare(a, b, n_boot=50).set_index("metric")
    assert np.allclose(table["difference"], 0)
    assert (table["estimate_a"] == table["estimate_b"]).all()


def test_compare_reports_pairs_usable_in_only_one_ledger():
    p, o = synthetic()
    worse = p.copy()
    worse.loc[:2, "status"] = "failed"
    table = ag2kit.compare(ag2kit.join(p, o), ag2kit.join(worse, o), n_boot=0)
    assert table["n_usable_a_only"].unique().tolist() == [3]
    assert table["n_usable_b_only"].unique().tolist() == [0]


def test_group_intervals_need_enough_clusters():
    p, o = synthetic(n_genes=40, per_gene=6)
    o["tissue"] = np.where(o["gene_id"].isin(["G0", "G1"]), "rare", "common")
    table = ag2kit.evaluate(p, o, group="tissue", n_boot=100).per_group.set_index("group")
    assert table.loc["rare", "n_clusters"].min() == 2
    assert table.loc["rare", "ci95_low"].isna().all()
    assert table.loc["common", "ci95_low"].notna().all()


def test_default_interval_needs_twenty_clusters():
    p, o = synthetic(n_genes=10)
    assert ag2kit.evaluate(p, o, n_boot=50).metrics["ci95_low"].isna().all()
    assert ag2kit.evaluate(p, o, n_boot=50, min_clusters=5).metrics["ci95_low"].notna().all()
