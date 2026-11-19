"""Tests allele orientation, statistics, and run provenance."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import ag2kit
from ag2kit import client as ag2
from ag2kit import stats as ev
from ag2kit import variants as va
from ag2kit.provenance import (
    Run,
    make_run_id,
    read_table,
    stable_hash,
    write_table,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "eff,ref,alt,status",
    [
        ("G", "T", "C", "effect_allele_not_in_ref_alt"),
        ("A", "A", "T", "palindromic"),
        ("C", "C", "G", "palindromic"),
        ("AT", "AT", "A", "not_snv"),
        ("C", "C", "C", "ref_equals_alt"),
    ],
)
def test_unorientable_records_are_excluded(eff, ref, alt, status):
    out = va.harmonize_effect(0.1, eff, ref, alt)
    assert out.status == status
    assert out.estimate is None


@pytest.mark.parametrize("estimate", [None, np.nan, np.inf, -np.inf])
def test_missing_or_nonfinite_estimate_is_excluded(estimate):
    out = va.harmonize_effect(estimate, "C", "T", "C")
    assert out.status == "missing_statistics"
    assert out.estimate is None


def test_reference_mismatch_is_detectable_from_cached_bases(tmp_path):
    cache = tmp_path / "bases.json"
    cache.write_text('{"21:41422154": "T", "1:12345": "G"}')
    ref = va.ReferenceBase(cache_path=cache)
    assert ref.get("chr21", 41422154) == "T"
    assert ref.get("21", 41422154) != "A"
    assert ref.get("1", 12345) == "G"


def test_duplicate_variant_gene_pairs_are_detectable():
    df = pd.DataFrame({"variant_id": ["rs1", "rs1", "rs2"], "gene": ["GENE3", "GENE3", "OAS1"]})
    dup = df.duplicated(subset=["variant_id", "gene"], keep=False)
    assert dup.tolist() == [True, True, False]


def test_identical_conditions_give_exactly_zero_interaction():
    for v in (-1.3, 0.0, 2.7):
        assert ag2.interaction(v, v) == 0.0


def test_non_finite_scores_propagate_as_nan():
    assert math.isnan(ag2.interaction(np.nan, 0.1))
    assert math.isnan(ag2.interaction(np.inf, 0.1))
    assert math.isnan(ag2.interaction(None, 0.1))


def test_zero_prediction_is_a_no_call_not_a_concordance():
    assert ev.classify(0.0, -0.03) == ev.NO_CALL
    assert ev.classify(0.0, 0.03) == ev.NO_CALL
    con, calls = ev.concordance([0.0] * 5, [-0.03, 0.02, -0.01, 0.04, -0.02])
    assert con.k == 0 and con.n_called == 0 and con.n_no_call == 5
    assert math.isnan(con.estimate)
    assert set(calls) == {ev.NO_CALL}


def test_no_call_threshold_from_same_condition_control():
    null = [0.001, -0.002, 0.004, -0.0005, 0.003, 0.01, -0.0015, 0.002, 0.0025, -0.003]
    thr = ev.no_call_threshold(null, 0.90)
    assert thr == pytest.approx(float(np.quantile([abs(x) for x in null], 0.90)))
    assert ev.classify(thr * 0.5, -0.03, thr) == ev.NO_CALL
    assert ev.classify(thr * 2, -0.03, thr) == ev.DISCORDANT
    assert ev.classify(-thr * 2, -0.03, thr) == ev.CONCORDANT


def test_concordance_reports_numerator_and_denominator():
    deltas = [-0.2, 0.3, -0.4, 0.5, 0.0]
    betas = [-0.03, 0.02, 0.01, 0.04, -0.02]
    con, calls = ev.concordance(deltas, betas)
    assert (con.k, con.n_called, con.n_no_call, con.n_total) == (3, 4, 1, 5)
    assert con.estimate == pytest.approx(0.75)
    assert calls[2] == ev.DISCORDANT
    row = con.as_row("primary")
    assert row["k_concordant"] == 3 and row["n_called"] == 4
    assert 0 <= row["ci95_low"] <= row["concordance"] <= row["ci95_high"] <= 1
    assert row["concordance_conservative"] == pytest.approx(3 / 5)


def test_mcnemar_pairs_only_disagreements():
    a = [ev.CONCORDANT, ev.DISCORDANT, ev.CONCORDANT, ev.CONCORDANT]
    b = [ev.DISCORDANT, ev.DISCORDANT, ev.CONCORDANT, ev.DISCORDANT]
    out = ev.mcnemar(a, b)
    assert out["a_better"] == 2 and out["b_better"] == 0
    assert out["n_discordant_pairs"] == 2


def test_spearman_is_rank_only_and_scale_invariant():
    deltas = [0.1, 0.2, 0.3, 0.4, 0.5]
    betas = [0.01, 0.02, 0.03, 0.04, 0.05]
    a = ev.spearman(deltas, betas, absolute=True)
    b = ev.spearman(deltas, [-x * 1000 for x in betas], absolute=True)
    assert a["rho"] == pytest.approx(1.0) and b["rho"] == pytest.approx(1.0)
    assert a["n"] == 5


def test_spearman_not_estimable_with_too_few_points():
    assert math.isnan(ev.spearman([0.1, 0.2], [0.01, 0.02])["rho"])


def test_correlated_variants_collapse_to_one_lead_per_locus():
    df = pd.DataFrame(
        {
            "variant_id": ["rsA", "rsB", "rsC", "rsD", "rsE"],
            "chrom": ["21", "21", "21", "21", "1"],
            "pos": [41422154, 41422500, 43500000, 41900000, 1000],
            "gene": ["GENE3", "GENE3", "FARAWAY", "OTHER", "ELSE"],
            "pval": [1e-6, 1e-4, 1e-5, 1e-3, 1e-2],
        }
    )
    out = va.cluster_loci(df, window=1_000_000)
    leads = out[out["is_lead"]]["variant_id"].tolist()
    assert "rsA" in leads and "rsB" not in leads and "rsD" not in leads
    assert "rsC" in leads and "rsE" in leads
    assert out.loc[out["variant_id"] == "rsB", "locus_group"].item() == "rsA"
    assert out["locus_group"].nunique() == 3
    assert list(out["variant_id"]) == list(df["variant_id"])


def test_lead_is_the_most_significant_pair_in_its_cluster():
    df = pd.DataFrame(
        {
            "variant_id": ["weak", "strong"],
            "chrom": ["7", "7"],
            "pos": [100_000, 200_000],
            "gene": ["G1", "G2"],
            "pval": [1e-2, 1e-8],
        }
    )
    out = va.cluster_loci(df)
    assert out[out["is_lead"]]["variant_id"].tolist() == ["strong"]


@pytest.mark.parametrize("index", [pd.Index([7, 7, 2]), pd.Index([3, 1, 4], name="sample")])
def test_cluster_loci_preserves_row_index_and_order(index):
    df = pd.DataFrame(
        {
            "variant_id": ["weak", "strong", "other"],
            "chrom": ["7", "7", "1"],
            "pos": [100_000, 200_000, 10_000],
            "gene": ["G1", "G2", "G3"],
            "pval": [1e-2, 1e-8, 1e-5],
        },
        index=index,
    )

    out = va.cluster_loci(df)

    pd.testing.assert_frame_equal(out[df.columns], df)
    assert out["locus_group"].tolist() == ["strong", "strong", "other"]
    assert out["is_lead"].tolist() == [False, True, True]


def test_ld_clusters_requires_its_columns():
    with pytest.raises(ValueError, match="requires columns"):
        va.cluster_loci(pd.DataFrame({"variant_id": ["rs1"]}))


def test_cache_key_refuses_nan_identity():
    with pytest.raises(ValueError):
        stable_hash({"delta": float("nan")})


def test_unknown_status_is_rejected(tmp_path):
    run = Run("stage", tmp_path)
    with pytest.raises(ValueError, match="unknown status"):
        run.status("x", "almost_done")


def test_non_finite_values_round_trip_stably(tmp_path):
    df = pd.DataFrame(
        {
            "locus": ["a", "b", "c", "d"],
            "delta": [0.12345678901234567, np.nan, np.inf, -np.inf],
            "beta": [-0.031166, 0.0, 1e-12, -5.0],
        }
    )
    path = write_table(df, tmp_path / "t.tsv", "run-1")
    back = read_table(path)
    assert back["run_id"].unique().tolist() == ["run-1"]
    assert back.loc[0, "delta"] == 0.12345678901234567
    assert pd.isna(back.loc[1, "delta"])
    assert np.isposinf(back.loc[2, "delta"]) and np.isneginf(back.loc[3, "delta"])
    assert back["delta"].dtype.kind == "f"
    assert back.loc[0, "beta"] == pytest.approx(-0.031166, abs=1e-12)


def test_run_ids_are_unique_within_a_second():
    assert len({make_run_id() for _ in range(5)}) == 5


def test_results_are_written_under_a_run_id_and_not_overwritten(tmp_path):
    r1 = Run("primary", tmp_path, run_id="runA")
    r2 = Run("primary", tmp_path, run_id="runB")
    r1.table(pd.DataFrame({"x": [1]}), "out.tsv")
    r2.table(pd.DataFrame({"x": [2]}), "out.tsv")
    assert read_table(tmp_path / "results/primary/runA/out.tsv").loc[0, "x"] == 1
    assert read_table(tmp_path / "results/primary/runB/out.tsv").loc[0, "x"] == 2


def test_run_manifest_records_provenance(tmp_path):
    run = Run("stage", tmp_path)
    run.note("source_sha256", "deadbeef")
    man = read_table(run.finish()).set_index("item")["value"]
    for key in ("run_id", "git_sha", "worktree_dirty", "version.python"):
        assert key in man.index and man[key]
    assert man["source_sha256"] == "deadbeef"


def test_no_credentials_in_tracked_files():
    if not (REPO_ROOT / ".git").exists():
        pytest.skip("repository hygiene check; needs a git checkout")
    env_example = (REPO_ROOT / ".env.example").read_text()
    assert "ALPHA_GENOME_API_KEY=" in env_example
    assert env_example.strip().endswith("ALPHA_GENOME_API_KEY=")
    assert "cache/" in (REPO_ROOT / ".gitignore").read_text()
    assert not (REPO_ROOT / ".env").exists() or ".env" in (REPO_ROOT / ".gitignore").read_text()


def test_missing_comparator_leaves_the_analysis_rather_than_counting_against_the_model():
    deltas = [0.2, -0.3, 0.4]
    betas = [0.02, np.nan, 0.0]
    con, calls = ev.concordance(deltas, betas)
    assert calls == [ev.CONCORDANT, ev.NO_COMPARATOR, ev.NO_COMPARATOR]
    assert (con.k, con.n_called, con.n_no_call, con.n_no_comparator) == (1, 1, 0, 2)
    assert con.n_total == 1
    row = con.as_row("primary")
    assert row["n_no_comparator"] == 2
    assert row["concordance_conservative"] == pytest.approx(1.0)


def test_mcnemar_counts_both_directions():
    a = [ev.CONCORDANT, ev.DISCORDANT, ev.CONCORDANT, ev.DISCORDANT]
    b = [ev.DISCORDANT, ev.CONCORDANT, ev.CONCORDANT, ev.CONCORDANT]
    out = ev.mcnemar(a, b)
    assert out["a_better"] == 1
    assert out["b_better"] == 2
    assert out["n_discordant_pairs"] == 3
    assert 0 < out["p_exact"] <= 1
    mirrored = ev.mcnemar(b, a)
    assert mirrored["a_better"] == out["b_better"]
    assert mirrored["p_exact"] == pytest.approx(out["p_exact"])


def test_monte_carlo_summary_matches_hand_calculation():
    vals = [0.010, 0.012, 0.008, 0.011, 0.009, 0.010]
    out = ev.monte_carlo_summary(vals, seeds=range(1, 7), first_half=range(1, 4))
    sd = float(np.std(vals, ddof=1))
    assert out["mean"] == pytest.approx(0.010)
    assert out["mcse"] == pytest.approx(sd / math.sqrt(6))
    assert out["t_critical"] == pytest.approx(2.570582, rel=1e-6)
    assert out["half1_mean"] == pytest.approx(0.010) and out["half2_mean"] == pytest.approx(0.010)
    assert out["directionally_resolved"]


def test_monte_carlo_summary_needs_both_the_interval_and_the_halves():
    noisy = [0.02, -0.03, 0.01, -0.01, 0.025, -0.02]
    assert not ev.monte_carlo_summary(noisy, range(6), range(3))["ci_excludes_zero"]
    split = [0.05, 0.05, 0.05, 0.05, -0.001, 0.05]
    out = ev.monte_carlo_summary(split, range(6), [4, 5, 9])
    assert out["ci_excludes_zero"]
    assert (
        ev.monte_carlo_summary([0.05, 0.05, -0.05, -0.04], range(4), [0, 1])["halves_agree_in_sign"]
        is False
    )
    zero_half = ev.monte_carlo_summary([0.0, 0.0, 0.03, 0.02], range(4), [0, 1])
    assert not zero_half["halves_agree_in_sign"] and not zero_half["directionally_resolved"]
    with pytest.raises(ValueError):
        ev.monte_carlo_summary([0.1, float("nan")], [1, 2], [1])
    with pytest.raises(ValueError):
        ev.monte_carlo_summary([0.1, 0.2], [1, 2], [1, 2])


def v01_cluster_loci(df, window):
    work = df.reset_index(drop=True).sort_values(["pval", "variant_id"], kind="mergesort")
    chrom = work["chrom"].map(lambda c: str(c).removeprefix("chr"))
    pos, gene = work["pos"].astype(float), work["gene"].astype(str)
    group = pd.Series(pd.NA, index=work.index, dtype="object")
    for i in work.index:
        if pd.notna(group[i]):
            continue
        near = (chrom == chrom[i]) & (pos - pos[i]).abs().le(window)
        group[(near | (gene == gene[i])) & group.isna()] = work.at[i, "variant_id"]
    return group.sort_index().tolist()


@settings(max_examples=50, deadline=None)
@given(
    st.lists(
        st.tuples(
            st.sampled_from(["1", "chr1", "2"]),
            st.integers(1, 5000),
            st.sampled_from(["G1", "G2", "G3", "G4"]),
            st.floats(1e-10, 1),
        ),
        min_size=1,
        max_size=40,
    ),
    st.sampled_from([0, 100, 1000]),
)
def test_cluster_loci_matches_the_reference_scan(rows, window):
    df = pd.DataFrame(rows, columns=["chrom", "pos", "gene", "pval"])
    df["variant_id"] = [f"v{i}" for i in range(len(df))]
    assert va.cluster_loci(df, window)["locus_group"].tolist() == v01_cluster_loci(df, window)


def test_cluster_loci_accepts_gtex_column_names():
    df = pd.DataFrame(
        {
            "variant_id": ["a", "b"],
            "chrom": "chr1",
            "pos": [1, 5_000_000],
            "gene_id": ["G", "G"],
            "pvalue": [0.1, 0.01],
        }
    )
    out = va.cluster_loci(df, gene="gene_id", pval="pvalue")
    assert out["locus_group"].tolist() == ["b", "b"]


def test_annotate_folds_labels_by_containing_split(tmp_path):
    pytest.importorskip("alphagenome")
    bed = tmp_path / "regions.bed"
    bed.write_text(
        "chr1\t0\t100\tfold4\nchr1\t50\t150\tfold0\nchr1\t200\t300\tfold3\n"
        "chr2\t0\t1000\tfold4\nchr2\t10\t20\tfold1\n"
    )
    df = pd.DataFrame(
        {"chrom": ["1", "chr1", "1", "1", "2", "2", "3"], "pos": [10, 60, 250, 175, 15, 500, 5]}
    )
    out = ag2kit.annotate_folds(df, "FOLD_1", str(bed))
    assert out["fold_subset"].tolist() == [
        "test",
        "train",
        "valid",
        "none",
        "train",
        "test",
        "none",
    ]
    with pytest.raises(ValueError, match="ALL_FOLDS"):
        ag2kit.annotate_folds(df, "ALL_FOLDS", str(bed))
