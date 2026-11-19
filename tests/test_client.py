"""Tests request identity, caching, retries, failure isolation, and SDK wiring.

Offline tests use SDK types with a fake backend. A marked live test checks
one request against the API.
"""

from __future__ import annotations

import json
import threading

import numpy as np
import pandas as pd
import pytest

import ag2kit
from ag2kit import client as ag2

pytest.importorskip("alphagenome")

GENES = ["ENSG00000157601.14", "ENSG00000000003.15"]


def spec(**kw) -> ag2.RequestSpec:
    args = dict(
        variant_id="rs0001",
        chrom="21",
        pos=41422154,
        ref="T",
        alt="C",
        gene_id="ENSG00000157601",
        condition="UBERON:0013756",
        scorer_name="RNA_SEQ",
    )
    return ag2.RequestSpec(**{**args, **kw})


def cell_spec(**kw) -> ag2.RequestSpec:
    args = dict(
        condition="",
        n_cells=4,
        condition_label="control",
        scorer_name="SC_GENE_EXPRESSION_DENOISED",
    )
    return spec(**{**args, **kw})


def cells(n=4, seed=0):
    import anndata as ad

    x = np.random.default_rng(seed).poisson(1.0, size=(n, 3)).astype(np.float32)
    adata = ad.AnnData(X=x)
    adata.var_names = ["ENSG00000157601", "ENSG00000111640", "ENSG00000075624"]
    return adata


def scorer_output(x, obs, var, scorer="RNA_SEQ", variant="chr21:41422154:T>C"):
    """Builds a synthetic AnnData with the metadata the SDK attaches.

    Args:
        x: Gene-by-track score matrix.
        obs: Gene metadata for rows.
        var: Track metadata for columns.
        scorer: SDK scorer name to attach.
        variant: SDK variant string to parse and attach.

    Returns:
        AnnData shaped like an SDK scorer result.
    """
    import anndata as ad
    from alphagenome.data import genome
    from alphagenome.models.v2 import variant_scorers

    var = pd.DataFrame(var)
    if "imputation_mode" not in var:
        var["imputation_mode"] = "none"
    adata = ad.AnnData(X=np.asarray(x, dtype=float), obs=pd.DataFrame(obs), var=var)
    # Use the SDK object types so this fixture exercises Parquet conversion.
    adata.uns["interval"] = genome.Interval("chr21", 40897866, 41946442)
    adata.uns["variant"] = genome.Variant.from_str(str(variant))
    adata.uns["variant_scorer"] = variant_scorers.RECOMMENDED_VARIANT_SCORERS[scorer]
    return adata


class FakeModel:
    """Mimics the SDK with two genes and three tracks whose scores depend on ALT."""

    def __init__(self, fail=None):
        self.calls = []
        self.fail = fail or (lambda kw: None)
        self.lock = threading.Lock()

    def score_variant(self, **kw):
        with self.lock:
            self.calls.append(kw)
        if (exc := self.fail(kw)) is not None:
            raise exc
        v = kw["variant"]
        base = {"C": 1.0, "G": 2.0, "A": 3.0}.get(v.alternate_bases, 4.0)
        x = [[base, -9.0, base + 0.2], [-9.0, -base, -base]]
        obs = {"gene_id": GENES, "strand": ["+", "-"], "gene_name": ["MX1", "TSPAN6"]}
        var = {"strand": ["+", "-", "."], "name": ["p", "m", "u"]}
        scorer = "SC_GENE_EXPRESSION_DENOISED" if "cell_conditioning" in kw else "RNA_SEQ"
        return [scorer_output(x, obs, var, scorer, str(v))]


def live(tmp_path, model=None, **kw) -> ag2.AG2Client:
    return ag2.AG2Client(dry_run=False, cache_dir=tmp_path, model=model or FakeModel(), **kw)


class FakeRpcError(Exception):
    def __init__(self, name):
        super().__init__(f"{name}: details")
        self._name = name

    def code(self):
        return type("Code", (), {"name": self._name})()


@pytest.mark.parametrize(
    "kw,match",
    [
        ({"pos": 0}, "1-based"),
        ({"pos": 1.5}, "1-based"),
        ({"ref": ""}, "A/C/G/T/N"),
        ({"alt": "X"}, "A/C/G/T/N"),
        ({"alt": "t"}, "identical"),
        ({"sequence_length": 1_000_000}, "sequence_length"),
        ({"organism": "MUS_MUSCULUS"}, "HOMO_SAPIENS"),
        ({"condition": ""}, "exactly one"),
        ({"n_cells": 4, "condition_label": "x"}, "exactly one"),
        ({"condition": "whole blood"}, "CURIE"),
        ({"n_cells": 300}, "n_cells"),
    ],
)
def test_invalid_requests_are_rejected_at_construction(kw, match):
    with pytest.raises(ValueError, match=match):
        spec(**kw)


def test_cell_conditioning_needs_a_label():
    with pytest.raises(ValueError, match="condition_label"):
        spec(condition="", n_cells=4)


def test_interval_matches_the_sdk_convention():
    genome = pytest.importorskip("alphagenome.data.genome")
    for pos in (41422154, 2_000_001):
        for length in sorted(ag2.SUPPORTED_SEQUENCE_LENGTHS):
            for ref, alt in (("T", "C"), ("TA", "T"), ("TAC", "T"), ("TACG", "T")):
                s = spec(pos=pos, sequence_length=length, ref=ref, alt=alt)
                sdk = genome.Variant(s.chrom, s.pos, s.ref, s.alt).reference_interval.resize(length)
                assert s.interval() == (sdk.chromosome, sdk.start, sdk.end)


def test_sdk_constants_match():
    from alphagenome.models.v2 import dna_client

    assert set(dna_client.SUPPORTED_SEQUENCE_LENGTHS.values()) == ag2.SUPPORTED_SEQUENCE_LENGTHS
    assert dna_client.MAX_CONDITIONING_CELLS == ag2.MAX_CONDITIONING_CELLS


def test_window_offset_near_a_chromosome_start(tmp_path):
    plan = ag2.AG2Client(cache_dir=tmp_path).plan(spec(pos=100))
    assert plan["interval"].startswith("chr21:0-") and plan["variant_offset_in_interval"] == 99


def test_labels_and_genes_share_one_identity_but_science_does_not(tmp_path):
    c = ag2.AG2Client(cache_dir=tmp_path)
    key = c.cache_key(spec())
    for label in (
        {"variant_id": "other"},
        {"gene_id": "ENSG00000000003"},
        {"gene_symbol": "X"},
        {"seed": 9},
        {"conditioning_set_id": "z"},
        {"chrom": "chr21", "ref": "t", "alt": "c"},
    ):
        assert c.cache_key(spec(**label)) == key, label
    for science in (
        {"alt": "G"},
        {"pos": 41422155, "ref": "A"},
        {"condition": "UBERON:0002048"},
        {"sequence_length": 2**17},
        {"model_version": "FOLD_1"},
        {"scorer_name": "CAGE"},
    ):
        assert c.cache_key(spec(**science)) != key, science
    assert ag2.AG2Client(cache_dir=tmp_path, sdk_version="99").cache_key(spec()) != key
    assert ag2.AG2Client(cache_dir=tmp_path, cache_namespace="2027-01").cache_key(spec()) != key
    assert c.cache_key(cell_spec(), cells(seed=0)) != c.cache_key(cell_spec(), cells(seed=1))


def test_custom_models_never_share_cache_with_the_api(tmp_path):
    live(tmp_path).score(spec())
    api = ag2.AG2Client(dry_run=False, cache_dir=tmp_path)
    assert api.cached(spec()) is None


def test_condition_path_builds_a_1_based_variant_and_reduces_on_strand(tmp_path):
    model = FakeModel()
    client = live(tmp_path, model)
    rec = client.score(spec(), strict=True)
    # MX1 uses the positive-strand and unstranded tracks.
    assert rec["raw_effect"] == pytest.approx(1.1) and rec["n_reduced_values"] == 2
    kw = model.calls[0]
    v, i = kw["variant"], kw["interval"]
    assert (v.chromosome, v.position, v.reference_bases, v.alternate_bases) == (
        "chr21",
        41422154,
        "T",
        "C",
    )
    assert v.start - i.start == client.plan(spec())["variant_offset_in_interval"]
    assert kw["condition"] == "UBERON:0013756" and "cell_conditioning" not in kw
    assert str(kw["variant_scorers"][0]).startswith("GeneMaskLFCScorer")


def test_cell_conditioning_path(tmp_path):
    model = FakeModel()
    c = cells()
    rec = live(tmp_path, model).score(cell_spec(), c, strict=True)
    assert rec["status"] == "completed"
    assert model.calls[0]["cell_conditioning"] is c
    assert model.calls[0]["cell_condition_name"] == "control"
    assert "condition" not in model.calls[0]


@pytest.mark.parametrize(
    "s,conditioning,match",
    [
        (cell_spec(), None, "no conditioning"),
        (spec(), "cells", "reference condition"),
        (cell_spec(), cells(n=3), "n_cells"),
    ],
)
def test_conditioning_mismatches_are_validation_failures(tmp_path, s, conditioning, match):
    model = FakeModel()
    rec = live(tmp_path, model).score(s, conditioning)
    assert rec["error_type"] == "validation" and match in rec["error"]
    assert model.calls == []


def test_cell_only_scorer_is_refused_with_a_reference_condition(tmp_path):
    rec = live(tmp_path).score(spec(scorer_name="SC_GENE_EXPRESSION_DENOISED"))
    assert rec["status"] == "failed" and "cell conditioning" in rec["error"]


def test_unsigned_and_unknown_scorers_are_refused(tmp_path):
    client = ag2.AG2Client(cache_dir=tmp_path)
    with pytest.raises(KeyError):
        client.recommended_scorer(spec(scorer_name="NOPE"))
    from alphagenome.models.v2 import variant_scorers

    unsigned = [
        n for n, s in variant_scorers.RECOMMENDED_VARIANT_SCORERS.items() if not s.is_signed
    ]
    if unsigned:
        with pytest.raises(ValueError, match="unsigned"):
            client.recommended_scorer(spec(scorer_name=unsigned[0]))


def test_gene_reduction_errors_are_explicit_and_do_not_refetch(tmp_path):
    model = FakeModel()
    client = live(tmp_path, model)
    rec = client.score(spec(gene_id="ENSG99999999999"))
    assert rec["status"] == "failed" and "absent" in rec["error"]
    assert client.score(spec())["from_cache"] is True and len(model.calls) == 1


def test_reduce_gene_effect_matches_sdk_strand_rule():
    from alphagenome.models.v2 import variant_scorers

    obs = {"gene_id": ["ENSG1.1", "ENSG2"], "strand": ["-", "."]}
    var = {"strand": ["+", "-", "."], "name": ["p", "m", "u"]}
    adata = scorer_output([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], obs, var)
    sdk = variant_scorers.tidy_anndata(adata, match_gene_strand=True)
    for gene in ("ENSG1", "ENSG2"):
        ours, meta = ag2.extract_gene_effect(adata, gene)
        expected = sdk.loc[sdk["gene_id"] == gene, "raw_score"].mean()
        assert ours == pytest.approx(expected)
        assert meta["n_reduced_values"] == (sdk["gene_id"] == gene).sum()
    with pytest.raises(ValueError, match="non-finite"):
        one = {"gene_id": ["E"], "strand": ["+"]}
        ag2.extract_gene_effect(
            scorer_output([[np.nan]], one, {"strand": ["+"], "name": ["t"]}), "E"
        )
    with pytest.raises(ValueError, match="no track on the strand"):
        ag2.extract_gene_effect(scorer_output([[0.3]], one, {"strand": ["-"], "name": ["t"]}), "E")


def test_cache_round_trips_the_full_sdk_table(tmp_path):
    client = live(tmp_path)
    first = client.score(spec())
    raw = client.raw_scores(spec())
    assert set(raw["gene_id"]) == {"ENSG00000157601", "ENSG00000000003"}
    assert {"raw_score", "track_strand", "gene_strand", "variant_scorer"} <= set(raw.columns)
    assert len(raw) == 6
    again = live(tmp_path, FakeModel(fail=lambda kw: RuntimeError("must not be called")))
    rec = again.score(spec())
    assert rec["from_cache"] is True and rec["raw_effect"] == first["raw_effect"]
    assert rec["requested_utc"] == first["requested_utc"]


@pytest.mark.parametrize("garbage", [b"\x00\xffnot parquet", b""])
def test_corrupt_cache_file_is_a_miss_and_is_replaced(tmp_path, garbage):
    client = live(tmp_path)
    path = client._cache_path(client.cache_key(spec()))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(garbage)
    rec = client.score(spec())
    assert rec["status"] == "completed" and rec["from_cache"] is False
    assert client.score(spec())["from_cache"] is True


def test_cache_file_under_the_wrong_name_is_a_miss(tmp_path):
    client = live(tmp_path)
    client.score(spec(alt="G"))
    src = client._cache_path(client.cache_key(spec(alt="G")))
    src.rename(client._cache_path(client.cache_key(spec())))
    assert client.cached(spec()) is None


def test_validation_errors_are_not_retried_and_strict_mode_raises(tmp_path):
    model = FakeModel(fail=lambda kw: FakeRpcError("INVALID_ARGUMENT"))
    with pytest.raises(ag2kit.ScoringError) as info:
        live(tmp_path, model, backoff=0).score(spec(), strict=True)
    assert info.value.error_type == "validation" and len(model.calls) == 1
    assert info.value.record["status"] == "failed"


def test_quota_errors_are_retried_up_to_the_limit_then_succeed(tmp_path):
    model = FakeModel(fail=lambda kw: FakeRpcError("RESOURCE_EXHAUSTED"))
    rec = live(tmp_path, model, backoff=0, max_retries=3).score(spec())
    assert rec["error_type"] == "quota" and rec["attempts"] == 3 and len(model.calls) == 3
    n = {"i": 0}

    def flaky(kw):
        n["i"] += 1
        return TimeoutError("slow") if n["i"] < 2 else None

    rec = live(tmp_path, FakeModel(fail=flaky), backoff=0).score(spec())
    assert rec["status"] == "completed" and rec["attempts"] == 2


def test_failures_are_not_cached(tmp_path):
    live(tmp_path, FakeModel(fail=lambda kw: FakeRpcError("INTERNAL"))).score(spec())
    assert not list(tmp_path.glob("*.parquet"))


@pytest.mark.parametrize(
    "exc,kind",
    [
        (FakeRpcError("RESOURCE_EXHAUSTED"), "quota"),
        (FakeRpcError("UNAVAILABLE"), "transient"),
        (FakeRpcError("INVALID_ARGUMENT"), "validation"),
        (FakeRpcError("UNAUTHENTICATED"), "credentials"),
        (FakeRpcError("INTERNAL"), "upstream"),
        (ValueError("bad gene"), "validation"),
        (PermissionError("denied"), "credentials"),
        (ConnectionError("reset"), "network"),
        (TimeoutError("slow"), "network"),
        (RuntimeError("DEADLINE_EXCEEDED after 30s"), "transient"),
    ],
)
def test_errors_are_classified(exc, kind):
    assert ag2.classify_error(exc) == kind


def test_api_key_never_appears_in_error_records(tmp_path, monkeypatch):
    monkeypatch.setenv(ag2.API_KEY_ENV, "sekrit-key-123")
    model = FakeModel(fail=lambda kw: RuntimeError("bad key sekrit-key-123"))
    rec = live(tmp_path, model).score(spec())
    assert "sekrit" not in rec["error"] and "<redacted>" in rec["error"]


def test_dry_run_and_missing_credentials(tmp_path, monkeypatch):
    monkeypatch.delenv(ag2.API_KEY_ENV, raising=False)
    assert ag2.AG2Client(cache_dir=tmp_path).score(spec())["status"] == "not_run"
    rec = ag2.AG2Client(dry_run=False, cache_dir=tmp_path).score(spec())
    assert rec["status"] == "blocked" and rec["error_type"] == "blocked"
    with pytest.raises(ag2kit.CredentialsError):
        ag2.AG2Client(dry_run=False, cache_dir=tmp_path).connect()


def test_model_version_mismatch_is_refused(tmp_path):
    rec = live(tmp_path, model_version="ALL_FOLDS").score(spec(model_version="FOLD_1"))
    assert rec["error_type"] == "validation" and "model_version" in rec["error"]


def test_score_many_keeps_order_dedups_requests_and_isolates_failures(tmp_path):
    model = FakeModel(
        fail=lambda kw: ValueError("bad") if kw["variant"].alternate_bases == "G" else None
    )
    client = live(tmp_path, model)
    specs = [
        spec(alt=a, gene_id=g, variant_id=f"{a}{g}")
        for a in ("C", "G", "A")
        for g in ("ENSG00000157601", "ENSG00000000003")
    ]
    recs = client.score_many(specs, max_workers=3)
    assert [r["variant_id"] for r in recs] == [s.variant_id for s in specs]
    assert len(model.calls) == 3
    assert [r["status"] for r in recs] == ["completed", "completed", "failed", "failed"] + [
        "completed"
    ] * 2
    assert recs[1]["raw_effect"] == pytest.approx(-1.0)
    assert recs[4]["raw_effect"] == pytest.approx(3.1)


def test_score_many_resumes_after_interruption(tmp_path):
    specs = [spec(alt=a) for a in "CGA"]
    live(tmp_path).score_many(specs[:2])
    model = FakeModel()
    recs = live(tmp_path, model).score_many(specs)
    assert [r["from_cache"] for r in recs] == [True, True, False] and len(model.calls) == 1


def test_score_many_maps_conditioning_by_label(tmp_path):
    model = FakeModel()
    a, b = cells(seed=0), cells(seed=1)
    specs = [cell_spec(), cell_spec(condition_label="IAV")]
    recs = live(tmp_path, model).score_many(specs, {"control": a, "IAV": b})
    assert {id(k["cell_conditioning"]) for k in model.calls} == {id(a), id(b)}
    assert recs[0]["conditioning_sha256"] != recs[1]["conditioning_sha256"]


def test_score_variants_table_keeps_every_row_and_explodes_genes(tmp_path):
    variants = pd.DataFrame(
        {
            "variant_id": ["v1", "v2", "v3", "v4"],
            "chrom": ["chr21", "chr21", "chr21", "chr21"],
            "pos": [41422154, 41422154, 41422154, 41422154],
            "ref": ["T", "T", "T", "T"],
            "alt": ["C", "C", "<DEL>", "G"],
            "input_status": ["ok", "ok", "symbolic_allele", "ok"],
            "assembly": ["GRCh38", "GRCh38", "GRCh38", "GRCh37"],
        }
    )
    model = FakeModel()
    out = live(tmp_path, model).score_variants(variants, condition="UBERON:0013756")
    assert len(model.calls) == 1
    assert out["row"].tolist() == [0, 0, 1, 1, 2, 3]
    assert out["status"].tolist() == ["completed"] * 4 + ["failed", "failed"]
    assert "symbolic_allele" in out.loc[4, "error"] and "GRCh38" in out.loc[5, "error"]
    assert set(out.loc[:1, "gene_name"]) == {"MX1", "TSPAN6"}
    assert out.loc[0, "input_variant_id"] == "v1" and out.loc[0, "source"].startswith("custom:")


def test_score_variants_pairs_mode(tmp_path):
    pairs = pd.DataFrame(
        {
            "chrom": ["21", "21"],
            "pos": [41422154, 41422154],
            "ref": ["T", "T"],
            "alt": ["C", "C"],
            "gene_id": ["ENSG00000000003.15", "ENSG00000157601"],
        }
    )
    out = live(tmp_path).score_variants(pairs, condition="UBERON:0013756", gene_column="gene_id")
    assert out["raw_effect"].tolist() == pytest.approx([-1.0, 1.1])
    out.to_parquet(tmp_path / "p.parquet")
    back = pd.read_parquet(tmp_path / "p.parquet")
    assert back["raw_effect"].tolist() == out["raw_effect"].tolist()


@pytest.mark.live
def test_live_score_of_one_variant(tmp_path):
    """Checks live request wiring with one Whole Blood SNV near MX1."""
    if not ag2.api_key_available():
        pytest.skip(f"{ag2.API_KEY_ENV} not set")
    s = spec()
    client = ag2.AG2Client(dry_run=False, cache_dir=tmp_path / "ag2")
    rec = client.score(s, strict=True)
    assert rec["status"] == "completed" and np.isfinite(rec["raw_effect"])
    assert client.score(s)["from_cache"] is True


def test_specs_sharing_a_request_are_validated_individually(tmp_path):
    c = cells(n=4)
    for order in ((5, 4), (4, 5)):
        recs = live(tmp_path).score_many([cell_spec(n_cells=n) for n in order], c)
        status = {s: r["status"] for s, r in zip(order, recs, strict=True)}
        assert status == {4: "completed", 5: "failed"}


def test_score_variants_edge_cases(tmp_path):
    client = live(tmp_path)
    empty = pd.DataFrame(columns=["chrom", "pos", "ref", "alt"])
    assert client.score_variants(empty, condition="UBERON:0013756").empty
    with pytest.raises(TypeError):
        client.score_variants(empty, conditioning={"a": cells()}, condition_label="a")
    bad = pd.DataFrame({"chrom": ["21"], "pos": [5], "ref": ["A"], "alt": ["A"], "gene_id": ["G"]})
    out = client.score_variants(bad, condition="UBERON:0013756", gene_column="gene_id")
    assert out.loc[0, "status"] == "failed" and out.loc[0, "chrom"] == "chr21"
    assert out.loc[0, "gene_id"] == "G"


class BloodModel(FakeModel):
    """Mimics Whole Blood output with an observed GTEx track and imputed ENCODE tracks."""

    def score_variant(self, **kw):
        self.calls.append(kw)
        obs = {"gene_id": GENES[:1], "strand": ["+"], "gene_name": ["MX1"]}
        var = {
            "strand": [".", ".", "."],
            "name": ["gtex", "encode_a", "encode_b"],
            "imputation_mode": ["observed", "imputed_exploratory", "imputed_exploratory"],
            "data_source": ["gtex", "encode", "encode"],
        }
        return [scorer_output([[0.3, -0.6, -0.9]], obs, var, variant=str(kw["variant"]))]


def test_track_filter_selects_tracks_without_refetching(tmp_path):
    model = BloodModel()
    client = live(tmp_path, model)
    assert client.score(spec())["raw_effect"] == pytest.approx(-0.4)
    rec = client.score(spec(), tracks={"imputation_mode": "observed"})
    assert rec["raw_effect"] == pytest.approx(0.3) and rec["n_reduced_values"] == 1
    assert rec["from_cache"] and len(model.calls) == 1
    assert json.loads(rec["track_filter"]) == {"imputation_mode": ["observed"]}
    rec = client.score(spec(), tracks={"data_source": ["encode", "gtex"]})
    assert rec["n_reduced_values"] == 3


@pytest.mark.parametrize(
    "tracks,match",
    [({"imputation_mode": "measured"}, "no track"), ({"assay": "x"}, "not in")],
)
def test_track_filter_errors_are_per_record(tmp_path, tracks, match):
    rec = live(tmp_path, BloodModel()).score(spec(), tracks=tracks)
    assert rec["status"] == "failed" and match in rec["error"]


def test_track_filter_through_score_variants(tmp_path):
    pairs = pd.DataFrame(
        {
            "chrom": ["21"],
            "pos": [41422154],
            "ref": ["T"],
            "alt": ["C"],
            "gene_id": ["ENSG00000157601"],
        }
    )
    out = live(tmp_path, BloodModel()).score_variants(
        pairs,
        condition="UBERON:0013756",
        gene_column="gene_id",
        tracks={"imputation_mode": "observed"},
    )
    assert out.loc[0, "raw_effect"] == pytest.approx(0.3)
