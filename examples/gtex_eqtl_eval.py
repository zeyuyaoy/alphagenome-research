"""Evaluate AlphaGenome 2 RNA-seq predictions against GTEx eQTLs in one tissue.

    python examples/gtex_eqtl_eval.py --genes MX1 OAS1 IFIT1 --tissue Whole_Blood --out results/

Steps: download significant eQTLs from the GTEx Portal (public, cached), map the tissue to
its UBERON condition, score each variant-gene pair with AlphaGenome 2 (``RNA_SEQ`` scorer,
cached and resumable; by default only tracks marked ``observed`` are averaged, which for
Whole Blood is the GTEx RNA-seq track rather than imputed ENCODE tracks), join and orient
effects, compute metrics with gene-cluster bootstrap intervals, and write the tables, a
figure and provenance to ``--out``.

Without ``ALPHA_GENOME_API_KEY`` the script stops after step 2 and reports how many API
requests the run would make; nothing is fabricated. Predictions are cached in
``$AG2KIT_CACHE``, so a rerun only scores what is missing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import ag2kit
from ag2kit import gtex


def main() -> int:
    """Runs the GTEx evaluation example from command-line arguments.

    Returns:
        Zero after planning or writing results, or one when no eQTLs match.
    """
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--genes", nargs="+", default=["MX1", "OAS1", "IFIT1"])
    ap.add_argument("--tissue", default="Whole_Blood")
    ap.add_argument("--dataset", default="gtex_v10", choices=sorted(gtex.GENCODE))
    ap.add_argument("--out", type=Path, default=Path("results/gtex_eqtl_eval"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument(
        "--all-tracks",
        action="store_true",
        help="average imputed tracks too (default: tracks observed in training data only)",
    )
    args = ap.parse_args()

    observed = gtex.eqtls(args.genes, args.tissue, args.dataset)
    if observed.empty:
        print("no significant eQTLs for these genes in this tissue", file=sys.stderr)
        return 1
    tissues = gtex.tissues(args.dataset).set_index("tissueSiteDetailId")
    condition = str(tissues.loc[args.tissue, "ontologyId"])
    print(f"{len(observed)} GTEx {args.dataset} pairs; condition {condition}")

    client = ag2kit.AG2Client(dry_run=not ag2kit.client.api_key_available())
    predictions = client.score_variants(
        observed,
        condition=condition,
        scorer="RNA_SEQ",
        gene_column="gene_id",
        max_workers=args.workers,
        tracks=None if args.all_tracks else {"imputation_mode": "observed"},
    )
    if client.dry_run:
        n = predictions["cache_key"].nunique()
        cached = sum(client._cache_path(k).exists() for k in predictions["cache_key"].unique())
        print(f"ALPHA_GENOME_API_KEY not set: would make {n - cached} requests ({cached} cached).")
        return 0
    print(predictions["status"].value_counts().to_string())

    evaluation = ag2kit.evaluate(
        predictions, observed, match={"ontology_id": "condition"}, cluster="gene_id"
    )
    evaluation.summary()
    out = evaluation.save(args.out)
    predictions.to_parquet(out / "predictions.parquet", index=False)
    observed.to_parquet(out / "observed.parquet", index=False)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ok = evaluation.pairs[evaluation.pairs["pair_status"] == "ok"]
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(ok["beta_alt"], ok["predicted"], s=8)
    ax.axhline(0, lw=0.5, color="grey")
    ax.axvline(0, lw=0.5, color="grey")
    ax.set_xlabel(f"GTEx {args.dataset} NES (ALT vs REF)")
    ax.set_ylabel("AlphaGenome 2 RNA_SEQ score (ALT vs REF)")
    ax.set_title(f"{args.tissue}, {len(ok)} pairs")
    fig.tight_layout()
    fig.savefig(out / "scatter.png", dpi=150)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
