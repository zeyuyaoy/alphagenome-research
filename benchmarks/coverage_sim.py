"""Empirical 95% CI coverage of ag2kit.evaluate's sign-AUROC bootstrap.

Compares the cluster bootstrap (cluster="gene_id") with the pair bootstrap
(cluster=None) on data where pairs within a gene share a gene-level random
effect in both predicted and observed effects.

    uv run python benchmarks/coverage_sim.py            # 200 replicates, n_boot=500
    uv run python benchmarks/coverage_sim.py --quick    # smoke run
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

import ag2kit

# Generative model: gene effect u_g drives beta; prediction sees u_g plus its
# own gene-level error v_g, so both sides are correlated within a gene.
SD_BETA_NOISE, PRED_SLOPE, SD_PRED_GENE, SD_PRED_NOISE = 0.5, 0.6, 0.8, 0.5


def draw(n_genes, per_gene, rng):
    g = np.repeat(np.arange(n_genes), per_gene)
    u, v = rng.normal(size=n_genes), rng.normal(scale=SD_PRED_GENE, size=n_genes)
    beta = u[g] + rng.normal(scale=SD_BETA_NOISE, size=g.size)
    pred = PRED_SLOPE * u[g] + v[g] + rng.normal(scale=SD_PRED_NOISE, size=g.size)
    return g, pred, beta


def auroc(pred, beta):
    pos, neg = pred[beta > 0], pred[beta < 0]
    return mannwhitneyu(pos, neg).statistic / (pos.size * neg.size)


def tables(g, pred, beta):
    n = g.size
    keys = pd.DataFrame(
        {
            "chrom": "chr1",
            "pos": np.arange(n) * 100 + 1_000_000,
            "ref": "A",
            "alt": "G",
            "gene_id": [f"ENSG{x:011d}" for x in g],
        }
    )
    return keys.assign(raw_effect=pred, status="completed"), keys.assign(
        beta=beta, effect_allele="G"
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--genes", type=int, default=60)
    ap.add_argument("--per-gene", type=int, default=6)
    ap.add_argument("--quick", action="store_true", help="20 replicates x n_boot=200")
    args = ap.parse_args(argv)
    if args.quick:
        args.reps, args.n_boot = 20, 200
    print(f"ag2kit {ag2kit.__version__} from {ag2kit.__file__}")

    # Population AUROC from one very large draw of the same model.
    truth = auroc(*draw(500_000, args.per_gene, np.random.default_rng(12345))[1:])
    print(
        f"genes={args.genes} per_gene={args.per_gene} reps={args.reps} "
        f"n_boot={args.n_boot}; true sign AUROC (3M-pair MC) = {truth:.4f}\n"
    )

    hits = {"gene_id": [], None: []}
    widths = {"gene_id": [], None: []}
    estimates = []
    t0 = time.perf_counter()
    for r in range(args.reps):
        g, pred, beta = draw(args.genes, args.per_gene, np.random.default_rng(r))
        predictions, observed = tables(g, pred, beta)
        for cluster in hits:
            m = ag2kit.evaluate(predictions, observed, cluster=cluster, n_boot=args.n_boot, seed=r)
            row = m.metrics.set_index("metric").loc["sign_auroc"]
            hits[cluster].append(row["ci95_low"] <= truth <= row["ci95_high"])
            widths[cluster].append(row["ci95_high"] - row["ci95_low"])
        estimates.append(row["estimate"])
    secs = time.perf_counter() - t0

    print("| bootstrap | coverage | binomial SE | mean CI width |")
    print("|---|---:|---:|---:|")
    for cluster, h in hits.items():
        p = float(np.mean(h))
        se = np.sqrt(p * (1 - p) / len(h))
        name = "cluster (gene_id)" if cluster else "pair (cluster=None)"
        print(f"| {name} | {p:.3f} | {se:.3f} | {np.mean(widths[cluster]):.4f} |")
    print(
        f"\nmean estimate {np.mean(estimates):.4f} (truth {truth:.4f}), "
        f"empirical SD of estimate {np.std(estimates, ddof=1):.4f}; {secs:.0f} s"
    )


if __name__ == "__main__":
    main()
