"""Evaluates predicted variant effects against observed molecular QTLs.

Variants sharing a gene or locus can be correlated through linkage
disequilibrium. Resampling whole clusters preserves that dependence
within each bootstrap replicate.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ag2kit.provenance import software_versions, utc_now
from ag2kit.stats import by_group, sign_auroc, spearman
from ag2kit.variants import harmonize_effect

Metric = Callable[[np.ndarray, np.ndarray], float]


def sign_accuracy(deltas, betas) -> float:
    """Computes the fraction of comparable effects with matching signs.

    Args:
        deltas: Predicted effects.
        betas: Observed effects in the same order and allele orientation.

    Returns:
        Sign accuracy after excluding non-finite pairs and zero betas,
        or NaN if no pairs remain. Zero predictions count as incorrect.
    """
    d = np.asarray(deltas, dtype=float)
    b = np.asarray(betas, dtype=float)
    ok = np.isfinite(d) & np.isfinite(b) & (b != 0)
    if not ok.any():
        return float("nan")
    return float(np.mean(np.sign(d[ok]) == np.sign(b[ok])))


def _rho(deltas, betas) -> float:
    return spearman(deltas, betas)["rho"]


METRICS: dict[str, Metric] = {
    "sign_auroc": sign_auroc,
    "spearman_rho": _rho,
    "sign_accuracy": sign_accuracy,
}


def _key(df: pd.DataFrame) -> pd.DataFrame:
    """Builds an order-independent allele key using unversioned gene IDs."""
    ref = df["ref"].astype(str).str.upper()
    alt = df["alt"].astype(str).str.upper()
    return pd.DataFrame(
        {
            "_chrom": "chr" + df["chrom"].astype(str).str.removeprefix("chr"),
            "_pos": pd.to_numeric(df["pos"], errors="coerce").astype("Int64"),
            "_alleles": np.where(ref < alt, ref + "/" + alt, alt + "/" + ref),
            "_gene": df["gene_id"].astype(str).str.split(".").str[0],
        },
        index=df.index,
    )


@dataclass
class Evaluation:
    """Stores the result of evaluate.

    Attributes:
        pairs: Ledger with one row per observed effect and its match status.
        metrics: Overall estimates, confidence intervals, and pair counts.
        per_group: Group estimates when requested, otherwise an empty table.
        settings: Evaluation parameters and input counts.
    """

    pairs: pd.DataFrame
    metrics: pd.DataFrame
    per_group: pd.DataFrame
    settings: dict = field(default_factory=dict)

    def summary(self) -> pd.DataFrame:
        """Prints the metrics and counts by pair status.

        Returns:
            The overall metrics table.
        """
        counts = self.pairs["pair_status"].value_counts().rename_axis("pair_status")
        print(self.metrics.to_string(index=False))
        print(counts.to_string())
        return self.metrics

    def save(self, directory: str | Path) -> Path:
        """Writes pairs.parquet, metrics.tsv, per_group.tsv, and evaluation.json.

        Args:
            directory: Output directory. Existing files with these names are replaced.

        Returns:
            Path to the output directory.

        Raises:
            OSError: If the directory or a file cannot be written.
        """
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        self.pairs.to_parquet(out / "pairs.parquet", index=False)
        self.metrics.to_csv(out / "metrics.tsv", sep="\t", index=False)
        self.per_group.to_csv(out / "per_group.tsv", sep="\t", index=False)
        meta = {**self.settings, "software": software_versions(), "written_utc": utc_now()}
        (out / "evaluation.json").write_text(json.dumps(meta, indent=2, default=str))
        return out


def join(
    predictions: pd.DataFrame,
    observed: pd.DataFrame,
    *,
    match: Mapping[str, str] | None = None,
    effect: str = "raw_effect",
    forward_strand: bool = True,
) -> pd.DataFrame:
    """Matches observed effects to predictions and orients them to predicted ALT.

    Matching uses chromosome, position, unordered REF/ALT, and unversioned
    gene ID. Equivalent prediction rows with the same score and status
    count as one candidate.

    Args:
        predictions: Table with chrom, pos, ref, alt, gene_id, status, and effect.
        observed: Table with chrom, pos, ref, alt, gene_id, beta, and effect_allele.
        match: Optional mapping from observed to prediction column names for
            additional keys, such as {"ontology_id": "condition"}.
        effect: Prediction column containing the signed effect.
        forward_strand: Whether observed alleles are on the GRCh38 forward
            strand. If true, a swapped observed REF is flagged as ref_mismatch.

    Returns:
        One row per observed record with prediction fields, ALT-oriented
        beta_alt, allele_flipped, and pair_status identifying usable pairs
        ("ok"), unmatched or failed predictions, or allele orientation problems.

    Raises:
        ValueError: If either input lacks a required column.
    """
    match = dict(match or {})
    need_obs = {"chrom", "pos", "ref", "alt", "gene_id", "beta", "effect_allele", *match}
    need_pred = {"chrom", "pos", "ref", "alt", "gene_id", effect, "status", *match.values()}
    for name, df, need in (
        ("observed", observed, need_obs),
        ("predictions", predictions, need_pred),
    ):
        if missing := need - set(df.columns):
            raise ValueError(f"{name} is missing columns {sorted(missing)}")

    obs = observed.reset_index(drop=True)
    pred = predictions[predictions["chrom"].notna()].reset_index(drop=True)
    keys = ["_chrom", "_pos", "_alleles", "_gene"]
    o = pd.concat([_key(obs), obs[list(match)].rename(columns=lambda c: f"_m_{c}")], axis=1)
    p = pd.concat(
        [
            _key(pred),
            pred[list(match.values())].rename(columns={v: f"_m_{k}" for k, v in match.items()}),
        ],
        axis=1,
    )
    keys += [f"_m_{c}" for c in match]
    p["_effect"] = pred[effect].to_numpy()
    p["_status"] = pred["status"].to_numpy()
    p = p.drop_duplicates([*keys, "_effect", "_status"])
    p["_n_pred"] = p.groupby(keys, dropna=False)["_gene"].transform("size")
    p = p.drop_duplicates(keys)
    p["_pred_row"] = p.index
    merged = o.merge(p[[*keys, "_n_pred", "_pred_row"]], on=keys, how="left")
    # Without gene_column, a request failure applies to every gene for that variant.
    variant_keys = [k for k in keys if k != "_gene"]
    failed = p[(p["_status"] != "completed") & p["_gene"].isin(["", "nan", "None"])]
    failed_variants = set(map(tuple, failed[variant_keys].astype(str).to_numpy()))
    merged["_variant_failed"] = [
        tuple(r) in failed_variants for r in merged[variant_keys].astype(str).to_numpy()
    ]

    pred_cols = ["ref", "alt", effect, "status", "error", "source", "condition", "scorer_name"]
    pred_cols = [c for c in pred_cols if c in pred.columns]
    matched = pred.reindex(merged["_pred_row"].fillna(-1).astype(int))[pred_cols]
    matched.index = obs.index
    out = obs.join(matched.add_prefix("pred_"))
    out["predicted"] = out[f"pred_{effect}"].astype(float)

    status, beta_alt, flipped = [], [], []
    for i, row in out.iterrows():
        n = merged.at[i, "_n_pred"]
        if pd.isna(n):
            gone = "prediction_failed" if merged.at[i, "_variant_failed"] else "no_prediction"
            s, b, f = gone, np.nan, False
        elif n > 1:
            s, b, f = "ambiguous_prediction", np.nan, False
        elif row["pred_status"] != "completed" or not np.isfinite(row["predicted"]):
            s, b, f = "prediction_failed", np.nan, False
        elif forward_strand and str(row["ref"]).upper() != str(row["pred_ref"]).upper():
            # Swapping genomic REF and ALT changes the model input, so negation
            # cannot recover the effect of the intended comparison.
            s, b, f = "ref_mismatch", np.nan, False
        else:
            h = harmonize_effect(
                row["beta"], row["effect_allele"], row["pred_ref"], row["pred_alt"], forward_strand
            )
            s, f = h.status, h.flipped
            b = np.nan if h.estimate is None else h.estimate
        status.append(s)
        beta_alt.append(b)
        flipped.append(f)
    out["beta_alt"] = beta_alt
    out["allele_flipped"] = flipped
    out["pair_status"] = status
    return out


def _bootstrap(
    metric: Metric,
    d: np.ndarray,
    b: np.ndarray,
    clusters: np.ndarray,
    groups: np.ndarray | None,
    n_boot: int,
    rng: np.random.Generator,
    min_group: int,
) -> np.ndarray:
    labels, inverse = np.unique(clusters, return_inverse=True)
    members = [np.flatnonzero(inverse == k) for k in range(labels.size)]
    stats = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.concatenate([members[k] for k in rng.integers(0, labels.size, labels.size)])
        if groups is None:
            stats[i] = metric(d[idx], b[idx])
        else:
            stats[i] = by_group(metric, d[idx], b[idx], groups[idx], min_group)[0]
    return stats


def evaluate(
    predictions: pd.DataFrame,
    observed: pd.DataFrame,
    *,
    match: Mapping[str, str] | None = None,
    group: str | None = None,
    cluster: str | None = "gene_id",
    effect: str = "raw_effect",
    forward_strand: bool = True,
    n_boot: int = 1000,
    seed: int = 0,
    min_group: int = 2,
    min_clusters: int = 5,
) -> Evaluation:
    """Measures sign AUROC, signed Spearman correlation, and sign accuracy.

    Metrics use matched pairs with nonzero observed effects. Confidence
    intervals are 95% percentiles over bootstrap replicates. They describe
    resampling of the supplied clusters and do not model GTEx sampling.

    Args:
        predictions: Prediction table accepted by join.
        observed: Observed-effects table accepted by join.
        match: Additional keys mapping observed columns to prediction columns.
        group: Observed column for group-size-weighted metrics, or None to pool.
        cluster: Observed column defining bootstrap units. None resamples
            individual pairs and can understate uncertainty for related pairs.
        effect: Prediction column containing the signed effect.
        forward_strand: Whether to require forward-strand REF agreement in join.
        n_boot: Number of bootstrap resamples. Zero disables intervals.
        seed: Random seed for bootstrapping.
        min_group: Minimum pair count for a group to contribute to a weighted metric.
        min_clusters: Minimum number of distinct clusters for an interval.

    Returns:
        Pair ledger, metrics with confidence intervals and counts, per-group
        estimates if requested, and evaluation settings. Intervals are NaN
        when disabled or when there are too few clusters or no finite replicates.

    Raises:
        ValueError: If a required join or cluster column is missing.
        KeyError: If the requested group column is missing.
    """
    pairs = join(predictions, observed, match=match, effect=effect, forward_strand=forward_strand)
    usable = pairs[(pairs["pair_status"] == "ok") & (pairs["beta_alt"] != 0)]
    d = usable["predicted"].to_numpy(float)
    b = usable["beta_alt"].to_numpy(float)
    if cluster is not None and cluster not in usable:
        raise ValueError(f"cluster column {cluster!r} not in observed")
    clusters = usable[cluster].astype(str).to_numpy() if cluster else np.arange(len(usable))
    groups = usable[group].astype(str).to_numpy() if group else None
    rng = np.random.default_rng(seed)

    rows, per_group = [], []
    for name, metric in METRICS.items():
        n_used = len(d)
        if groups is None:
            estimate = metric(d, b)
        else:
            estimate, table = by_group(metric, d, b, groups, min_group)
            per_group.append(table.assign(metric=name))
            n_used = int(table.loc[(table["n"] >= min_group) & table["value"].notna(), "n"].sum())
        enough = len(np.unique(clusters)) >= min_clusters
        boots = (
            _bootstrap(metric, d, b, clusters, groups, n_boot, rng, min_group)
            if len(d) and n_boot and enough
            else np.array([np.nan])
        )
        finite = boots[np.isfinite(boots)]
        lo, hi = np.quantile(finite, [0.025, 0.975]) if finite.size else (np.nan, np.nan)
        rows.append(
            {
                "metric": name,
                "estimate": estimate,
                "ci95_low": float(lo),
                "ci95_high": float(hi),
                "n_pairs": len(d),
                "n_pairs_used": n_used,
                "n_clusters": len(np.unique(clusters)),
                "n_boot_finite": int(finite.size),
                "weighting": f"by {group}" if group else "pooled",
            }
        )
    settings = {
        "match": dict(match or {}),
        "group": group,
        "cluster": cluster,
        "effect": effect,
        "forward_strand": forward_strand,
        "n_boot": n_boot,
        "seed": seed,
        "min_group": min_group,
        "min_clusters": min_clusters,
        "n_observed": len(observed),
        "n_predictions": len(predictions),
    }
    groups_table = (
        pd.concat(per_group, ignore_index=True)
        if per_group
        else pd.DataFrame(columns=["group", "n", "value", "metric"])
    )
    return Evaluation(pairs, pd.DataFrame(rows), groups_table, settings)
