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
from ag2kit.stats import sign_accuracy, sign_auroc, spearman
from ag2kit.variants import harmonize_effect, normalize_chrom

Metric = Callable[[np.ndarray, np.ndarray], float]


def _rho(deltas: np.ndarray, betas: np.ndarray) -> float:
    return spearman(deltas, betas)["rho"]


METRICS: dict[str, Metric] = {
    "sign_auroc": sign_auroc,
    "spearman_rho": _rho,
    "sign_accuracy": sign_accuracy,
}


def _key(df: pd.DataFrame) -> pd.DataFrame:
    """Builds an order-independent allele key using unversioned gene IDs."""
    ref = df["ref"].astype("string").str.upper()
    alt = df["alt"].astype("string").str.upper()
    return pd.DataFrame(
        {
            "_chrom": df["chrom"].map(normalize_chrom, na_action="ignore").astype("string"),
            "_pos": pd.to_numeric(df["pos"], errors="coerce").astype("Int64"),
            "_alleles": (ref + "/" + alt).where(ref < alt, alt + "/" + ref),
            "_gene": df["gene_id"].astype("string").str.split(".").str[0],
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
    gene ID. Equivalent predictions with the same REF, score, and status
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
    pred = predictions.reset_index(drop=True)
    keys = ["_chrom", "_pos", "_alleles", "_gene"]
    match_keys = [f"_m_{c}" for c in match]
    o = pd.concat([_key(obs), obs[list(match)].set_axis(match_keys, axis=1)], axis=1)
    p = pd.concat(
        [
            _key(pred),
            pred[list(match.values())].set_axis(match_keys, axis=1),
        ],
        axis=1,
    )
    keys += match_keys
    variant_keys = [k for k in keys if k != "_gene"]
    p["_ref"] = pred["ref"].astype("string").str.upper()
    p["_effect"] = pred[effect].to_numpy()
    p["_status"] = pred["status"].to_numpy()
    p = p.dropna(subset=variant_keys).drop_duplicates([*keys, "_ref", "_effect", "_status"])
    p["_n_pred"] = p.groupby(keys, dropna=False)["_gene"].transform("size")
    p = p.drop_duplicates(keys)
    p["_pred_row"] = p.index
    merged = o.merge(p.dropna(subset=keys)[[*keys, "_n_pred", "_pred_row"]], on=keys, how="left")
    # Without gene_column, a request failure applies to every gene for that variant.
    failed = p[(p["_status"] != "completed") & p["_gene"].fillna("").isin(["", "nan", "None"])]
    failed_variants = set(map(tuple, failed[variant_keys].astype(str).to_numpy()))
    merged["_variant_failed"] = [
        tuple(r) in failed_variants for r in merged[variant_keys].astype(str).to_numpy()
    ]

    pred_cols = ["ref", "alt", effect, "status", "error", "source", "condition", "scorer_name"]
    pred_cols += ["model_version", "track_filter", "n_reduced_values"]
    pred_cols = [c for c in pred_cols if c in pred.columns]
    matched = pred.reindex(merged["_pred_row"].fillna(-1).astype(int))[pred_cols]
    matched.index = obs.index
    out = obs.join(matched.add_prefix("pred_"))
    score = out[f"pred_{effect}"]
    out["predicted"] = score.where(score.notna(), np.nan).astype(float)

    status, beta_alt, flipped = [], [], []
    counts = merged["_n_pred"].fillna(0).to_numpy(dtype=int)
    rows = zip(
        counts,
        merged["_variant_failed"],
        out["pred_status"].astype(str),
        out["predicted"],
        out["ref"].astype(str).str.upper(),
        out["pred_ref"].astype(str).str.upper(),
        out["pred_alt"],
        out["beta"],
        out["effect_allele"],
        strict=True,
    )
    for n, variant_failed, pred_status, predicted, ref, pred_ref, pred_alt, beta, allele in rows:
        if n == 0:
            s, b, f = ("prediction_failed" if variant_failed else "no_prediction"), np.nan, False
        elif n > 1:
            s, b, f = "ambiguous_prediction", np.nan, False
        elif pred_status != "completed" or not np.isfinite(predicted):
            s, b, f = "prediction_failed", np.nan, False
        elif forward_strand and ref != pred_ref:
            s, b, f = "ref_mismatch", np.nan, False
        else:
            h = harmonize_effect(beta, allele, pred_ref, pred_alt, forward_strand)
            s, f = h.status, h.flipped
            b = np.nan if h.estimate is None else h.estimate
        status.append(s)
        beta_alt.append(b)
        flipped.append(f)
    out["beta_alt"] = beta_alt
    out["allele_flipped"] = flipped
    out["pair_status"] = status
    return out


def _usable(pairs: pd.DataFrame) -> pd.DataFrame:
    missing = {"pair_status", "predicted", "beta_alt"} - set(pairs.columns)
    if missing:
        raise ValueError(f"pairs is missing ledger columns {sorted(missing)}; build it with join")
    return pairs[(pairs["pair_status"] == "ok") & (pairs["beta_alt"] != 0)]


def _codes(values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    labels, codes = np.unique(values.astype(str).to_numpy(), return_inverse=True)
    return labels, codes.astype(np.intp)


class _ClusterSampler:
    """Draws cluster bootstrap resamples as row indices, vectorized over clusters."""

    def __init__(self, clusters: np.ndarray, rng: np.random.Generator):
        _, inverse = np.unique(clusters, return_inverse=True)
        self.order = np.argsort(inverse, kind="stable")
        self.sizes = np.bincount(inverse)
        self.starts = np.concatenate([[0], np.cumsum(self.sizes)[:-1]])
        self.n_clusters = self.sizes.size
        self.rng = rng

    def draw(self) -> np.ndarray:
        chosen = self.rng.integers(0, self.n_clusters, self.n_clusters)
        sizes = self.sizes[chosen]
        offsets = np.arange(sizes.sum()) - np.repeat(np.cumsum(sizes) - sizes, sizes)
        return self.order[np.repeat(self.starts[chosen], sizes) + offsets]


def _estimate(
    metric: Metric,
    d: np.ndarray,
    b: np.ndarray,
    codes: np.ndarray | None,
    n_groups: int,
    min_group: int,
    fixed: np.ndarray | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Returns the pooled or group-size-weighted estimate, group values, and group sizes.

    With fixed, only those groups can contribute, so groups never join the
    average in a resample; a fixed group that is undefined in a resample is
    left out, as in by_group.
    """
    if codes is None:
        return float(metric(d, b)), np.empty(0), np.empty(0, dtype=int)
    sizes = np.bincount(codes, minlength=n_groups)
    members = np.split(np.argsort(codes, kind="stable"), np.cumsum(sizes)[:-1])
    values = np.array([float(metric(d[m], b[m])) if m.size else np.nan for m in members])
    use = (sizes >= min_group) & np.isfinite(values)
    if fixed is not None:
        use &= fixed
    if not use.any():
        return np.nan, values, sizes
    return float(np.average(values[use], weights=sizes[use])), values, sizes


def _interval(replicates: np.ndarray) -> tuple[float, float, int]:
    finite = replicates[np.isfinite(replicates)]
    if not finite.size:
        return np.nan, np.nan, 0
    lo, hi = np.quantile(finite, [0.025, 0.975])
    return float(lo), float(hi), int(finite.size)


def evaluate_pairs(
    pairs: pd.DataFrame,
    *,
    group: str | None = None,
    cluster: str | None = "gene_id",
    n_boot: int = 1000,
    seed: int = 0,
    min_group: int = 2,
    min_clusters: int = 20,
) -> Evaluation:
    """Measures sign AUROC, signed Spearman correlation, and sign accuracy on a ledger.

    Metrics use pairs with pair_status "ok" and nonzero beta_alt. Every metric,
    and every group, is evaluated on the same cluster bootstrap resamples, so
    intervals do not depend on metric order. Intervals are 95% percentiles and
    describe resampling of the supplied clusters only.

    Args:
        pairs: Ledger from join. Columns can be added first for stratified
            analyses, for example bins of absolute predicted effect.
        group: Ledger column for group-size-weighted metrics, or None to pool.
        cluster: Ledger column defining bootstrap units. None resamples
            individual pairs and can understate uncertainty for related pairs.
        n_boot: Number of bootstrap resamples. Zero disables intervals.
        seed: Random seed for bootstrapping.
        min_group: Minimum pair count for a group to contribute to a weighted metric.
        min_clusters: Minimum number of distinct clusters for an interval.

    Returns:
        The ledger, overall metrics with intervals and counts, per-group
        estimates with intervals if group is set, and settings.

    Raises:
        ValueError: If ledger or cluster columns are missing.
        KeyError: If the group column is missing.
    """
    usable = _usable(pairs)
    if cluster is not None and cluster not in usable:
        raise ValueError(f"cluster column {cluster!r} not in pairs")
    d = usable["predicted"].to_numpy(float)
    b = usable["beta_alt"].to_numpy(float)
    clusters = usable[cluster].astype(str).to_numpy() if cluster else np.arange(len(usable))
    n_clusters = len(np.unique(clusters))
    labels, codes = _codes(usable[group]) if group else (np.empty(0), None)
    n_groups = labels.size

    estimates = {m: _estimate(f, d, b, codes, n_groups, min_group) for m, f in METRICS.items()}
    fixed = {m: (e[2] >= min_group) & np.isfinite(e[1]) for m, e in estimates.items()}
    group_clusters = (
        np.array([len(np.unique(clusters[codes == j])) for j in range(n_groups)])
        if codes is not None
        else np.empty(0, dtype=int)
    )
    run_boot = bool(len(d) and n_boot and n_clusters >= min_clusters)
    reps = {m: np.full((n_boot if run_boot else 0, max(n_groups, 1) + 1), np.nan) for m in METRICS}
    if run_boot:
        sampler = _ClusterSampler(clusters, np.random.default_rng(seed))
        for i in range(n_boot):
            idx = sampler.draw()
            sub = None if codes is None else codes[idx]
            for m, f in METRICS.items():
                weighted, values, _ = _estimate(
                    f, d[idx], b[idx], sub, n_groups, min_group, fixed[m] if group else None
                )
                reps[m][i, 0] = weighted
                reps[m][i, 1 : 1 + values.size] = values

    rows, per_group = [], []
    for m, (estimate, values, sizes) in estimates.items():
        lo, hi, n_finite = _interval(reps[m][:, 0])
        used = sizes[(sizes >= min_group) & np.isfinite(values)].sum() if group else len(d)
        rows.append(
            {
                "metric": m,
                "estimate": estimate,
                "ci95_low": lo,
                "ci95_high": hi,
                "n_pairs": len(d),
                "n_pairs_used": int(used),
                "n_clusters": n_clusters,
                "n_boot_finite": n_finite,
                "weighting": f"by {group}" if group else "pooled",
            }
        )
        for j, label in enumerate(labels):
            g_lo, g_hi, g_finite = _interval(reps[m][:, 1 + j])
            if group_clusters[j] < min_clusters:
                g_lo, g_hi, g_finite = np.nan, np.nan, 0
            per_group.append(
                {
                    "group": label,
                    "n": int(sizes[j]),
                    "value": values[j],
                    "metric": m,
                    "ci95_low": g_lo,
                    "ci95_high": g_hi,
                    "n_clusters": int(group_clusters[j]),
                    "n_boot_finite": g_finite,
                }
            )
    settings = {
        "group": group,
        "cluster": cluster,
        "n_boot": n_boot,
        "seed": seed,
        "min_group": min_group,
        "min_clusters": min_clusters,
        "n_ledger_rows": len(pairs),
    }
    columns = ["group", "n", "value", "metric", "ci95_low", "ci95_high", "n_clusters"]
    columns.append("n_boot_finite")
    return Evaluation(pairs, pd.DataFrame(rows), pd.DataFrame(per_group, columns=columns), settings)


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
    min_clusters: int = 20,
) -> Evaluation:
    """Joins predictions to observed effects and evaluates them.

    Equivalent to evaluate_pairs(join(predictions, observed, ...), ...).

    Args:
        predictions: Prediction table accepted by join.
        observed: Observed-effects table accepted by join.
        match: Additional keys mapping observed columns to prediction columns.
        group: Observed column for group-size-weighted metrics, or None to pool.
        cluster: Observed column defining bootstrap units.
        effect: Prediction column containing the signed effect.
        forward_strand: Whether to require forward-strand REF agreement in join.
        n_boot: Number of bootstrap resamples. Zero disables intervals.
        seed: Random seed for bootstrapping.
        min_group: Minimum pair count for a group to contribute to a weighted metric.
        min_clusters: Minimum number of distinct clusters for an interval.

    Returns:
        Pair ledger, metrics with confidence intervals and counts, per-group
        estimates if requested, and evaluation settings.

    Raises:
        ValueError: If a required join or cluster column is missing.
        KeyError: If the requested group column is missing.
    """
    pairs = join(predictions, observed, match=match, effect=effect, forward_strand=forward_strand)
    result = evaluate_pairs(
        pairs,
        group=group,
        cluster=cluster,
        n_boot=n_boot,
        seed=seed,
        min_group=min_group,
        min_clusters=min_clusters,
    )
    result.settings = {
        "match": dict(match or {}),
        "effect": effect,
        "forward_strand": forward_strand,
        **result.settings,
        "n_observed": len(observed),
        "n_predictions": len(predictions),
    }
    return result


def compare(
    pairs_a: pd.DataFrame,
    pairs_b: pd.DataFrame,
    *,
    group: str | None = None,
    cluster: str | None = "gene_id",
    n_boot: int = 1000,
    seed: int = 0,
    min_group: int = 2,
    min_clusters: int = 20,
) -> pd.DataFrame:
    """Estimates metric differences between two prediction sets on the same observations.

    Both ledgers must come from join with the same observed table, so row i
    describes the same observed record in each. Every observed column is
    checked for this, including match and group columns. The comparison uses rows that
    are usable in both ledgers, each oriented to its own predicted ALT, and a
    paired cluster bootstrap that applies
    each resample to both prediction sets. This accounts for the dependence
    that the McNemar test in ag2kit.stats ignores.

    Args:
        pairs_a: Ledger for the first prediction set.
        pairs_b: Ledger for the second prediction set.
        group: Ledger column for group-size-weighted metrics, or None to pool.
        cluster: Ledger column defining bootstrap units.
        n_boot: Number of bootstrap resamples. Zero disables intervals.
        seed: Random seed for bootstrapping.
        min_group: Minimum pair count for a group to contribute to a weighted metric.
        min_clusters: Minimum number of distinct clusters for an interval.

    Returns:
        One row per metric with estimate_a, estimate_b, difference (a minus b),
        its 95% percentile interval, the number of common pairs, pairs usable
        in only one ledger, and cluster and replicate counts.

    Raises:
        ValueError: If the ledgers are not row-aligned or lack required columns.
    """
    derived = {"predicted", "beta_alt", "allele_flipped", "pair_status"}
    keys = [
        c
        for c in pairs_a.columns
        if c in pairs_b.columns and c not in derived and not str(c).startswith("pred_")
    ]
    keys = list(dict.fromkeys([*keys, *(c for c in (group, cluster) if c)]))
    for name, ledger in (("pairs_a", pairs_a), ("pairs_b", pairs_b)):
        if missing := set(keys) - set(ledger.columns):
            raise ValueError(f"{name} is missing columns {sorted(missing)}")
    left = pairs_a[keys].reset_index(drop=True).astype(str)
    right = pairs_b[keys].reset_index(drop=True).astype(str)
    if len(pairs_a) != len(pairs_b) or not left.equals(right):
        raise ValueError("pairs_a and pairs_b must be joined against the same observed table")
    a = pairs_a.reset_index(drop=True)
    b_ = pairs_b.reset_index(drop=True)
    ok_a, ok_b = a.index.isin(_usable(a).index), b_.index.isin(_usable(b_).index)
    both = ok_a & ok_b
    a, b_ = a[both], b_[both]
    beta_a, beta_b = a["beta_alt"].to_numpy(float), b_["beta_alt"].to_numpy(float)
    da, db = a["predicted"].to_numpy(float), b_["predicted"].to_numpy(float)
    clusters = a[cluster].astype(str).to_numpy() if cluster else np.arange(len(a))
    n_clusters = len(np.unique(clusters))
    labels, codes = _codes(a[group]) if group else (np.empty(0), None)

    fixed: dict[str, np.ndarray] = {}

    def diff(m: str, f: Metric, idx: np.ndarray | slice) -> tuple[float, float]:
        sub = None if codes is None else codes[idx]
        ea, values_a, sizes = _estimate(
            f, da[idx], beta_a[idx], sub, labels.size, min_group
        )
        eb, values_b, _ = _estimate(
            f, db[idx], beta_b[idx], sub, labels.size, min_group
        )
        if codes is None:
            return ea, eb
        use = (sizes >= min_group) & np.isfinite(values_a) & np.isfinite(values_b)
        if m not in fixed:
            fixed[m] = use.copy()
        use &= fixed[m]
        if not use.any():
            return np.nan, np.nan
        return (
            float(np.average(values_a[use], weights=sizes[use])),
            float(np.average(values_b[use], weights=sizes[use])),
        )

    originals = {m: diff(m, f, slice(None)) for m, f in METRICS.items()}

    run_boot = bool(len(beta_a) and n_boot and n_clusters >= min_clusters)
    reps = {m: np.full(n_boot if run_boot else 0, np.nan) for m in METRICS}
    if run_boot:
        sampler = _ClusterSampler(clusters, np.random.default_rng(seed))
        for i in range(n_boot):
            idx = sampler.draw()
            for m, f in METRICS.items():
                ea, eb = diff(m, f, idx)
                reps[m][i] = ea - eb
    rows = []
    for m, (ea, eb) in originals.items():
        lo, hi, n_finite = _interval(reps[m])
        rows.append(
            {
                "metric": m,
                "estimate_a": ea,
                "estimate_b": eb,
                "difference": ea - eb,
                "ci95_low": lo,
                "ci95_high": hi,
                "n_pairs": len(beta_a),
                "n_usable_a_only": int((ok_a & ~ok_b).sum()),
                "n_usable_b_only": int((ok_b & ~ok_a).sum()),
                "n_clusters": n_clusters,
                "n_boot_finite": n_finite,
                "weighting": f"by {group}" if group else "pooled",
            }
        )
    return pd.DataFrame(rows)
