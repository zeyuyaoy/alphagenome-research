"""Directional concordance and rank metrics for predicted and observed effects.

Sign agreement with an observed eQTL benchmarks predictions against association data.
Experimental validation would require an independent measurement of the variant effect.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import stats

CONCORDANT, DISCORDANT, NO_CALL = "concordant", "discordant", "no_call"
NO_COMPARATOR = "no_comparator"


def no_call_threshold(null_deltas: Iterable[float], quantile: float = 0.90) -> float:
    """Derives a no-call threshold from absolute scores under a null comparison.

    Args:
        null_deltas: Null scores, for example from identical-condition contrasts.
            Non-finite scores are discarded.
        quantile: Quantile in [0, 1].

    Returns:
        Magnitude at or below which classify treats predictions as no-calls.

    Raises:
        ValueError: If the quantile is outside [0, 1] or no finite scores remain.
    """
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be in [0, 1]")
    vals = np.asarray([abs(float(v)) for v in null_deltas if np.isfinite(v)], dtype=float)
    if vals.size == 0:
        raise ValueError("no finite null deltas to derive a threshold from")
    return float(np.quantile(vals, quantile))


def classify(delta: float, beta: float, threshold: float = 0.0) -> str:
    """Classifies agreement between effects oriented to the same allele.

    Args:
        delta: Predicted effect.
        beta: Observed effect.
        threshold: Maximum absolute prediction treated as a no-call.

    Returns:
        "no_comparator" for missing, non-finite, or zero beta. Otherwise,
    "no_call" for missing, non-finite, or zero delta, or when its absolute
    value is at or below the threshold. Otherwise, "concordant" or
    "discordant" by sign.
    """
    if beta is None or not np.isfinite(beta) or beta == 0:
        return NO_COMPARATOR
    if delta is None or not np.isfinite(delta):
        return NO_CALL
    if abs(delta) <= threshold or delta == 0:
        return NO_CALL
    return CONCORDANT if np.sign(delta) == np.sign(beta) else DISCORDANT


@dataclass
class Concordance:
    """Summarizes sign agreement with exact binomial inference.

    Attributes:
        k: Number of concordant calls.
        n_called: Number of concordant and discordant calls.
        n_no_call: Number of comparable pairs with no prediction call.
        n_total: Number of pairs with a comparator.
        threshold: Maximum absolute prediction treated as a no-call.
        n_no_comparator: Number of pairs without a usable observed effect.
        estimate: Concordance among called pairs, or NaN if none were called.
        ci_low: Lower bound of the exact 95% interval for estimate.
        ci_high: Upper bound of the exact 95% interval for estimate.
        p_value: Two-sided exact binomial p-value against 0.5 among calls.
        n_conservative: Number of pairs when no-calls count as discordant.
        p_conservative: Exact p-value with no-calls counted as discordant,
            or NaN if there are no comparable pairs.
    """

    k: int
    n_called: int
    n_no_call: int
    n_total: int
    threshold: float
    n_no_comparator: int = 0
    estimate: float = field(init=False)
    ci_low: float = field(init=False)
    ci_high: float = field(init=False)
    p_value: float = field(init=False)
    n_conservative: int = field(init=False)
    p_conservative: float = field(init=False)

    def __post_init__(self) -> None:
        if self.n_called > 0:
            test = stats.binomtest(self.k, self.n_called, 0.5, alternative="two-sided")
            ci = test.proportion_ci(confidence_level=0.95, method="exact")
            self.estimate = self.k / self.n_called
            self.ci_low, self.ci_high = float(ci.low), float(ci.high)
            self.p_value = float(test.pvalue)
        else:
            self.estimate = self.ci_low = self.ci_high = self.p_value = float("nan")
        self.n_conservative = self.n_called + self.n_no_call
        self.p_conservative = (
            float(stats.binomtest(self.k, self.n_conservative, 0.5, alternative="two-sided").pvalue)
            if self.n_conservative > 0
            else float("nan")
        )

    def as_row(self, label: str) -> dict:
        """Formats concordance and its conservative counterpart for export.

        Args:
            label: Comparison name.

        Returns:
            A table row with the label, counts, estimates, interval, and p-values.
        """
        return {
            "comparison": label,
            "k_concordant": self.k,
            "n_called": self.n_called,
            "n_no_call": self.n_no_call,
            "n_no_comparator": self.n_no_comparator,
            "n_total": self.n_total,
            "no_call_threshold": self.threshold,
            "concordance": self.estimate,
            "ci95_low": self.ci_low,
            "ci95_high": self.ci_high,
            "p_binomial_vs_0.5": self.p_value,
            "concordance_conservative": (
                self.k / self.n_conservative if self.n_conservative else float("nan")
            ),
            "p_conservative": self.p_conservative,
        }


def concordance(
    deltas: Iterable[float], betas: Iterable[float], threshold: float = 0.0
) -> tuple[Concordance, list[str]]:
    """Tests sign concordance against 0.5 with a two-sided exact binomial test.

    Args:
        deltas: Predicted effects in pair order.
        betas: Observed effects in the same order and allele orientation.
        threshold: Maximum absolute prediction treated as a no-call.

    Returns:
        A summary and the classification of each input pair in order.
        Pairs without a comparator are excluded from both tests.

    Raises:
        ValueError: If the inputs differ in length.
    """
    deltas = list(deltas)
    betas = list(betas)
    if len(deltas) != len(betas):
        raise ValueError("deltas and betas must have the same length")
    calls = [classify(d, b, threshold) for d, b in zip(deltas, betas, strict=True)]
    k = sum(c == CONCORDANT for c in calls)
    n_called = sum(c in (CONCORDANT, DISCORDANT) for c in calls)
    n_no_comparator = sum(c == NO_COMPARATOR for c in calls)
    return (
        Concordance(
            k=k,
            n_called=n_called,
            n_no_call=sum(c == NO_CALL for c in calls),
            n_total=len(calls) - n_no_comparator,
            threshold=float(threshold),
            n_no_comparator=n_no_comparator,
        ),
        calls,
    )


def _pairs(deltas: ArrayLike, betas: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    d = np.asarray(deltas, dtype=float)
    b = np.asarray(betas, dtype=float)
    if d.shape != b.shape:
        raise ValueError("deltas and betas must have the same length")
    ok = np.isfinite(d) & np.isfinite(b)
    return d[ok], b[ok]


def spearman(deltas: ArrayLike, betas: ArrayLike, absolute: bool = False) -> dict:
    """Computes Spearman correlation after dropping non-finite pairs.

    Args:
        deltas: Predicted effects.
        betas: Observed effects aligned to deltas.
        absolute: Whether to correlate magnitudes. The default uses signed
            effects, as in the AlphaGenome eQTL benchmark.

    Returns:
        The correlation (rho), p_value, and pair count (n). The correlation
        and p-value are NaN with fewer than four finite pairs.

    Raises:
        ValueError: If the inputs have different shapes.
    """
    d, b = _pairs(deltas, betas)
    if absolute:
        d, b = np.abs(d), np.abs(b)
    if d.size < 4:
        return {"rho": float("nan"), "p_value": float("nan"), "n": int(d.size)}
    res = stats.spearmanr(d, b)
    return {"rho": float(res.statistic), "p_value": float(res.pvalue), "n": int(d.size)}


def sign_auroc(deltas: ArrayLike, betas: ArrayLike) -> float:
    """Computes how well signed predictions rank positive above negative effects.

    This is the sign metric used in the Borzoi and AlphaGenome eQTL benchmarks.

    Args:
        deltas: Signed predicted effects.
        betas: Observed effects aligned to deltas.

    Returns:
        AUROC after excluding non-finite pairs and zero betas, or NaN if
        either sign class is empty.

    Raises:
        ValueError: If the inputs have different shapes.
    """
    d, b = _pairs(deltas, betas)
    pos, neg = d[b > 0], d[b < 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    u = stats.mannwhitneyu(pos, neg, alternative="two-sided").statistic
    return float(u / (pos.size * neg.size))


def sign_accuracy(deltas: ArrayLike, betas: ArrayLike) -> float:
    """Computes the fraction of comparable effects with matching signs.

    Args:
        deltas: Predicted effects.
        betas: Observed effects in the same order and allele orientation.

    Returns:
        Sign accuracy after excluding non-finite pairs and zero betas,
        or NaN if no pairs remain. Zero predictions count as incorrect.

    Raises:
        ValueError: If the inputs have different shapes.
    """
    d, b = _pairs(deltas, betas)
    ok = b != 0
    if not ok.any():
        return float("nan")
    return float(np.mean(np.sign(d[ok]) == np.sign(b[ok])))


def recall_at_sign_accuracy(deltas: ArrayLike, betas: ArrayLike, accuracy: float = 0.9) -> dict:
    """Finds the longest ranked prefix meeting a target sign accuracy.

    Pairs are ranked by decreasing absolute prediction, with ties resolved
    in input order. Zero predictions count as incorrect.

    Args:
        deltas: Predicted effects.
        betas: Observed effects aligned to deltas.
        accuracy: Minimum sign accuracy in (0, 1].

    Returns:
        A dict with recall (fraction of comparable pairs retained), n_called,
        n (finite pairs with nonzero beta), and min_abs_delta (the last
        retained magnitude, or NaN if none qualify). Recall is NaN when n is zero.

    Raises:
        ValueError: If accuracy is outside (0, 1] or input shapes differ.
    """
    if not 0 < accuracy <= 1:
        raise ValueError("accuracy must be in (0, 1]")
    d, b = _pairs(deltas, betas)
    keep = b != 0
    d, b = d[keep], b[keep]
    if d.size == 0:
        return {"recall": float("nan"), "n_called": 0, "n": 0, "min_abs_delta": float("nan")}
    order = np.argsort(-np.abs(d), kind="stable")
    correct = (np.sign(d[order]) == np.sign(b[order])) & (d[order] != 0)
    running = np.cumsum(correct) / np.arange(1, d.size + 1)
    ok = np.flatnonzero(running >= accuracy)
    k = int(ok[-1]) + 1 if ok.size else 0
    cutoff = float(abs(d[order][k - 1])) if k else float("nan")
    return {"recall": k / d.size, "n_called": k, "n": int(d.size), "min_abs_delta": cutoff}


def by_group(
    metric: Callable[[np.ndarray, np.ndarray], float],
    deltas: ArrayLike,
    betas: ArrayLike,
    groups: ArrayLike,
    min_size: int = 2,
) -> tuple[float, pd.DataFrame]:
    """Computes a metric per group and averages it by group size.

    Args:
        metric: Callable taking predicted and observed arrays and returning a float.
        deltas: Predicted effects.
        betas: Observed effects aligned to deltas.
        groups: Group label for each pair, such as a tissue ID.
        min_size: Minimum group size for inclusion in the average.

    Returns:
        The weighted mean and a table of group, n, and value for each group.
        Small groups and NaN values are excluded from the mean, which is NaN
        if no groups qualify.
    """
    df = pd.DataFrame({"delta": deltas, "beta": betas, "group": groups})
    rows = []
    for name, g in df.groupby("group", sort=True, observed=False):
        value = float(metric(g["delta"].to_numpy(), g["beta"].to_numpy()))
        rows.append({"group": name, "n": len(g), "value": value})
    table = pd.DataFrame(rows, columns=["group", "n", "value"])
    used = table[(table["n"] >= min_size) & table["value"].notna()]
    weighted = float(np.average(used["value"], weights=used["n"])) if len(used) else float("nan")
    return weighted, table


def mcnemar(calls_a: Iterable[str], calls_b: Iterable[str]) -> dict:
    """Compares two sets of calls on the same pairs with an exact McNemar test.

    Args:
        calls_a: Classifications from the first method.
        calls_b: Classifications from the second method in the same order.

    Returns:
        Counts a_better and b_better for pairs where one method is concordant
        and the other discordant, their sum (n_discordant_pairs), and the
        two-sided p_exact, which is NaN when there are no disagreements.

    Raises:
        ValueError: If the lists differ in length.
    """
    a = list(calls_a)
    b = list(calls_b)
    if len(a) != len(b):
        raise ValueError("call lists must have the same length")
    a_only = sum(x == CONCORDANT and y == DISCORDANT for x, y in zip(a, b, strict=True))
    b_only = sum(x == DISCORDANT and y == CONCORDANT for x, y in zip(a, b, strict=True))
    n = a_only + b_only
    p = (
        float(stats.binomtest(a_only, n, 0.5, alternative="two-sided").pvalue)
        if n > 0
        else float("nan")
    )
    return {"a_better": a_only, "b_better": b_only, "n_discordant_pairs": n, "p_exact": p}


def monte_carlo_summary(values: ArrayLike, seeds: ArrayLike, first_half: Iterable[Any]) -> dict:
    """Summarizes sampling variability across conditioning draws for one variant.

    A direction is resolved when the absolute mean meets the 95% t margin,
    the mean is nonzero, and both half-means have the same nonzero sign.
    The interval describes Monte Carlo error conditional on the draw procedure.

    Args:
        values: Scores from at least two finite draws.
        seeds: Seed label for each score in the same order.
        first_half: Seed labels assigned to the first half.

    Returns:
        Draw count, mean, sample standard deviation, Monte Carlo standard
        error, t interval, half-means, and direction diagnostics.

    Raises:
        ValueError: If there are fewer than two draws, any score is non-finite,
            or either half is empty.
    """
    v = np.asarray(values, dtype=float)
    s = np.asarray(seeds)
    if v.size < 2 or not np.all(np.isfinite(v)):
        raise ValueError("need at least two finite draws")
    in_first = np.isin(s, list(first_half))
    if not in_first.any() or in_first.all():
        raise ValueError("both halves need at least one draw")
    n = v.size
    mean = float(v.mean())
    sd = float(v.std(ddof=1))
    mcse = sd / np.sqrt(n)
    t_crit = float(stats.t.ppf(0.975, n - 1))
    half1, half2 = float(v[in_first].mean()), float(v[~in_first].mean())
    excludes_zero = bool(abs(mean) >= t_crit * mcse and mean != 0)
    halves_agree = bool(half1 != 0 and half2 != 0 and np.sign(half1) == np.sign(half2))
    return {
        "n_draws": n,
        "mean": mean,
        "sd_between_draws": sd,
        "mcse": float(mcse),
        "t_critical": t_crit,
        "ci95_low": mean - t_crit * mcse,
        "ci95_high": mean + t_crit * mcse,
        "abs_mean_over_mcse": abs(mean) / mcse if mcse > 0 else float("inf"),
        "half1_mean": half1,
        "half2_mean": half2,
        "ci_excludes_zero": excludes_zero,
        "halves_agree_in_sign": halves_agree,
        "directionally_resolved": excludes_zero and halves_agree,
    }
