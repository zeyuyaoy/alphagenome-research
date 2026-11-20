"""Allele harmonization, GRCh38 reference-allele checks and locus clustering.

For gene symbols, IDs and coordinates use the SDK's ``alphagenome.data.gene_annotation``
with a GENCODE GTF, so genes are defined by an annotation release you can pin.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from ag2kit.provenance import cache_dir

log = logging.getLogger("ag2kit")

BASES = frozenset("ACGT")
ALLELE = re.compile(r"^[ACGTN]+$")
PALINDROMIC = frozenset({("A", "T"), ("T", "A"), ("C", "G"), ("G", "C")})

SEQUENCE_URL = "https://rest.ensembl.org/sequence/region/human/{region}?content-type=text/plain"


@dataclass(frozen=True)
class Harmonized:
    """Stores an allele-oriented effect or its exclusion reason.

    Attributes:
        estimate: Effect relative to ALT, or None for an excluded record.
        flipped: Whether the original estimate was negated.
        status: "ok" or the reason the record was excluded.
    """

    estimate: float | None
    flipped: bool
    status: str


def is_snv(ref: str, alt: str) -> bool:
    """Checks whether both alleles are single A, C, G, or T bases.

    Args:
        ref: Reference allele.
        alt: Alternate allele.

    Returns:
        Whether both are canonical single bases, ignoring case. Identical
        bases also pass this check.
    """
    return (
        isinstance(ref, str)
        and isinstance(alt, str)
        and len(ref) == 1
        and len(alt) == 1
        and ref.upper() in BASES
        and alt.upper() in BASES
    )


def is_palindromic(ref: str, alt: str) -> bool:
    """Checks for a single-base allele pair with ambiguous strand.

    Args:
        ref: Reference allele.
        alt: Alternate allele.

    Returns:
        Whether the two alleles are A/T or C/G in either order, ignoring case.
    """
    return (str(ref).upper(), str(alt).upper()) in PALINDROMIC


def normalize_chrom(chrom: object) -> str:
    """Formats a chromosome name as in GRCh38 UCSC style.

    Args:
        chrom: Name with or without the chr prefix. MT and chrMT become chrM.

    Returns:
        The chr-prefixed name.
    """
    name = "chr" + str(chrom).removeprefix("chr")
    return "chrM" if name == "chrMT" else name


def _chrom(chrom: object) -> str:
    return str(chrom).removeprefix("chr")


def harmonize_effect(
    estimate: float | None,
    effect_allele: str,
    ref: str,
    alt: str,
    forward_strand: bool = False,
) -> Harmonized:
    """Orients an observed effect to ALT relative to REF.

    REF and ALT should be checked against the GRCh38 forward strand before
    calling. The function compares allele labels directly without complementing
    bases. With unknown source strand, it excludes indels and palindromic SNVs.

    Args:
        estimate: Observed effect, or None if missing.
        effect_allele: Allele to which the observed effect refers.
        ref: Forward-strand GRCh38 reference allele.
        alt: Forward-strand GRCh38 alternate allele.
        forward_strand: Whether the source reports GRCh38 forward-strand
            alleles with genomic REF, as GTEx does.

    Returns:
        The original effect if it refers to ALT, the negated effect if it
        refers to REF, or an exclusion status if orientation fails.
    """
    if estimate is None or pd.isna(estimate):
        return Harmonized(None, False, "missing_statistics")
    estimate = float(estimate)
    if not math.isfinite(estimate):
        return Harmonized(None, False, "missing_statistics")
    ref, alt = str(ref).upper(), str(alt).upper()
    effect = str(effect_allele).upper()
    if ref == alt:
        return Harmonized(None, False, "ref_equals_alt")
    if not forward_strand and not is_snv(ref, alt):
        return Harmonized(None, False, "not_snv")
    if not forward_strand and is_palindromic(ref, alt):
        return Harmonized(None, False, "palindromic")
    if effect == alt:
        return Harmonized(estimate, False, "ok")
    if effect == ref:
        return Harmonized(-estimate, True, "ok")
    return Harmonized(None, False, "effect_allele_not_in_ref_alt")


def _load(path: Path) -> dict:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, sort_keys=True))
    tmp.replace(path)


class ReferenceBase:
    """Looks up GRCh38 reference bases through Ensembl and caches successful lookups.

    Checking REF before scoring helps prevent swapped REF and ALT labels
    from reversing the interpretation of downstream effects.

    Attributes:
        cache_path: Path to the local JSON cache.
    """

    def __init__(self, cache_path: str | Path | None = None):
        """Loads a reference-base cache.

        Args:
            cache_path: Cache file, defaulting to derived/grch38_bases.json under
                the ag2kit cache directory.
        """
        self.cache_path = Path(cache_path or cache_dir() / "derived/grch38_bases.json")
        self._cache: dict[str, str] = _load(self.cache_path)

    def get(self, chrom: str, pos: int, offline: bool = False) -> str | None:
        """Looks up the reference base for one position.

        Args:
            chrom: Chromosome name with or without the chr prefix.
            pos: 1-based GRCh38 position.
            offline: Whether to use only cached results.

        Returns:
            An uppercase base or None if the lookup fails or the offline cache misses.

        Raises:
            ValueError: If pos is below one.
            OSError: If a successful lookup cannot be saved to the cache.
        """
        if int(pos) < 1:
            raise ValueError(f"pos must be 1-based, got {pos}")
        key = f"{_chrom(chrom)}:{int(pos)}"
        if key in self._cache:
            return self._cache[key]
        if offline:
            return None
        url = SEQUENCE_URL.format(region=f"{_chrom(chrom)}:{pos}..{pos}")
        for attempt in range(3):
            try:
                response = requests.get(url, timeout=30)
            except requests.RequestException as exc:
                log.warning("Ensembl sequence lookup %s failed: %s", key, exc)
            else:
                base = response.text.strip().upper() if response.ok else ""
                if len(base) == 1 and base in "ACGTN":
                    self._cache[key] = base
                    _save(self.cache_path, self._cache)
                    return base
                if 400 <= response.status_code < 500 and response.status_code != 429:
                    log.warning("Ensembl rejected region %s (HTTP %s)", key, response.status_code)
                    return None
            time.sleep(1 + attempt)
        return None


def cluster_loci(
    df: pd.DataFrame,
    window: int = 1_000_000,
    *,
    gene: str = "gene",
    pval: str = "pval",
) -> pd.DataFrame:
    """Labels one lead per locus using distance and gene as proxies for LD.

    Rows are visited by ascending p-value, breaking ties by variant ID.
    Each unassigned row becomes a lead and absorbs unassigned rows within
    the window on the same chromosome, plus rows with the same gene.

    Args:
        df: Table with variant_id, chrom, pos, gene, and p-value columns.
        window: Maximum distance from the lead in base pairs.
        gene: Gene column, for example "gene_id" for GTEx tables.
        pval: P-value column, for example "pvalue" for GTEx tables.

    Returns:
        The input rows in original order with locus_group (the lead's variant
        ID) and is_lead.

    Raises:
        ValueError: If a required column is missing.
    """
    required = {"variant_id", "chrom", "pos", gene, pval}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"cluster_loci requires columns {sorted(missing)}")
    work = df.reset_index(drop=True)
    order = work.sort_values([pval, "variant_id"], kind="mergesort").index.to_numpy()
    _, chrom = np.unique(work["chrom"].map(normalize_chrom).to_numpy(str), return_inverse=True)
    _, genes = np.unique(work[gene].astype(str).to_numpy(), return_inverse=True)
    pos = work["pos"].to_numpy(float)
    ids = work["variant_id"].to_numpy(object)
    group = np.full(len(work), None, dtype=object)
    lead = np.zeros(len(work), dtype=bool)
    for i in order:
        if group[i] is not None:
            continue
        lead[i] = True
        near = ((chrom == chrom[i]) & (np.abs(pos - pos[i]) <= window)) | (genes == genes[i])
        group[near & (group == None)] = ids[i]  # noqa: E711
    out = df.copy()
    out["locus_group"] = group
    out["is_lead"] = lead
    return out


def _covered(pos0: np.ndarray, chrom: np.ndarray, regions: pd.DataFrame) -> np.ndarray:
    hit = np.zeros(pos0.size, dtype=bool)
    for name, iv in regions.groupby(regions["chromosome"].map(normalize_chrom)):
        mask = chrom == name
        if not mask.any():
            continue
        iv = iv.sort_values("start")
        starts = iv["start"].to_numpy()
        reach = np.maximum.accumulate(iv["end"].to_numpy())
        i = np.searchsorted(starts, pos0[mask], side="right") - 1
        hit[mask] = (i >= 0) & (reach[np.maximum(i, 0)] > pos0[mask])
    return hit


def annotate_folds(
    df: pd.DataFrame, model_version: str = "FOLD_1", regions_path: str | None = None
) -> pd.DataFrame:
    """Labels variants by the training split of a cross-validation checkpoint.

    A variant is "train" if any training region of the checkpoint contains it,
    otherwise "valid" or "test" if a region of that split contains it, and
    "none" if no region does. Evaluating with group="fold_subset" separates
    performance on held-out regions from regions seen in training.

    Args:
        df: Table with chrom and 1-based pos columns on GRCh38.
        model_version: Cross-validation checkpoint name, such as "FOLD_1".
        regions_path: Local copy of the SDK's example-regions BED file. None
            downloads the default file, which needs network access.

    Returns:
        A copy of df with a fold_subset column.

    Raises:
        ValueError: If model_version is not FOLD_1, the only AlphaGenome 2
            checkpoint that holds out regions.
        ImportError: If the AlphaGenome SDK is unavailable.
    """
    from alphagenome.data import fold_intervals
    from alphagenome.models.v1 import dna_client

    if model_version != "FOLD_1":
        raise ValueError(
            f"{model_version!r}: AlphaGenome 2 offers one cross-validation checkpoint, FOLD_1"
        )
    version = dna_client.ModelVersion[model_version]
    organism = dna_client.Organism.HOMO_SAPIENS
    pos0 = df["pos"].to_numpy(dtype=np.int64) - 1
    chrom = df["chrom"].map(normalize_chrom).to_numpy(str)
    label = np.full(len(df), "none", dtype=object)
    for subset in ("TEST", "VALID", "TRAIN"):
        regions = fold_intervals.get_fold_intervals(
            version, organism, fold_intervals.Subset[subset], regions_path
        )
        label[_covered(pos0, chrom, regions)] = subset.lower()
    out = df.copy()
    out["fold_subset"] = label
    return out


def locus_clusters(
    df: pd.DataFrame, window: int = 1_000_000, *, gene: str = "gene_id"
) -> pd.Series:
    """Groups rows into loci for use as bootstrap clusters.

    Rows are linked when they share a gene or lie on the same chromosome
    within window base pairs of each other, and loci are the connected
    components of these links. Unlike cluster_loci, every row of a gene and
    every chain of nearby variants falls in one locus, so dependence through
    shared genes, shared variants, and nearby LD stays within a cluster.

    Args:
        df: Table with chrom, pos, and gene columns.
        window: Maximum gap in base pairs between linked neighbouring rows.
        gene: Gene column.

    Returns:
        Integer locus labels aligned to df's index.

    Raises:
        ValueError: If a required column is missing.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    missing = {"chrom", "pos", gene} - set(df.columns)
    if missing:
        raise ValueError(f"locus_clusters requires columns {sorted(missing)}")
    n = len(df)
    chrom = df["chrom"].map(normalize_chrom).to_numpy(str)
    pos = df["pos"].to_numpy(float)
    order = np.lexsort((pos, chrom))
    near = (chrom[order][1:] == chrom[order][:-1]) & (np.diff(pos[order]) <= window)
    _, genes = np.unique(df[gene].astype(str).to_numpy(), return_inverse=True)
    by_gene = np.argsort(genes, kind="stable")
    same = genes[by_gene][1:] == genes[by_gene][:-1]
    rows = np.concatenate([order[:-1][near], by_gene[:-1][same]])
    cols = np.concatenate([order[1:][near], by_gene[1:][same]])
    graph = coo_matrix((np.ones(rows.size), (rows, cols)), shape=(n, n))
    labels = connected_components(graph, directed=False)[1] if n else np.empty(0, dtype=int)
    return pd.Series(labels, index=df.index, name="locus")
