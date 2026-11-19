"""Allele harmonization, GRCh38 reference-allele checks and locus clustering.

For gene symbols, IDs and coordinates use the SDK's ``alphagenome.data.gene_annotation``
with a GENCODE GTF, so genes are defined by an annotation release you can pin.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

from ag2kit.provenance import cache_dir

log = logging.getLogger("ag2kit")

BASES = frozenset("ACGT")
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
    ref, alt = str(ref).upper(), str(alt).upper()
    effect = str(effect_allele).upper()
    if ref == alt:
        return Harmonized(None, False, "ref_equals_alt")
    if not forward_strand and not is_snv(ref, alt):
        return Harmonized(None, False, "not_snv")
    if not forward_strand and is_palindromic(ref, alt):
        return Harmonized(None, False, "palindromic")
    if effect == alt:
        return Harmonized(float(estimate), False, "ok")
    if effect == ref:
        return Harmonized(-float(estimate), True, "ok")
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


def cluster_loci(df: pd.DataFrame, window: int = 1_000_000) -> pd.DataFrame:
    """Labels one lead per locus using distance and gene as proxies for LD.

    Rows are visited by ascending p-value, breaking ties by variant ID.
    Each unassigned row becomes a lead and absorbs unassigned rows within
    the window on the same chromosome, plus rows with the same gene.

    Args:
        df: Table with variant_id, chrom, pos, gene, and pval columns.
        window: Maximum distance from the lead in base pairs.

    Returns:
        The input rows in original order with locus_group (the lead's variant
        ID) and is_lead.

    Raises:
        ValueError: If a required column is missing.
    """
    # shortcut: Quadratic scan suits small tables. Index by position if this becomes slow.
    required = {"variant_id", "chrom", "pos", "gene", "pval"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"cluster_loci requires columns {sorted(missing)}")
    work = df.sort_values(["pval", "variant_id"], kind="mergesort").reset_index()
    chrom = work["chrom"].map(_chrom)
    pos = work["pos"].astype(float)
    gene = work["gene"].astype(str)
    group = pd.Series(pd.NA, index=work.index, dtype="object")
    lead = pd.Series(False, index=work.index)
    for i in work.index:
        if pd.notna(group[i]):
            continue
        lead[i] = True
        near = (chrom == chrom[i]) & (pos - pos[i]).abs().le(window)
        group[(near | (gene == gene[i])) & group.isna()] = work.at[i, "variant_id"]
    work["locus_group"] = group
    work["is_lead"] = lead
    out = work.set_index("index").loc[df.index, ["locus_group", "is_lead"]]
    return df.join(out)
