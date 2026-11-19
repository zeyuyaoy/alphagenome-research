"""Loads significant GTEx eQTL pairs from the Portal API or release files.

Both sources report normalized expression slopes for ALT dosage on GRCh38.
Use signs and ranks to compare these slopes with model scores. Restricting
evaluation to significant pairs favors strong effects (see docs/methods.md).
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import requests

from ag2kit.provenance import cache_dir, stable_hash, utc_now

log = logging.getLogger("ag2kit")

API = "https://gtexportal.org/api/v2"
SOURCE = "GTEx Portal API v2"
# GENCODE releases come from the Portal metadata for each GRCh38 dataset.
GENCODE = {"gtex_v8": "v26", "gtex_v10": "v39", "gtex_v11": "v47"}
COLUMNS = (
    "variant_id",
    "chrom",
    "pos",
    "ref",
    "alt",
    "source_id",
    "rsid",
    "gene_id",
    "gencode_id",
    "gene_symbol",
    "tissue",
    "ontology_id",
    "beta",
    "se",
    "pvalue",
    "effect_allele",
    "effect_scale",
    "assembly",
    "dataset_id",
    "source",
)


def _dataset(dataset: str) -> str:
    if dataset not in GENCODE:
        raise ValueError(f"dataset must be one of {sorted(GENCODE)} (GRCh38 releases)")
    return dataset


def _get(path: str, params: dict, refresh: bool = False) -> dict:
    """Fetches JSON with disk caching for successful responses.

    Args:
        path: Endpoint relative to the GTEx API base URL.
        params: Query parameters, also used in the cache key.
        refresh: Whether to bypass the cache read.

    Returns:
        Decoded response body.

    Raises:
        RuntimeError: If the HTTP request fails after the applicable retries.
    """
    cache = cache_dir() / "gtex" / f"{stable_hash({'path': path, **params})}.json"
    if not refresh:
        try:
            return json.loads(cache.read_text())["response"]
        except (OSError, ValueError, KeyError):
            pass
    for attempt in range(3):
        try:
            response = requests.get(f"{API}/{path}", params=params, timeout=60)
        except requests.RequestException as exc:
            error: str = str(exc)
        else:
            if response.ok:
                body = response.json()
                cache.parent.mkdir(parents=True, exist_ok=True)
                payload = {"url": response.url, "retrieved_utc": utc_now(), "response": body}
                tmp = cache.with_suffix(".tmp")
                tmp.write_text(json.dumps(payload))
                tmp.replace(cache)
                return body
            error = f"HTTP {response.status_code}: {response.text[:200]}"
            if 400 <= response.status_code < 500 and response.status_code != 429:
                break
        time.sleep(2**attempt)
    raise RuntimeError(f"GTEx API {path} {params} failed: {error}")


def _pages(path: str, params: dict, refresh: bool) -> list[dict]:
    rows: list[dict] = []
    page = 0
    while True:
        body = _get(path, {**params, "page": page, "itemsPerPage": 1000}, refresh)
        rows += body.get("data", [])
        info = body.get("paging_info", {})
        page += 1
        if page >= int(info.get("numberOfPages", 1)):
            return rows


def gencode_id(gene: str, dataset: str = "gtex_v10", refresh: bool = False) -> str:
    """Resolves a gene to the GENCODE ID used by a GTEx release.

    The Portal matches versioned IDs exactly, so a suffix from a different
    GENCODE release may prevent a match.

    Args:
        gene: Gene symbol or Ensembl ID.
        dataset: Supported GTEx dataset ID.
        refresh: Whether to bypass the cached response.

    Returns:
        The versioned GENCODE ID.

    Raises:
        ValueError: If the dataset is unsupported or the match is absent or ambiguous.
        RuntimeError: If the Portal request fails.
    """
    body = _get(
        "reference/gene",
        {
            "geneId": gene,
            "gencodeVersion": GENCODE[_dataset(dataset)],
            "genomeBuild": "GRCh38/hg38",
        },
        refresh,
    )
    hits = body.get("data", [])
    if len(hits) != 1:
        raise ValueError(f"{gene!r} matched {len(hits)} genes in GENCODE {GENCODE[dataset]}")
    return str(hits[0]["gencodeId"])


def tissues(dataset: str = "gtex_v10", refresh: bool = False) -> pd.DataFrame:
    """Lists tissue metadata for a GTEx release.

    Args:
        dataset: Supported GTEx dataset ID.
        refresh: Whether to bypass the cached response.

    Returns:
        A table of tissue metadata, including UBERON IDs and eGene counts.

    Raises:
        ValueError: If the dataset is unsupported.
        RuntimeError: If the Portal request fails.
    """
    body = _get("dataset/tissueSiteDetail", {"datasetId": _dataset(dataset)}, refresh)
    return pd.DataFrame(body.get("data", body if isinstance(body, list) else []))


def _split_variant(variant: pd.Series) -> pd.DataFrame:
    parts = variant.astype(str).str.extract(r"^([^_]+)_(0*[1-9][0-9]*)_([^_]+)_([^_]+)_b38\Z")
    invalid = parts[0].isna()
    if invalid.any():
        bad = variant[invalid].iloc[0]
        raise ValueError(f"unexpected GTEx variant id {bad!r} (want chr_pos_ref_alt_b38)")
    return pd.DataFrame(
        {"chrom": parts[0], "pos": parts[1].astype("int64"), "ref": parts[2], "alt": parts[3]}
    )


def _finish(df: pd.DataFrame, dataset: str, source: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=list(COLUMNS))
    df = df.drop(columns=["chrom", "chromosome", "pos", "ref", "alt"], errors="ignore")
    df = pd.concat([_split_variant(df["source_id"]), df], axis=1)
    df["variant_id"] = df["chrom"] + ":" + df["pos"].astype(str) + ":" + df["ref"] + ">" + df["alt"]
    df["gene_id"] = df["gencode_id"].str.split(".").str[0]
    df["effect_allele"] = df["alt"]
    df["effect_scale"] = "normalized_expression_slope"
    df["assembly"] = "GRCh38"
    df["dataset_id"] = dataset
    df["source"] = source
    for col in COLUMNS:
        if col not in df:
            df[col] = pd.NA
    return df.loc[:, list(COLUMNS)].reset_index(drop=True)


def eqtls(
    genes: Iterable[str],
    tissue: str,
    dataset: str = "gtex_v10",
    refresh: bool = False,
) -> pd.DataFrame:
    """Fetches significant single-tissue eQTLs from the GTEx Portal.

    Args:
        genes: Gene symbols or unversioned Ensembl IDs to resolve. IDs with
            a version suffix are passed through and must match the dataset.
        tissue: GTEx tissue ID, such as "Whole_Blood".
        dataset: Supported GTEx dataset ID.
        refresh: Whether to bypass cached responses.

    Returns:
        An observed-effects table with the columns in COLUMNS. Beta is the
        ALT-oriented normalized effect size (nes). The API leaves se missing.

    Raises:
        ValueError: If the dataset is unsupported, a gene cannot be resolved
            uniquely, or returned variant IDs are invalid.
        RuntimeError: If a Portal request fails.
    """
    _dataset(dataset)
    frames = []
    for gene in genes:
        gid = gene if "." in str(gene) else gencode_id(str(gene), dataset, refresh)
        rows = _pages(
            "association/singleTissueEqtl",
            {"gencodeId": gid, "tissueSiteDetailId": tissue, "datasetId": dataset},
            refresh,
        )
        if not rows:
            log.info("no significant %s eQTLs for %s in %s", dataset, gid, tissue)
        frames.append(pd.DataFrame(rows))
    raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    df = raw.rename(
        columns={
            "variantId": "source_id",
            "snpId": "rsid",
            "gencodeId": "gencode_id",
            "geneSymbol": "gene_symbol",
            "tissueSiteDetailId": "tissue",
            "ontologyId": "ontology_id",
            "nes": "beta",
            "pValue": "pvalue",
        }
    )
    return _finish(df, dataset, SOURCE)


def read_signif_pairs(path: str | Path, tissue: str, dataset: str) -> pd.DataFrame:
    """Loads significant GTEx variant-gene pairs from a tissue release file.

    Args:
        path: Per-tissue Parquet file (v10/v11) or tab-delimited file (v8).
        tissue: Tissue ID to assign to every row.
        dataset: Supported GTEx dataset ID.

    Returns:
        An observed-effects table as in eqtls, with beta from slope, se
        from slope_se when present, and pvalue from pval_nominal.

    Raises:
        ValueError: If the dataset is unsupported, required columns are missing,
            or variant IDs are invalid.
        OSError: If the file cannot be read.
    """
    path = Path(path)
    _dataset(dataset)
    raw = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, sep="\t")
    missing = {"variant_id", "gene_id", "slope", "pval_nominal"} - set(raw.columns)
    if missing:
        raise ValueError(f"{path.name} lacks GTEx columns {sorted(missing)}")
    df = raw.rename(
        columns={
            "variant_id": "source_id",
            "gene_id": "gencode_id",
            "slope": "beta",
            "slope_se": "se",
            "pval_nominal": "pvalue",
        }
    )
    df["tissue"] = tissue
    return _finish(df, dataset, f"GTEx {dataset} download: {path.name}")
