"""Variant input adapters that preserve source order and 1-based coordinates."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

OUTPUT_COLUMNS = (
    "variant_id",
    "chrom",
    "pos",
    "ref",
    "alt",
    "source_id",
    "assembly",
    "record_index",
    "alt_index",
    "input_status",
)
_ALIASES = {
    "chrom": ("chrom", "chromosome", "chr", "#chrom"),
    "pos": ("pos", "position", "bp"),
    "ref": ("ref", "reference", "ref_allele"),
    "alt": ("alt", "alternate", "alt_allele"),
}
_ID_COLUMNS = ("source_id", "id", "variant_id", "rsid")
_VCF_SUFFIXES = (".vcf", ".vcf.gz", ".vcf.bgz", ".bcf")
_ALLELE = re.compile(r"^[ACGTN]+$")
_MISSING = frozenset({"", "."})
_BUILDS = {
    "GRCh37": "grch37|hg19|b37|hs37d5|human_g1k_v37|g1k_v37|assembly19",
    "GRCh38": "grch38|hg38|b38|hs38dh|assembly38",
}
# Chromosome 1 length identifies the build when the contig lacks an assembly label.
_CHR1_LENGTH = {249250621: "GRCh37", 248956422: "GRCh38"}


def _pysam() -> Any:
    try:
        import pysam
    except ImportError as exc:
        raise ImportError(
            'Reading VCF/BCF or FASTA needs pysam: pip install "ag2kit[vcf]"'
        ) from exc
    return pysam


def _chrom(chrom: object) -> str:
    chrom = "chr" + str(chrom).removeprefix("chr")
    return "chrM" if chrom == "chrMT" else chrom


def _allele(allele: object) -> str:
    return "" if pd.isna(allele) else str(allele).strip().upper()


def _builds(text: str) -> set[str]:
    text = text.lower()
    return {b for b, p in _BUILDS.items() if re.search(rf"(?<![a-z0-9])(?:{p})(?![a-z0-9])", text)}


def _check_header(header: Any, assembly: str) -> None:
    declared = _builds(assembly) or {assembly}
    for record in header.records:
        if record.key == "reference":
            value = record.value
        elif record.type == "CONTIG":
            value = record.get("assembly")
            length = _CHR1_LENGTH.get(int(record.get("length") or 0))
            if record.get("ID") in ("1", "chr1") and length and not value:
                value = f"{length} (chr1 length {record.get('length')})"
        else:
            continue
        found = _builds(str(value or ""))
        if found and not found & declared:
            raise ValueError(
                f"VCF header names build {sorted(found)[0]} ({value!r}) but assembly={assembly!r};"
                " ag2kit never converts coordinates between assemblies"
            )


def _from_vcf(path: Path, assembly: str) -> pd.DataFrame:
    pysam = _pysam()
    rows = []
    with pysam.VariantFile(str(path)) as vcf:
        _check_header(vcf.header, assembly)
        for i, rec in enumerate(vcf):
            for j, alt in enumerate(rec.alts or (".",), start=1):
                rows.append((rec.chrom, rec.pos, rec.ref or ".", alt, rec.id or "", i, j))
    columns = ["chrom", "pos", "ref", "alt", "source_id", "record_index", "alt_index"]
    return pd.DataFrame(rows, columns=columns)


def _from_table(table: pd.DataFrame, assembly: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    table = table.reset_index(drop=True)
    lower: dict[str, Any] = {}
    for column in table.columns:
        lower.setdefault(str(column).lower(), column)
    found = {k: next((lower[a] for a in v if a in lower), None) for k, v in _ALIASES.items()}
    missing = [k for k, v in found.items() if v is None]
    if missing:
        raise ValueError(f"variant table is missing required columns {missing}")
    declared = _builds(assembly) or {assembly}
    if "assembly" in table.columns and any(
        (_builds(str(a)) or {str(a)}) != declared for a in table["assembly"].unique()
    ):
        raise ValueError(f"table has an assembly column that differs from assembly={assembly!r}")
    raw_pos = table[found["pos"]]
    pos = pd.to_numeric(raw_pos, errors="coerce")
    bad = (pos.isna() | (pos < 1) | (pos % 1 != 0)).to_numpy()
    if bad.any():
        i = int(np.argmax(bad))
        raise ValueError(f"row {i}: pos {raw_pos.iloc[i]!r} is not a positive integer")
    id_column = next((lower[a] for a in _ID_COLUMNS if a in lower), None)
    source_id = (
        table[id_column].map(lambda v: "" if pd.isna(v) or str(v) == "." else str(v))
        if id_column is not None
        else pd.Series("", index=table.index, dtype=object)
    )
    core = pd.DataFrame(
        {
            "chrom": table[found["chrom"]],
            "pos": pos,
            "ref": table[found["ref"]],
            "alt": table[found["alt"]],
            "source_id": source_id,
            "record_index": table.get("record_index", pd.Series(table.index)),
            "alt_index": table.get("alt_index", pd.Series(1, index=table.index)),
        }
    )
    used = {*found.values(), id_column, *OUTPUT_COLUMNS}
    return core, table[[c for c in table.columns if c not in used]]


def _contig(chrom: str, references: set[str]) -> str | None:
    bare = chrom.removeprefix("chr")
    candidates = [chrom, bare, *(["MT", "chrMT"] if bare == "M" else [])]
    return next((c for c in candidates if c in references), None)


def _status(chrom: str, pos: int, ref: str, alt: str, fasta: Any, references: set[str]) -> str:
    if ref in _MISSING or alt in _MISSING:
        return "missing_allele"
    if alt == "*" or alt.startswith("<") or "[" in alt or "]" in alt:
        return "symbolic_allele"
    if not (_ALLELE.match(ref) and _ALLELE.match(alt)):
        return "invalid_allele"
    if ref == alt:
        return "ref_equals_alt"
    if fasta is None:
        return "ok"
    contig = _contig(chrom, references)
    if contig is None:
        return "contig_not_in_reference"
    end = pos - 1 + len(ref)
    if end > fasta.get_reference_length(contig):
        return "position_out_of_range"
    if fasta.fetch(contig, pos - 1, end).upper() != ref:
        return "ref_mismatch"
    return "ok"


def read_variants(
    source: str | Path | pd.DataFrame,
    *,
    assembly: str = "GRCh38",
    fasta: str | Path | None = None,
) -> pd.DataFrame:
    """Loads a canonical variant table with one row per record and ALT allele.

    Coordinates stay on the declared assembly. Indels retain their source
    representation, so normalize upstream if downstream matching requires it.

    Args:
        source: VCF/BCF, CSV, TSV, TXT, Parquet, or DataFrame. Compressed VCF
            and gzip-compressed text are supported. Tabular input needs
            chromosome, position, REF, and ALT columns. Column aliases are
            case-insensitive, and unused columns pass through.
        assembly: Genome assembly that the input coordinates use.
        fasta: Optional indexed FASTA on that assembly, used to check the
            contig, bounds, and full REF sequence.

    Returns:
        A table in source order with chr-prefixed chromosomes (MT becomes
        chrM), 1-based positions, uppercase alleles, canonical variant IDs,
        and source IDs when available. record_index is zero-based and shared
        by split records. alt_index is one-based. input_status records the
        first allele or reference check that fails, or "ok".

    Raises:
        ValueError: If the format is unsupported, required columns are missing,
            table positions are invalid, or the assembly is blank or conflicts
            with input metadata.
        ImportError: If VCF/BCF or FASTA input is used without pysam.
        OSError: If an input file cannot be read.
    """
    if not isinstance(assembly, str) or not assembly.strip():
        raise ValueError(f"assembly must be a non-empty string, got {assembly!r}")
    extra = None
    if isinstance(source, pd.DataFrame):
        core, extra = _from_table(source, assembly)
    else:
        path = Path(source)
        name = path.name.lower()
        if name.endswith(_VCF_SUFFIXES):
            core = _from_vcf(path, assembly)
        elif name.endswith(".parquet"):
            core, extra = _from_table(pd.read_parquet(path), assembly)
        elif name.endswith((".csv", ".csv.gz")):
            core, extra = _from_table(pd.read_csv(path), assembly)
        elif name.endswith((".tsv", ".txt", ".tsv.gz", ".txt.gz")):
            core, extra = _from_table(pd.read_csv(path, sep="\t"), assembly)
        else:
            raise ValueError(f"unrecognised variant file type: {path.name}")

    out = pd.DataFrame(index=core.index)
    out["chrom"] = core["chrom"].map(_chrom).astype(object)
    out["pos"] = core["pos"].astype("int64")
    out["ref"] = core["ref"].map(_allele).astype(object)
    out["alt"] = core["alt"].map(_allele).astype(object)
    out["source_id"] = core["source_id"].astype(str).astype(object)
    out["assembly"] = pd.Series(assembly, index=core.index, dtype=object)
    out["record_index"] = core["record_index"].astype("int64")
    out["alt_index"] = core["alt_index"].astype("int64")
    out["variant_id"] = (
        out["chrom"] + ":" + out["pos"].astype(str) + ":" + out["ref"] + ">" + out["alt"]
    ).astype(object)
    rows = zip(out["chrom"], out["pos"], out["ref"], out["alt"], strict=True)
    if fasta is None:
        status = [_status(c, int(p), r, a, None, set()) for c, p, r, a in rows]
    else:
        with _pysam().FastaFile(str(fasta)) as handle:
            refs = set(handle.references)
            status = [_status(c, int(p), r, a, handle, refs) for c, p, r, a in rows]
    out["input_status"] = pd.Series(status, index=core.index, dtype=object)
    out = out[list(OUTPUT_COLUMNS)]
    return out if extra is None else pd.concat([out, extra], axis=1)
