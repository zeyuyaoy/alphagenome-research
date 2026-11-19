"""Tests canonical variant input across file formats and tables."""

from __future__ import annotations

import re
import sys

import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from ag2kit.io import OUTPUT_COLUMNS, read_variants

SEQ = "ACGTACGTAC" * 6


def write_vcf(path, records, contigs=("1",), header=()):
    lines = ["##fileformat=VCFv4.2", *header]
    lines += [f"##contig=<ID={c},length=1000>" for c in contigs]
    lines.append("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO")
    lines += ["\t".join([*map(str, r), ".", ".", "."]) for r in records]
    path.write_text("\n".join(lines) + "\n")
    return path


def write_fasta(path, contigs):
    pysam = pytest.importorskip("pysam")
    path.write_text("".join(f">{c}\n{SEQ}\n" for c in contigs))
    pysam.faidx(str(path))
    return path


def test_vcf_split_order_and_statuses(tmp_path):
    pytest.importorskip("pysam")
    vcf = write_vcf(
        tmp_path / "v.vcf",
        [
            (1, 10, "rs1", "a", "C,G"),
            (1, 20, ".", "A", "<DEL>"),
            (1, 30, ".", "A", "*"),
            (1, 40, ".", "A", "A]1:100]"),
            (1, 50, ".", "A", "."),
            (1, 55, ".", "A", "A"),
            (1, 60, ".", "A", "R"),
        ],
    )
    out = read_variants(vcf)
    assert list(out.columns) == list(OUTPUT_COLUMNS)
    assert out["record_index"].tolist() == [0, 0, 1, 2, 3, 4, 5, 6]
    assert out["alt_index"].tolist() == [1, 2, 1, 1, 1, 1, 1, 1]
    assert out["variant_id"].tolist()[:2] == ["chr1:10:A>C", "chr1:10:A>G"]
    assert out["source_id"].tolist()[:3] == ["rs1", "rs1", ""]
    assert out["input_status"].tolist() == [
        "ok",
        "ok",
        "symbolic_allele",
        "symbolic_allele",
        "symbolic_allele",
        "missing_allele",
        "ref_equals_alt",
        "invalid_allele",
    ]
    assert out["pos"].dtype == "int64" and (out["assembly"] == "GRCh38").all()


def test_compressed_and_bcf_inputs_match_plain_vcf(tmp_path):
    pysam = pytest.importorskip("pysam")
    vcf = write_vcf(tmp_path / "v.vcf", [(1, 10, "rs1", "A", "C,G"), (1, 20, ".", "C", "T")])
    expected = read_variants(vcf)
    bgz = tmp_path / "v.vcf.bgz"
    pysam.tabix_compress(str(vcf), str(bgz))
    gz = pysam.tabix_index(str(vcf), preset="vcf", keep_original=True)
    bcf = tmp_path / "v.bcf"
    with (
        pysam.VariantFile(str(vcf)) as src,
        pysam.VariantFile(str(bcf), "wb", header=src.header) as dst,
    ):
        for rec in src:
            dst.write(rec)
    for path in (bgz, gz, bcf):
        pd.testing.assert_frame_equal(read_variants(path), expected)


@pytest.mark.parametrize("vcf_prefix,fasta_prefix", [("", "chr"), ("chr", "")])
def test_fasta_checks_across_contig_naming(tmp_path, vcf_prefix, fasta_prefix):
    mito_vcf = "chrM" if vcf_prefix else "MT"
    mito_fasta = "chrM" if fasta_prefix else "MT"
    fasta = write_fasta(tmp_path / "ref.fa", [f"{fasta_prefix}1", mito_fasta])
    c = f"{vcf_prefix}1"
    vcf = write_vcf(
        tmp_path / "v.vcf",
        [
            (c, 1, ".", "ACG", "A"),
            (c, 2, ".", "A", "G"),
            (c, 59, ".", "ACG", "A"),
            (f"{vcf_prefix}2", 5, ".", "A", "G"),
            (mito_vcf, 3, ".", "G", "T"),
        ],
        contigs=(c, f"{vcf_prefix}2", mito_vcf),
    )
    out = read_variants(vcf, fasta=fasta)
    assert out["input_status"].tolist() == [
        "ok",
        "ref_mismatch",
        "position_out_of_range",
        "contig_not_in_reference",
        "ok",
    ]
    assert out["chrom"].tolist() == ["chr1", "chr1", "chr1", "chr2", "chrM"]


@pytest.mark.parametrize(
    "line",
    ["##reference=file:///refs/hs37d5.fa.gz", "##contig=<ID=2,length=1000,assembly=b37>"],
)
def test_vcf_header_with_other_build_raises(tmp_path, line):
    pytest.importorskip("pysam")
    vcf = write_vcf(tmp_path / "v.vcf", [(1, 10, ".", "A", "C")], header=[line])
    with pytest.raises(ValueError, match="GRCh37.*GRCh38"):
        read_variants(vcf, assembly="GRCh38")
    assert read_variants(vcf, assembly="GRCh37")["assembly"].tolist() == ["GRCh37"]


def test_vcf_header_without_build_is_accepted(tmp_path):
    pytest.importorskip("pysam")
    vcf = write_vcf(tmp_path / "v.vcf", [(1, 10, ".", "A", "C")], header=["##reference=ref.fa"])
    assert read_variants(vcf)["input_status"].tolist() == ["ok"]


def test_empty_assembly_raises():
    with pytest.raises(ValueError, match="assembly"):
        read_variants(
            pd.DataFrame({"chrom": [1], "pos": [1], "ref": ["A"], "alt": ["C"]}), assembly=""
        )


def test_missing_pysam_gives_install_hint(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "pysam", None)
    with pytest.raises(ImportError, match=r"ag2kit\[vcf\]"):
        read_variants(write_vcf(tmp_path / "v.vcf", [(1, 10, ".", "A", "C")]))


TABLE = pd.DataFrame(
    {
        "CHROM": ["2", "chr1", "MT"],
        "Position": [300, 100, 5],
        "REF_allele": ["a", "C", "G"],
        "alt": ["G", "<INS>", "G"],
        "rsid": ["rs3", None, "."],
        "score": [0.5, 1.5, 2.5],
        "note": ["x", "y", "z"],
    }
)


@pytest.mark.parametrize("kind", ["frame", "csv", "tsv.gz", "parquet"])
def test_tabular_inputs_keep_order_and_passthrough(tmp_path, kind):
    path = tmp_path / f"t.{kind}"
    if kind == "frame":
        source = TABLE
    elif kind == "parquet":
        TABLE.to_parquet(path)
        source = path
    else:
        TABLE.to_csv(path, sep="," if kind == "csv" else "\t", index=False)
        source = path
    out = read_variants(source)
    assert list(out.columns) == [*OUTPUT_COLUMNS, "score", "note"]
    assert out["variant_id"].tolist() == ["chr2:300:A>G", "chr1:100:C><INS>", "chrM:5:G>G"]
    assert out["input_status"].tolist() == ["ok", "symbolic_allele", "ref_equals_alt"]
    assert out["source_id"].tolist() == ["rs3", "", ""]
    assert out["record_index"].tolist() == [0, 1, 2]
    assert out["alt_index"].tolist() == [1, 1, 1]
    assert out["score"].tolist() == [0.5, 1.5, 2.5] and out["note"].tolist() == ["x", "y", "z"]


def test_missing_columns_are_listed():
    with pytest.raises(ValueError, match=r"\['pos', 'alt'\]"):
        read_variants(pd.DataFrame({"chr": ["1"], "ref": ["A"]}))


@pytest.mark.parametrize(
    "bad,dtype",
    [
        (0, None),
        (-5, None),
        (1.5, None),
        ("abc", None),
        (None, None),
        (2**63, None),
        (str(2**63), None),
        (float(2**63), None),
        (1e30, None),
        (None, "int64[pyarrow]"),
        (2**63, "uint64[pyarrow]"),
        (1.5, "double[pyarrow]"),
        (float(2**63), "double[pyarrow]"),
    ],
)
def test_bad_positions_raise_naming_the_row(bad, dtype):
    table = pd.DataFrame(
        {"chrom": "1", "pos": pd.Series([10, bad], dtype=dtype), "ref": "A", "alt": "C"}
    )
    with pytest.raises(ValueError, match="row 1"):
        read_variants(table)


@pytest.mark.parametrize(
    "pos,dtype",
    [
        (2**63 - 1, "int64"),
        (2**63 - 1, "int64[pyarrow]"),
        (2**63 - 1, "uint64[pyarrow]"),
        (float(2**63 - 1024), "double[pyarrow]"),
    ],
)
def test_large_valid_position_preserves_value(pos, dtype):
    table = pd.DataFrame(
        {"chrom": "1", "pos": pd.Series([pos], dtype=dtype), "ref": "A", "alt": "G"}
    )
    assert read_variants(table)["pos"].item() == pos


def test_categorical_alleles_preserve_missing_values_and_normalize_text():
    table = pd.DataFrame(
        {"chrom": "1", "pos": [1, 2], "ref": pd.Categorical([" a ", None]), "alt": "G"}
    )
    result = read_variants(table)
    assert result["ref"].tolist() == ["A", ""]
    assert result["input_status"].tolist() == ["ok", "missing_allele"]


def test_parquet_round_trip_preserves_output(tmp_path):
    first = read_variants(TABLE)
    first.to_parquet(tmp_path / "v.parquet")
    pd.testing.assert_frame_equal(read_variants(tmp_path / "v.parquet"), first)
    with pytest.raises(ValueError, match="assembly"):
        read_variants(tmp_path / "v.parquet", assembly="GRCh37")


def test_parquet_round_trip_of_split_vcf(tmp_path):
    pytest.importorskip("pysam")
    vcf = write_vcf(tmp_path / "v.vcf", [(1, 10, "rs1", "A", "C,G"), (1, 20, ".", "C", ".")])
    first = read_variants(vcf)
    first.to_parquet(tmp_path / "v.parquet")
    pd.testing.assert_frame_equal(read_variants(tmp_path / "v.parquet"), first)


VARIANT_ID = re.compile(r"^(chr[^:]+):(\d+):([ACGTN]+)>([ACGTN]+)$")
snv = st.tuples(
    st.sampled_from(["1", "chr2", "X", "chrY", "MT"]),
    st.integers(1, 250_000_000),
    st.sampled_from("ACGT"),
    st.sampled_from("ACGT"),
).filter(lambda r: r[2] != r[3])


@settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(st.lists(snv, max_size=30))
def test_snv_tables_round_trip_through_variant_id(rows):
    df = pd.DataFrame(rows, columns=["chrom", "pos", "ref", "alt"])
    out = read_variants(df)
    assert len(out) == len(df) and out["record_index"].tolist() == list(range(len(df)))
    parsed = [VARIANT_ID.match(v).groups() for v in out["variant_id"]]  # type: ignore[union-attr]
    expected = [
        ("chrM" if c in ("MT", "chrMT") else "chr" + c.removeprefix("chr"), str(p), r, a)
        for c, p, r, a in rows
    ]
    assert parsed == expected
    assert (out["input_status"] == "ok").all()


@pytest.mark.parametrize(
    "header",
    [
        "##reference=file:///ref/human_g1k_v37.fasta",
        "##reference=file:///ref/Homo_sapiens_assembly19.fasta",
        "##contig=<ID=1,length=249250621>",
    ],
)
def test_grch37_vcfs_are_detected_by_name_or_chr1_length(tmp_path, header):
    pytest.importorskip("pysam")
    contig = "" if "contig" in header else "##contig=<ID=1>\n"
    path = tmp_path / "b37.vcf"
    path.write_text(
        f"##fileformat=VCFv4.2\n{header}\n{contig}"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n1\t100\t.\tA\tG\t.\t.\t.\n"
    )
    with pytest.raises(ValueError, match="GRCh37"):
        read_variants(path, assembly="GRCh38")


def test_assembly_aliases_are_equivalent():
    df = pd.DataFrame(
        {"chrom": ["1"], "pos": [5], "ref": ["A"], "alt": ["G"], "assembly": ["hg38"]}
    )
    assert read_variants(df, assembly="GRCh38")["assembly"][0] == "GRCh38"
