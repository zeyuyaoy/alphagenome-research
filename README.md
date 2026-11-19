### """ag2kit: reproducible variant-effect analysis with the AlphaGenome SDK

AlphaGenome predicts how non-coding DNA variants affect gene regulation [1]. AlphaGenome 2 also supports predictions conditioned on a reference cell type, tissue, or single-cell dataset [4]. Its SDK handles individual predictions and batch scoring. Evaluating those predictions against observed effects requires additional steps, including input validation, allele matching, error handling, and uncertainty estimation that accounts for linkage disequilibrium.

`ag2kit` brings these steps into a Python workflow. It scores variants with the official [`alphagenome`](https://github.com/google-deepmind/alphagenome) SDK, caches results to resume interrupted runs, and compares predicted gene-expression effects with expression quantitative trait locus (eQTL) data from sources such as the Genotype-Tissue Expression (GTEx) project.

The package covers four tasks:

| Capability       | Description                                                                                                                                                                                                                                                        |
|------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Variant input    | `read_variants` loads VCF/BCF (via `pysam`), delimited text, Parquet, and DataFrames into a common GRCh38 schema. It splits multiallelic records, assigns an `input_status` to every row, and optionally checks REF alleles against an indexed FASTA.              |
| Scoring          | `AG2Client.score_variants` caches complete SDK results for each unique request. Scoring one variant against multiple genes uses one API call. Runs can resume after interruption, and failed requests receive error records. Output rows retain their input order. |
| Observed effects | `ag2kit.gtex` retrieves significant single-tissue eQTLs from GTEx Portal API v2 (GTEx v8, v10, and v11) or reads downloaded significant-pairs files [2].                                                                                                           |
| Evaluation       | `evaluate` orients observed effects to the predicted ALT allele and records a `pair_status` for every pair. It reports sign AUROC, signed Spearman correlation, and sign accuracy, with gene-cluster bootstrap intervals and optional tissue weighting.            |

> [!NOTE]
> Use the official SDK for single predictions, track plots, in-silico mutagenesis, and gene annotation (`alphagenome.data.gene_annotation`). Use separate tools for liftover and summary-statistic quality control. For indel normalisation, use `bcftools norm` [3].

#### Installation

```bash
pip install "ag2kit[model,vcf]"
```

Python 3.11 or later is required.

Install the `model` extra for the AlphaGenome SDK (2.x) and the `vcf` extra for VCF/BCF and FASTA support through `pysam`.

For live scoring, set the `ALPHA_GENOME_API_KEY` environment variable (see [`.env.example`](.env.example)).

#### Quick start

```python
import pandas as pd

import ag2kit

variants = ag2kit.read_variants(
    pd.DataFrame(
        {
            "chrom": ["21", "21"],
            "pos": [41422154, 41422154],
            "ref": ["T", "T"],
            "alt": ["C", "<DEL>"],
            "id": ["rs_example", "sv1"],
        }
    )
)
print(variants[["variant_id", "source_id", "input_status"]])

client = ag2kit.AG2Client()
plan = client.score_variants(variants, condition="UBERON:0013756", scorer="RNA_SEQ")
print(plan[["input_variant_id", "status", "error"]])
```

> [!NOTE]
> By default, `dry_run=True` plans requests without contacting the API. It also shows how input validation rejects a symbolic ALT allele.

#### Evaluating predictions against GTEx

Using the `ALPHA_GENOME_API_KEY`, the next example scores significant Whole Blood eQTL pairs for `MX1` and `OAS1`, then evaluates the results:

```python
import ag2kit
from ag2kit import gtex

observed = gtex.eqtls(["MX1", "OAS1"], tissue="Whole_Blood", dataset="gtex_v10")

client = ag2kit.AG2Client(dry_run=False)
predictions = client.score_variants(
    observed,
    condition="UBERON:0013756",  # Whole Blood; gtex.tissues() lists each tissue's UBERON ID
    scorer="RNA_SEQ",
    gene_column="gene_id",
    tracks={"imputation_mode": "observed"},
)
predictions.to_parquet("predictions.parquet")

evaluation = ag2kit.evaluate(predictions, observed, match={"ontology_id": "condition"})
evaluation.summary()
evaluation.save("results/")
```

The `tracks={"imputation_mode": "observed"}` option limits gene-effect calculations to tracks observed during training; see [Methods](docs/methods.md#gene-level-effect).

For a command-line workflow, use [`examples/gtex_eqtl_eval.py`](examples/gtex_eqtl_eval.py). It also generates a scatter plot with `matplotlib`, which must be installed separately. 

> [!WARNING]
> Without an API key, the script downloads GTEx data and estimates the API requests needed for scoring.

You can also call `evaluate` on any two tables with the required columns. The following example uses synthetic data:

```python
import pandas as pd

import ag2kit

pred = pd.DataFrame(
    {
        "chrom": "chr1",
        "pos": [100, 200, 300, 400],
        "ref": "A",
        "alt": "G",
        "gene_id": ["G1", "G1", "G2", "G2"],
        "raw_effect": [0.4, -0.1, 0.3, -0.5],
        "status": "completed",
    }
)
obs = pd.DataFrame(
    {
        "chrom": "1",
        "pos": [100, 200, 300, 400],
        "ref": "A",
        "alt": "G",
        "gene_id": ["G1", "G1", "G2", "G2"],
        "beta": [0.2, 0.1, -0.3, 0.6],
        "effect_allele": ["G", "A", "A", "A"],
    }
)
result = ag2kit.evaluate(pred, obs, n_boot=200)
print(result.metrics[["metric", "estimate", "ci95_low", "ci95_high", "n_clusters"]])
print(result.pairs[["pos", "beta", "beta_alt", "allele_flipped", "pair_status"]])
```

Confidence intervals are `NaN` when fewer than five gene clusters are available.

#### Conditioning

Each AlphaGenome 2 request uses one conditioning input. You can choose one of two formats:

1. **Reference condition:** Set `condition` to an ontology CURIE such as `UBERON:0013756`, or a composite such as `UBERON:0000178_CL:0001054`. Use a scorer such as `RNA_SEQ`.
2. **Single-cell data:** Set `conditioning` to an `AnnData` object containing raw UMI counts. Use Ensembl IDs for `var_names` and include up to 256 cells. Set `condition_label` and use `SC_GENE_EXPRESSION_DENOISED`.

To find valid reference conditions, call `supported_conditions()`. For finer control, build `RequestSpec` objects and use `score` or `score_many`. You can then inspect cached SDK tracks through `AG2Client.raw_scores`.

#### Interpretation and limitations

- `RNA_SEQ` predicts log fold changes, while GTEx reports slopes for normalised expression. Accordingly, `ag2kit` compares the direction and rank of effects.
- GTEx significant-pairs files contain selected eQTL associations, and many variants linked to the same gene are in linkage disequilibrium. As a result, sign metrics may be higher than those measured across all tested pairs. By default, bootstrap intervals resample whole genes to account for within-gene dependence.
- Sign agreement describes performance at the tested loci. It does not establish causality or resolve individual variant effects. In addition, the p-values from `concordance`, `spearman`, and `mcnemar` assume independent pairs, so `evaluate` leaves them out.
- All coordinates are GRCh38 and 1-based; the package performs no liftover. The API does not expose a model revision, so reproducible reruns depend on cached results ([Reproducibility](docs/api.md#reproducibility)). Tests cover the implementation, but the package has not reproduced a published AlphaGenome benchmark. See [Validation](docs/methods.md#validation) and [Methods](docs/methods.md#limitations) for details.
- Finally, AlphaGenome outputs are subject to the SDK's usage restrictions: non-commercial research only, with no use for training other machine-learning models or making clinical decisions. The AlphaGenome terms also govern cached and shared outputs. When using GTEx data, acknowledge the GTEx Project [2].

#### Documentation

| Page                               | Content                                                                                                             |
|------------------------------------|---------------------------------------------------------------------------------------------------------------------|
| [docs/methods.md](docs/methods.md) | Request identity, gene-level reduction, allele orientation, metrics, bootstrap, statistical assumptions, validation |
| [docs/api.md](docs/api.md)         | Public interface, output schemas, environment variables, caches, provenance, reproducibility                        |

#### Development

```bash
uv sync --extra model --extra vcf
uv run pytest
uv run ruff check . && uv run ruff format --check . && uv run mypy
```

By default, tests run offline using a pinned GTEx Portal response. To run the live test, set `ALPHA_GENOME_API_KEY` and execute `uv run pytest -m live`.

#### Licence

`ag2kit` is licensed under MIT ([LICENSE](LICENSE)); the `alphagenome` SDK is licensed separately under Apache 2.0. AlphaGenome's terms of use govern API access and outputs.

#### References

> [!NOTE]
> For GTEx v10 and v11, cite reference 2 and include the Portal release and access date.

1. Avsec, Ž. *et al.* Advancing regulatory variant effect prediction with AlphaGenome. *Nature* **649**, 1206–1218 (2026). doi:[10.1038/s41586-025-10014-0](https://doi.org/10.1038/s41586-025-10014-0)
2. GTEx Consortium. The GTEx Consortium atlas of genetic regulatory effects across human tissues. *Science* **369**, 1318–1330 (2020). doi:[10.1126/science.aaz1776](https://doi.org/10.1126/science.aaz1776)
3. Danecek, P. *et al.* Twelve years of SAMtools and BCFtools. *GigaScience* **10**, giab008 (2021). doi:[10.1093/gigascience/giab008](https://doi.org/10.1093/gigascience/giab008)
4. Google DeepMind. AlphaGenome 2: deciphering how cells read DNA to better understand disease (announcement, 11 November 2026). <https://deepmind.google/blog/alphagenome-2-deciphering-how-cells-read-dna-to-better-understand-disease/>
5. Hanley, J. A. & McNeil, B. J. The meaning and use of the area under a receiver operating characteristic (ROC) curve. *Radiology* **143**, 29–36 (1982). doi:[10.1148/radiology.143.1.7063747](https://doi.org/10.1148/radiology.143.1.7063747)
6. Davison, A. C. & Hinkley, D. V. *Bootstrap Methods and their Application* (Cambridge University Press, 1997). doi:[10.1017/CBO9780511802843](https://doi.org/10.1017/CBO9780511802843)
7. Field, C. A. & Welsh, A. H. Bootstrapping clustered data. *J. R. Stat. Soc. B* **69**, 369–390 (2007). doi:[10.1111/j.1467-9868.2007.00593.x](https://doi.org/10.1111/j.1467-9868.2007.00593.x)
8. McNemar, Q. Note on the sampling error of the difference between correlated proportions or percentages. *Psychometrika* **12**, 153–157 (1947). doi:[10.1007/BF02295996](https://doi.org/10.1007/BF02295996)
9. Mohammadi, P., Castel, S. E., Brown, A. A. & Lappalainen, T. Quantifying the regulatory effect size of cis-acting genetic variation using allelic fold change. *Genome Res.* **27**, 1872–1884 (2017). doi:[10.1101/gr.216747.116](https://doi.org/10.1101/gr.216747.116)
