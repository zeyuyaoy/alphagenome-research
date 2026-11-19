### API reference

#### Variant input

```python
ag2kit.read_variants(source, *, assembly="GRCh38", fasta=None) -> pandas.DataFrame
```

`read_variants` accepts a pandas DataFrame or one of these file formats:

- VCF/BCF: `.vcf`, `.vcf.gz`, `.vcf.bgz`, `.bcf` (requires `pysam`)
- Delimited text: `.csv`, `.tsv`, `.txt` (optionally gzip-compressed)
- Parquet: `.parquet`

The result contains one row per input record and ALT allele, in source order. Additional columns from input tables are preserved:

| Column                       | Description                                                                    |
|------------------------------|--------------------------------------------------------------------------------|
| `variant_id`                 | `chr<chrom>:<pos>:<REF>><ALT>`                                                 |
| `chrom`, `pos`, `ref`, `alt` | `chr`-prefixed chromosome, 1-based position, and uppercase alleles             |
| `source_id`                  | Input identifier; `""` if missing                                              |
| `assembly`                   | Value passed to `assembly`                                                     |
| `record_index`, `alt_index`  | 0-based record index and 1-based ALT index; split alleles share a record index |
| `input_status`               | `ok` or the first applicable validation status                                 |

Possible validation statuses are `missing_allele`, `symbolic_allele`, `invalid_allele`, and `ref_equals_alt`. Providing `fasta` also enables `contig_not_in_reference`, `position_out_of_range`, and `ref_mismatch` checks.

A mismatch between the `assembly` argument and the assembly declared in a VCF header or input table raises `ValueError`. The argument records the assembly label; it does not verify the coordinates. Scoring accepts only `GRCh38` or `hg38`. 

> [!NOTE]
> The package does not perform liftover or indel normalization.

#### Scoring

```python
client = ag2kit.AG2Client(
    dry_run=True,
    cache_dir=...,
    model_version="ALL_FOLDS",
    max_retries=2,
    backoff=30.0,
    cache_namespace="",
    model=None,
)
```

`AG2Client` manages scoring requests and their cache. By default, `dry_run=True` plans requests without calling the API.

| Method                                                                                                                                                                    | Description                                                                                                                                                              |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `score_variants(variants, *, condition="", scorer="RNA_SEQ", gene_column=None, conditioning=None, condition_label="", sequence_length=2**20, max_workers=4, tracks=None)` | Score a table of variants. Returns one row per input row and gene, or one row per input row when `gene_column` identifies a gene. Input columns use the `input_` prefix. |
| `score(spec, conditioning=None, strict=False, tracks=None)`                                                                                                               | Score one `RequestSpec`. With `strict=True`, raise `ScoringError` for `failed` or `blocked` results.                                                                     |
| `score_many(specs, conditioning=None, max_workers=4, tracks=None)`                                                                                                        | Score requests concurrently, deduplicate identical requests, and preserve input order.                                                                                   |
| `plan(spec, conditioning=None)`                                                                                                                                           | Return the interval, cache key, and cache state without calling the API.                                                                                                 |
| `raw_scores(spec, conditioning=None)`                                                                                                                                     | Return the cached SDK table for all genes and tracks, or `None`.                                                                                                         |
| `cached(spec)`                                                                                                                                                            | Return a cached record, or `None`.                                                                                                                                       |
| `connect()`                                                                                                                                                               | Open the API connection. Connections also open automatically when needed.                                                                                                |
| `request_identity(spec)`                                                                                                                                                  | Return the fields that identify a request.                                                                                                                               |
| `cache_key(spec)`                                                                                                                                                         | Return the request's SHA-256 cache key.                                                                                                                                  |

##### Request validation

`RequestSpec` is an immutable dataclass. It validates the following fields:

| Field             | Requirement                               |
|-------------------|-------------------------------------------|
| Position          | 1-based integer                           |
| Alleles           | Match `[ACGTN]+`; REF and ALT must differ |
| `sequence_length` | `2**14`, `2**17`, `2**19`, or `2**20`     |
| `n_cells`         | 0–256                                     |
| Species           | `HOMO_SAPIENS`                            |
| Conditioning      | Exactly one conditioning mode per request |

Use `condition` for an ontology CURIE such as `UBERON:0013756` or an underscore-joined composite. For cell conditioning, set `n_cells > 0` and provide `condition_label`.

Reference conditions use scorers such as `RNA_SEQ`; the default scorer for cell conditioning is `SC_GENE_EXPRESSION_DENOISED`.

##### Scoring results

Each scoring record includes the request fields and these output fields:

| Field                                                                  | Description                                                                                                  |
|------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------|
| `raw_effect`                                                           | Mean signed ALT − REF `raw_score` across selected gene tracks; `NaN` if no gene is selected or scoring fails |
| `n_reduced_values`                                                     | Number of tracks included in the mean                                                                        |
| `gene_strand`                                                          | Gene strand                                                                                                  |
| `track_names`                                                          | First four distinct track names                                                                              |
| `track_filter`                                                         | JSON representation of `tracks`; empty when no filter is set                                                 |
| `status`                                                               | Request status (see below)                                                                                   |
| `error`, `error_type`                                                  | Error message and category                                                                                   |
| `attempts`, `from_cache`, `requested_utc`                              | Retry count, cache indicator, and UTC timestamp of the originating API call                                  |
| `cache_key`, `scorer_definition`, `sdk_version`, `conditioning_sha256` | Request identity and provenance                                                                              |

The full status set is `completed`, `completed_with_warning`, `not_estimable`, `blocked`, `failed`, and `not_run` (`ag2kit.provenance.STATUSES`). `score_variants` returns `completed`, `failed`, `blocked`, or `not_run`.

Possible `error_type` values are `quota`, `transient`, `network`, `validation`, `credentials`, `upstream`, and `blocked`.

##### Scoring helpers

| Function                                         | Description                                                                              |
|--------------------------------------------------|------------------------------------------------------------------------------------------|
| `reduce_gene_effect(tidy, gene_id, tracks=None)` | Reduce a tidy scorer table to a gene-level effect.                                       |
| `extract_gene_effect(adata, gene_id)`            | Extract a gene-level effect from one SDK scorer output.                                  |
| `interaction(score_treated, score_control)`      | Subtract the control effect from the treated effect; returns `NaN` if either is missing. |

#### GTEx observed effects

```python
ag2kit.gtex.eqtls(genes, tissue, dataset="gtex_v10", refresh=False)
ag2kit.gtex.read_signif_pairs(path, tissue, dataset)
ag2kit.gtex.tissues(dataset="gtex_v10")
ag2kit.gtex.gencode_id(gene, dataset="gtex_v10")
```

Supported datasets are `gtex_v8` (GENCODE v26), `gtex_v10` (v39), and `gtex_v11` (v47). All use GRCh38.

| Function            | Description                                                                                                                          |
|---------------------|--------------------------------------------------------------------------------------------------------------------------------------|
| `eqtls`             | Retrieve significant eQTL pairs from GTEx Portal API v2. `beta` is the ALT-allele normalized expression slope (`nes`); `se` is `NA`. |
| `read_signif_pairs` | Load a downloaded `*.signif_pairs.parquet` or `*.signif_variant_gene_pairs.txt.gz` file into the same schema, including `se`.        |
| `tissues`           | List tissue IDs, UBERON IDs, and eGene counts.                                                                                       |
| `gencode_id`        | Resolve a gene to its release-specific GENCODE identifier.                                                                           |

`eqtls` resolves gene symbols and unversioned Ensembl IDs to the selected GENCODE release. Versioned Ensembl IDs (containing `.`) must match that release. `tissues` and `gencode_id` also accept `refresh`. GTEx responses are cached at `$AG2KIT_CACHE/gtex`.

#### Evaluation

```python
ag2kit.evaluate(
    predictions,
    observed,
    *,
    match=None,
    group=None,
    cluster="gene_id",
    effect="raw_effect",
    forward_strand=True,
    n_boot=1000,
    seed=0,
    min_group=2,
    min_clusters=5,
) -> Evaluation
```

`evaluate` matches predicted effects to observed effects and calculates agreement metrics. Both inputs require `chrom`, `pos`, `ref`, `alt`, and `gene_id`. Additional required columns are:

| Input         | Required columns                                                  |
|---------------|-------------------------------------------------------------------|
| `predictions` | `status` and the column named by `effect` (default: `raw_effect`) |
| `observed`    | `beta`, `effect_allele`                                           |

Use `match` to require agreement on additional columns. For example, `match={"ontology_id": "condition"}` matches the observed `ontology_id` to the predicted `condition`.

##### Evaluation output

| Attribute or method | Description                                                                                                                                           |
|---------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|
| `pairs`             | One row per observed record, including `predicted`, `beta_alt`, `allele_flipped`, `pair_status`, and matched prediction fields prefixed with `pred_`. |
| `metrics`           | `sign_auroc`, `spearman_rho`, and `sign_accuracy`, with estimates and uncertainty fields.                                                             |
| `per_group`         | Group-level results: `group`, `n`, `value`, `metric`.                                                                                                 |
| `settings`          | Evaluation configuration.                                                                                                                             |
| `summary()`         | Print metrics and counts by `pair_status`.                                                                                                            |
| `save(directory)`   | Save `pairs.parquet`, `metrics.tsv`, `per_group.tsv`, and `evaluation.json`.                                                                          |

The `metrics` table includes `estimate`, `ci95_low`, `ci95_high`, `n_pairs`, `n_pairs_used`, `n_clusters`, `n_boot_finite`, and `weighting`. `evaluation.json` includes the settings, software versions, and write timestamp.

`pair_status` can be `ok`, `no_prediction`, `prediction_failed`, `ambiguous_prediction`, `ref_mismatch`, `missing_statistics`, `ref_equals_alt`, `not_snv`, `palindromic`, or `effect_allele_not_in_ref_alt`. The `not_snv` and `palindromic` statuses apply only when `forward_strand=False`.

#### Statistics and utilities

| Function or type                                                            | Description                                                                                                    |
|-----------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------|
| `sign_auroc`, `spearman`, `sign_accuracy`                                   | Evaluation metrics defined in [Methods](methods.md#evaluation-metrics).                                        |
| `by_group(metric, deltas, betas, groups, min_size=2)`                       | Pair-count-weighted mean of a metric across groups.                                                            |
| `recall_at_sign_accuracy(deltas, betas, accuracy=0.9)`                      | Fraction of pairs in the largest top-*k* set by absolute predicted effect that meets the target sign accuracy. |
| `concordance`, `classify`, `no_call_threshold`                              | Sign concordance and exact binomial testing; classification and null-derived no-call threshold.                |
| `ag2kit.stats.mcnemar`                                                      | Exact McNemar test for discordant pairs.                                                                       |
| `ag2kit.stats.monte_carlo_summary`                                          | Summary across conditioning draws.                                                                             |
| `harmonize_effect(estimate, effect_allele, ref, alt, forward_strand=False)` | Orient an observed effect to ALT. The default is `False`; `evaluate` defaults to `True`.                       |
| `cluster_loci(df, window=1_000_000)`                                        | Greedy clustering by distance and gene, with one lead row per locus. This is not LD-based clustering.          |
| `AG2KitError`, `CredentialsError`, `ScoringError`                           | Exceptions. `ScoringError` provides `error_type` and `record`.                                                 |
| `Concordance`                                                               | Return type of `concordance`.                                                                                  |
| `ReferenceBase`                                                             | Cached GRCh38 reference-base lookup through Ensembl REST.                                                      |

#### Run provenance

`Run(stage, root)` creates `<root>/results/<stage>/<run_id>/`. Set `AG2KIT_RESULTS_ROOT` to override `<root>/results`. The run ID follows `<UTC timestamp>-<git short SHA>[-dirty]`.

`Run.status` and `Run.check` collect entries in memory. `Run.table` and `Run.finish` write `status.tsv`, `verification.tsv`, and `run_manifest.tsv` using the `write_table` and `read_table` utilities.

The manifest records the Git state and versions of Python, `ag2kit`, `alphagenome`, NumPy, pandas, SciPy, and AnnData.

#### Configuration

| Environment variable   | Description                                                            |
|------------------------|------------------------------------------------------------------------|
| `ALPHA_GENOME_API_KEY` | Required for live scoring. Error records redact the key.               |
| `AG2KIT_CACHE`         | Cache root. Defaults to `$XDG_CACHE_HOME/ag2kit` or `~/.cache/ag2kit`. |
| `AG2KIT_AG2_CACHE`     | Prediction cache directory. Defaults to `<cache root>/ag2`.            |
| `AG2KIT_RESULTS_ROOT`  | Overrides the results directory used by `Run`.                         |

#### Reproducibility

Requests with the same identity reuse cached predictions. Each cache entry stores its `requested_utc` timestamp. See [Prediction requests and caching](methods.md#prediction-requests-and-caching).

Cell-conditioned requests use separate, ordered lists of cell and gene names in the cache fingerprint. Entries created with the earlier fingerprint require rescoring.

The AlphaGenome API does not expose a model revision or guarantee identical results across calls. Results may change after cache deletion or a server-side model update. For reproducible analyses:

1. Archive the supporting `predictions.parquet`, `evaluation.json`, and cache entries.
2. Record the `ag2kit` and `alphagenome` versions. `Evaluation.save()` includes them in its output.
3. Use a new `cache_namespace` after an announced server-side model change.

> [!IMPORTANT]
> AlphaGenome's terms of use also apply to cached outputs. Review the terms before sharing cache files.
