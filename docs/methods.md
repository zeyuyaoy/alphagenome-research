### Methods

#### Analysis workflow

`ag2kit` standardizes variant records to GRCh38 using `read_variants`, predicts variant effects with `AG2Client.score_variants`, and reduces the model outputs to one signed effect per variant–gene pair. `evaluate` then compares these predictions with observed expression quantitative trait locus (eQTL) effects. Each stage retains input records and records their status, allowing users to account for excluded and failed pairs.

#### Prediction requests and caching

An AlphaGenome request predicts the effects of a variant across genes in its sequence window. `ag2kit` stores the complete SDK output for each distinct request and performs gene-level reduction after retrieval. This allows several genes to share a single prediction request.

Each request has a SHA-256 key derived from canonical JSON containing:

- The variant (`chrom:pos:ref>alt`) and its 0-based, half-open input interval. The interval follows `Variant.reference_interval.resize`, is centered on the REF allele, and has a start coordinate of at least zero.
- The model version, organism, ontology condition or cell-condition name, and the digest of conditioning `AnnData` counts, cell names, and gene names.
- The full SDK scorer `repr`, backend identity (SDK or user-supplied model class), installed SDK version, and optional `cache_namespace`.

The key excludes the target gene and labels that do not affect predictions: `variant_id`, `gene_symbol`, `seed`, and `conditioning_set_id`.

Cache entries are Parquet files containing `variant_scorers.tidy_anndata(match_gene_strand=False, include_extended_metadata=True)`. File metadata records the request identity and time. Invalid or unreadable entries are treated as cache misses. Writes are atomic, and failed requests are not cached. Updating the SDK changes the request key. For user-supplied models, the key records the class name; assign distinct `cache_namespace` values to instances with different weights.

#### Gene-level effect estimation

For variant–gene pair $i$, the predicted effect $d_i$ (`raw_effect`) is the arithmetic mean of signed `raw_score` values (ALT − REF) across the selected tracks for that gene. Track selection follows `tidy_anndata(match_gene_strand=True)`: opposite-strand tracks are excluded, and unstranded tracks are retained. Ensembl gene version suffixes are ignored. A missing gene, empty track selection, or non-finite selected value produces a validation error.

AlphaGenome 2 provides condition-specific predictions and imputes some modalities [4]. Track metadata identifies provenance through `imputation_mode`: `observed`, `imputed_supported`, or `imputed_exploratory`. For comparisons with GTEx, users can select `tracks={"imputation_mode": "observed"}` to restrict the reduction to observed tracks. Filtering occurs after prediction, is recorded in `track_filter`, and leaves the cached request unchanged.

In one live Whole Blood request (`UBERON:0013756`, `RNA_SEQ`, `alphagenome` 2.0.0; November 2026), the output contained one observed GTEx RNA-seq track and four `imputed_exploratory` tracks. This observation has not been added to the automated tests.

The interpretation of `raw_effect` depends on the scorer. `RNA_SEQ` (`GeneMaskLFCScorer`) predicts a gene-expression log fold change. `SC_GENE_EXPRESSION_DENOISED` returns a difference in summed predicted expression. Both are model-derived quantities.

#### Variant matching and allele orientation

Observed records are matched to predictions using chromosome, position, the unordered REF/ALT allele pair, the unversioned gene identifier, and any additional `match` columns. The observed effect $\beta_i$ is oriented to the predicted ALT allele:

$$
\beta_{\mathrm{alt},i} =
\begin{cases}
\beta_i & \text{if the effect allele equals the predicted ALT},\\
-\beta_i & \text{if the effect allele equals the predicted REF}.
\end{cases}
$$

Records with an effect allele matching neither allele are excluded. With `forward_strand=True` (the default), the procedure accepts forward-strand palindromic single-nucleotide variants and indels, as reported by sources such as GTEx. With `forward_strand=False`, these variants are excluded because allele labels alone cannot establish strand orientation.

When `forward_strand=True`, a difference between observed and predicted REF alleles produces `ref_mismatch`, including when the observed REF equals the predicted ALT. The procedure does not reverse the predicted substitution because its effect need not equal the negative of the forward substitution. Identical duplicate predictions are collapsed; conflicting duplicates receive `ambiguous_prediction`.

#### Evaluation metrics

Let $d_i$ denote the predicted effect and $b_i=\beta_{\mathrm{alt},i}$ the ALT-oriented observed effect for pair $i$. Pairs with non-finite effects or $b_i=0$ are excluded from metric calculations. `evaluate` reports three metrics:

- **Sign AUROC:** The area under the receiver operating characteristic curve for distinguishing $b_i>0$ from $b_i<0$ using $d_i$. It is calculated as $U/(n_+n_-)$, where $U$ is the Mann–Whitney statistic [5]. The metric is undefined if either observed sign class is absent.
- **Signed Spearman correlation ($\rho$):** The rank correlation between $d_i$ and $b_i$, defined for at least four pairs. The separate helper `ag2kit.spearman(..., absolute=True)` instead compares effect magnitudes.
- **Sign accuracy:** The fraction of pairs for which $\operatorname{sign}(d_i)=\operatorname{sign}(b_i)$. A zero prediction counts as incorrect.

For grouped analyses, such as analyses by tissue, each metric is calculated within group $g$. Defined estimates from groups with $n_g\geq$ `min_group` (default: 2) are combined using pair-count weights:

$$
M=\frac{\sum_g n_g m_g}{\sum_g n_g}.
$$

`n_pairs_used` counts pairs in contributing groups. This aggregation was intended to resemble tissue-weighted AlphaGenome evaluations [1], but equivalence to their published methods or code has not been verified. Comparisons with published benchmarks also require matching variant selection, tissue definitions, and score construction.

#### Uncertainty estimation

`evaluate` calculates percentile 95% confidence intervals from cluster bootstrap resampling [6, 7]. It samples whole clusters with replacement for `n_boot` iterations (default: 1,000). The default cluster is `gene_id`, reflecting dependence among eQTL pairs that share a gene and may be in linkage disequilibrium. Users can supply another clustering column. The `cluster_loci` helper groups records by genomic distance and gene; it does not calculate linkage disequilibrium.

Intervals require at least `min_clusters` clusters (default: 5). Resamples with undefined metrics are omitted, and `n_boot_finite` records the number retained. An interval is `NaN` when it cannot be estimated. The implementation uses one seeded generator (`seed=0` by default) across metrics in sequence. Each metric therefore receives different resamples, and its interval depends on the metric evaluation order. Cluster counts include all scored pairs, including pairs in groups excluded by `min_group`.

Cluster bootstrap inference assumes that clusters are independent and exchangeable. Related genes or shared loci across tissues may violate this assumption. The intervals describe variation from resampling the supplied clusters; they do not represent GTEx sampling uncertainty or model-prediction uncertainty. `evaluate` does not report pair-independence-based p-values.

#### Statistical interpretation

Several features of the data and model affect interpretation:

- GTEx significant-pairs datasets exclude non-significant tests and underrepresent small effects. Sign-based performance on these datasets may be higher than performance across all tested pairs.
- Users define tissue-to-model matching, for example through `match={"ontology_id": "condition"}`. API support for a condition does not establish that it was represented in model training.
- GTEx normalized-expression slopes and `RNA_SEQ` predicted log fold changes use different scales. The evaluation compares effect signs and ranks. Conversion to allelic fold change [9] is not implemented.
- The exact binomial test in `concordance` and the McNemar test in `ag2kit.stats.mcnemar` [8] assume independent pairs. A lead pair per cluster, selected using a suitable procedure such as `cluster_loci`, can reduce dependence but does not establish independence.

#### Implementation validation

The offline test suite (`uv run pytest`) checks numerical metrics against hand calculations and independent implementations. It also tests allele orientation, exclusion statuses, retention of observed records, seeded bootstrap reproducibility, and wider cluster-bootstrap intervals than pair-bootstrap intervals on clustered test data. Additional tests cover cache keys, atomic writes, recovery after failed requests with a fake model, VCF headers, FASTA checks, and execution of README Python examples.

A pinned GTEx v10 Portal response tests real-data parsing. Real-data fixtures do not cover the GTEx v8 and v11 paths or the v8 text reader. One optional live test (`uv run pytest -m live`) makes an API request using `ALPHA_GENOME_API_KEY`.

These tests assess software behavior. They do not establish biological validity. No published AlphaGenome benchmark result has been reproduced using `ag2kit`.

#### Scope and limitations

- Scoring supports GRCh38/hg38 and *Homo sapiens*. Conflicting assembly labels in input headers or tables raise errors, and non-GRCh38 rows fail scoring. Coordinates are not lifted over.
- Indels are not left-aligned or normalized. Use `bcftools norm` when normalization is required.
- The API does not expose a model revision or document identical outputs across calls. Cached results support repeatable reruns; see [reproducibility](api.md#reproducibility).
- Precomputed AlphaGenome Atlas scores and Open Targets credible sets are not supported.
- eQTL concordance measures agreement at evaluated loci. It does not establish causality or isolate the effect of an individual variant.

> [!NOTE]
> References [1]–[9] appear in the [README](../README.md#references).
