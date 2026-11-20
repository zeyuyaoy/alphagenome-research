### Methods

#### Analysis workflow

`ag2kit` standardizes variant records to GRCh38 using `read_variants`, predicts variant effects with `AG2Client.score_variants`, and reduces the model outputs to one signed effect per variant–gene pair. `join` and `evaluate_pairs` (combined in `evaluate`) then compare these predictions with observed expression quantitative trait locus (eQTL) effects. Each stage retains input records and records their status, allowing users to account for excluded and failed pairs.

#### Prediction requests and caching

An AlphaGenome request predicts the effects of a variant across genes in its sequence window. `ag2kit` stores the complete SDK output for each distinct request and performs gene-level reduction after retrieval. This allows several genes to share a single prediction request.

Each request has a SHA-256 key derived from canonical JSON containing:

- The variant (`chrom:pos:ref>alt`) and its 0-based, half-open input interval. The interval follows `Variant.reference_interval.resize`, is centered on the REF allele, and has a start coordinate of at least zero.
- The model version, organism, ontology condition or cell-condition name, and the digest of conditioning `AnnData` counts, cell names, and gene names.
- The full SDK scorer `repr`, backend identity (SDK or user-supplied model class), installed SDK version, and optional `cache_namespace`.

The key excludes the target gene and labels that do not affect predictions: `variant_id`, `gene_symbol`, `seed`, and `conditioning_set_id`.

Cache entries are Parquet files containing `variant_scorers.tidy_anndata(match_gene_strand=False, include_extended_metadata=True)`. File metadata records the request identity and time. Pages carry CRC checksums that are verified on read. Invalid, corrupted, or unreadable entries are logged and treated as cache misses. Writes are atomic, and failed requests are not cached. Batch scoring fetches each distinct key once and reduces its output to gene records in the worker, so only in-flight outputs are held in memory. Updating the SDK changes the request key. For user-supplied models, the key records the class name; assign distinct `cache_namespace` values to instances with different weights.

#### Gene-level effect estimation

For variant–gene pair $i$, the predicted effect $d_i$ (`raw_effect`) is the arithmetic mean of signed `raw_score` values (ALT − REF) across the selected tracks for that gene. Track selection follows `tidy_anndata(match_gene_strand=True)`: opposite-strand tracks are excluded, and unstranded tracks are retained. Ensembl gene version suffixes are ignored. A missing gene, empty track selection, or non-finite selected value produces a validation error.

AlphaGenome 2 provides condition-specific predictions and imputes some modalities [4]. Track metadata identifies provenance through `imputation_mode`: `observed`, `imputed_supported`, or `imputed_exploratory`. For comparisons with GTEx, users can select `tracks={"imputation_mode": "observed"}` to restrict the reduction to observed tracks. Filtering occurs after prediction, is recorded in `track_filter`, and leaves the cached request unchanged.

In one live Whole Blood request (`UBERON:0013756`, `RNA_SEQ`, `alphagenome` 2.0.0; November 2026), the output contained one observed GTEx RNA-seq track and four `imputed_exploratory` tracks. This observation has not been added to the automated tests.

The interpretation of `raw_effect` depends on the scorer. `RNA_SEQ` (`GeneMaskLFCScorer`) predicts a gene-expression log fold change. `SC_GENE_EXPRESSION_DENOISED` returns a difference in summed predicted expression. Both are model-derived quantities.

#### Variant matching and allele orientation

Chromosome names are normalized to UCSC style everywhere (`1` and `chr1` become `chr1`; `MT` and `chrMT` become `chrM`). Observed records are matched to predictions using chromosome, position, the unordered REF/ALT allele pair, the unversioned gene identifier, and any additional `match` columns. The observed effect $\beta_i$ is oriented to the predicted ALT allele:

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

`evaluate_pairs` calculates percentile 95% confidence intervals from cluster bootstrap resampling [6, 7]. It samples whole clusters with replacement for `n_boot` iterations (default: 1,000). All metrics, all groups, and both prediction sets in `compare` are evaluated on the same resamples, so intervals do not depend on metric order and differences are paired.

The cluster column determines validity. The default, `gene_id`, keeps pairs that share a gene together, but it ignores dependence between genes: one variant tested against several genes, neighbouring genes that share a causal variant, LD that spans genes, and the same variant across tissues. For genome-wide or multi-gene sets, use `locus_clusters`, which links rows that share a gene or lie within `window` of each other and returns the connected components. `cluster_loci` selects one lead per locus; it is not a bootstrap unit, because its greedy rule can split a gene across loci. Neither helper computes LD; clumping output from a reference panel can be supplied as the cluster column instead.

Intervals require at least `min_clusters` clusters (default: 20), and a per-group interval requires that many clusters within the group. In simulations with 40 equally sized clusters, intervals for `compare` differences that excluded zero under the null occurred at 5–6%; with 8 clusters this rose to 8–13%, and resampling pairs instead of clusters gave 12–26%. Resamples with undefined metrics are omitted, and `n_boot_finite` records the number retained. For weighted estimates, the contributing groups are fixed by the original sample.

Three sources of variation should be kept apart:

1. **Sampling uncertainty over loci.** This is what the cluster bootstrap describes. It treats predictions and observed effects as fixed and does not include GTEx estimation error in `beta`.
2. **Model-output variability.** Cell-conditioned predictions depend on the conditioning draw. `ag2kit.stats.monte_carlo_summary` reports the Monte Carlo error of one variant's score across draws; it says nothing about accuracy. Average draws before evaluation, or report the metric across draws.
3. **Model choice.** `ALL_FOLDS` versus `FOLD_1`, observed versus imputed tracks, and different scorers or conditions are different estimators, not noise. Compare them with `compare` on the same observations, not by overlapping intervals.

#### Comparing prediction sets

`compare` estimates metric differences on pairs that are usable in both ledgers, each oriented to its own predicted ALT. This is a fair paired comparison for that subset. If one prediction set fails non-randomly, for example on indels or variants near window edges, the subset differs from either full set, so report `n_usable_a_only`, `n_usable_b_only`, and each set's own `evaluate_pairs` result. When a few clusters drive all disagreement between two highly correlated predictors, percentile intervals can exclude zero although an exact count of discordant clusters would not. DeLong and McNemar tests assume independent pairs and are not used.

#### Held-out regions

`annotate_folds` labels each variant by the training split of the `FOLD_1` checkpoint: `train` if any training region contains it, otherwise `valid` or `test`, and `none` if no region does. The regions are the Borzoi `sequences_human.bed` folds, mapped to splits by the SDK's `fold_intervals` module. That AlphaGenome 2's `FOLD_1` uses this split (validation `fold3`, test `fold4`) is stated in the SDK but not otherwise documented.

The labels are meaningful only for predictions from `FOLD_1`; `ALL_FOLDS` was trained on every region. Test regions form contiguous multi-megabase blocks, so most of a test variant's 1 Mb input window is also held out, but the target gene's TSS or exons can lie in another split, and repetitive or paralogous sequence occurs across folds. The eQTL itself was never a training target: "train" means the reference sequence-to-expression mapping of that locus was fitted. Differences between train and test regions within `FOLD_1` are confounded by region composition; the defensible comparisons are `FOLD_1` performance on test regions, and `compare` between `FOLD_1` and `ALL_FOLDS` on test-region variants.

#### Statistical interpretation

Several features of the data and model affect interpretation:

- GTEx significant pairs contain every variant passing the gene-level threshold, most of which are non-causal LD tags. A tag's `beta` is a marginal association whose sign relative to its ALT allele reflects LD phase with the causal variant. Selecting on significance removes near-zero effects and inflates magnitudes (the winner's curse). The defensible claim is agreement between the predicted ALT effect and the direction of marginal association among variants selected for strong association in a tissue. It is not a claim about causal-variant effects or about all tested variants. Fine-mapped variants (for example SuSiE PIP ≥ 0.9) or one lead per gene and tissue give a cleaner primary analysis; stratification by `|beta|` or p-value is descriptive only.
- Pooled metrics weight genes by their number of significant pairs, which tracks LD-block length and tissue sample size. Size-weighted group averages approximate a pooled within-group metric, not a mean over groups.
- The default gene reduction averages all tracks for the condition, including imputed tracks. In one live Whole Blood request, one track was observed and four were `imputed_exploratory`. For GTEx comparisons use `tracks={"imputation_mode": "observed"}`, or stratify by track mode.
- Users define tissue-to-model matching, for example through `match={"ontology_id": "condition"}`. API support for a condition does not establish that it was represented in model training.
- GTEx normalized-expression slopes and `RNA_SEQ` predicted log fold changes use different scales. The evaluation compares effect signs and ranks. Conversion to allelic fold change [9] is not implemented.
- The exact binomial test in `concordance` and the McNemar test in `ag2kit.stats.mcnemar` [8] assume independent pairs.
- `forward_strand=True` is valid only when the source reports GRCh38 forward-strand alleles with genomic REF, as GTEx and the eQTL Catalogue do. In that mode, palindromic SNVs are accepted and a strand error cannot be detected; spot-check REF with `ReferenceBase`. Records with swapped REF and ALT become `ref_mismatch` and are excluded.
- Indels match only by exact allele strings. Normalize both sides with `bcftools norm -f GRCh38.fa -m-any`; GTEx indels are left-aligned with an anchor base. Differences between GENCODE releases can drop pairs whose gene IDs were retired or remapped.

#### Implementation validation

The offline test suite (`uv run pytest`) checks numerical metrics against hand calculations and independent implementations. It also tests allele orientation, exclusion statuses, retention of observed records, seeded bootstrap reproducibility, and wider cluster-bootstrap intervals than pair-bootstrap intervals on clustered test data. Additional tests cover cache keys, atomic writes, recovery after failed requests with a fake model, VCF headers, FASTA checks, and execution of README Python examples.

A pinned GTEx v10 Portal response tests real-data parsing. Real-data fixtures do not cover the GTEx v8 and v11 paths or the v8 text reader. One optional live test (`uv run pytest -m live`) makes an API request using `ALPHA_GENOME_API_KEY`.

Simulations in `benchmarks/coverage_sim.py` check the coverage of cluster and pair bootstrap intervals on clustered data with a known population AUROC. With 60 genes of 6 pairs each (true sign AUROC 0.719, 200 replicates), 95% intervals covered the truth in 93.0% of replicates (SE 1.8%) when resampling genes and 65.5% (SE 3.4%) when resampling pairs.

These tests assess software behavior. They do not establish biological validity. No published AlphaGenome or Borzoi eQTL result has been reproduced, for these reasons:

- The published AlphaGenome results used AlphaGenome 1. `ag2kit` scores through the AlphaGenome 2 API, which offers only the `ALL_FOLDS` and `FOLD_1` checkpoints.
- The published benchmarks used fine-mapped GTEx v8 variants, including matched non-causal negatives. `ag2kit` loads significant pairs and measures sign and rank agreement among them; it has no fine-mapping loader and no causal-versus-non-causal metric.
- Equivalence of track selection, aggregation, tissue matching, and tissue weighting to the published methods has not been verified.

A user who supplies a fine-mapped set can compute an AlphaGenome 2 analogue of the sign AUROC and Spearman correlation.

#### Scope and limitations

- Scoring supports GRCh38/hg38 and *Homo sapiens*. Conflicting assembly labels in input headers or tables raise errors, and non-GRCh38 rows fail scoring. Coordinates are not lifted over.
- Indels are not left-aligned or normalized. Use `bcftools norm` when normalization is required.
- The API does not expose a model revision or document identical outputs across calls. Cached results support repeatable reruns; see [reproducibility](api.md#reproducibility).
- Precomputed AlphaGenome Atlas scores and Open Targets credible sets are not supported.
- eQTL concordance measures agreement at evaluated loci. It does not establish causality or isolate the effect of an individual variant.

> [!NOTE]
> References [1]–[9] appear in the [README](../README.md#references).
