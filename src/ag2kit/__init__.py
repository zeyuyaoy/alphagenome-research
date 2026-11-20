"""Public API for AlphaGenome variant scoring and evaluation."""

from importlib.metadata import PackageNotFoundError, version

from ag2kit import gtex
from ag2kit.client import (
    AG2Client,
    AG2KitError,
    CredentialsError,
    RequestSpec,
    ScoringError,
    extract_gene_effect,
    interaction,
    reduce_gene_effect,
)
from ag2kit.evaluate import Evaluation, compare, evaluate, evaluate_pairs, join
from ag2kit.io import read_variants
from ag2kit.provenance import Run, read_table, write_table
from ag2kit.stats import (
    Concordance,
    by_group,
    classify,
    concordance,
    no_call_threshold,
    recall_at_sign_accuracy,
    sign_accuracy,
    sign_auroc,
    spearman,
)
from ag2kit.variants import (
    ReferenceBase,
    annotate_folds,
    cluster_loci,
    harmonize_effect,
    locus_clusters,
)

try:
    __version__ = version("ag2kit")
except PackageNotFoundError:
    __version__ = "dev"

__all__ = [
    "AG2Client",
    "AG2KitError",
    "Concordance",
    "CredentialsError",
    "Evaluation",
    "ReferenceBase",
    "RequestSpec",
    "Run",
    "ScoringError",
    "__version__",
    "annotate_folds",
    "by_group",
    "classify",
    "cluster_loci",
    "compare",
    "concordance",
    "evaluate",
    "evaluate_pairs",
    "extract_gene_effect",
    "gtex",
    "harmonize_effect",
    "interaction",
    "join",
    "locus_clusters",
    "no_call_threshold",
    "read_table",
    "read_variants",
    "recall_at_sign_accuracy",
    "reduce_gene_effect",
    "sign_accuracy",
    "sign_auroc",
    "spearman",
    "write_table",
]
