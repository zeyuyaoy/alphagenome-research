"""AlphaGenome scoring with request-level caching and gene-level reduction."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import logging
import os
import random
import re
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ag2kit.provenance import cache_dir, stable_hash, utc_now
from ag2kit.variants import ALLELE, normalize_chrom

log = logging.getLogger("ag2kit")

API_KEY_ENV = "ALPHA_GENOME_API_KEY"
DEFAULT_SEQUENCE_LENGTH = 2**20
SUPPORTED_SEQUENCE_LENGTHS = frozenset({2**14, 2**17, 2**19, 2**20})
MAX_CONDITIONING_CELLS = 256
SOURCE = "alphagenome"
_CURIES = re.compile(r"^[A-Za-z]+:[A-Za-z0-9.]+(?:_[A-Za-z]+:[A-Za-z0-9.]+)*$")
_CACHE_FORMAT = 2
MAX_BACKOFF = 600.0
PROGRESS_EVERY = 100
_HASH_ROWS = 64
_T = TypeVar("_T")

_ERROR_TYPES = {
    "RESOURCE_EXHAUSTED": "quota",
    "UNAVAILABLE": "transient",
    "DEADLINE_EXCEEDED": "transient",
    "ABORTED": "transient",
    "INVALID_ARGUMENT": "validation",
    "FAILED_PRECONDITION": "validation",
    "OUT_OF_RANGE": "validation",
    "NOT_FOUND": "validation",
    "UNIMPLEMENTED": "validation",
    "UNAUTHENTICATED": "credentials",
    "PERMISSION_DENIED": "credentials",
}
_RETRYABLE = frozenset({"quota", "transient", "network"})

Tracks = Mapping[str, str | Sequence[str]]


class AG2KitError(Exception):
    """Base class for ag2kit errors."""


class CredentialsError(AG2KitError):
    """The API key is missing or was rejected."""


class ScoringError(AG2KitError):
    """Reports a failed or blocked scoring request in strict mode.

    Attributes:
        error_type: Category of the failure.
        record: Scoring record that would otherwise have been returned.
    """

    def __init__(self, message: str, error_type: str, record: dict):
        """Initializes the exception with its scoring context.

        Args:
            message: Human-readable error message.
            error_type: Failure category.
            record: Failed or blocked scoring record.
        """
        super().__init__(message)
        self.error_type = error_type
        self.record = record


@dataclass(frozen=True)
class RequestSpec:
    """Specifies one GRCh38 variant under one reference or cell condition.

    Use exactly one condition input. Reference conditions use condition,
    typically with RNA_SEQ. Cell conditioning uses a positive n_cells and
    a condition_label. Pass the actual AnnData to the scoring method.
    Construction raises ValueError for invalid coordinates, alleles, sequence
    length, cell count, organism, or condition settings.

    Attributes:
        variant_id: Caller-provided variant label.
        chrom: Chromosome name, normalized to include the chr prefix.
        pos: 1-based position of the first REF base.
        ref: Forward-strand reference allele, stored in uppercase.
        alt: Forward-strand alternate allele, stored in uppercase.
        gene_id: Gene to reduce to a single effect. Empty skips gene reduction.
        gene_symbol: Caller-provided gene label.
        condition_label: Cell-condition name used by the SDK to name output tracks.
        conditioning_set_id: Caller-provided label for the conditioning set.
        n_cells: Number of conditioning cells, or zero for reference conditions.
        seed: Caller-provided seed label for the conditioning draw.
        model_version: AlphaGenome model version.
        scorer_name: SDK recommended scorer name.
        sequence_length: Supported input window length in bases.
        organism: SDK organism name. Only HOMO_SAPIENS is supported.
        condition: Reference condition as one or more ontology CURIEs.
        cell_conditioned: Whether the spec uses cell conditioning.
        variant_string: Variant in SDK-compatible chrom:pos:ref>alt form.
    """

    variant_id: str
    chrom: str
    pos: int
    ref: str
    alt: str
    gene_id: str = ""
    gene_symbol: str = ""
    condition_label: str = ""
    conditioning_set_id: str = ""
    n_cells: int = 0
    seed: int = 0
    model_version: str = "ALL_FOLDS"
    scorer_name: str = "SC_GENE_EXPRESSION_DENOISED"
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH
    organism: str = "HOMO_SAPIENS"
    condition: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "chrom", normalize_chrom(self.chrom))
        object.__setattr__(self, "ref", str(self.ref).upper())
        object.__setattr__(self, "alt", str(self.alt).upper())
        if isinstance(self.pos, bool) or int(self.pos) != self.pos or self.pos < 1:
            raise ValueError(f"pos must be a 1-based integer position, got {self.pos!r}")
        object.__setattr__(self, "pos", int(self.pos))
        for name in ("ref", "alt"):
            if not ALLELE.match(getattr(self, name)):
                raise ValueError(f"{name} must be a non-empty A/C/G/T/N string")
        if self.ref == self.alt:
            raise ValueError("ref and alt are identical")
        if self.sequence_length not in SUPPORTED_SEQUENCE_LENGTHS:
            supported = sorted(SUPPORTED_SEQUENCE_LENGTHS)
            raise ValueError(f"sequence_length {self.sequence_length} not in {supported}")
        if not 0 <= self.n_cells <= MAX_CONDITIONING_CELLS:
            raise ValueError(f"n_cells must be between 0 and {MAX_CONDITIONING_CELLS}")
        if self.organism != "HOMO_SAPIENS":
            raise ValueError("only HOMO_SAPIENS is supported")
        if self.condition and not _CURIES.match(self.condition):
            raise ValueError(
                f"condition must be ontology CURIE(s) like UBERON:0013756, got {self.condition!r}"
            )
        if bool(self.condition) == self.cell_conditioned:
            raise ValueError(
                "set exactly one of condition (ontology CURIE) or n_cells > 0 (cell conditioning)"
            )
        if self.cell_conditioned and not self.condition_label:
            raise ValueError("cell conditioning needs a condition_label")

    @property
    def cell_conditioned(self) -> bool:
        """Whether the request uses cell conditioning."""
        return self.n_cells > 0

    @property
    def variant_string(self) -> str:
        """Variant string accepted by the SDK variant parser."""
        return f"{self.chrom}:{self.pos}:{self.ref}>{self.alt}"

    def interval(self) -> tuple[str, int, int]:
        """Builds the input window around the reference allele span.

        Near the chromosome start, the window shifts right to start at zero.
        The API rejects windows that extend beyond the chromosome end.

        Returns:
            Chromosome, zero-based inclusive start, and exclusive end.
        """
        centre = self.pos - 1 + (len(self.ref) + 1) // 2
        start = max(0, centre - self.sequence_length // 2)
        return (self.chrom, start, start + self.sequence_length)


def sdk_version() -> str:
    """Returns the installed AlphaGenome SDK version or "not_installed"."""
    try:
        return importlib.metadata.version("alphagenome")
    except importlib.metadata.PackageNotFoundError:
        return "not_installed"


def api_key_available() -> bool:
    """Checks whether the API key environment variable contains a nonblank value."""
    return bool(os.environ.get(API_KEY_ENV, "").strip())


def sdk_available() -> bool:
    """Checks whether Python can locate the AlphaGenome SDK."""
    return importlib.util.find_spec("alphagenome") is not None


def classify_error(exc: BaseException) -> str:
    """Categorizes scoring errors for records and retry decisions.

    Args:
        exc: Exception from the SDK or a custom model.

    Returns:
        An error category, using the RPC status when available and
        "upstream" for unrecognized failures.
    """
    code = getattr(exc, "code", None)
    if callable(code):
        try:
            name = getattr(code(), "name", "")
        except Exception:
            name = ""
        if name:
            return _ERROR_TYPES.get(name, "upstream")
    if isinstance(exc, PermissionError):
        return "credentials"
    if isinstance(exc, ValueError | KeyError | IndexError | NotImplementedError):
        return "validation"
    if isinstance(exc, ConnectionError | TimeoutError):
        return "network"
    text = str(exc).upper()
    return next((kind for name, kind in _ERROR_TYPES.items() if name in text), "upstream")


def _redact(text: str) -> str:
    key = os.environ.get(API_KEY_ENV, "").strip()
    return text.replace(key, "<redacted>") if key else text


@dataclass
class _Outcome:
    """Stores a request result with status and cache metadata."""

    tidy: pd.DataFrame | None
    status: str
    error: str = ""
    error_type: str = ""
    attempts: int = 0
    from_cache: bool = False
    requested_utc: str = ""
    _rows: dict[Any, Any] | None = field(default=None, repr=False)

    def reduce(self, gene_id: str, tracks: Tracks | None) -> tuple[float, dict]:
        """Applies reduce_gene_effect to the rows of one gene, indexing genes once."""
        assert self.tidy is not None
        if self._rows is None:
            genes = self.tidy["gene_id"].astype(str).str.split(".").str[0]
            self._rows = dict(genes.groupby(genes, sort=False).indices)
        rows = self._rows.get(str(gene_id).split(".")[0])
        if rows is None:
            raise ValueError(f"target gene {gene_id} absent from scorer output")
        return reduce_gene_effect(self.tidy.iloc[rows], gene_id, tracks)


@dataclass(frozen=True)
class _Request:
    """Holds a spec with its conditioning, identity, and validation result."""

    spec: RequestSpec
    conditioning: Any
    fingerprint: str
    key: str
    problem: str


@dataclass
class AG2Client:
    """Scores request specs and caches full scorer outputs on disk.

    One SDK connection is shared across worker threads. The client retries
    quota, transient, and network failures with jittered exponential backoff,
    in addition to the SDK's own retries.
    The API does not report a model revision, so use a new namespace after a
    server-side change. Custom models are identified by class, so give
    differently trained instances of the same class distinct namespaces.

    Attributes:
        dry_run: Whether to validate and plan requests without live scoring.
        cache_dir: Directory for cached scores, overridden by AG2KIT_AG2_CACHE.
        model_version: SDK model version required by requests on this client.
        max_retries: Maximum attempts per request, including the initial attempt.
            Construction raises ValueError for values below one.
        backoff: Base retry delay in seconds. Attempt k waits a jittered
            delay between half and all of backoff * 2**(k - 1), capped at
            MAX_BACKOFF.
        sdk_version: SDK version recorded in request identity and provenance.
        cache_namespace: Label for isolating caches across model revisions.
        model: Optional backend implementing the SDK score_variant interface.
            Live API connections are assigned here on first use.
    """

    dry_run: bool = True
    cache_dir: Path = field(default_factory=lambda: cache_dir() / "ag2")
    model_version: str = "ALL_FOLDS"
    max_retries: int = 4
    backoff: float = 30.0
    sdk_version: str = field(default_factory=sdk_version)
    cache_namespace: str = ""
    model: Any = None

    def __post_init__(self) -> None:
        self.cache_dir = Path(os.environ.get("AG2KIT_AG2_CACHE") or self.cache_dir)
        if self.max_retries < 1:
            raise ValueError("max_retries must be at least 1")
        self._custom_model = self.model is not None
        self._connect_lock = threading.Lock()
        self._scorers: dict[tuple[str, bool], tuple[str, str]] = {}

    def connect(self) -> None:
        """Initializes the live client if needed.

        Called lazily when scoring. Dry runs and clients with an existing backend
        skip initialization.

        Raises:
            CredentialsError: If live connection is needed and the key is missing.
            ImportError: If the AlphaGenome SDK is unavailable.
            AttributeError: If model_version is unknown to the SDK.
        """
        with self._connect_lock:
            if self.model is not None or self.dry_run:
                return
            if not api_key_available():
                raise CredentialsError(f"{API_KEY_ENV} is not set; live scoring is unavailable")
            from alphagenome.models.v2 import dna_client

            version = getattr(dna_client.ModelVersion, self.model_version)
            self.model = dna_client.create(os.environ[API_KEY_ENV].strip(), model_version=version)

    def recommended_scorer(self, spec: RequestSpec) -> Any:
        """Resolves the SDK scorer and checks compatibility with the request.

        Args:
            spec: Request with the scorer name and conditioning mode.

        Returns:
            The SDK scorer object.

        Raises:
            ImportError: If the AlphaGenome SDK is unavailable.
            KeyError: If the scorer name is unknown.
            ValueError: If the scorer is unsigned or requires cell conditioning
                and the request uses a reference condition.
        """
        from alphagenome.models.v2 import variant_scorers

        scorers = variant_scorers.RECOMMENDED_VARIANT_SCORERS
        if spec.scorer_name not in scorers:
            raise KeyError(
                f"{spec.scorer_name!r} not in RECOMMENDED_VARIANT_SCORERS: {sorted(scorers)}"
            )
        scorer = scorers[spec.scorer_name]
        if not getattr(scorer, "is_signed", True):
            raise ValueError(f"scorer {spec.scorer_name!r} is unsigned; ag2kit needs ALT-REF")
        cell_only: frozenset = variant_scorers.CELL_CONDITIONING_ONLY_OUTPUT_TYPES
        if not spec.cell_conditioned and getattr(scorer, "requested_output", None) in cell_only:
            raise ValueError(f"scorer {spec.scorer_name!r} needs cell conditioning")
        return scorer

    def _scorer_info(self, spec: RequestSpec) -> tuple[str, str]:
        key = (spec.scorer_name, spec.cell_conditioned)
        if key not in self._scorers:
            if not sdk_available():
                info = (spec.scorer_name, "")
            else:
                try:
                    info = (repr(self.recommended_scorer(spec)), "")
                except (KeyError, ValueError) as exc:
                    info = (spec.scorer_name, str(exc).strip("'\""))
            self._scorers[key] = info
        return self._scorers[key]

    def scorer_definition(self, spec: RequestSpec) -> str:
        """Describes the scorer used for request identity.

        Args:
            spec: Request whose scorer to describe.

        Returns:
            The SDK scorer representation, or its name if the SDK or a
            compatible scorer is unavailable.
        """
        return self._scorer_info(spec)[0]

    def _backend(self) -> str:
        if not self._custom_model:
            return SOURCE
        cls = type(self.model)
        return f"custom:{cls.__module__}.{cls.__qualname__}"

    def _identity(self, spec: RequestSpec, fingerprint: str) -> dict:
        chrom, start, end = spec.interval()
        return {
            "cache_format": _CACHE_FORMAT,
            "variant": spec.variant_string,
            "interval": f"{chrom}:{start}-{end}",
            "model_version": spec.model_version,
            "organism": spec.organism,
            "condition": spec.condition,
            "cell_condition_name": spec.condition_label if spec.cell_conditioned else "",
            "conditioning_sha256": fingerprint,
            "scorer_definition": self.scorer_definition(spec),
            "backend": self._backend(),
            "sdk_version": self.sdk_version,
            "cache_namespace": self.cache_namespace,
        }

    def request_identity(self, spec: RequestSpec, conditioning: Any = None) -> dict:
        """Builds the identity shared by requests for the same full scorer output.

        Gene selection and provenance labels are excluded. The cell-condition
        name participates because it affects output track names.

        Args:
            spec: Request to identify.
            conditioning: Cell-conditioning AnnData, or None for reference conditions.

        Returns:
            A dictionary of request inputs and cache provenance.
        """
        return self._identity(spec, conditioning_fingerprint(conditioning))

    def cache_key(self, spec: RequestSpec, conditioning: Any = None) -> str:
        """Hashes the request identity for cache lookup.

        Args:
            spec: Request to identify.
            conditioning: Cell-conditioning AnnData, or None for reference conditions.

        Returns:
            SHA-256 digest of request_identity.
        """
        return stable_hash(self.request_identity(spec, conditioning))

    def _request(
        self, spec: RequestSpec, conditioning: Any, fingerprints: dict[int, str] | None = None
    ) -> _Request:
        if fingerprints is None:
            fingerprint = conditioning_fingerprint(conditioning)
        else:
            if id(conditioning) not in fingerprints:
                fingerprints[id(conditioning)] = conditioning_fingerprint(conditioning)
            fingerprint = fingerprints[id(conditioning)]
        key = stable_hash(self._identity(spec, fingerprint))
        return _Request(spec, conditioning, fingerprint, key, self._check(spec, conditioning))

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.parquet"

    def _read_cache(self, key: str) -> tuple[pd.DataFrame, dict] | None:
        path = self._cache_path(key)
        try:
            table = pq.read_table(path, page_checksum_verification=True)
            meta = json.loads((table.schema.metadata or {})[b"ag2kit"])
        except FileNotFoundError:
            return None
        except (OSError, KeyError, ValueError, pa.ArrowException) as exc:
            log.warning("ignoring unreadable cache entry %s: %s", path.name, exc)
            return None
        if meta.get("cache_key") != key or table.num_rows == 0:
            log.warning("ignoring cache entry %s with mismatched identity", path.name)
            return None
        return table.to_pandas(), meta

    def _write_cache(self, key: str, tidy: pd.DataFrame, meta: dict) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self._cache_path(key)
        table = pa.Table.from_pandas(tidy, preserve_index=False)
        table = table.replace_schema_metadata(
            {**(table.schema.metadata or {}), b"ag2kit": json.dumps(meta, sort_keys=True)}
        )
        tmp = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
        pq.write_table(table, tmp, write_page_checksum=True)
        tmp.replace(path)

    def raw_scores(self, spec: RequestSpec, conditioning: Any = None) -> pd.DataFrame | None:
        """Reads the full cached scorer output without calling the API.

        Args:
            spec: Request to look up. Gene selection does not affect the lookup.
            conditioning: Cell-conditioning AnnData used by the request, if any.

        Returns:
            Tidy gene-by-track scores for all strands, including raw_score and
            quantile_score when supplied by the SDK, or None on a cache miss.
        """
        hit = self._read_cache(self.cache_key(spec, conditioning))
        return None if hit is None else hit[0]

    def _base(self, req: _Request) -> dict:
        spec = req.spec
        return {
            **{f.name: getattr(spec, f.name) for f in fields(spec)},
            "variant": spec.variant_string,
            "source": self._backend(),
            "cache_key": req.key,
            "conditioning_sha256": req.fingerprint,
            "sdk_version": self.sdk_version,
            "scorer_definition": self.scorer_definition(spec),
        }

    def plan(self, spec: RequestSpec, conditioning: Any = None) -> dict:
        """Describes a scoring request without contacting the API.

        Args:
            spec: Request to describe.
            conditioning: Cell-conditioning AnnData, or None for reference conditions.

        Returns:
            Request metadata, interval, zero-based offset of the first REF base
            within the interval, and whether the expected cache file exists.
        """
        req = self._request(spec, conditioning)
        chrom, start, end = spec.interval()
        return {
            **self._base(req),
            "interval": f"{chrom}:{start}-{end}",
            "variant_offset_in_interval": spec.pos - 1 - start,
            "cached": self._cache_path(req.key).exists(),
        }

    def cached(self, spec: RequestSpec, conditioning: Any = None) -> dict | None:
        """Reads a gene-level scoring record from the cache.

        Args:
            spec: Request and optional gene selection.
            conditioning: Cell-conditioning AnnData used by the request, if any.

        Returns:
            A scoring record, or None on a cache miss.
        """
        req = self._request(spec, conditioning)
        hit = self._read_cache(req.key)
        if hit is None:
            return None
        tidy, meta = hit
        out = _Outcome(tidy, "completed", from_cache=True, requested_utc=meta["requested_utc"])
        return self._record(req, out)

    def _check(self, spec: RequestSpec, conditioning: Any) -> str:
        n_obs = getattr(conditioning, "n_obs", None)
        if spec.cell_conditioned and conditioning is None:
            return "cell-conditioned spec but no conditioning AnnData was given"
        if not spec.cell_conditioned and conditioning is not None:
            return "conditioning AnnData given for a spec with a reference condition"
        if n_obs is not None and n_obs != spec.n_cells:
            return f"spec.n_cells={spec.n_cells} but conditioning has {n_obs} cells"
        if problem := self._scorer_info(spec)[1]:
            return problem
        if spec.model_version != self.model_version:
            return (
                f"spec.model_version {spec.model_version!r} differs from the client's "
                f"{self.model_version!r}; use one client per model version"
            )
        return ""

    def _delay(self, attempt: int) -> float:
        ceiling = min(MAX_BACKOFF, self.backoff * 2 ** (attempt - 1))
        return ceiling / 2 + random.uniform(0, ceiling / 2)

    def _obtain(self, req: _Request) -> _Outcome:
        if req.problem:
            return _Outcome(None, "failed", req.problem, "validation")
        if self.dry_run:
            return _Outcome(None, "not_run", "dry_run")
        if (hit := self._read_cache(req.key)) is not None:
            return _Outcome(
                hit[0], "completed", from_cache=True, requested_utc=hit[1]["requested_utc"]
            )
        if not sdk_available():
            return _Outcome(
                None,
                "blocked",
                "alphagenome SDK not installed (pip install 'ag2kit[model]')",
                "blocked",
            )
        if not self._custom_model and not api_key_available():
            return _Outcome(None, "blocked", f"{API_KEY_ENV} not set", "blocked")

        spec = req.spec
        error, error_type, attempt = "", "upstream", 0
        for attempt in range(1, self.max_retries + 1):
            requested = utc_now()
            try:
                tidy = self._call(spec, req.conditioning)
            except Exception as exc:
                error = _redact(f"{type(exc).__name__}: {exc}")
                error_type = classify_error(exc)
                if error_type not in _RETRYABLE or attempt == self.max_retries:
                    break
                delay = self._delay(attempt)
                log.info(
                    "retrying %s after %s in %.0fs (attempt %d)",
                    spec.variant_string,
                    error_type,
                    delay,
                    attempt,
                )
                time.sleep(delay)
                continue
            meta = {
                **self._identity(spec, req.fingerprint),
                "cache_key": req.key,
                "requested_utc": requested,
            }
            try:
                self._write_cache(req.key, tidy, meta)
            except (OSError, pa.ArrowException) as exc:
                log.warning("could not cache %s: %s", spec.variant_string, exc)
            return _Outcome(tidy, "completed", attempts=attempt, requested_utc=requested)
        return _Outcome(None, "failed", error, error_type, attempts=attempt)

    def _call(self, spec: RequestSpec, conditioning: Any) -> pd.DataFrame:
        from alphagenome.data import genome
        from alphagenome.models.v2 import variant_scorers

        if self.model is None:
            self.connect()
        chrom, start, end = spec.interval()
        kwargs: dict[str, Any] = {
            "interval": genome.Interval(chromosome=chrom, start=start, end=end),
            "variant": genome.Variant(
                chromosome=chrom,
                position=spec.pos,
                reference_bases=spec.ref,
                alternate_bases=spec.alt,
            ),
            "variant_scorers": [self.recommended_scorer(spec)],
        }
        if spec.cell_conditioned:
            kwargs |= {
                "cell_conditioning": conditioning,
                "cell_condition_name": spec.condition_label,
            }
        else:
            kwargs["condition"] = spec.condition
        outputs = self.model.score_variant(**kwargs)
        if not outputs:
            raise ValueError("AlphaGenome returned no scores")
        tidy = variant_scorers.tidy_anndata(
            outputs[0], match_gene_strand=False, include_extended_metadata=True
        )
        if tidy.empty:
            raise ValueError("AlphaGenome returned no scores")
        for col in tidy.columns[tidy.dtypes == np.dtype("object")]:
            tidy[col] = tidy[col].map(lambda v: v if v is None or isinstance(v, str) else str(v))
        return tidy

    def _record(self, req: _Request, out: _Outcome, tracks: Tracks | None = None) -> dict:
        rec = {
            **self._base(req),
            "raw_effect": np.nan,
            "status": out.status,
            "error": out.error,
            "error_type": out.error_type,
            "attempts": out.attempts,
            "from_cache": out.from_cache,
            "requested_utc": out.requested_utc,
            "track_filter": json.dumps(_normalise(tracks), sort_keys=True) if tracks else "",
        }
        if out.tidy is not None and req.spec.gene_id:
            try:
                effect, meta = out.reduce(req.spec.gene_id, tracks)
            except ValueError as exc:
                rec |= {"status": "failed", "error": str(exc), "error_type": "validation"}
            else:
                rec |= {"raw_effect": effect, **meta}
        return rec

    def score(
        self,
        spec: RequestSpec,
        conditioning: Any = None,
        strict: bool = False,
        tracks: Tracks | None = None,
    ) -> dict:
        """Scores one variant under one condition.

        The score is a predicted ALT-minus-REF effect. RNA_SEQ uses a log fold
        change, while SC_GENE_EXPRESSION_DENOISED uses a difference in summed
        predicted expression.

        Args:
            spec: Variant, model inputs, and optional gene selection.
            conditioning: Cell-conditioning AnnData, or None for reference conditions.
            strict: Whether to raise for failed or blocked records.
            tracks: Optional track metadata filters used in gene reduction.
                Filters share the cache entry for the full scorer output.

        Returns:
            A record with provenance, status, and raw_effect for a successful
            gene reduction. Otherwise raw_effect is NaN. Failed or blocked
            records include error details.

        Raises:
            ScoringError: If strict is true and scoring fails or is blocked.
        """
        req = self._request(spec, conditioning)
        rec = self._record(req, self._obtain(req), tracks)
        if strict and rec["status"] in ("failed", "blocked"):
            raise ScoringError(f"{rec['variant']}: {rec['error']}", rec["error_type"], rec)
        return rec

    def score_many(
        self,
        specs: Iterable[RequestSpec],
        conditioning: Any = None,
        max_workers: int = 4,
        tracks: Tracks | None = None,
    ) -> list[dict]:
        """Scores specs concurrently, fetching each distinct request once.

        Individual request failures are recorded without cancelling the rest.
        Cached requests can be reused after an interruption.

        Args:
            specs: Requests in the desired output order.
            conditioning: One AnnData for all requests, a mapping from
                condition_label to AnnData, or None for reference conditions.
            max_workers: Maximum number of concurrent workers. Must be positive.
            tracks: Optional track metadata filters used in gene reduction.

        Returns:
            One scoring record per spec, in input order.

        Raises:
            ValueError: If max_workers is below one.
        """
        fingerprints: dict[int, str] = {}

        def cells(spec: RequestSpec) -> Any:
            if isinstance(conditioning, Mapping):
                return conditioning.get(spec.condition_label)
            return conditioning

        reqs = [self._request(s, cells(s), fingerprints) for s in specs]
        return self._run(reqs, max_workers, lambda r, o: self._record(r, o, tracks))

    def _run(
        self,
        reqs: list[_Request],
        max_workers: int,
        finish: Callable[[_Request, _Outcome], _T],
    ) -> list[_T]:
        """Obtains each distinct request once and finishes every spec that shares it.

        Each worker reduces its output before returning, so full scorer tables
        are released as soon as their records exist.
        """
        results: list[Any] = [None] * len(reqs)
        shared: dict[str, list[int]] = {}
        for i, req in enumerate(reqs):
            if req.problem:
                results[i] = finish(req, _Outcome(None, "failed", req.problem, "validation"))
            else:
                shared.setdefault(req.key, []).append(i)

        def work(members: list[int]) -> list[tuple[int, _T]]:
            out = self._obtain(reqs[members[0]])
            return [(i, finish(reqs[i], out)) for i in members]

        pool = ThreadPoolExecutor(max_workers=max_workers)
        try:
            futures = [pool.submit(work, members) for members in shared.values()]
            for n, future in enumerate(as_completed(futures), start=1):
                for i, value in future.result():
                    results[i] = value
                if n % PROGRESS_EVERY == 0 or n == len(futures):
                    log.info("finished %d of %d requests", n, len(futures))
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
        return results

    def score_variants(
        self,
        variants: pd.DataFrame,
        *,
        condition: str = "",
        scorer: str = "RNA_SEQ",
        gene_column: str | None = None,
        conditioning: Any = None,
        condition_label: str = "",
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        max_workers: int = 4,
        tracks: Tracks | None = None,
    ) -> pd.DataFrame:
        """Scores a variant table and reduces each result by gene.

        Supply either a reference condition or cell conditioning with a label.
        Memory use is bounded by the number of concurrent requests because
        each request is reduced to gene records as soon as it completes.

        Args:
            variants: Table with chrom, 1-based pos, ref, and alt. Uses variant_id,
                assembly, and input_status when present.
            condition: Reference condition as an ontology CURIE.
            scorer: SDK recommended scorer name.
            gene_column: Column naming the target gene for each row. None reports
                every gene in the scorer output.
            conditioning: One AnnData for a cell-conditioned batch.
            condition_label: SDK track label for the cell condition.
            sequence_length: Supported input window length in bases.
            max_workers: Maximum number of concurrent workers. Must be positive.
            tracks: Optional track metadata filters used in gene reduction.

        Returns:
            Scoring rows with a zero-based input row number and input_ columns.
            Failures are represented by status and reason, including variant-level
            failures when gene selection depends on missing scorer output.

        Raises:
            ValueError: If required variant columns are missing or max_workers
                is below one for a nonempty batch.
            KeyError: If the named gene column is missing from a nonempty batch.
            TypeError: If conditioning is a mapping.
        """
        missing = {"chrom", "pos", "ref", "alt"} - set(variants.columns)
        if missing:
            raise ValueError(f"variants is missing columns {sorted(missing)}")
        if isinstance(conditioning, Mapping):
            raise TypeError("score_variants takes one AnnData; call it once per condition")
        if variants.empty:
            return pd.DataFrame(columns=["row", "status", "error", "error_type"])
        n_cells = int(getattr(conditioning, "n_obs", 0) or 0)
        fingerprints: dict[int, str] = {}
        input_rows = variants.to_dict("records")
        records: list[list[dict]] = [[] for _ in input_rows]
        reqs: list[_Request] = []
        rows: list[int] = []
        for i, row in enumerate(input_rows):
            problem = str(row.get("input_status", "ok") or "ok")
            problem = "" if problem == "ok" else f"input_status={problem}"
            assembly = str(row.get("assembly", "GRCh38") or "GRCh38")
            if not problem and assembly.upper() not in ("GRCH38", "HG38"):
                problem = f"assembly {assembly}: AlphaGenome scores GRCh38; lift over first"
            if not problem:
                try:
                    spec = RequestSpec(
                        variant_id=str(row.get("variant_id") or ""),
                        chrom=row["chrom"],
                        pos=row["pos"],
                        ref=row["ref"],
                        alt=row["alt"],
                        gene_id=str(row[gene_column]) if gene_column else "",
                        condition=condition,
                        condition_label=condition_label,
                        n_cells=n_cells,
                        model_version=self.model_version,
                        scorer_name=scorer,
                        sequence_length=sequence_length,
                    )
                except (ValueError, TypeError) as exc:
                    problem = str(exc)
                else:
                    reqs.append(self._request(spec, conditioning, fingerprints))
                    rows.append(i)
                    continue
            records[i] = [
                {
                    "chrom": normalize_chrom(row["chrom"]),
                    "pos": row["pos"],
                    "ref": str(row["ref"]).upper(),
                    "alt": str(row["alt"]).upper(),
                    "gene_id": str(row[gene_column]) if gene_column else "",
                    "condition": condition,
                    "status": "failed",
                    "error": problem,
                    "error_type": "validation",
                }
            ]

        def finish(req: _Request, out: _Outcome) -> list[dict]:
            if gene_column or out.tidy is None:
                return [self._record(req, out, tracks)]
            names = out.tidy.drop_duplicates("gene_id").set_index("gene_id")["gene_name"]
            return [
                {
                    **self._record(
                        replace(req, spec=replace(req.spec, gene_id=str(g))), out, tracks
                    ),
                    "gene_name": name,
                }
                for g, name in names.items()
            ]

        for i, recs in zip(rows, self._run(reqs, max_workers, finish), strict=True):
            records[i] = recs
        table = pd.DataFrame.from_records(
            [{"row": i, **rec} for i, recs in enumerate(records) for rec in recs]
        )
        inputs = variants.reset_index(drop=True).add_prefix("input_")
        return inputs.join(table.set_index("row"), how="right").rename_axis("row").reset_index()


def _normalise(tracks: Tracks | None) -> dict[str, list[str]]:
    return {
        str(k): [v] if isinstance(v, str) else [str(x) for x in v]
        for k, v in (tracks or {}).items()
    }


def reduce_gene_effect(
    tidy: pd.DataFrame, gene_id: str, tracks: Tracks | None = None
) -> tuple[float, dict]:
    """Averages signed scores for a gene over strand-compatible tracks.

    Unstranded tracks are included, matching the SDK's strand-filtering
    rule. Filter imputation_mode to "observed" to average tracks observed
    in training data separately from imputed tracks.

    Args:
        tidy: SDK tidy scorer output with all strands.
        gene_id: Ensembl gene ID. Version suffixes are ignored.
        tracks: Optional mapping from metadata columns to allowed values.

    Returns:
        Mean raw_score and metadata with the number of values, gene strand,
        and up to four distinct track names.

    Raises:
        ValueError: If the gene or a filter column is absent, no tracks match,
            or any selected score is non-finite.
    """
    target = str(gene_id).split(".")[0]
    rows = tidy[tidy["gene_id"].astype(str).str.split(".").str[0] == target]
    if rows.empty:
        raise ValueError(f"target gene {gene_id} absent from scorer output")
    for column, allowed in _normalise(tracks).items():
        if column not in rows:
            raise ValueError(f"track filter column {column!r} not in {sorted(tidy.columns)}")
        rows = rows[rows[column].astype(str).isin(allowed)]
        if rows.empty:
            raise ValueError(f"no track of gene {gene_id} has {column} in {allowed}")
    gs, ts = rows["gene_strand"], rows["track_strand"]
    rows = rows[~(((gs == "+") & (ts == "-")) | ((gs == "-") & (ts == "+")))]
    values = rows["raw_score"].to_numpy(dtype=float)
    if values.size == 0:
        raise ValueError(f"no track on the strand of gene {gene_id}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"non-finite score for gene {gene_id}")
    meta = {
        "n_reduced_values": int(values.size),
        "gene_strand": str(rows["gene_strand"].iloc[0]),
        "track_names": ";".join(map(str, rows["track_name"].unique()[:4])),
    }
    return float(values.mean()), meta


def extract_gene_effect(adata: Any, gene_id: str) -> tuple[float, dict]:
    """Reduces a scorer AnnData to one gene's strand-compatible effect.

    Args:
        adata: SDK scorer result with genes as rows and tracks as columns.
        gene_id: Ensembl ID of the target gene.

    Returns:
        Mean signed effect and reduction metadata, including the total number
        of tracks in adata before filtering.

    Raises:
        ImportError: If the AlphaGenome SDK is unavailable.
        ValueError: If the gene is absent, has no compatible tracks, or has
            a non-finite selected score.
    """
    from alphagenome.models.v2 import variant_scorers

    tidy = variant_scorers.tidy_anndata(
        adata, match_gene_strand=False, include_extended_metadata=False
    )
    effect, meta = reduce_gene_effect(tidy, gene_id)
    return effect, {**meta, "n_tracks": int(adata.shape[1])}


def conditioning_fingerprint(conditioning: Any) -> str:
    """Hashes cell-conditioning inputs for request identity.

    Sparse matrices are hashed in row blocks, giving the same digest as the
    dense matrix without materializing it.

    Args:
        conditioning: AnnData with a count matrix and ordered cell and gene names,
            or None. Objects without a matrix are identified by their repr.

    Returns:
        SHA-256 digest, or an empty string for None.
    """
    if conditioning is None:
        return ""
    import scipy.sparse as sp

    h = hashlib.sha256()
    x = getattr(conditioning, "X", None)
    if x is None:
        h.update(repr(conditioning).encode())
        return h.hexdigest()
    if sp.issparse(x):
        x = sp.csr_matrix(x)
        h.update(f"{x.dtype}{x.shape}".encode())
        for start in range(0, x.shape[0], _HASH_ROWS):
            h.update(np.ascontiguousarray(x[start : start + _HASH_ROWS].toarray()).tobytes())
    else:
        x = np.ascontiguousarray(np.asarray(x))
        h.update(f"{x.dtype}{x.shape}".encode())
        h.update(x.tobytes())
    names = [list(map(str, names)) for names in (conditioning.obs_names, conditioning.var_names)]
    h.update(json.dumps(names).encode())
    return h.hexdigest()


def interaction(score_treated: float | None, score_control: float | None) -> float:
    """Computes a treated-minus-control difference in predicted effects.

    Args:
        score_treated: Predicted effect under treatment.
        score_control: Predicted effect for the same variant, gene, and scorer
            under control.

    Returns:
        Difference in predicted effects, or NaN if either score is missing
        or non-finite.
    """
    if score_treated is None or score_control is None:
        return float("nan")
    if not (np.isfinite(score_treated) and np.isfinite(score_control)):
        return float("nan")
    return float(score_treated) - float(score_control)
