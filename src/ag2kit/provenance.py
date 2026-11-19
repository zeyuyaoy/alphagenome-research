"""Run identifiers, typed statuses, lossless TSV tables and run manifests."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

log = logging.getLogger("ag2kit")

STATUSES = (
    "completed",
    "completed_with_warning",
    "not_estimable",
    "blocked",
    "failed",
    "not_run",
)

_VERSIONED_PACKAGES = ("ag2kit", "alphagenome", "numpy", "pandas", "scipy", "anndata")


def cache_dir() -> Path:
    """Resolves the cache directory from the current environment.

    Returns:
        AG2KIT_CACHE if set, otherwise the ag2kit subdirectory under
        XDG_CACHE_HOME or ~/.cache.
    """
    if env := os.environ.get("AG2KIT_CACHE"):
        return Path(env)
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "ag2kit"


def utc_now() -> str:
    """Returns the current UTC timestamp with fractional seconds omitted."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git(root: Path, *args: str) -> str:
    try:
        return subprocess.run(
            ("git", "-C", str(root), *args), capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def git_sha(root: str | Path = ".") -> str:
    """Identifies the current commit of a Git checkout.

    Args:
        root: Directory from which to locate the checkout.

    Returns:
        The short HEAD hash, or "nogit" if Git cannot resolve it.
    """
    return _git(Path(root), "rev-parse", "--short", "HEAD") or "nogit"


def worktree_dirty(root: str | Path = ".") -> bool:
    """Checks for uncommitted changes, including untracked files.

    Args:
        root: Directory from which to locate the checkout.

    Returns:
        Whether Git reports changes. False if Git status is unavailable.
    """
    return bool(_git(Path(root), "status", "--porcelain"))


def make_run_id(root: str | Path = ".") -> str:
    """Builds a timestamped run identifier.

    Args:
        root: Directory from which to locate the Git checkout.

    Returns:
        UTC timestamp to milliseconds, short commit hash or "nogit",
        and a -dirty suffix if Git reports uncommitted changes.
    """
    now = datetime.now(UTC)
    stamp = f"{now:%Y%m%dT%H%M%S}{now.microsecond // 1000:03d}Z"
    suffix = "-dirty" if worktree_dirty(root) else ""
    return f"{stamp}-{git_sha(root)}{suffix}"


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    """Hashes a file without loading it into memory.

    Args:
        path: File to hash.
        chunk: Number of bytes to read at a time. Use a positive value.

    Returns:
        Hexadecimal SHA-256 digest.

    Raises:
        OSError: If the file cannot be opened or read.
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def stable_hash(obj: Any) -> str:
    """Hashes canonical JSON with sorted keys.

    Args:
        obj: JSON-serializable value.

    Returns:
        Hexadecimal SHA-256 digest of the encoded JSON.

    Raises:
        ValueError: If the value contains NaN or infinity, or is circular.
        TypeError: If the value cannot be encoded or its keys cannot be sorted.
    """
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def software_versions() -> dict[str, str]:
    """Collects software versions for run provenance.

    Returns:
        Python, platform, and package version strings, with "not_installed"
        for missing packages.
    """
    versions = {"python": sys.version.split()[0], "platform": platform.platform()}
    for name in _VERSIONED_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not_installed"
    return versions


def write_table(df: pd.DataFrame, path: str | Path, run_id: str) -> Path:
    """Writes a TSV for use with read_table, which preserves float precision.

    Args:
        df: Data to write, with any existing run_id column replaced in the output.
        path: Destination file. Parent directories are created as needed.
        run_id: Run identifier to store on every row.

    Returns:
        Path to the written TSV.

    Raises:
        OSError: If the destination cannot be created or written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out["run_id"] = run_id
    out.to_csv(path, sep="\t", index=False, na_rep="")
    return path


def read_table(path: str | Path) -> pd.DataFrame:
    """Reads a TSV with round-trip float precision.

    Args:
        path: Path to the TSV.

    Returns:
        Parsed table, with empty fields and pandas default NA tokens treated
        as missing.

    Raises:
        OSError: If the file cannot be read.
    """
    # pandas 3 parses floats exactly by default and deprecates the option.
    if int(pd.__version__.split(".")[0]) < 3:
        return pd.read_csv(path, sep="\t", float_precision="round_trip")
    return pd.read_csv(path, sep="\t")


@dataclass
class Run:
    """Collects output tables, checks, and provenance for one analysis stage.

    Attributes:
        stage: Stage name used in the output path and manifest.
        root: Project directory whose Git state is recorded.
        run_id: Output directory name, generated from root when omitted.
        started_utc: UTC start timestamp recorded in the manifest.
        outdir: Run output directory, created on access. AG2KIT_RESULTS_ROOT
            overrides the default results directory under root.
        failed_checks: Recorded verification rows whose pass flag is false.
    """

    stage: str
    root: Path
    run_id: str = ""
    started_utc: str = field(default_factory=utc_now)
    _status: list[dict] = field(default_factory=list, init=False, repr=False)
    _checks: list[dict] = field(default_factory=list, init=False, repr=False)
    _manifest: dict[str, str] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.run_id = self.run_id or make_run_id(self.root)

    @property
    def outdir(self) -> Path:
        """Output directory for this stage and run ID, created on access."""
        results = os.environ.get("AG2KIT_RESULTS_ROOT") or self.root / "results"
        d = Path(results) / self.stage / self.run_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def status(self, component: str, status: str, reason: str = "") -> None:
        """Records and logs a component status.

        Args:
            component: Name of the component reporting its status.
            status: One of the values in STATUSES.
            reason: Optional explanation recorded alongside the status.

        Raises:
            ValueError: If status is unknown.
        """
        if status not in STATUSES:
            raise ValueError(f"unknown status {status!r}; expected one of {STATUSES}")
        self._status.append({"component": component, "status": status, "reason": reason})
        log.info("[%s] %s%s", status, component, f" - {reason}" if reason else "")

    def check(self, area: str, name: str, passed: bool, value: Any = "", note: str = "") -> bool:
        """Records a verification result.

        Args:
            area: Category for the check.
            name: Check name.
            passed: Whether the check passed.
            value: Optional observed value.
            note: Optional context for the result.

        Returns:
            The recorded pass flag.
        """
        self._checks.append(
            {"area": area, "check": name, "pass": bool(passed), "value": value, "note": note}
        )
        return bool(passed)

    def note(self, key: str, value: Any) -> None:
        """Adds or replaces a manifest field.

        Args:
            key: Manifest key, which can also override a built-in field.
            value: Value to store as text.
        """
        self._manifest[key] = str(value)

    def table(self, df: pd.DataFrame, name: str) -> Path:
        """Writes a table under the run output directory.

        Args:
            df: Data to export.
            name: Relative filename for the TSV.

        Returns:
            Path to the written table.

        Raises:
            OSError: If the output cannot be written.
        """
        return write_table(df, self.outdir / name, self.run_id)

    def finish(self) -> Path:
        """Writes the manifest and any recorded statuses and checks.

        Returns:
            Path to run_manifest.tsv.

        Raises:
            OSError: If an output table cannot be written.
        """
        if self._status:
            self.table(pd.DataFrame(self._status), "status.tsv")
        if self._checks:
            self.table(pd.DataFrame(self._checks), "verification.tsv")
        items = {
            "run_id": self.run_id,
            "stage": self.stage,
            "started_utc": self.started_utc,
            "finished_utc": utc_now(),
            "git_sha": git_sha(self.root),
            "worktree_dirty": str(worktree_dirty(self.root)).lower(),
        }
        items |= {f"version.{k}": v for k, v in software_versions().items()}
        items |= self._manifest
        return self.table(
            pd.DataFrame({"item": list(items), "value": list(items.values())}), "run_manifest.tsv"
        )

    @property
    def failed_checks(self) -> list[dict]:
        """Verification rows whose pass flag is false."""
        return [c for c in self._checks if not c["pass"]]
