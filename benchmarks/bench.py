"""Wall-time and peak-memory benchmark for ag2kit's public API.

Uses only public names so it runs unchanged against v0.1 and v0.2. Never calls
the AlphaGenome API: scoring goes through an in-process fake model.

    uv run python benchmarks/bench.py            # full run
    uv run python benchmarks/bench.py --quick    # CI smoke run (<~20 s)
    PYTHONPATH=<checkout>/src uv run python benchmarks/bench.py   # other version
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import tracemalloc
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

# Keep any user cache overrides from redirecting benchmark I/O.
for _var in ("AG2KIT_AG2_CACHE", "AG2KIT_RESULTS_ROOT"):
    os.environ.pop(_var, None)

import ag2kit  # noqa: E402

N_GENES, N_TRACKS = 30, 50


def scorer_output(x, obs, var, scorer, variant):
    """SDK-shaped AnnData accepted by variant_scorers.tidy_anndata (copied from tests)."""
    import anndata as ad
    from alphagenome.data import genome
    from alphagenome.models.v2 import variant_scorers

    obs, var = pd.DataFrame(obs), pd.DataFrame(var)
    obs.index, var.index = obs.index.astype(str), var.index.astype(str)
    adata = ad.AnnData(X=np.asarray(x, dtype=float), obs=obs, var=var)
    adata.uns["interval"] = genome.Interval("chr21", 40897866, 41946442)
    adata.uns["variant"] = genome.Variant.from_str(str(variant))
    adata.uns["variant_scorer"] = variant_scorers.RECOMMENDED_VARIANT_SCORERS[scorer]
    return adata


class FakeModel:
    """SDK stand-in returning N_GENES x N_TRACKS scores seeded by the variant."""

    def __init__(self):
        self.calls = 0
        self.lock = threading.Lock()
        strand = np.array(["+", "-", "."])[np.arange(N_TRACKS) % 3]
        self.var = {
            "strand": strand,
            "name": [f"track{j}" for j in range(N_TRACKS)],
            "imputation_mode": ["none"] * N_TRACKS,
        }

    def score_variant(self, **kw):
        with self.lock:
            self.calls += 1
        v = kw["variant"]
        rng = np.random.default_rng(zlib.crc32(str(v).encode()))
        locus = v.position // 100_000
        genes = [f"ENSG{locus * 100 + k:011d}" for k in range(N_GENES)]
        obs = {
            "gene_id": genes,
            "strand": np.array(["+", "-"])[np.arange(N_GENES) % 2],
            "gene_name": [f"G{locus}_{k}" for k in range(N_GENES)],
        }
        x = rng.normal(size=(N_GENES, N_TRACKS))
        scorer = "SC_GENE_EXPRESSION_DENOISED" if "cell_conditioning" in kw else "RNA_SEQ"
        return [scorer_output(x, obs, self.var, scorer, str(v))]


def make_variants(n, seed=0):
    rng = np.random.default_rng(seed)
    pos = rng.choice(np.arange(1_000_000, 40_000_000), size=n, replace=False)
    ref = rng.choice(list("ACGT"), size=n)
    alt = (pd.Series(ref).map({"A": "C", "C": "G", "G": "T", "T": "A"})).to_numpy()
    return pd.DataFrame(
        {
            "variant_id": [f"v{i}" for i in range(n)],
            "chrom": "chr1",
            "pos": pos,
            "ref": ref,
            "alt": alt,
        }
    )


def make_cells(n_cells=256, n_genes=20_000, seed=0):
    import anndata as ad
    import scipy.sparse as sp

    rng = np.random.default_rng(seed)
    x = sp.random(n_cells, n_genes, density=0.1, format="csr", dtype=np.float32, random_state=rng)
    x.data = rng.poisson(3.0, size=x.data.size).astype(np.float32) + 1
    adata = ad.AnnData(X=x)
    adata.var_names = [f"ENSG{i:011d}" for i in range(n_genes)]
    adata.obs_names = [f"cell{i}" for i in range(n_cells)]
    return adata


def make_eval(n_pairs, n_genes, seed=0):
    rng = np.random.default_rng(seed)
    gene_idx = rng.integers(0, n_genes, n_pairs)
    g_eff = rng.normal(size=n_genes)
    beta = g_eff[gene_idx] + rng.normal(size=n_pairs)
    pred = 0.5 * g_eff[gene_idx] + rng.normal(size=n_pairs)
    ref = rng.choice(list("ACGT"), size=n_pairs)
    alt = pd.Series(ref).map({"A": "C", "C": "G", "G": "T", "T": "A"}).to_numpy()
    keys = pd.DataFrame(
        {
            "chrom": "chr1",
            "pos": np.arange(n_pairs) * 50 + 1_000_000,
            "ref": ref,
            "alt": alt,
            "gene_id": [f"ENSG{g:011d}" for g in gene_idx],
        }
    )
    predictions = keys.assign(raw_effect=pred, status="completed")
    tissues = np.array([f"tissue{t}" for t in range(5)])
    observed = keys.assign(
        beta=beta, effect_allele=alt, tissue=tissues[rng.integers(0, 5, n_pairs)]
    )
    return predictions, observed


def make_loci(n, seed=0):
    rng = np.random.default_rng(seed)
    chrom = rng.integers(1, 23, n)
    return pd.DataFrame(
        {
            "variant_id": [f"rs{i}" for i in range(n)],
            "chrom": [f"chr{c}" for c in chrom],
            "pos": rng.integers(1, 150_000_000, n),
            "gene": [f"ENSG{g:011d}" for g in rng.integers(0, n // 10 + 1, n)],
            "pval": rng.uniform(size=n),
        }
    )


def measure(fn, setup=lambda: None, repeats=1):
    """Best-of-`repeats` untraced wall time plus one traced run for peak memory.

    repeats=0 reports the traced run's own time (slow cases; tracemalloc adds overhead).
    """
    times = []
    for _ in range(repeats):
        arg = setup()
        t0 = time.perf_counter()
        fn(arg)
        times.append(time.perf_counter() - t0)
    arg = setup()
    tracemalloc.start()
    t0 = time.perf_counter()
    fn(arg)
    times.append(time.perf_counter() - t0) if not repeats else None
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return min(times), peak / 2**20


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=float, default=1.0, help="scale factor for problem sizes")
    ap.add_argument("--quick", action="store_true", help="small sizes for a CI smoke run")
    ap.add_argument("--only", default="", help="comma-separated case prefixes (a,b,c,d)")
    args = ap.parse_args(argv)
    s = args.n * (0.01 if args.quick else 1.0)
    sz = lambda n: max(1, int(round(n * s)))  # noqa: E731
    only = set(filter(None, args.only.split(",")))
    want = lambda c: not only or c in only  # noqa: E731

    root = Path(tempfile.mkdtemp(prefix="ag2kit-bench-"))
    os.environ["AG2KIT_CACHE"] = str(root / "ag2kit-cache")
    fresh = lambda: Path(tempfile.mkdtemp(dir=root))  # noqa: E731
    src = Path(ag2kit.__file__).resolve().parent
    sha = (
        subprocess.run(
            ["git", "-C", str(src), "rev-parse", "--short", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
        or "unknown"
    )
    print(f"ag2kit {ag2kit.__version__} @ {sha} from {ag2kit.__file__}")
    print(f"python {sys.version.split()[0]}, scale={s:g}\n")
    rows = []

    def client(cache):
        return ag2kit.AG2Client(dry_run=False, model=FakeModel(), cache_dir=cache, backoff=0)

    def score(cache, variants, **kw):
        out = client(cache).score_variants(variants, **kw)
        assert (out["status"] == "completed").all(), out["status"].value_counts()
        return out

    if want("a"):
        kw = dict(condition="UBERON:0013756", scorer="RNA_SEQ")
        for n in (sz(500), sz(2000)):
            v = make_variants(n)
            reps = 3 if n <= 10 or n == 500 else 1
            t, m = measure(lambda c, v=v: score(c, v, **kw), fresh, repeats=reps)
            rows.append(("a score_variants RNA_SEQ cold", n, t, m))
            warm = fresh()
            score(warm, v, **kw)
            t, m = measure(lambda c, v=v: score(c, v, **kw), lambda w=warm: w, repeats=reps)
            rows.append(("a score_variants RNA_SEQ warm", n, t, m))

    if want("b"):
        # ponytail: one traced run each (cold, then warm on the same cache); this case is slow.
        n = 1 if args.quick else sz(200)
        v, cells = make_variants(n, seed=1), make_cells()
        kw = dict(
            scorer="SC_GENE_EXPRESSION_DENOISED", conditioning=cells, condition_label="control"
        )
        cache = fresh()
        t, m = measure(lambda c: score(c, v, **kw), lambda: cache, repeats=0)
        rows.append(("b score_variants cells 256x20000 cold (traced time)", n, t, m))
        t, m = measure(lambda c: score(c, v, **kw), lambda: cache, repeats=0)
        rows.append(("b score_variants cells 256x20000 warm (traced time)", n, t, m))

    if want("c"):
        n = sz(20_000)
        pred, obs = make_eval(n, max(5, n // 10))
        nb = 1000 if not args.quick else 50
        for group in (None, "tissue"):
            fn = lambda _, g=group: ag2kit.evaluate(pred, obs, n_boot=nb, group=g)  # noqa: E731
            t, m = measure(fn, repeats=1)
            rows.append((f"c evaluate n_boot={nb} group={group}", n, t, m))

    if want("d"):
        n = sz(50_000)
        loci = make_loci(n)
        t, m = measure(lambda _: ag2kit.cluster_loci(loci), repeats=1 if n > 10_000 else 3)
        rows.append(("d cluster_loci", n, t, m))

    print("| case | N | seconds | peak MiB |")
    print("|---|---:|---:|---:|")
    for case, n, t, m in rows:
        print(f"| {case} | {n} | {t:.3f} | {m:.1f} |")
    shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
