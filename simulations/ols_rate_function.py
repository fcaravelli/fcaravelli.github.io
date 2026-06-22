"""
ols_rate_function.py

Compare Monte Carlo OLS cost samples against the theoretical rate function
I(e), and visualise convergence to theory as the number of samples increases.

Usage on the cluster
--------------------

    python ols_rate_function.py --data-dir /path/to/Data/OLS --out-dir ./figures

By default the script:

  1. Scans --data-dir for files of the form
         ols_output{samples}_p{p}_N{N}_{rep}.txt        (text, one xi per line)
         ols_output{samples}_p{p}_N{N}.txt
         ols_output{samples}.txt                          (legacy single file)
     and also accepts binary output from ols_sampler.c
         *.bin  (raw little-endian float64) with a *.bin.meta sidecar that
                carries N, P and the sample count.
     Use --prefix to match a different stem (e.g. "regression").

  2. Groups replicates by (samples, p, N) and produces, for each (p, N)
     pair, a two-panel figure:
         - top:    rate function  I(e)  (theory, Gaussian approx, MC at
                   several total-sample counts)
         - bottom: |I_MC(e) - I_theory(e)|  on a log scale.

  3. Writes one PNG per (p, N) pair into --out-dir, plus a summary CSV.

Theory (exact, closed form)
---------------------------
For OLS with p < N and Gaussian design, the residual is r = (I - H) y with
H = A (A'A)^{-1} A' the hat matrix.  Since (I - H) A = 0 the signal is
annihilated and  RSS = eps' (I - H) eps ~ chi^2_{N-p}, so

    xi = E/P = RSS / (2 p) = (1 / 2p) * chi^2_{N-p},

independent of beta and of the realisation of A.  With  a = (1 - r) / (2 r),
r = p/N, the scaled cumulant generating function and rate function are

    Phi(s) = a * log(1 + s),     xi(s) = Phi'(s) = a / (1 + s),
    I(e)   = sup_s [Phi(s) - s e] = e - a - a * log(e / a),     e > 0,

with minimum I(a) = 0 (so <xi> = a) and curvature I''(a) = 1/a, giving the
Gaussian (CLT) approximation  I_G(e) = (e - a)^2 / (2a)  and  Var[xi] ~ a/p.
This matches compute_ols_limit_theory() in lasso_rate_function.py.

The cost emitted by ols_sampler.c is exactly e = xi = E/P in this notation.

Author: helper script for Francesco
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

# Headless backend for cluster use.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =====================================================================
# Theory: exact closed-form OLS rate function
# =====================================================================

def a_coeff(r: float) -> float:
    """a = (1 - r) / (2 r),  the mean cost <xi> = E[E/P] for OLS, r = p/N < 1."""
    if not (0.0 < r < 1.0):
        raise ValueError(f"OLS theory needs 0 < r < 1 (got r={r}).")
    return (1.0 - r) / (2.0 * r)


def ols_rate_closed(e: np.ndarray, a: float) -> np.ndarray:
    """I(e) = e - a - a log(e/a), the Cramer rate of xi = chi^2_{N-p}/(2p).

    Defined for e > 0; returns +inf at e <= 0.  I(a) = 0 is the minimum.
    """
    e = np.asarray(e, dtype=float)
    out = np.full(e.shape, np.inf)
    pos = e > 0.0
    out[pos] = e[pos] - a - a * np.log(e[pos] / a)
    return out


def ols_rate_exact_finite(e: np.ndarray, p: int, N: int) -> np.ndarray:
    """Exact finite-N analogue of the empirical rate, -(1/p) log f_xi(e).

    The samples obey xi = chi^2_{N-p} / (2p) exactly, a Gamma(shape=(N-p)/2,
    scale=1/p).  Its density is
        f(e) = (2 p e)^{m/2 - 1} e^{-p e} (2 p) / (2^{m/2} Gamma(m/2)),  m = N-p,
    so the finite-p "rate"  -(1/p) log f(e)  is what the Monte Carlo empirical
    rate converges to *at finite p* (the asymptotic I(e) is its p -> inf limit).
    Anchored to its own minimum, exactly like the MC and theory curves.
    """
    e = np.asarray(e, dtype=float)
    m = N - p
    out = np.full(e.shape, np.inf)
    pos = e > 0.0
    log_f = ((m / 2.0 - 1.0) * np.log(2.0 * p * e[pos])
             - p * e[pos]
             - (m / 2.0) * math.log(2.0)
             - math.lgamma(m / 2.0)
             + math.log(2.0 * p))
    out[pos] = -log_f / p
    if np.any(np.isfinite(out)):
        out = out - np.nanmin(out[np.isfinite(out)])
    return out


def ols_scgf(s: np.ndarray, a: float) -> np.ndarray:
    """Phi(s) = a log(1 + s), the SCGF (domain s > -1)."""
    s = np.asarray(s, dtype=float)
    out = np.full(s.shape, np.nan)
    ok = s > -1.0
    out[ok] = a * np.log1p(s[ok])
    return out


@dataclass
class TheoryCurve:
    """Everything needed to overlay theory on the plots."""
    r: float
    a: float
    e_mean: float          # = a
    v_per_p: float         # = a  (so Var[xi] ~ v_per_p / p)
    e_grid: np.ndarray
    I_grid: np.ndarray
    # parametric cross-check (Phi, xi, psi) along an s-grid
    s_grid: np.ndarray
    Phi_grid: np.ndarray
    xi_grid: np.ndarray
    psi_grid: np.ndarray


def compute_theory(r: float,
                   e_lo_factor: float = 0.05,
                   e_hi_factor: float = 4.0,
                   Ne: int = 800,
                   Ns: int = 4000) -> TheoryCurve:
    """Closed-form OLS theory on a dense e-grid, plus a parametric cross-check.

    e-grid spans [e_lo_factor*a, e_hi_factor*a]; the plotter clips to MC support.
    """
    a = a_coeff(r)

    e_lo = max(1e-12, e_lo_factor * a)
    e_hi = e_hi_factor * a
    e_grid = np.linspace(e_lo, e_hi, Ne)
    I_grid = ols_rate_closed(e_grid, a)

    # Parametric cross-check: build I(e) from (xi(s), psi(s)) and confirm it
    # agrees with the closed form to machine precision.
    s_grid = np.linspace(-0.95, 60.0, Ns)
    Phi_grid = ols_scgf(s_grid, a)
    xi_grid = a / (1.0 + s_grid)
    psi_grid = Phi_grid - s_grid * xi_grid
    i0 = int(np.argmin(np.abs(s_grid)))
    psi_grid = psi_grid - psi_grid[i0]

    return TheoryCurve(r=r, a=a, e_mean=a, v_per_p=a,
                       e_grid=e_grid, I_grid=I_grid,
                       s_grid=s_grid, Phi_grid=Phi_grid,
                       xi_grid=xi_grid, psi_grid=psi_grid)


# =====================================================================
# File discovery and indexing
# =====================================================================

@dataclass
class Run:
    path: Path
    samples_str: str
    samples: int
    p: int
    N: int
    rep: int
    binary: bool


@dataclass
class Group:
    """All replicates sharing (samples_str, p, N)."""
    samples_str: str
    samples: int          # samples per replicate
    p: int
    N: int
    runs: list[Run] = field(default_factory=list)

    @property
    def total_samples(self) -> int:
        return self.samples * len(self.runs)

    @property
    def key(self) -> tuple[int, int]:
        return (self.p, self.N)


def _parse_samples(s: str) -> int:
    """'1e8' -> 100000000;  '100000' -> 100000;  '10e8' -> 1000000000."""
    if "e" in s:
        base, exp = s.split("e")
        return int(base) * 10 ** int(exp)
    return int(s)


def _read_meta(path: Path) -> dict[str, str]:
    """Parse a key=value .meta sidecar written by ols_sampler.c."""
    meta: dict[str, str] = {}
    try:
        with open(path, "r") as fh:
            for line in fh:
                line = line.strip()
                if not line or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                meta[k.strip()] = v.strip()
    except OSError:
        pass
    return meta


def discover_files(data_dir: Path, prefix: str,
                   legacy_p: int, legacy_N: int) -> list[Group]:
    """Find text and binary OLS sample files under data_dir."""
    # Text:   prefix_output{samples}_p{p}_N{N}[_{rep}].txt   or legacy ..{samples}.txt
    pat_full = re.compile(
        rf"^{re.escape(prefix)}_output(?P<samples>\d+(?:e\d+)?)"
        rf"_p(?P<p>\d+)_N(?P<N>\d+)(?:_(?P<rep>\d+))?\.txt$"
    )
    pat_legacy = re.compile(
        rf"^{re.escape(prefix)}_output(?P<samples>\d+(?:e\d+)?)\.txt$"
    )

    groups: dict[tuple[str, int, int], Group] = {}

    def add(samples_str, samples, p, N, rep, path, binary):
        run = Run(path=path, samples_str=samples_str, samples=samples,
                  p=p, N=N, rep=rep, binary=binary)
        gkey = (samples_str, p, N)
        if gkey not in groups:
            groups[gkey] = Group(samples_str=samples_str, samples=samples,
                                 p=p, N=N)
        groups[gkey].runs.append(run)

    for entry in sorted(data_dir.iterdir()):
        if not entry.is_file():
            continue
        name = entry.name

        if entry.suffix == ".txt":
            m = pat_full.match(name)
            if m:
                add(m["samples"], _parse_samples(m["samples"]),
                    int(m["p"]), int(m["N"]),
                    int(m["rep"]) if m["rep"] is not None else 0,
                    entry, binary=False)
                continue
            m = pat_legacy.match(name)
            if m:
                add(m["samples"], _parse_samples(m["samples"]),
                    legacy_p, legacy_N, 0, entry, binary=False)
                continue

        elif entry.suffix == ".bin":
            # Binary from ols_sampler.c: read N, P, n from the .meta sidecar.
            meta = _read_meta(entry.with_name(entry.name + ".meta"))
            try:
                p = int(meta["P"]); N = int(meta["N"])
            except (KeyError, ValueError):
                p, N = legacy_p, legacy_N
            n = int(meta["n"]) if meta.get("n", "").isdigit() else \
                entry.stat().st_size // 8
            # optional trailing _rep in stem
            mrep = re.search(r"_(\d+)$", entry.stem)
            rep = int(mrep.group(1)) if mrep else 0
            add(str(n), n, p, N, rep, entry, binary=True)

    for g in groups.values():
        g.runs.sort(key=lambda r: r.rep)

    return sorted(groups.values(), key=lambda g: (g.p, g.N, g.samples))


# =====================================================================
# Streaming readers (text and binary), histogram + Welford stats
# =====================================================================

def _iter_chunks(run: Run, chunk_lines: int) -> Iterable[np.ndarray]:
    """Yield float64 chunks from a Run, text or binary, without loading it all."""
    if run.binary:
        with open(run.path, "rb") as fh:
            while True:
                arr = np.fromfile(fh, dtype="<f8", count=chunk_lines)
                if arr.size == 0:
                    break
                yield arr
        return
    # text
    try:
        import pandas as pd
        reader = pd.read_csv(run.path, header=None, names=["x"],
                             dtype=np.float64, chunksize=chunk_lines, engine="c")
        for chunk in reader:
            yield chunk["x"].to_numpy()
    except ImportError:
        with open(run.path, "r") as fh:
            buf: list[float] = []
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    buf.append(float(line))
                except ValueError:
                    continue
                if len(buf) >= chunk_lines:
                    yield np.asarray(buf, dtype=np.float64)
                    buf.clear()
            if buf:
                yield np.asarray(buf, dtype=np.float64)


def streaming_histogram(runs: Iterable[Run], bins: np.ndarray,
                        chunk_lines: int = 5_000_000) -> tuple[np.ndarray, int]:
    counts = np.zeros(len(bins) - 1, dtype=np.int64)
    total = 0
    for run in runs:
        for arr in _iter_chunks(run, chunk_lines):
            c, _ = np.histogram(arr, bins=bins)
            counts += c
            total += arr.size
    return counts, total


def streaming_stats(runs: Iterable[Run],
                    chunk_lines: int = 5_000_000
                    ) -> tuple[float, float, float, float, int]:
    """(mean, std, min, max, n) via chunked Welford (Chan et al.)."""
    n = 0
    mean = 0.0
    M2 = 0.0
    vmin = float("inf")
    vmax = float("-inf")
    for run in runs:
        for arr in _iter_chunks(run, chunk_lines):
            if arr.size == 0:
                continue
            vmin = min(vmin, float(arr.min()))
            vmax = max(vmax, float(arr.max()))
            cn = arr.size
            cmean = float(arr.mean())
            cM2 = float(((arr - cmean) ** 2).sum())
            delta = cmean - mean
            new_n = n + cn
            mean += delta * cn / new_n
            M2 += cM2 + delta * delta * n * cn / new_n
            n = new_n
    var = M2 / n if n > 1 else 0.0
    return mean, math.sqrt(var), vmin, vmax, n


# =====================================================================
# Empirical rate function
# =====================================================================

def _empirical_rate(counts: np.ndarray, total_n: int, p: int,
                    bin_centers: np.ndarray, min_count: int = 50
                    ) -> tuple[np.ndarray, np.ndarray]:
    """I_emp(e) = -log(density)/p, anchored so its minimum is 0."""
    mask = counts >= min_count
    if not np.any(mask):
        return np.array([]), np.array([])
    binwidth = bin_centers[1] - bin_centers[0]
    density = counts[mask] / (total_n * binwidth)
    I_emp = -np.log(density) / p
    I_emp = I_emp - I_emp.min()
    return bin_centers[mask], I_emp


# =====================================================================
# Plotting
# =====================================================================

def plot_group(p: int, N: int, groups_for_pn: list[Group],
               theory: TheoryCurve, out_path: Path,
               n_bins: int = 200, min_count: int = 50,
               chunk_lines: int = 5_000_000,
               show_exact_finite: bool = True) -> dict:
    """Two-panel figure for one (p, N) pair."""
    # First pass: global stats for a common bin grid.
    stats: list[tuple[Group, tuple]] = []
    for g in groups_for_pn:
        s = streaming_stats(g.runs, chunk_lines=chunk_lines)
        stats.append((g, s))

    means = [s[0] for _, s in stats]
    stds = [s[1] for _, s in stats]
    mins = [s[2] for _, s in stats]
    maxs = [s[3] for _, s in stats]

    e_min = max(min(mins), max(means) - 8.0 * max(stds))
    e_min = max(e_min, 1e-9)                        # OLS support is e > 0
    e_max = min(max(maxs), max(means) + 8.0 * max(stds))
    if not (e_max > e_min):
        e_max = e_min + max(1e-6, abs(e_min))
    bins = np.linspace(e_min, e_max, n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])

    # Second pass: cumulative empirical rate per group (more samples -> theory).
    curves: list[dict] = []
    for g, s in stats:
        counts, total_n = streaming_histogram(g.runs, bins=bins,
                                              chunk_lines=chunk_lines)
        e_emp, I_emp = _empirical_rate(counts, total_n, p, centers,
                                       min_count=min_count)
        curves.append({"group": g, "stats": s, "e": e_emp, "I": I_emp,
                       "total_n": total_n})
    curves.sort(key=lambda c: c["total_n"])

    # ---------- Plot ----------
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(8.5, 9.0),
        gridspec_kw={"height_ratios": [3.0, 2.0], "hspace": 0.08},
        sharex=True,
    )

    # Theory (closed form)
    ax_top.plot(theory.e_grid, theory.I_grid, color="black", lw=2.2,
                label=fr"Theory (exact)  $r=p/N={theory.r:.3f}$")

    # Gaussian / CLT approximation around the MC mean of the densest curve.
    if curves:
        ref = curves[-1]
        mu, sd, *_ = ref["stats"]
        I_g = (theory.e_grid - mu) ** 2 / (2.0 * p * sd * sd)
        ax_top.plot(theory.e_grid, I_g, "k--", lw=1.4,
                    label=r"Gaussian approx. $(e-\mu)^2/(2p\,\sigma^2)$")

    # Exact finite-N rate: what the MC empirical rate converges to at this p
    # (isolates the O(1/p) prefactor correction from the asymptotic theory).
    if show_exact_finite:
        I_fp = ols_rate_exact_finite(theory.e_grid, p, N)
        ax_top.plot(theory.e_grid, I_fp, "-", color="tab:blue", lw=1.4,
                    alpha=0.9, label=r"Exact finite-$N$  $-\frac{1}{p}\log f_\xi$")

    # MC curves, light -> dark with sample count.
    cmap = plt.get_cmap("plasma")
    n_curves = max(len(curves), 1)
    for k, c in enumerate(curves):
        col = cmap(0.15 + 0.7 * k / max(n_curves - 1, 1))
        g = c["group"]
        label = (rf"MC  {g.samples_str}/run $\times$ {len(g.runs)}"
                 rf"  ($N_{{\rm tot}}\approx{c['total_n']:.1e}$)")
        ax_top.plot(c["e"], c["I"], "o", ms=2.5, color=col,
                    label=label, alpha=0.85)

    ax_top.axvline(theory.e_mean, color="green", ls=":", lw=1.2,
                   label=fr"$\langle e\rangle_{{\rm th}}=a={theory.e_mean:.4f}$")

    ax_top.set_ylabel(r"$I(e)$  (rate function)")
    ax_top.set_title(f"OLS rate function — $p={p}$, $N={N}$, $r={p/N:.3f}$")
    ax_top.grid(True, alpha=0.3)
    ax_top.legend(fontsize=8, loc="upper right")

    if curves:
        all_I = np.concatenate([c["I"] for c in curves if c["I"].size])
        if all_I.size:
            ymax = min(np.percentile(all_I, 99.5),
                       float(np.max(theory.I_grid)) * 1.05)
            if ymax > 0:
                ax_top.set_ylim(-0.02 * ymax, ymax)

    if curves:
        e_concat = np.concatenate([c["e"] for c in curves if c["e"].size])
        if e_concat.size:
            xmin = float(e_concat.min()); xmax = float(e_concat.max())
            pad = 0.05 * (xmax - xmin)
            ax_top.set_xlim(xmin - pad, xmax + pad)

    # ---------- Residual panel ----------
    # Theory is closed form, so evaluate it exactly at the MC e-values
    # (no interpolation error).  Anchor both to their minima over the support.
    for k, c in enumerate(curves):
        col = cmap(0.15 + 0.7 * k / max(n_curves - 1, 1))
        if c["e"].size == 0:
            continue
        I_th = ols_rate_closed(c["e"], theory.a)
        ok = np.isfinite(I_th)
        if not np.any(ok):
            continue
        I_th_anch = I_th[ok] - np.nanmin(I_th[ok])
        I_mc_anch = c["I"][ok] - c["I"][ok].min()
        diff = np.abs(I_mc_anch - I_th_anch)
        diff = np.where(diff > 0, diff, np.nan)
        ax_bot.plot(c["e"][ok], diff, "-", color=col, lw=1.0,
                    label=rf"$N_{{\rm tot}}\approx{c['total_n']:.1e}$")

    ax_bot.set_yscale("log")
    ax_bot.set_xlabel(r"$e=E/P$  (cost per feature)")
    ax_bot.set_ylabel(r"$|I_{\rm MC}(e)-I_{\rm th}(e)|$")
    ax_bot.grid(True, which="both", alpha=0.3)
    ax_bot.legend(fontsize=8, loc="upper right")

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "p": p, "N": N, "r": p / N,
        "e_mean_theory": theory.e_mean,
        "groups": [
            {"samples_str": c["group"].samples_str,
             "n_replicates": len(c["group"].runs),
             "total_n": c["total_n"],
             "mean_mc": c["stats"][0],
             "std_mc": c["stats"][1]}
            for c in curves
        ],
    }


# =====================================================================
# Main
# =====================================================================

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True,
                    help="Directory containing OLS sample files (.txt or .bin)")
    ap.add_argument("--out-dir", type=Path, default=Path("./figures"),
                    help="Where to write PNGs and the summary CSV")
    ap.add_argument("--prefix", type=str, default="ols",
                    help="Filename stem to match: <prefix>_output*.txt (default: ols)")
    ap.add_argument("--legacy-p", type=int, default=70,
                    help="Assumed p for legacy/headerless files (default: 70)")
    ap.add_argument("--legacy-N", type=int, default=100,
                    help="Assumed N for legacy/headerless files (default: 100)")
    ap.add_argument("--bins", type=int, default=200,
                    help="Histogram bins (default: 200)")
    ap.add_argument("--min-count", type=int, default=50,
                    help="Drop bins with fewer counts than this (default: 50)")
    ap.add_argument("--chunk-lines", type=int, default=5_000_000,
                    help="Streaming chunk size (default: 5e6)")
    ap.add_argument("--no-exact-finite", action="store_true",
                    help="Do not overlay the exact finite-N rate curve")
    args = ap.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    groups = discover_files(args.data_dir, args.prefix,
                            args.legacy_p, args.legacy_N)
    if not groups:
        print(f"No matching files in {args.data_dir} "
              f"(prefix '{args.prefix}', .txt or .bin)", file=sys.stderr)
        return 1

    print("Discovered groups:")
    for g in groups:
        reps = ",".join(str(r.rep) for r in g.runs)
        kind = "bin" if g.runs[0].binary else "txt"
        print(f"  p={g.p:>4d}  N={g.N:>4d}  samples={g.samples_str:>10s}"
              f"  [{kind}]  reps=[{reps}]  total={g.total_samples:.2e}")

    by_pn: dict[tuple[int, int], list[Group]] = {}
    for g in groups:
        by_pn.setdefault(g.key, []).append(g)

    summary_rows: list[dict] = []
    for (p, N), gs in sorted(by_pn.items()):
        r = p / N
        if not (0.0 < r < 1.0):
            print(f"  skipping p={p}, N={N}: OLS needs p<N (r={r:.3f})",
                  file=sys.stderr)
            continue
        print(f"\nComputing theory for p={p}, N={N}, r={r:.4f} ...")
        theory = compute_theory(r=r)
        # parametric vs closed-form self-check
        I_param = np.interp(theory.e_grid,
                            *(_sorted := _xy_sorted(theory.xi_grid, theory.psi_grid)))
        max_dev = float(np.nanmax(np.abs(I_param - theory.I_grid)))
        print(f"  <e>_th = a = {theory.e_mean:.6f},  "
              f"closed-form vs parametric max dev = {max_dev:.2e}")

        out_png = args.out_dir / f"ols_rate_p{p}_N{N}.png"
        print(f"  Building figure -> {out_png}")
        info = plot_group(p=p, N=N, groups_for_pn=gs, theory=theory,
                          out_path=out_png, n_bins=args.bins,
                          min_count=args.min_count, chunk_lines=args.chunk_lines,
                          show_exact_finite=not args.no_exact_finite)
        for grow in info["groups"]:
            summary_rows.append({
                "p": p, "N": N, "r": r,
                "samples_str": grow["samples_str"],
                "n_replicates": grow["n_replicates"],
                "total_n": grow["total_n"],
                "mean_mc": grow["mean_mc"],
                "std_mc": grow["std_mc"],
                "mean_theory": info["e_mean_theory"],
                "mean_diff": grow["mean_mc"] - info["e_mean_theory"],
            })

    csv_path = args.out_dir / "summary.csv"
    if summary_rows:
        keys = list(summary_rows[0].keys())
        with open(csv_path, "w") as fh:
            fh.write(",".join(keys) + "\n")
            for row in summary_rows:
                fh.write(",".join(
                    f"{row[k]:.8g}" if isinstance(row[k], float) else str(row[k])
                    for k in keys) + "\n")
        print(f"\nSummary written to {csv_path}")

    print(f"\nDone. {len(by_pn)} figure(s) in {args.out_dir}")
    return 0


def _xy_sorted(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sort (x,y) by x and drop duplicate x for use with np.interp."""
    order = np.argsort(x)
    xs = np.asarray(x)[order]
    ys = np.asarray(y)[order]
    keep = np.concatenate(([True], np.diff(xs) > 0))
    return xs[keep], ys[keep]


if __name__ == "__main__":
    sys.exit(main())
