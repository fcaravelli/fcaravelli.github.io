"""
ridge_rate_function.py

Compare Monte Carlo ridge regression cost samples against the theoretical
rate function I(e), and visualise convergence to theory as the number of
samples increases.

Usage on the cluster
--------------------

    python ridge_rate_function.py --data-dir /path/to/Data --out-dir ./figures

By default the script:

  1. Scans --data-dir for files of the form
         ridge_output{samples}_p{p}_N{N}_{rep}.txt
     plus the legacy single file ridge_output10e8.txt (treated as p=70,
     N=100 by default; override with --legacy-p / --legacy-N).

  2. Groups replicates by (samples, p, N) and produces, for each (p, N)
     pair, a two-panel figure:
         - top:    rate function  I(e)  (theory, Gaussian approx, MC at
                   several total-sample counts)
         - bottom: |I_MC(e) - I_theory(e)|  on a log scale, for the same
                   sample counts, showing convergence.

  3. Writes one PNG per (p, N) pair into --out-dir, plus a summary CSV.

Theory follows rate_function_ridge.m (Grabert / large-deviation framework
for Ridge), with r = p/N and beta^2 = 1 (since the C code uses
beta_i = 1/sqrt(p), so ||beta||^2 = 1).

The cost from the C code is
    cost = (RSS + p*lambda*||w||^2) / (2p)
which is exactly e = E/P in the theory's notation.

Author: helper script for Francesco
"""

from __future__ import annotations

import argparse
import math
import os
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
# Theory: port of rate_function_ridge.m
# =====================================================================

def phi1(r: float, lam: float) -> float:
    """Eq. (113)."""
    return (math.sqrt(4.0 * lam * r * r + ((lam - 1.0) * r + 1.0) ** 2)
            - lam * r + r - 1.0) / (2.0 * lam * r)


def phi2(r: float, lam: float) -> float:
    """Eq. (114)."""
    return (math.sqrt((lam + 1.0) ** 2 * r * r + 2.0 * (lam - 1.0) * r + 1.0)
            - r * (lam + 1.0) + 1.0) / (4.0 * r)


def avg_loss(r: float, lam: float, b2: float = 1.0) -> float:
    """Eq. (119): average loss <E>/P."""
    return phi2(r, lam) - 0.5 * b2 * lam * (lam * phi1(r, lam) - 1.0)


def _solve_x(s: float, r: float, lam: float, b2: float = 1.0) -> float:
    """Solve Eq. (136) by bisection.  Returns NaN if no root in (0, +inf).

    Mirrors solve_x() in the MATLAB code line by line.
    """
    def f(xx: float) -> float:
        denom = lam + 1.0 / (r * xx)
        return xx - 1.0 - s - 1.0 / denom - b2 * lam * lam * s / (denom * denom)

    xlo = 1e-14
    xhi = max(200.0, 200.0 + 200.0 * abs(s))

    flo = f(xlo)
    fhi = f(xhi)

    if flo * fhi > 0.0:
        return float("nan")

    for _ in range(400):
        xm = 0.5 * (xlo + xhi)
        fm = f(xm)
        if fm * flo < 0.0:
            xhi, fhi = xm, fm
        else:
            xlo, flo = xm, fm
        if (xhi - xlo) < 1e-14 * max(1.0, abs(xm)):
            break
    return 0.5 * (xlo + xhi)


def _find_s_crit(r: float, lam: float, b2: float = 1.0) -> float:
    """Bisect the boundary s_crit below which Eq. (136) has no root."""
    s_good = 0.0       # x = 1 + phi1 is a valid root at s = 0
    s_bad = -10.0
    for _ in range(10):
        if math.isnan(_solve_x(s_bad, r, lam, b2)):
            break
        s_bad *= 2.0

    for _ in range(200):
        s_try = 0.5 * (s_good + s_bad)
        if math.isnan(_solve_x(s_try, r, lam, b2)):
            s_bad = s_try
        else:
            s_good = s_try
        if abs(s_good - s_bad) < 1e-10:
            break
    return 0.5 * (s_good + s_bad)


def Phi_scgf(s: float, r: float, lam: float, b2: float = 1.0) -> float:
    """SCGF Phi(s).  Returns NaN outside its domain."""
    p1 = phi1(r, lam)
    p2 = phi2(r, lam)
    x = _solve_x(s, r, lam, b2)
    if math.isnan(x):
        return float("nan")
    y = lam + 1.0 / (r * x)
    return (p1 * p2
            - (x - 1.0 - s) / (2.0 * r * x)
            + math.log(x / (1.0 + p1)) / (2.0 * r)
            + math.log(y / (lam + 2.0 * p2)) / 2.0
            + b2 * lam * s / 2.0
            - b2 * lam * lam * s / (2.0 * y))


@dataclass
class TheoryCurve:
    """Output of compute_theory(): everything needed to overlay on plots."""
    r: float
    lam: float
    b2: float
    e_mean: float
    s_crit: float
    s_grid: np.ndarray
    Phi_grid: np.ndarray
    e_grid: np.ndarray
    I_grid: np.ndarray


def compute_theory(r: float, lam: float, b2: float = 1.0,
                   target_e_hi_factor: float = 5.0,
                   s_max: float = 60.0,
                   Ns: int = 5000,
                   Ne: int = 600) -> TheoryCurve:
    """Replicates rate_function_ridge.m and returns the theory curves."""
    e_mean = avg_loss(r, lam, b2)
    s_crit = _find_s_crit(r, lam, b2)

    # Approach s_crit until Phi'(s_min) is large enough to cover the
    # right tail we care about (5x the mean by default).
    s_min = s_crit + 0.5 * (0.0 - s_crit)
    ds = 1e-5
    target_e_hi = target_e_hi_factor * e_mean
    for _ in range(60):
        d_phi = (Phi_scgf(s_min + ds, r, lam, b2)
                 - Phi_scgf(s_min - ds, r, lam, b2)) / (2.0 * ds)
        if not math.isnan(d_phi) and d_phi > target_e_hi:
            break
        s_min = 0.5 * (s_min + s_crit)

    s_grid = np.linspace(s_min, s_max, Ns)
    Phi_grid = np.array([Phi_scgf(float(s), r, lam, b2) for s in s_grid])

    # Drop any NaNs (shouldn't happen for s > s_crit but be safe).
    ok = np.isfinite(Phi_grid)
    s_grid = s_grid[ok]
    Phi_grid = Phi_grid[ok]

    dPhi = np.diff(Phi_grid) / np.diff(s_grid)
    e_lo = max(0.0, float(np.min(dPhi)) * 0.8)
    e_hi = float(np.max(dPhi)) * 1.05
    e_hi = max(e_hi, e_mean * 2.0)

    e_grid = np.linspace(e_lo, e_hi, Ne)

    # Legendre transform: I(e) = sup_s [Phi(s) - s*e]
    # Vectorised: outer product gives (Ns x Ne) matrix; max over axis 0.
    I_grid = np.max(Phi_grid[:, None] - s_grid[:, None] * e_grid[None, :],
                    axis=0)

    return TheoryCurve(r=r, lam=lam, b2=b2,
                       e_mean=e_mean, s_crit=s_crit,
                       s_grid=s_grid, Phi_grid=Phi_grid,
                       e_grid=e_grid, I_grid=I_grid)


# =====================================================================
# File discovery and indexing
# =====================================================================

# Matches:  ridge_output10e9_p70_N100_3.txt
_PAT_FULL = re.compile(
    r"^ridge_output(?P<samples>\d+e\d+)_p(?P<p>\d+)_N(?P<N>\d+)_(?P<rep>\d+)\.txt$"
)
# Matches the legacy single file:  ridge_output10e8.txt
_PAT_LEGACY = re.compile(
    r"^ridge_output(?P<samples>\d+e\d+)\.txt$"
)


@dataclass
class Run:
    path: Path
    samples_str: str
    samples: int
    p: int
    N: int
    rep: int           # 0 for the legacy file


@dataclass
class Group:
    """All replicates sharing (samples_str, p, N)."""
    samples_str: str
    samples: int       # samples per replicate
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
    """'10e8' -> 1_000_000_000.  '10e9' -> 10_000_000_000."""
    base, exp = s.split("e")
    return int(base) * 10 ** int(exp)


def discover_files(data_dir: Path,
                   legacy_p: int, legacy_N: int) -> list[Group]:
    groups: dict[tuple[str, int, int], Group] = {}

    for entry in sorted(data_dir.iterdir()):
        if not entry.is_file() or entry.suffix != ".txt":
            continue
        name = entry.name

        m = _PAT_FULL.match(name)
        if m:
            p = int(m["p"])
            N = int(m["N"])
            samples_str = m["samples"]
            rep = int(m["rep"])
        else:
            m = _PAT_LEGACY.match(name)
            if not m:
                continue
            p = legacy_p
            N = legacy_N
            samples_str = m["samples"]
            rep = 0

        samples = _parse_samples(samples_str)
        run = Run(path=entry, samples_str=samples_str, samples=samples,
                  p=p, N=N, rep=rep)
        gkey = (samples_str, p, N)
        if gkey not in groups:
            groups[gkey] = Group(samples_str=samples_str, samples=samples,
                                 p=p, N=N)
        groups[gkey].runs.append(run)

    for g in groups.values():
        g.runs.sort(key=lambda r: r.rep)

    return sorted(groups.values(),
                  key=lambda g: (g.p, g.N, g.samples))


# =====================================================================
# Streaming histogram
# =====================================================================

def streaming_histogram(paths: Iterable[Path],
                        bins: np.ndarray,
                        chunk_lines: int = 5_000_000
                        ) -> tuple[np.ndarray, int]:
    """Histogram one or more text files of one float per line, in chunks.

    Avoids loading 10^9 floats into memory at once.
    Returns (counts, total_n).
    """
    counts = np.zeros(len(bins) - 1, dtype=np.int64)
    total = 0

    for path in paths:
        # np.loadtxt is too slow / memory-heavy.  Use a streaming reader
        # via pandas if available; else fall back to a pure-numpy chunked
        # reader using io.
        try:
            import pandas as pd  # noqa: WPS433
            reader = pd.read_csv(path, header=None, names=["x"],
                                 dtype=np.float64,
                                 chunksize=chunk_lines,
                                 engine="c")
            for chunk in reader:
                arr = chunk["x"].to_numpy()
                c, _ = np.histogram(arr, bins=bins)
                counts += c
                total += arr.size
        except ImportError:
            with open(path, "r") as fh:
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
                        arr = np.asarray(buf, dtype=np.float64)
                        c, _ = np.histogram(arr, bins=bins)
                        counts += c
                        total += arr.size
                        buf.clear()
                if buf:
                    arr = np.asarray(buf, dtype=np.float64)
                    c, _ = np.histogram(arr, bins=bins)
                    counts += c
                    total += arr.size
    return counts, total


def streaming_stats(paths: Iterable[Path],
                    chunk_lines: int = 5_000_000
                    ) -> tuple[float, float, float, float, int]:
    """Return (mean, std, min, max, n) over all files via Welford."""
    n = 0
    mean = 0.0
    M2 = 0.0
    vmin = float("inf")
    vmax = float("-inf")

    try:
        import pandas as pd
        for path in paths:
            reader = pd.read_csv(path, header=None, names=["x"],
                                 dtype=np.float64,
                                 chunksize=chunk_lines,
                                 engine="c")
            for chunk in reader:
                arr = chunk["x"].to_numpy()
                if arr.size == 0:
                    continue
                vmin = min(vmin, float(arr.min()))
                vmax = max(vmax, float(arr.max()))
                # Chunked Welford (Chan et al.)
                cn = arr.size
                cmean = float(arr.mean())
                cM2 = float(((arr - cmean) ** 2).sum())
                delta = cmean - mean
                new_n = n + cn
                mean += delta * cn / new_n
                M2 += cM2 + delta * delta * n * cn / new_n
                n = new_n
    except ImportError:
        # Pure-Python fallback (slower)
        for path in paths:
            with open(path, "r") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        v = float(line)
                    except ValueError:
                        continue
                    n += 1
                    if v < vmin:
                        vmin = v
                    if v > vmax:
                        vmax = v
                    delta = v - mean
                    mean += delta / n
                    M2 += delta * (v - mean)

    var = M2 / n if n > 1 else 0.0
    return mean, math.sqrt(var), vmin, vmax, n


# =====================================================================
# Plotting
# =====================================================================

def _empirical_rate(counts: np.ndarray, total_n: int, p: int,
                    bin_centers: np.ndarray, min_count: int = 50
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Empirical rate function from a histogram.

    I_emp(e) = -log(hat{p}(e)) / p,  hat{p}(e) = counts / (N * binwidth).
    Anchored so its minimum is 0 for direct comparison with theory.
    Returns (e, I_emp) over bins with >= min_count counts.
    """
    mask = counts >= min_count
    if not np.any(mask):
        return np.array([]), np.array([])

    # Probability density (the additive log(N*binwidth) becomes a constant
    # offset that the min-shift below removes anyway, but include it for
    # numerical sanity).
    binwidth = bin_centers[1] - bin_centers[0]
    density = counts[mask] / (total_n * binwidth)
    I_emp = -np.log(density) / p
    I_emp = I_emp - I_emp.min()
    return bin_centers[mask], I_emp


def plot_group(p: int, N: int, lam: float,
               groups_for_pn: list[Group],
               theory: TheoryCurve,
               out_path: Path,
               n_bins: int = 200,
               min_count: int = 50) -> dict:
    """Produce the two-panel figure for one (p, N) pair.

    `groups_for_pn` is a list of Groups all sharing (p, N) but with
    different sample counts.  Each Group may itself have several
    replicates that we stack.
    """
    # First pass: get global min/max and means across all groups so we
    # can pick a single bin grid for fair comparison.
    stats: list[tuple[Group, tuple]] = []
    for g in groups_for_pn:
        s = streaming_stats([r.path for r in g.runs])
        stats.append((g, s))

    # Histogram range: union of all (min, max) clamped to a few sigma
    # around the largest mean so we don't waste bins on rare outliers.
    means = [s[0] for _, s in stats]
    stds = [s[1] for _, s in stats]
    mins = [s[2] for _, s in stats]
    maxs = [s[3] for _, s in stats]

    e_min = max(min(mins), max(means) - 8.0 * max(stds))
    e_max = min(max(maxs), max(means) + 8.0 * max(stds))
    bins = np.linspace(e_min, e_max, n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])

    # Second pass: build cumulative empirical rate functions for
    # increasing total sample counts.
    # We produce one curve per Group (i.e., per samples_str), using the
    # full set of replicates inside that group.  This gives a clean
    # "more samples -> closer to theory" story.
    curves: list[dict] = []
    for g, s in stats:
        counts, total_n = streaming_histogram([r.path for r in g.runs],
                                              bins=bins)
        e_emp, I_emp = _empirical_rate(counts, total_n, p, centers,
                                       min_count=min_count)
        curves.append({"group": g, "stats": s,
                       "e": e_emp, "I": I_emp,
                       "total_n": total_n})

    # Sort by total samples ascending so darker = more samples reads
    # naturally on the legend.
    curves.sort(key=lambda c: c["total_n"])

    # ---------- Plot ----------
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(8.5, 9.0),
        gridspec_kw={"height_ratios": [3.0, 2.0], "hspace": 0.08},
        sharex=True,
    )

    # Theory
    ax_top.plot(theory.e_grid, theory.I_grid, color="black", lw=2.2,
                label=fr"Theory  $r=p/N={theory.r:.3f}$, $\lambda={lam}$")

    # Gaussian approximation around the mean of the most-sampled curve.
    # Equivalent to I_G(e) = (e - mu)^2 / (2 p sd^2), which we anchor to 0
    # at its minimum.  Computed in closed form to avoid log(0) in the tails.
    if curves:
        ref = curves[-1]
        mu, sd, *_ = ref["stats"]
        I_g = (theory.e_grid - mu) ** 2 / (2.0 * p * sd * sd)
        ax_top.plot(theory.e_grid, I_g, "k--", lw=1.4,
                    label=r"Gaussian approx. (around $\langle e\rangle_{\rm MC}$)")

    # MC curves with a colour gradient from light to dark
    cmap = plt.get_cmap("plasma")
    n_curves = max(len(curves), 1)
    for k, c in enumerate(curves):
        col = cmap(0.15 + 0.7 * k / max(n_curves - 1, 1))
        g = c["group"]
        label = (rf"MC  ${g.samples_str}$/run × {len(g.runs)}"
                 rf"  ($N_{{\rm tot}}\approx {c['total_n']:.1e}$)")
        ax_top.plot(c["e"], c["I"], "o", ms=2.5, color=col,
                    label=label, alpha=0.85)

    # Reference vertical line at the theoretical mean
    ax_top.axvline(theory.e_mean, color="green", ls=":", lw=1.2,
                   label=fr"$\langle e\rangle_{{\rm th}}={theory.e_mean:.4f}$")

    ax_top.set_ylabel(r"$I(e)$  (rate function)")
    ax_top.set_title(
        f"Ridge rate function — $p={p}$, $N={N}$, "
        fr"$r={p/N:.3f}$, $\lambda={lam}$"
    )
    ax_top.grid(True, alpha=0.3)
    ax_top.legend(fontsize=8, loc="upper right")

    # Restrict y-axis so far-tail fluctuations don't squash the basin.
    if curves:
        all_I = np.concatenate([c["I"] for c in curves if c["I"].size])
        if all_I.size:
            ymax = min(np.percentile(all_I, 99.5),
                       float(np.max(theory.I_grid)) * 1.05)
            ax_top.set_ylim(-0.02 * ymax, ymax)

    # Restrict x-axis to where MC has meaningful content (otherwise the
    # theory grid extending to ~5x the mean dominates the view).
    if curves:
        e_concat = np.concatenate([c["e"] for c in curves if c["e"].size])
        if e_concat.size:
            xmin = float(e_concat.min())
            xmax = float(e_concat.max())
            pad = 0.05 * (xmax - xmin)
            ax_top.set_xlim(xmin - pad, xmax + pad)

    # ---------- Residual panel ----------
    # Interpolate theory onto each MC curve's e values and plot |diff|.
    for k, c in enumerate(curves):
        col = cmap(0.15 + 0.7 * k / max(n_curves - 1, 1))
        if c["e"].size == 0:
            continue
        I_th_at_e = np.interp(c["e"], theory.e_grid, theory.I_grid,
                              left=np.nan, right=np.nan)
        # The empirical curve is min-anchored; anchor theory the same way
        # over the same support so we compare apples to apples.
        ok = np.isfinite(I_th_at_e)
        if not np.any(ok):
            continue
        I_th_anchored = I_th_at_e[ok] - np.nanmin(I_th_at_e[ok])
        I_mc_anchored = c["I"][ok] - c["I"][ok].min()
        diff = np.abs(I_mc_anchored - I_th_anchored)
        # Avoid log(0)
        diff = np.where(diff > 0, diff, np.nan)
        g = c["group"]
        ax_bot.plot(c["e"][ok], diff, "-", color=col, lw=1.0,
                    label=rf"$N_{{\rm tot}}\approx {c['total_n']:.1e}$")

    ax_bot.set_yscale("log")
    ax_bot.set_xlabel(r"$e$  (cost per feature)")
    ax_bot.set_ylabel(r"$|I_{\rm MC}(e) - I_{\rm th}(e)|$")
    ax_bot.grid(True, which="both", alpha=0.3)
    ax_bot.legend(fontsize=8, loc="upper right")

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "p": p, "N": N, "lambda": lam, "r": p / N,
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
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True,
                    help="Directory containing ridge_output*.txt files")
    ap.add_argument("--out-dir", type=Path, default=Path("./figures"),
                    help="Where to write PNGs and the summary CSV")
    ap.add_argument("--lambda", dest="lam", type=float, default=0.5,
                    help="Ridge regularisation lambda (default: 0.5)")
    ap.add_argument("--b2", type=float, default=1.0,
                    help="||beta||^2 (default: 1.0, matches the C code)")
    ap.add_argument("--legacy-p", type=int, default=70,
                    help="Assumed p for the legacy ridge_output*.txt file")
    ap.add_argument("--legacy-N", type=int, default=100,
                    help="Assumed N for the legacy ridge_output*.txt file")
    ap.add_argument("--bins", type=int, default=200,
                    help="Histogram bins (default: 200)")
    ap.add_argument("--min-count", type=int, default=50,
                    help="Drop bins with fewer counts than this (default: 50)")
    args = ap.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    groups = discover_files(args.data_dir, args.legacy_p, args.legacy_N)
    if not groups:
        print(f"No matching files in {args.data_dir}", file=sys.stderr)
        return 1

    print("Discovered groups:")
    for g in groups:
        reps = ",".join(str(r.rep) for r in g.runs)
        print(f"  p={g.p:>4d}  N={g.N:>4d}  samples={g.samples_str:>5s}"
              f"  reps=[{reps}]  total={g.total_samples:.2e}")

    # Group by (p, N) to make one figure per pair.
    by_pn: dict[tuple[int, int], list[Group]] = {}
    for g in groups:
        by_pn.setdefault(g.key, []).append(g)

    summary_rows: list[dict] = []

    for (p, N), gs in sorted(by_pn.items()):
        r = p / N
        print(f"\nComputing theory for p={p}, N={N}, r={r:.4f} ...")
        theory = compute_theory(r=r, lam=args.lam, b2=args.b2)
        print(f"  <e>_th = {theory.e_mean:.6f}, s_crit = {theory.s_crit:.4f}")

        out_png = args.out_dir / f"ridge_rate_p{p}_N{N}.png"
        print(f"  Building figure -> {out_png}")
        info = plot_group(p=p, N=N, lam=args.lam,
                          groups_for_pn=gs, theory=theory,
                          out_path=out_png,
                          n_bins=args.bins, min_count=args.min_count)
        for grow in info["groups"]:
            summary_rows.append({
                "p": p, "N": N, "r": r, "lambda": args.lam,
                "samples_str": grow["samples_str"],
                "n_replicates": grow["n_replicates"],
                "total_n": grow["total_n"],
                "mean_mc": grow["mean_mc"],
                "std_mc": grow["std_mc"],
                "mean_theory": info["e_mean_theory"],
                "mean_diff": grow["mean_mc"] - info["e_mean_theory"],
            })

    # Summary CSV
    csv_path = args.out_dir / "summary.csv"
    if summary_rows:
        keys = list(summary_rows[0].keys())
        with open(csv_path, "w") as fh:
            fh.write(",".join(keys) + "\n")
            for row in summary_rows:
                fh.write(",".join(
                    f"{row[k]:.8g}" if isinstance(row[k], float)
                    else str(row[k])
                    for k in keys) + "\n")
        print(f"\nSummary written to {csv_path}")

    print(f"\nDone. {len(by_pn)} figure(s) in {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
