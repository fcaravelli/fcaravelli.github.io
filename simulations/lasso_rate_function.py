"""
lasso_rate_function.py

Compare Monte Carlo Lasso regression cost samples against the finite-s
saddle-point rate function used in ldlr_v3.m.

The normalization matches simulations/lasso_regression_omp.c and ldlr_v3.m:

    A_ij = N(0,1) / sqrt(p)
    y = A beta + epsilon,       epsilon_i ~ N(0,1)
    E(w) = 1/2 ||y - A w||^2 + (lambda/2) ||w||_1
    e = E(w*) / p

Usage:

    python lasso_rate_function.py --data-dir ./Data --out-dir ./figures

The script scans for files named like:

    lasso_output1e5_p70_N100_1.txt
    lasso_output100000_p70_N100_beta1_1.txt

Each file should contain one e = E/P sample per line.
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

import matplotlib

matplotlib.use("Agg")
from matplotlib import colors as mcolors
import matplotlib.pyplot as plt


# =====================================================================
# Beta distributions and quadrature
# =====================================================================


@dataclass(frozen=True)
class BetaSpec:
    kind: str = "constant"
    value: float = 1.0
    mean: float = 0.0
    std: float = 1.0
    rho: float = 0.2

    @property
    def label(self) -> str:
        if self.kind == "constant":
            return rf"constant $\beta={self.value:g}$"
        if self.kind == "normal":
            return rf"normal $\mu={self.mean:g}$, $\sigma={self.std:g}$"
        if self.kind == "rademacher":
            return rf"Rademacher $\sigma={self.std:g}$"
        if self.kind == "sparse_rademacher":
            return rf"sparse Rademacher $\rho={self.rho:g}$, $\sigma={self.std:g}$"
        return self.kind

    @property
    def second_moment(self) -> float:
        if self.kind == "constant":
            return self.value * self.value
        if self.kind == "normal":
            return self.std * self.std + self.mean * self.mean
        if self.kind in {"rademacher", "sparse_rademacher"}:
            return self.std * self.std
        return float("nan")


@dataclass
class Quadrature:
    z: np.ndarray
    beta: np.ndarray
    w: np.ndarray
    beta_second_moment: float


def gauss_hermite_normal(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Nodes and weights for expectation over a standard normal."""
    x, w = np.polynomial.hermite.hermgauss(n)
    return np.sqrt(2.0) * x, w / np.sqrt(np.pi)


def beta_quadrature(spec: BetaSpec, n: int) -> tuple[np.ndarray, np.ndarray]:
    kind = spec.kind
    if kind == "constant":
        return np.array([spec.value], dtype=float), np.array([1.0], dtype=float)
    if kind == "normal":
        x, w = np.polynomial.hermite.hermgauss(n)
        nodes = spec.mean + np.sqrt(2.0) * spec.std * x
        weights = w / np.sqrt(np.pi)
        return nodes.astype(float), weights.astype(float)
    if kind == "rademacher":
        amp = spec.std
        return np.array([-amp, amp], dtype=float), np.array([0.5, 0.5], dtype=float)
    if kind == "sparse_rademacher":
        rho = spec.rho
        if rho <= 0.0 or rho > 1.0:
            raise ValueError("rho must be in (0, 1] for sparse_rademacher")
        amp = spec.std / math.sqrt(rho)
        return (np.array([-amp, 0.0, amp], dtype=float),
                np.array([0.5 * rho, 1.0 - rho, 0.5 * rho], dtype=float))
    raise ValueError(f"Unsupported beta kind: {kind}")


def make_quadrature(spec: BetaSpec, n_hermite_z: int,
                    n_hermite_beta: int) -> Quadrature:
    z_nodes, z_weights = gauss_hermite_normal(n_hermite_z)
    b_nodes, b_weights = beta_quadrature(spec, n_hermite_beta)

    z_grid, b_grid = np.meshgrid(z_nodes, b_nodes, indexing="ij")
    wz_grid, wb_grid = np.meshgrid(z_weights, b_weights, indexing="ij")
    weights = (wz_grid * wb_grid).ravel()
    weights = weights / weights.sum()

    return Quadrature(
        z=z_grid.ravel(),
        beta=b_grid.ravel(),
        w=weights,
        beta_second_moment=spec.second_moment,
    )


# =====================================================================
# Lasso finite-s saddle theory: port of ldlr_v3.m
# =====================================================================


@dataclass
class TheoryCurve:
    r: float
    lam: float
    beta: BetaSpec
    e_mean: float
    v_per_p: float
    s_grid: np.ndarray
    Phi_grid: np.ndarray
    xi_grid: np.ndarray
    psi_grid: np.ndarray
    e_grid: np.ndarray
    I_grid: np.ndarray
    status: np.ndarray
    theta: np.ndarray


def default_s_grid() -> np.ndarray:
    grid = np.r_[
        np.linspace(-0.85, -0.05, 65),
        np.linspace(-0.045, 0.045, 41),
        np.linspace(0.05, 8.0, 140),
    ]
    return np.unique(grid)


def soft_threshold(x: np.ndarray, theta: float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - theta, 0.0)


def unpack_theta(theta: np.ndarray) -> tuple[float, float, float, float, float]:
    v = np.exp(theta)
    q0 = float(v[0])
    chi = float(v[1])
    chihat = float(v[2])
    tau = float(v[3])
    qhat0 = -0.5 * tau * tau
    return q0, chi, chihat, tau, qhat0


def theta_is_reasonable(theta: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(theta)) and np.all(theta > -80.0) and np.all(theta < 80.0))


def scalar_u_m_lasso(lam: float, chihat: float, tau: float,
                     beta: np.ndarray, z: np.ndarray
                     ) -> tuple[np.ndarray, np.ndarray]:
    a = beta - (tau / (2.0 * chihat)) * z
    theta = lam / (4.0 * chihat)
    wstar = soft_threshold(a, theta)
    u = beta - wstar
    m = chihat * u * u - tau * z * u + 0.5 * lam * np.abs(wstar)
    return u, m


def logsumexp(a: np.ndarray) -> float:
    mx = float(np.max(a))
    return mx + math.log(float(np.sum(np.exp(a - mx))))


def tilted_moments(theta: np.ndarray, s: float, lam: float,
                   quad: Quadrature) -> tuple[float, float]:
    _, _, chihat, tau, _ = unpack_theta(theta)
    u, m = scalar_u_m_lasso(lam, chihat, tau, quad.beta, quad.z)
    logw = np.log(np.maximum(quad.w, np.finfo(float).tiny)) - s * m
    mx = float(np.max(logw))
    wtilt = np.exp(logw - mx)
    wtilt = wtilt / wtilt.sum()
    tm_u2 = float(np.sum(wtilt * u * u))
    tm_zu = float(np.sum(wtilt * quad.z * u))
    return tm_u2, tm_zu


def saddle_residual(theta: np.ndarray, s: float, r: float, lam: float,
                    quad: Quadrature) -> np.ndarray:
    if not theta_is_reasonable(theta):
        return np.full(4, 1e6, dtype=float)
    q0, chi, chihat, tau, qhat0 = unpack_theta(theta)
    x = 1.0 + chi + s * (q0 + 1.0)
    if x <= 0.0 or chi <= 0.0 or chihat <= 0.0 or tau <= 0.0 or not math.isfinite(x):
        return np.full(4, 1e6, dtype=float)

    tm_u2, tm_zu = tilted_moments(theta, s, lam, quad)
    F = np.empty(4, dtype=float)
    F[0] = chihat - 1.0 / (2.0 * r * (1.0 + chi))
    F[1] = qhat0 - (1.0 / s) * (1.0 / (2.0 * r * x) - chihat)
    F[2] = q0 - tm_u2
    F[3] = chi + s * q0 - tm_zu / tau

    scale = np.array([
        max(1.0, abs(chihat)),
        max(1.0, abs(qhat0)),
        max(1.0, abs(q0)),
        max(1.0, abs(chi + s * q0)),
    ])
    return F / scale


def action_general(theta: np.ndarray, s: float, r: float, lam: float,
                   quad: Quadrature) -> tuple[float, bool]:
    if not theta_is_reasonable(theta):
        return float("nan"), False
    q0, chi, chihat, tau, qhat0 = unpack_theta(theta)
    x = 1.0 + chi + s * (q0 + 1.0)
    valid = x > 0.0 and chi > 0.0 and chihat > 0.0 and tau > 0.0
    if not valid:
        return float("nan"), False

    _, m = scalar_u_m_lasso(lam, chihat, tau, quad.beta, quad.z)
    log_avg = logsumexp(np.log(np.maximum(quad.w, np.finfo(float).tiny)) - s * m)
    S = (
        s * (qhat0 * chi + q0 * chihat)
        + s * s * q0 * qhat0
        - (1.0 / (2.0 * r)) * math.log(x / (1.0 + chi))
        + log_avg
    )
    return S, math.isfinite(S)


def initial_theta(r: float) -> np.ndarray:
    if r < 1.0:
        chi = r / (1.0 - r)
        chihat = (1.0 - r) / (2.0 * r)
        q0 = max(chi, 1e-3)
        tau = math.sqrt(max((1.0 - r) / r, 1e-12))
    else:
        chi = 1.0
        chihat = 1.0 / (2.0 * r * (1.0 + chi))
        q0 = 1.0
        tau = 1.0
    return np.log(np.array([q0, chi, chihat, tau], dtype=float))


def _objective_norm(fun, theta: np.ndarray) -> float:
    val = fun(theta)
    return float(np.linalg.norm(val))


def solve_saddle_general(s: float, r: float, lam: float, quad: Quadrature,
                         theta0: np.ndarray, solver_tol: float,
                         solver_max_iter: int, rng: np.random.Generator
                         ) -> tuple[np.ndarray, float, bool]:
    try:
        from scipy import optimize
    except ImportError as exc:
        raise RuntimeError(
            "lasso_rate_function.py needs scipy for the nonlinear saddle solver."
        ) from exc

    fun = lambda th: saddle_residual(th, s, r, lam, quad)
    maxfev = max(20000, 10 * solver_max_iter)

    best_theta = theta0.copy()
    best_norm = float("inf")

    def try_root(start: np.ndarray) -> tuple[np.ndarray, float, bool]:
        res = optimize.root(
            fun, start, method="hybr",
            options={"xtol": solver_tol, "maxfev": maxfev},
        )
        rn = _objective_norm(fun, res.x)
        return res.x, rn, bool(res.success)

    def try_minimize(start: np.ndarray) -> tuple[np.ndarray, float, bool]:
        res = optimize.minimize(
            lambda th: float(np.dot(fun(th), fun(th))),
            start,
            method="Nelder-Mead",
            options={
                "xatol": solver_tol,
                "fatol": solver_tol,
                "maxiter": solver_max_iter,
                "maxfev": maxfev,
                "disp": False,
            },
        )
        rn = _objective_norm(fun, res.x)
        return res.x, rn, bool(res.success)

    for start in [theta0]:
        theta, rn, ok = try_root(start)
        if (not ok) or rn > 1e-5:
            theta, rn, ok = try_minimize(theta)
        if rn < best_norm:
            best_theta = theta
            best_norm = rn

    if best_norm > 1e-5:
        for scale in (0.1, 0.3, 1.0):
            start = theta0 + scale * rng.standard_normal(theta0.shape)
            theta, rn, ok = try_root(start)
            if (not ok) or rn > 1e-5:
                theta, rn, ok = try_minimize(theta)
            if rn < best_norm:
                best_theta = theta
                best_norm = rn

    S, valid = action_general(best_theta, s, r, lam, quad)
    ok = best_norm < 1e-5 and valid and math.isfinite(S)
    return best_theta, S, ok


def local_cumulants(s: np.ndarray, Phi: np.ndarray) -> tuple[float, float]:
    if s.size < 3:
        return float("nan"), float("nan")

    idx = np.argsort(np.abs(s))[: min(11, s.size)]
    ss = s[idx]
    pp = Phi[idx]
    order = np.argsort(ss)
    ss = ss[order]
    pp = pp[order]

    if ss.size >= 5:
        deg = min(4, ss.size - 1)
        poly = np.polyfit(ss, pp, deg)
        dpoly = np.polyder(poly)
        ddpoly = np.polyder(dpoly)
        mean_xi = float(np.polyval(dpoly, 0.0))
        v_per_p = float(-np.polyval(ddpoly, 0.0))
    else:
        d1 = np.gradient(Phi, s)
        d2 = np.gradient(d1, s)
        i0 = int(np.argmin(np.abs(s)))
        mean_xi = float(d1[i0])
        v_per_p = float(-d2[i0])

    if not math.isfinite(v_per_p) or v_per_p <= 0.0:
        d1 = np.gradient(Phi, s)
        d2 = np.gradient(d1, s)
        i0 = int(np.argmin(np.abs(s)))
        mean_xi = float(d1[i0])
        v_per_p = max(float(-d2[i0]), np.finfo(float).eps)

    return mean_xi, v_per_p


def _unique_sorted_curve(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size == 0:
        return x, y
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    xs: list[float] = []
    ys: list[float] = []
    for xv, yv in zip(x, y):
        if xs and abs(xv - xs[-1]) <= 1e-12 * max(1.0, abs(xs[-1])):
            ys[-1] = min(ys[-1], float(yv))
        else:
            xs.append(float(xv))
            ys.append(float(yv))
    return np.asarray(xs), np.asarray(ys)


def compute_ols_limit_theory(r: float, lam: float, beta: BetaSpec,
                             s_grid: np.ndarray) -> TheoryCurve:
    """Exact lambda=0 limit for p<N, where Lasso reduces to OLS."""
    if r >= 1.0:
        raise ValueError("lambda=0 is the OLS limit; this exact branch needs r < 1.")

    s_used = np.asarray(s_grid, dtype=float)
    s_used = s_used[s_used > -0.999]
    if s_used.size < 3:
        raise ValueError("Need at least three s points with s > -1 for lambda=0.")
    s_used = np.sort(s_used)

    a = (1.0 - r) / (2.0 * r)
    Phi = a * np.log1p(s_used)
    xi = a / (1.0 + s_used)
    psi = Phi - s_used * xi
    i0 = int(np.argmin(np.abs(s_used)))
    psi = psi - psi[i0]

    e_grid, I_grid = _unique_sorted_curve(xi, psi)
    if I_grid.size:
        I_grid = I_grid - np.nanmin(I_grid)

    return TheoryCurve(
        r=r,
        lam=lam,
        beta=beta,
        e_mean=a,
        v_per_p=a,
        s_grid=s_used,
        Phi_grid=Phi,
        xi_grid=xi,
        psi_grid=psi,
        e_grid=e_grid,
        I_grid=I_grid,
        status=np.ones(s_used.size, dtype=int),
        theta=np.empty((s_used.size, 0), dtype=float),
    )


def compute_theory(r: float, lam: float, beta: BetaSpec,
                   s_grid: np.ndarray | None = None,
                   n_hermite_z: int = 80,
                   n_hermite_beta: int = 80,
                   solver_tol: float = 1e-9,
                   solver_max_iter: int = 2000,
                   seed: int = 12345,
                   verbose: bool = False) -> TheoryCurve:
    if r <= 0.0:
        raise ValueError("r must be positive")
    if lam < 0.0:
        raise ValueError("lambda must be nonnegative")

    grid_input = default_s_grid() if s_grid is None else np.asarray(s_grid, dtype=float)
    if lam == 0.0:
        return compute_ols_limit_theory(r, lam, beta, grid_input)

    quad = make_quadrature(beta, n_hermite_z, n_hermite_beta)
    rng = np.random.default_rng(seed)

    Phi = np.full(grid_input.size, np.nan, dtype=float)
    theta = np.full((grid_input.size, 4), np.nan, dtype=float)
    status = np.zeros(grid_input.size, dtype=int)
    Sval = np.full(grid_input.size, np.nan, dtype=float)
    th0 = initial_theta(r)

    for k, s_original in enumerate(grid_input):
        s_eval = float(s_original)
        if abs(s_eval) < 1e-9:
            s_eval = 1e-9

        th, S, ok = solve_saddle_general(
            s_eval, r, lam, quad, th0, solver_tol, solver_max_iter, rng
        )
        theta[k, :] = th
        Sval[k] = S
        Phi[k] = -S
        status[k] = 1 if ok else 0
        if ok:
            th0 = th
        if verbose:
            print(
                f"  theory lasso: s={s_original: .5g} "
                f"Phi={Phi[k]: .10g} ok={int(ok)}",
                file=sys.stderr,
            )

    good = np.isfinite(Phi) & (status > 0)
    s_used = grid_input[good]
    Phi_used = Phi[good]
    theta_used = theta[good, :]
    status_used = status[good]

    if s_used.size < 3:
        raise RuntimeError(
            "Too few successful saddle points. Try a smaller s range, "
            "lower quadrature order, or inspect --verbose output."
        )

    order = np.argsort(s_used)
    s_used = s_used[order]
    Phi_used = Phi_used[order]
    theta_used = theta_used[order, :]
    status_used = status_used[order]

    xi = np.gradient(Phi_used, s_used)
    psi = Phi_used - s_used * xi
    i0 = int(np.argmin(np.abs(s_used)))
    psi = psi - psi[i0]

    e_grid, I_grid = _unique_sorted_curve(xi, psi)
    if I_grid.size:
        I_grid = I_grid - np.nanmin(I_grid)

    mean_xi, v_per_p = local_cumulants(s_used, Phi_used)

    return TheoryCurve(
        r=r,
        lam=lam,
        beta=beta,
        e_mean=mean_xi,
        v_per_p=v_per_p,
        s_grid=s_used,
        Phi_grid=Phi_used,
        xi_grid=xi,
        psi_grid=psi,
        e_grid=e_grid,
        I_grid=I_grid,
        status=status_used,
        theta=theta_used,
    )


# =====================================================================
# File discovery and indexing
# =====================================================================


_PAT_FULL = re.compile(
    r"^lasso_output(?P<samples>\d+(?:e\d+)?)_p(?P<p>\d+)_N(?P<N>\d+)"
    r"(?:_beta(?P<beta0>[\d.]+))?(?:_(?P<rep>\d+))?\.txt$"
)

_PAT_LEGACY = re.compile(r"^lasso_output(?P<samples>\d+(?:e\d+)?)\.txt$")


@dataclass
class Run:
    path: Path
    samples_str: str
    samples: int
    p: int
    N: int
    beta0: float
    rep: int


@dataclass
class Group:
    samples_str: str
    samples: int
    p: int
    N: int
    beta0: float
    runs: list[Run] = field(default_factory=list)

    @property
    def total_samples(self) -> int:
        return self.samples * len(self.runs)


def _parse_samples(s: str) -> int:
    if "e" in s:
        base, exp = s.split("e")
        return int(base) * 10 ** int(exp)
    return int(s)


def discover_files(data_dir: Path, legacy_p: int, legacy_N: int,
                   beta0_default: float = 1.0) -> list[Group]:
    groups: dict[tuple[str, int, int, float], Group] = {}

    for entry in sorted(data_dir.iterdir()):
        if not entry.is_file() or entry.suffix != ".txt":
            continue
        name = entry.name

        m = _PAT_FULL.match(name)
        if m:
            p = int(m["p"])
            N = int(m["N"])
            samples_str = m["samples"]
            beta0 = float(m["beta0"]) if m["beta0"] is not None else beta0_default
            rep = int(m["rep"]) if m["rep"] is not None else 0
        else:
            m = _PAT_LEGACY.match(name)
            if not m:
                continue
            p = legacy_p
            N = legacy_N
            samples_str = m["samples"]
            beta0 = beta0_default
            rep = 0

        run = Run(
            path=entry,
            samples_str=samples_str,
            samples=_parse_samples(samples_str),
            p=p,
            N=N,
            beta0=beta0,
            rep=rep,
        )
        key = (samples_str, p, N, beta0)
        if key not in groups:
            groups[key] = Group(
                samples_str=samples_str,
                samples=run.samples,
                p=p,
                N=N,
                beta0=beta0,
            )
        groups[key].runs.append(run)

    for g in groups.values():
        g.runs.sort(key=lambda r: r.rep)

    return sorted(groups.values(), key=lambda g: (g.p, g.N, g.samples, g.beta0))


# =====================================================================
# Streaming statistics and empirical rates
# =====================================================================


def streaming_histogram(paths: Iterable[Path], bins: np.ndarray,
                        chunk_lines: int = 5_000_000) -> tuple[np.ndarray, int]:
    counts = np.zeros(len(bins) - 1, dtype=np.int64)
    total = 0

    try:
        import pandas as pd
    except ImportError:
        pd = None

    for path in paths:
        if pd is not None:
            reader = pd.read_csv(
                path, header=None, names=["x"], dtype=np.float64,
                chunksize=chunk_lines, engine="c",
            )
            for chunk in reader:
                arr = chunk["x"].to_numpy()
                c, _ = np.histogram(arr, bins=bins)
                counts += c
                total += arr.size
        else:
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
    n = 0
    mean = 0.0
    M2 = 0.0
    vmin = float("inf")
    vmax = float("-inf")

    try:
        import pandas as pd
    except ImportError:
        pd = None

    if pd is not None:
        for path in paths:
            reader = pd.read_csv(
                path, header=None, names=["x"], dtype=np.float64,
                chunksize=chunk_lines, engine="c",
            )
            for chunk in reader:
                arr = chunk["x"].to_numpy()
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
    else:
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
                    vmin = min(vmin, v)
                    vmax = max(vmax, v)
                    delta = v - mean
                    mean += delta / n
                    M2 += delta * (v - mean)

    if n == 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), 0
    var = M2 / n if n > 1 else 0.0
    return mean, math.sqrt(var), vmin, vmax, n


def empirical_rate(counts: np.ndarray, total_n: int, p: int,
                   bin_centers: np.ndarray, min_count: int = 50
                   ) -> tuple[np.ndarray, np.ndarray]:
    mask = counts >= min_count
    if total_n <= 0 or not np.any(mask):
        return np.array([]), np.array([])

    binwidth = bin_centers[1] - bin_centers[0]
    density = counts[mask] / (total_n * binwidth)
    I_emp = -np.log(density) / p
    I_emp = I_emp - I_emp.min()
    return bin_centers[mask], I_emp


# =====================================================================
# Plotting
# =====================================================================


def _blend_with(color: tuple[float, float, float],
                target: tuple[float, float, float],
                amount: float) -> tuple[float, float, float]:
    return tuple((1.0 - amount) * c + amount * t for c, t in zip(color, target))


def _group_curve_colors(n: int, base_color: tuple[float, float, float]
                        ) -> list[tuple[float, float, float]]:
    if n <= 1:
        return [_blend_with(base_color, (1.0, 1.0, 1.0), 0.10)]

    n_dark = (n + 1) // 2
    n_light = n // 2
    dark_amounts = np.linspace(0.10, 0.28, n_dark)
    light_amounts = np.linspace(0.18, 0.45, n_light)

    colors: list[tuple[float, float, float]] = []
    for k in range(n):
        if k % 2 == 0:
            colors.append(_blend_with(base_color, (0.0, 0.0, 0.0),
                                      float(dark_amounts[k // 2])))
        else:
            colors.append(_blend_with(base_color, (1.0, 1.0, 1.0),
                                      float(light_amounts[k // 2])))
    return colors


def plot_group(lam: float, groups_for_r: list[Group], theory: TheoryCurve,
               out_path: Path, n_bins: int = 200, min_count: int = 50,
               chunk_lines: int = 5_000_000) -> dict:
    stats: list[tuple[Group, tuple[float, float, float, float, int]]] = []
    for g in groups_for_r:
        s = streaming_stats([rn.path for rn in g.runs], chunk_lines=chunk_lines)
        stats.append((g, s))

    means = [s[0] for _, s in stats if math.isfinite(s[0])]
    stds = [s[1] for _, s in stats if math.isfinite(s[1])]
    mins = [s[2] for _, s in stats if math.isfinite(s[2])]
    maxs = [s[3] for _, s in stats if math.isfinite(s[3])]
    if not means or not mins or not maxs:
        raise RuntimeError("No finite samples found for this group.")

    std_ref = max(stds) if stds and max(stds) > 0.0 else max(1e-3, 0.05 * abs(max(means)))
    e_min = max(min(mins), max(means) - 8.0 * std_ref)
    e_max = min(max(maxs), max(means) + 8.0 * std_ref)
    if not math.isfinite(e_min) or not math.isfinite(e_max) or e_max <= e_min:
        center = means[0]
        pad = max(1e-3, 0.1 * abs(center))
        e_min, e_max = center - pad, center + pad
    bins = np.linspace(e_min, e_max, n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])

    curves: list[dict] = []
    for g, s in stats:
        counts, total_n = streaming_histogram(
            [rn.path for rn in g.runs], bins=bins, chunk_lines=chunk_lines
        )
        declared_n = g.samples * len(g.runs)
        if declared_n > 0 and total_n > 0:
            rel_gap = abs(total_n - declared_n) / max(total_n, declared_n)
            if rel_gap > 0.05:
                print(
                    f"  warning: {g.samples_str} for p={g.p}, N={g.N} "
                    f"declares {declared_n:.3g} samples, but read {total_n:.3g} lines.",
                    file=sys.stderr,
                )
        e_emp, I_emp = empirical_rate(counts, total_n, g.p, centers,
                                      min_count=min_count)
        curves.append({
            "group": g,
            "stats": s,
            "e": e_emp,
            "I": I_emp,
            "total_n": total_n,
        })

    curves.sort(key=lambda c: (c["group"].p, c["group"].N, c["total_n"]))
    unique_pn = sorted({(c["group"].p, c["group"].N) for c in curves})

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(8.5, 9.0),
        gridspec_kw={"height_ratios": [3.0, 2.0], "hspace": 0.08},
        sharex=True,
    )

    ax_top.plot(
        theory.e_grid, theory.I_grid,
        color="black", lw=2.2,
        label=fr"Theory  $r={theory.r:.4f}$, $\lambda={lam:g}$",
    )

    if theory.e_grid.size and math.isfinite(theory.e_mean):
        v = theory.v_per_p
        if (not math.isfinite(v)) or v <= 0.0:
            ref = max(curves, key=lambda c: c["total_n"])
            v = ref["group"].p * ref["stats"][1] * ref["stats"][1]
        if math.isfinite(v) and v > 0.0:
            I_g = (theory.e_grid - theory.e_mean) ** 2 / (2.0 * v)
            ax_top.plot(theory.e_grid, I_g, "k--", lw=1.4,
                        label="Gaussian approximation")

    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    pn_to_marker = {pn: markers[k % len(markers)] for k, pn in enumerate(unique_pn)}
    base_palette = plt.get_cmap("tab10").colors
    pn_to_base_color = {
        pn: mcolors.to_rgb(base_palette[k % len(base_palette)])
        for k, pn in enumerate(unique_pn)
    }

    pn_curves: dict[tuple[int, int], list[dict]] = {}
    for c in curves:
        pn_curves.setdefault((c["group"].p, c["group"].N), []).append(c)
    for pn, lst in pn_curves.items():
        lst.sort(key=lambda c: c["total_n"])

    for pn, lst in pn_curves.items():
        marker = pn_to_marker[pn]
        curve_colors = _group_curve_colors(len(lst), pn_to_base_color[pn])
        for k, c in enumerate(lst):
            col = curve_colors[k]
            g = c["group"]
            label = (
                rf"MC $p={g.p}$, $N={g.N}$, "
                rf"files={len(g.runs)}, "
                rf"$N_{{tot}}\approx {c['total_n']:.1e}$"
            )
            ax_top.plot(c["e"], c["I"], marker, ms=2.8, color=col,
                        label=label, alpha=0.85)

    if math.isfinite(theory.e_mean):
        ax_top.axvline(theory.e_mean, color="green", ls=":", lw=1.2,
                       label=fr"$\langle e\rangle_{{th}}={theory.e_mean:.4f}$")

    pn_title = ", ".join(f"({p}, {N})" for p, N in unique_pn)
    ax_top.set_ylabel(r"$I(e)$  (rate function)")
    ax_top.set_title(
        fr"Lasso rate function - $r={theory.r:.4f}$, $\lambda={lam:g}$, "
        + theory.beta.label
        + (rf"   $(p, N) \in$ {{{pn_title}}}" if len(unique_pn) > 1
           else f"   $(p, N) = ({unique_pn[0][0]}, {unique_pn[0][1]})$")
    )
    ax_top.grid(True, alpha=0.3)
    ax_top.legend(fontsize=8, loc="upper right")

    if curves:
        all_I = np.concatenate([c["I"] for c in curves if c["I"].size])
        if all_I.size:
            ymax = min(np.percentile(all_I, 99.5),
                       float(np.nanmax(theory.I_grid)) * 1.05)
            if math.isfinite(ymax) and ymax > 0.0:
                ax_top.set_ylim(-0.02 * ymax, ymax)

        e_concat = np.concatenate([c["e"] for c in curves if c["e"].size])
        if e_concat.size:
            xmin = float(e_concat.min())
            xmax = float(e_concat.max())
            pad = 0.05 * max(xmax - xmin, 1e-9)
            ax_top.set_xlim(xmin - pad, xmax + pad)

    for pn, lst in pn_curves.items():
        curve_colors = _group_curve_colors(len(lst), pn_to_base_color[pn])
        for k, c in enumerate(lst):
            if c["e"].size == 0 or theory.e_grid.size == 0:
                continue
            I_th = np.interp(c["e"], theory.e_grid, theory.I_grid,
                             left=np.nan, right=np.nan)
            ok = np.isfinite(I_th)
            if not np.any(ok):
                continue
            I_th = I_th[ok] - np.nanmin(I_th[ok])
            I_mc = c["I"][ok] - np.nanmin(c["I"][ok])
            diff = np.abs(I_mc - I_th)
            diff = np.where(diff > 0.0, diff, np.nan)
            g = c["group"]
            ax_bot.plot(
                c["e"][ok], diff, "-", color=curve_colors[k], lw=1.0,
                label=rf"$p={g.p}$, $N={g.N}$, $N_{{tot}}\approx {c['total_n']:.1e}$",
            )

    ax_bot.set_yscale("log")
    ax_bot.set_xlabel(r"$e$  (cost per feature)")
    ax_bot.set_ylabel(r"$|I_{\rm MC}(e) - I_{\rm th}(e)|$")
    ax_bot.grid(True, which="both", alpha=0.3)
    ax_bot.legend(fontsize=8, loc="upper right")

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "r": theory.r,
        "lambda": lam,
        "beta": theory.beta.label,
        "e_mean_theory": theory.e_mean,
        "v_per_p_theory": theory.v_per_p,
        "groups": [
            {
                "p": c["group"].p,
                "N": c["group"].N,
                "samples_str": c["group"].samples_str,
                "n_replicates": len(c["group"].runs),
                "total_n": c["total_n"],
                "mean_mc": c["stats"][0],
                "std_mc": c["stats"][1],
            }
            for c in curves
        ],
    }


# =====================================================================
# Main
# =====================================================================


def make_s_grid_from_args(args: argparse.Namespace) -> np.ndarray:
    if args.s_points is None:
        return default_s_grid()
    return np.linspace(args.s_min, args.s_max, args.s_points)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--data-dir", type=Path, required=True,
                    help="Directory containing lasso_output*.txt files")
    ap.add_argument("--out-dir", type=Path, default=Path("./figures"),
                    help="Where to write PNGs and summary.csv")
    ap.add_argument("--lambda", dest="lam", type=float, default=0.5,
                    help="Lasso lambda; penalty is lambda/2 * ||w||_1")
    ap.add_argument("--beta-kind", choices=[
        "constant", "normal", "rademacher", "sparse_rademacher",
    ], default="constant")
    ap.add_argument("--beta0", type=float, default=1.0,
                    help="Default beta value for files without _beta{beta0}")
    ap.add_argument("--beta-mean", type=float, default=0.0)
    ap.add_argument("--beta-std", type=float, default=None,
                    help="Beta std/amplitude for non-constant beta; default uses beta0")
    ap.add_argument("--beta-rho", type=float, default=0.2)
    ap.add_argument("--legacy-p", type=int, default=70)
    ap.add_argument("--legacy-N", type=int, default=100)
    ap.add_argument("--bins", type=int, default=200)
    ap.add_argument("--min-count", type=int, default=50)
    ap.add_argument("--chunk-lines", type=int, default=5_000_000)
    ap.add_argument("--r-decimals", type=int, default=4)
    ap.add_argument("--n-hermite-z", type=int, default=80)
    ap.add_argument("--n-hermite-beta", type=int, default=80)
    ap.add_argument("--solver-tol", type=float, default=1e-9)
    ap.add_argument("--solver-max-iter", type=int, default=2000)
    ap.add_argument("--s-min", type=float, default=-0.85)
    ap.add_argument("--s-max", type=float, default=8.0)
    ap.add_argument("--s-points", type=int, default=None,
                    help="Use a simple linspace s-grid instead of the MATLAB default")
    ap.add_argument("--seed", type=int, default=12345,
                    help="Seed for deterministic saddle restarts")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    groups = discover_files(args.data_dir, args.legacy_p, args.legacy_N,
                            beta0_default=args.beta0)
    if not groups:
        print(f"No matching files in {args.data_dir}", file=sys.stderr)
        return 1

    print("Discovered groups:")
    for g in groups:
        reps = ",".join(str(r.rep) for r in g.runs)
        print(
            f"  p={g.p:>4d}  N={g.N:>4d}  beta0={g.beta0:g}  "
            f"samples={g.samples_str:>8s}  reps=[{reps}]  "
            f"declared_total={g.total_samples:.2e}"
        )

    by_r_beta0: dict[tuple[float, float], list[Group]] = {}
    for g in groups:
        r_key = round(g.p / g.N, args.r_decimals)
        by_r_beta0.setdefault((r_key, g.beta0), []).append(g)

    summary_rows: list[dict] = []
    s_grid = make_s_grid_from_args(args)

    for (r_key, beta0), gs in sorted(by_r_beta0.items()):
        unique_pn = sorted({(g.p, g.N) for g in gs})
        p_ref, N_ref = unique_pn[0]
        r = p_ref / N_ref

        beta_std = args.beta_std if args.beta_std is not None else beta0
        beta = BetaSpec(
            kind=args.beta_kind,
            value=beta0,
            mean=args.beta_mean,
            std=beta_std,
            rho=args.beta_rho,
        )

        pn_label = ", ".join(f"(p={p}, N={N})" for p, N in unique_pn)
        print(
            f"\nComputing Lasso theory for r={r:.4f}, lambda={args.lam:g}, "
            f"{beta.label}  {{{pn_label}}} ..."
        )
        theory = compute_theory(
            r=r,
            lam=args.lam,
            beta=beta,
            s_grid=s_grid,
            n_hermite_z=args.n_hermite_z,
            n_hermite_beta=args.n_hermite_beta,
            solver_tol=args.solver_tol,
            solver_max_iter=args.solver_max_iter,
            seed=args.seed,
            verbose=args.verbose,
        )
        print(
            f"  <e>_th = {theory.e_mean:.6f}, "
            f"v_per_p = {theory.v_per_p:.6g}, "
            f"saddle points = {theory.s_grid.size}/{s_grid.size}"
        )

        pn_tag = "_".join(f"p{p}N{N}" for p, N in unique_pn)
        out_png = args.out_dir / f"lasso_rate_r{r:.3f}_{pn_tag}_beta0{beta0:g}.png"
        print(f"  Building figure -> {out_png}")
        info = plot_group(
            lam=args.lam,
            groups_for_r=gs,
            theory=theory,
            out_path=out_png,
            n_bins=args.bins,
            min_count=args.min_count,
            chunk_lines=args.chunk_lines,
        )
        for grow in info["groups"]:
            summary_rows.append({
                "p": grow["p"],
                "N": grow["N"],
                "r": r,
                "lambda": args.lam,
                "beta_kind": args.beta_kind,
                "beta0": beta0,
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
                    f"{row[k]:.8g}" if isinstance(row[k], float)
                    else str(row[k])
                    for k in keys
                ) + "\n")
        print(f"\nSummary written to {csv_path}")

    print(f"\nDone. {len(by_r_beta0)} figure(s) in {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
