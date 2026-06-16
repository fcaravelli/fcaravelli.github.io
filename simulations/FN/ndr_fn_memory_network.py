"""
Network simulator for Laplacian-coupled NDR/FitzHugh-Nagumo memories.

This script implements the dimensionless model described in FN.pdf:

    dx/dt = c * (z + x - x^3/3 - y) - epsilon * L x + epsilon * u
    dy/dt = (x - a - b y) / c

The simplest bistable NDR memory element is obtained with

    a = 0, b = 2, z = 0

for which the isolated motif has three equilibria. The two outer roots are
stable memory states; the middle root is the saddle threshold separating
their basins. The code launches binary memory patterns near those two outer
states, couples the voltage variables through a graph Laplacian, and records
the asymptotic memories reached in weak and strong coupling regimes.

Example:

    python simulations/FN/ndr_fn_memory_network.py \
        --N 8 --graph ring --eps 0 0.01 0.05 0.1 0.25 0.5 1 2 5 \
        --patterns all --out-dir simulations/FN/output --plot

Fixed-point discovery from random initial conditions:

    python simulations/FN/ndr_fn_memory_network.py --mode discover \
        --N 4 --graph ring --n-initial 512 --eps 0 0.02 0.1 0.5 2 --plot
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class FNParams:
    """Dimensionless NDR/FitzHugh-Nagumo motif parameters."""

    a: float = 0.0
    b: float = 2.0
    c: float = 5.0
    z: float = 0.0


@dataclass
class SimulationConfig:
    t_final: float = 250.0
    max_step: float = 0.2
    rtol: float = 1e-8
    atol: float = 1e-10
    jitter: float = 1e-3
    refine: bool = True


@dataclass
class PatternResult:
    epsilon: float
    initial_bits: tuple[int, ...]
    final_bits: tuple[int, ...]
    exact_recall: bool
    hamming: int
    residual_norm: float
    max_real_eig: float
    stable: bool
    x_final: np.ndarray
    y_final: np.ndarray


@dataclass
class FixedPointSample:
    epsilon: float
    sample_index: int
    cluster_id: int
    final_bits: tuple[int, ...]
    residual_norm: float
    max_real_eig: float
    stable: bool
    x_initial: np.ndarray
    y_initial: np.ndarray
    x_final: np.ndarray
    y_final: np.ndarray


@dataclass
class FixedPointCluster:
    epsilon: float
    cluster_id: int
    count: int
    final_bits: tuple[int, ...]
    residual_norm: float
    max_real_eig: float
    stable: bool
    x_center: np.ndarray
    y_center: np.ndarray


def isolated_equilibria(params: FNParams) -> list[dict]:
    """Return isolated motif equilibria sorted by x.

    Equilibria solve

        -x^3/3 + (1 - 1/b) x + z + a/b = 0,
        y = (x - a)/b.
    """
    if params.b == 0:
        raise ValueError("b must be nonzero.")

    coeffs = np.array(
        [-1.0 / 3.0, 0.0, 1.0 - 1.0 / params.b, params.z + params.a / params.b],
        dtype=float,
    )
    roots = np.roots(coeffs)
    real_roots = sorted(float(r.real) for r in roots if abs(r.imag) < 1e-9)

    equilibria: list[dict] = []
    for x in real_roots:
        y = (x - params.a) / params.b
        J = isolated_jacobian(x, params)
        eig = np.linalg.eigvals(J)
        trace = float(np.trace(J))
        det = float(np.linalg.det(J))
        equilibria.append(
            {
                "x": x,
                "y": y,
                "trace": trace,
                "det": det,
                "eig": eig,
                "stable": bool(np.max(eig.real) < 0.0),
                "ndr_branch": bool(abs(x) > 1.0),
            }
        )
    return equilibria


def stable_memory_equilibria(params: FNParams) -> tuple[dict, dict, dict]:
    """Return left stable state, saddle threshold, right stable state."""
    eq = isolated_equilibria(params)
    if len(eq) != 3:
        raise ValueError(
            "The selected parameters do not give three isolated equilibria. "
            "Use a bistable regime, for example a=0, b=2, z=0."
        )
    left, middle, right = eq
    if not (left["stable"] and right["stable"] and not middle["stable"]):
        raise ValueError("Expected stable-saddle-stable equilibria, but did not get them.")
    return left, middle, right


def isolated_jacobian(x: float, params: FNParams) -> np.ndarray:
    return np.array(
        [
            [params.c * (1.0 - x * x), -params.c],
            [1.0 / params.c, -params.b / params.c],
        ],
        dtype=float,
    )


def graph_adjacency(kind: str, N: int, weight: float = 1.0,
                    p_edge: float = 0.3, seed: int = 1) -> np.ndarray:
    """Build a simple undirected weighted graph adjacency matrix."""
    if N <= 0:
        raise ValueError("N must be positive.")

    A = np.zeros((N, N), dtype=float)
    if kind == "path":
        for i in range(N - 1):
            A[i, i + 1] = A[i + 1, i] = weight
    elif kind == "ring":
        for i in range(N):
            A[i, (i + 1) % N] = A[(i + 1) % N, i] = weight
    elif kind == "complete":
        A[:, :] = weight
        np.fill_diagonal(A, 0.0)
    elif kind == "star":
        for i in range(1, N):
            A[0, i] = A[i, 0] = weight
    elif kind == "erdos":
        rng = np.random.default_rng(seed)
        mask = rng.random((N, N)) < p_edge
        mask = np.triu(mask, 1)
        A = weight * (mask + mask.T).astype(float)
    else:
        raise ValueError(f"Unknown graph kind: {kind}")
    return A


def laplacian_from_adjacency(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("adjacency matrix must be square.")
    if not np.allclose(A, A.T):
        raise ValueError("adjacency matrix must be symmetric.")
    return np.diag(A.sum(axis=1)) - A


def rhs(_t: float, state: np.ndarray, L: np.ndarray, epsilon: float,
        params: FNParams, u: np.ndarray | None = None) -> np.ndarray:
    N = L.shape[0]
    x = state[:N]
    y = state[N:]
    if u is None:
        u = np.zeros(N, dtype=float)

    xdot = (
        params.c * (params.z + x - (x ** 3) / 3.0 - y)
        - epsilon * (L @ x)
        + epsilon * u
    )
    ydot = (x - params.a - params.b * y) / params.c
    return np.concatenate([xdot, ydot])


def network_jacobian(x: np.ndarray, L: np.ndarray, epsilon: float,
                     params: FNParams) -> np.ndarray:
    N = L.shape[0]
    J = np.zeros((2 * N, 2 * N), dtype=float)
    J[:N, :N] = np.diag(params.c * (1.0 - x * x)) - epsilon * L
    J[:N, N:] = -params.c * np.eye(N)
    J[N:, :N] = (1.0 / params.c) * np.eye(N)
    J[N:, N:] = -(params.b / params.c) * np.eye(N)
    return J


def bits_to_state(bits: Iterable[int], params: FNParams, jitter: float,
                  rng: np.random.Generator) -> np.ndarray:
    left, _, right = stable_memory_equilibria(params)
    bits_tuple = tuple(int(b) for b in bits)
    N = len(bits_tuple)
    x = np.empty(N, dtype=float)
    y = np.empty(N, dtype=float)
    for i, bit in enumerate(bits_tuple):
        eq = right if bit else left
        x[i] = eq["x"]
        y[i] = eq["y"]
    if jitter > 0.0:
        x += jitter * rng.standard_normal(N)
        y += jitter * rng.standard_normal(N)
    return np.concatenate([x, y])


def classify_bits(x: np.ndarray, params: FNParams) -> tuple[int, ...]:
    _, middle, _ = stable_memory_equilibria(params)
    threshold = middle["x"]
    return tuple(int(v > threshold) for v in x)


def refine_equilibrium(state: np.ndarray, L: np.ndarray, epsilon: float,
                       params: FNParams, u: np.ndarray | None) -> np.ndarray:
    try:
        from scipy import optimize
    except ImportError:
        return state

    fun = lambda s: rhs(0.0, s, L, epsilon, params, u)
    sol = optimize.root(fun, state, method="hybr")
    if sol.success and np.linalg.norm(fun(sol.x)) < np.linalg.norm(fun(state)):
        return np.asarray(sol.x, dtype=float)
    return state


def integrate_to_equilibrium(state0: np.ndarray, L: np.ndarray, epsilon: float,
                             params: FNParams, cfg: SimulationConfig,
                             u: np.ndarray | None = None
                             ) -> tuple[np.ndarray, float, float, bool]:
    try:
        from scipy.integrate import solve_ivp
    except ImportError as exc:
        raise RuntimeError("This simulator needs scipy.integrate.solve_ivp.") from exc

    N = L.shape[0]
    fun = lambda t, s: rhs(t, s, L, epsilon, params, u)

    try:
        sol = solve_ivp(
            fun,
            (0.0, cfg.t_final),
            state0,
            method="LSODA",
            max_step=cfg.max_step,
            rtol=cfg.rtol,
            atol=cfg.atol,
        )
    except Exception:
        sol = solve_ivp(
            fun,
            (0.0, cfg.t_final),
            state0,
            method="RK45",
            max_step=cfg.max_step,
            rtol=cfg.rtol,
            atol=cfg.atol,
        )
    if not sol.success:
        raise RuntimeError(f"integration failed at epsilon={epsilon}: {sol.message}")

    state = sol.y[:, -1]
    if cfg.refine:
        state = refine_equilibrium(state, L, epsilon, params, u)

    x = state[:N]
    residual_norm = float(np.linalg.norm(rhs(0.0, state, L, epsilon, params, u)))
    J = network_jacobian(x, L, epsilon, params)
    max_real_eig = float(np.max(np.linalg.eigvals(J).real))
    stable = max_real_eig < 0.0
    return state, residual_norm, max_real_eig, stable


def simulate_pattern(bits: tuple[int, ...], L: np.ndarray, epsilon: float,
                     params: FNParams, cfg: SimulationConfig,
                     rng: np.random.Generator,
                     u: np.ndarray | None = None) -> PatternResult:
    N = L.shape[0]
    state0 = bits_to_state(bits, params, cfg.jitter, rng)
    state, residual_norm, max_real_eig, stable = integrate_to_equilibrium(
        state0, L, epsilon, params, cfg, u
    )
    x = state[:N]
    y = state[N:]
    final_bits = classify_bits(x, params)
    hamming = sum(a != b for a, b in zip(bits, final_bits))

    return PatternResult(
        epsilon=float(epsilon),
        initial_bits=bits,
        final_bits=final_bits,
        exact_recall=(hamming == 0),
        hamming=hamming,
        residual_norm=residual_norm,
        max_real_eig=max_real_eig,
        stable=stable,
        x_final=x.copy(),
        y_final=y.copy(),
    )


def random_initial_state(N: int, x_range: tuple[float, float],
                         y_range: tuple[float, float],
                         rng: np.random.Generator) -> np.ndarray:
    x = rng.uniform(x_range[0], x_range[1], size=N)
    y = rng.uniform(y_range[0], y_range[1], size=N)
    return np.concatenate([x, y])


def simulate_random_initial_condition(sample_index: int, L: np.ndarray,
                                      epsilon: float, params: FNParams,
                                      cfg: SimulationConfig,
                                      rng: np.random.Generator,
                                      x_range: tuple[float, float],
                                      y_range: tuple[float, float],
                                      u: np.ndarray | None = None
                                      ) -> FixedPointSample:
    N = L.shape[0]
    state0 = random_initial_state(N, x_range, y_range, rng)
    state, residual_norm, max_real_eig, stable = integrate_to_equilibrium(
        state0, L, epsilon, params, cfg, u
    )
    x = state[:N]
    y = state[N:]
    return FixedPointSample(
        epsilon=float(epsilon),
        sample_index=sample_index,
        cluster_id=-1,
        final_bits=classify_bits(x, params),
        residual_norm=residual_norm,
        max_real_eig=max_real_eig,
        stable=stable,
        x_initial=state0[:N].copy(),
        y_initial=state0[N:].copy(),
        x_final=x.copy(),
        y_final=y.copy(),
    )


def all_patterns(N: int) -> list[tuple[int, ...]]:
    return [tuple(map(int, bits)) for bits in itertools.product([0, 1], repeat=N)]


def random_patterns(N: int, count: int, rng: np.random.Generator) -> list[tuple[int, ...]]:
    return [tuple(map(int, rng.integers(0, 2, size=N))) for _ in range(count)]


def select_patterns(N: int, mode: str, rng: np.random.Generator,
                    random_count: int) -> list[tuple[int, ...]]:
    if mode == "all":
        if N > 12:
            raise ValueError("all patterns means 2^N simulations; use --patterns random for N > 12.")
        return all_patterns(N)
    if mode == "corners":
        return [tuple(0 for _ in range(N)), tuple(1 for _ in range(N))]
    if mode == "single-flip":
        pats = [tuple(0 for _ in range(N)), tuple(1 for _ in range(N))]
        for i in range(N):
            p0 = [0] * N
            p1 = [1] * N
            p0[i] = 1
            p1[i] = 0
            pats.append(tuple(p0))
            pats.append(tuple(p1))
        return sorted(set(pats))
    if mode == "random":
        return random_patterns(N, random_count, rng)
    raise ValueError(f"Unknown pattern mode: {mode}")


def summarize(results: list[PatternResult]) -> list[dict]:
    by_eps: dict[float, list[PatternResult]] = {}
    for res in results:
        by_eps.setdefault(res.epsilon, []).append(res)

    rows: list[dict] = []
    for eps, group in sorted(by_eps.items()):
        hamming = np.array([g.hamming for g in group], dtype=float)
        residual = np.array([g.residual_norm for g in group], dtype=float)
        maxeig = np.array([g.max_real_eig for g in group], dtype=float)
        rows.append(
            {
                "epsilon": eps,
                "n_patterns": len(group),
                "recall_fraction": float(np.mean([g.exact_recall for g in group])),
                "mean_hamming": float(hamming.mean()),
                "max_hamming": int(hamming.max()),
                "stable_fraction": float(np.mean([g.stable for g in group])),
                "max_residual": float(residual.max()),
                "max_real_eig_worst": float(maxeig.max()),
                "regime": "weak" if eps < 0.1 else ("intermediate" if eps < 1.0 else "strong"),
            }
        )
    return rows


def cluster_fixed_point_samples(samples: list[FixedPointSample],
                                tol: float,
                                params: FNParams) -> list[FixedPointCluster]:
    """Greedily cluster terminal equilibria by max-norm distance."""
    centers: list[np.ndarray] = []
    grouped: list[list[FixedPointSample]] = []

    for sample in samples:
        state = np.concatenate([sample.x_final, sample.y_final])
        assigned = False
        for idx, center in enumerate(centers):
            if float(np.max(np.abs(state - center))) <= tol:
                sample.cluster_id = idx
                grouped[idx].append(sample)
                n = len(grouped[idx])
                centers[idx] = center + (state - center) / n
                assigned = True
                break
        if not assigned:
            sample.cluster_id = len(centers)
            centers.append(state.copy())
            grouped.append([sample])

    clusters: list[FixedPointCluster] = []
    for idx, group in enumerate(grouped):
        center = centers[idx]
        N = group[0].x_final.size
        x_center = center[:N]
        y_center = center[N:]
        residual = np.array([g.residual_norm for g in group], dtype=float)
        maxeig = np.array([g.max_real_eig for g in group], dtype=float)
        bits = classify_bits(x_center, params)
        # Use the first sample's epsilon; each call clusters one epsilon group.
        clusters.append(
            FixedPointCluster(
                epsilon=group[0].epsilon,
                cluster_id=idx,
                count=len(group),
                final_bits=bits,
                residual_norm=float(residual.min()),
                max_real_eig=float(maxeig.max()),
                stable=bool(np.all([g.stable for g in group])),
                x_center=x_center.copy(),
                y_center=y_center.copy(),
            )
        )

    clusters.sort(key=lambda c: (-c.count, c.cluster_id))
    id_map = {old.cluster_id: new_id for new_id, old in enumerate(clusters)}
    for sample in samples:
        sample.cluster_id = id_map[sample.cluster_id]
    for new_id, cluster in enumerate(clusters):
        cluster.cluster_id = new_id
    return clusters


def summarize_clusters(clusters_by_eps: dict[float, list[FixedPointCluster]],
                       n_initial: int) -> list[dict]:
    rows: list[dict] = []
    for eps, clusters in sorted(clusters_by_eps.items()):
        counts = np.array([c.count for c in clusters], dtype=float)
        stable = np.array([c.stable for c in clusters], dtype=bool)
        residual = np.array([c.residual_norm for c in clusters], dtype=float)
        maxeig = np.array([c.max_real_eig for c in clusters], dtype=float)
        rows.append(
            {
                "epsilon": eps,
                "n_initial": n_initial,
                "n_fixed_points_found": len(clusters),
                "n_stable_found": int(stable.sum()),
                "largest_basin_fraction": float(counts.max() / n_initial) if counts.size else 0.0,
                "smallest_basin_count": int(counts.min()) if counts.size else 0,
                "max_residual": float(residual.max()) if residual.size else float("nan"),
                "max_real_eig_worst": float(maxeig.max()) if maxeig.size else float("nan"),
                "regime": "weak" if eps < 0.1 else ("intermediate" if eps < 1.0 else "strong"),
            }
        )
    return rows


def write_summary_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_patterns_csv(results: list[PatternResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        fieldnames = [
            "epsilon",
            "initial_bits",
            "final_bits",
            "exact_recall",
            "hamming",
            "residual_norm",
            "max_real_eig",
            "stable",
            "x_final",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for res in results:
            writer.writerow(
                {
                    "epsilon": res.epsilon,
                    "initial_bits": "".join(map(str, res.initial_bits)),
                    "final_bits": "".join(map(str, res.final_bits)),
                    "exact_recall": int(res.exact_recall),
                    "hamming": res.hamming,
                    "residual_norm": f"{res.residual_norm:.8g}",
                    "max_real_eig": f"{res.max_real_eig:.8g}",
                    "stable": int(res.stable),
                    "x_final": " ".join(f"{v:.8g}" for v in res.x_final),
                }
            )


def write_fixed_point_clusters_csv(clusters_by_eps: dict[float, list[FixedPointCluster]],
                                   path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epsilon",
        "cluster_id",
        "count",
        "final_bits",
        "stable",
        "residual_norm",
        "max_real_eig",
        "x_center",
        "y_center",
    ]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for eps in sorted(clusters_by_eps):
            for cluster in clusters_by_eps[eps]:
                writer.writerow(
                    {
                        "epsilon": cluster.epsilon,
                        "cluster_id": cluster.cluster_id,
                        "count": cluster.count,
                        "final_bits": "".join(map(str, cluster.final_bits)),
                        "stable": int(cluster.stable),
                        "residual_norm": f"{cluster.residual_norm:.8g}",
                        "max_real_eig": f"{cluster.max_real_eig:.8g}",
                        "x_center": " ".join(f"{v:.8g}" for v in cluster.x_center),
                        "y_center": " ".join(f"{v:.8g}" for v in cluster.y_center),
                    }
                )


def write_fixed_point_samples_csv(samples: list[FixedPointSample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epsilon",
        "sample_index",
        "cluster_id",
        "final_bits",
        "stable",
        "residual_norm",
        "max_real_eig",
        "x_initial",
        "y_initial",
        "x_final",
        "y_final",
    ]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for sample in samples:
            writer.writerow(
                {
                    "epsilon": sample.epsilon,
                    "sample_index": sample.sample_index,
                    "cluster_id": sample.cluster_id,
                    "final_bits": "".join(map(str, sample.final_bits)),
                    "stable": int(sample.stable),
                    "residual_norm": f"{sample.residual_norm:.8g}",
                    "max_real_eig": f"{sample.max_real_eig:.8g}",
                    "x_initial": " ".join(f"{v:.8g}" for v in sample.x_initial),
                    "y_initial": " ".join(f"{v:.8g}" for v in sample.y_initial),
                    "x_final": " ".join(f"{v:.8g}" for v in sample.x_final),
                    "y_final": " ".join(f"{v:.8g}" for v in sample.y_final),
                }
            )


def plot_summary(rows: list[dict], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eps = np.array([r["epsilon"] for r in rows], dtype=float)
    recall = np.array([r["recall_fraction"] for r in rows], dtype=float)
    stable = np.array([r["stable_fraction"] for r in rows], dtype=float)
    hamming = np.array([r["mean_hamming"] for r in rows], dtype=float)

    fig, ax1 = plt.subplots(figsize=(7.5, 4.8))
    ax1.plot(eps, recall, "o-", label="exact recall fraction")
    ax1.plot(eps, stable, "s-", label="stable final equilibria")
    ax1.set_xscale("symlog", linthresh=1e-3)
    ax1.set_xlabel("coupling epsilon")
    ax1.set_ylabel("fraction")
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(eps, hamming, "^-", color="tab:red", label="mean Hamming distance")
    ax2.set_ylabel("mean Hamming distance")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_discovery_summary(rows: list[dict], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eps = np.array([r["epsilon"] for r in rows], dtype=float)
    n_fixed = np.array([r["n_fixed_points_found"] for r in rows], dtype=float)
    largest = np.array([r["largest_basin_fraction"] for r in rows], dtype=float)

    fig, ax1 = plt.subplots(figsize=(7.5, 4.8))
    ax1.plot(eps, n_fixed, "o-", label="fixed points found")
    ax1.set_xscale("symlog", linthresh=1e-3)
    ax1.set_xlabel("coupling epsilon")
    ax1.set_ylabel("number of distinct terminal fixed points")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(eps, largest, "s-", color="tab:red", label="largest basin fraction")
    ax2.set_ylabel("largest basin fraction")
    ax2.set_ylim(-0.05, 1.05)

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def run_sweep(args: argparse.Namespace) -> tuple[list[PatternResult], list[dict]]:
    params = FNParams(a=args.a, b=args.b, c=args.c, z=args.z)
    left, middle, right = stable_memory_equilibria(params)

    A = graph_adjacency(
        args.graph,
        args.N,
        weight=args.weight,
        p_edge=args.p_edge,
        seed=args.seed,
    )
    L = laplacian_from_adjacency(A)
    rng = np.random.default_rng(args.seed)
    patterns = select_patterns(args.N, args.patterns, rng, args.random_patterns)
    cfg = SimulationConfig(
        t_final=args.t_final,
        max_step=args.max_step,
        rtol=args.rtol,
        atol=args.atol,
        jitter=args.jitter,
        refine=not args.no_refine,
    )

    print("Isolated motif equilibria:")
    for name, eq in [("left", left), ("middle", middle), ("right", right)]:
        print(
            f"  {name:>6s}: x={eq['x']:+.8f}, y={eq['y']:+.8f}, "
            f"stable={eq['stable']}, ndr={eq['ndr_branch']}, "
            f"max_real_eig={np.max(eq['eig'].real):+.6g}"
        )
    print(f"\nGraph={args.graph}, N={args.N}, patterns={len(patterns)}, eps={args.eps}")
    print("Running asymptotic-state sweep...")

    u = np.zeros(args.N, dtype=float)
    results: list[PatternResult] = []
    for eps in args.eps:
        for bits in patterns:
            results.append(simulate_pattern(bits, L, eps, params, cfg, rng, u=u))
        rows = summarize([r for r in results if math.isclose(r.epsilon, eps)])
        row = rows[0]
        print(
            f"  epsilon={eps:>8g}  recall={row['recall_fraction']:.3f}  "
            f"mean_hamming={row['mean_hamming']:.3f}  "
            f"stable={row['stable_fraction']:.3f}  "
            f"max_Re(lambda)={row['max_real_eig_worst']:+.4g}"
        )

    summary_rows = summarize(results)
    return results, summary_rows


def run_discovery(
    args: argparse.Namespace,
) -> tuple[list[FixedPointSample], dict[float, list[FixedPointCluster]], list[dict]]:
    params = FNParams(a=args.a, b=args.b, c=args.c, z=args.z)
    left, middle, right = stable_memory_equilibria(params)

    A = graph_adjacency(
        args.graph,
        args.N,
        weight=args.weight,
        p_edge=args.p_edge,
        seed=args.seed,
    )
    L = laplacian_from_adjacency(A)
    rng = np.random.default_rng(args.seed)
    cfg = SimulationConfig(
        t_final=args.t_final,
        max_step=args.max_step,
        rtol=args.rtol,
        atol=args.atol,
        jitter=args.jitter,
        refine=not args.no_refine,
    )
    x_range = (float(args.x_range[0]), float(args.x_range[1]))
    y_range = (float(args.y_range[0]), float(args.y_range[1]))

    print("Isolated motif equilibria:")
    for name, eq in [("left", left), ("middle", middle), ("right", right)]:
        print(
            f"  {name:>6s}: x={eq['x']:+.8f}, y={eq['y']:+.8f}, "
            f"stable={eq['stable']}, ndr={eq['ndr_branch']}, "
            f"max_real_eig={np.max(eq['eig'].real):+.6g}"
        )
    print(
        f"\nGraph={args.graph}, N={args.N}, random_initial_conditions={args.n_initial}, "
        f"x_range={x_range}, y_range={y_range}, eps={args.eps}"
    )
    print("Discovering fixed points from random initial conditions...")

    u = np.zeros(args.N, dtype=float)
    all_samples: list[FixedPointSample] = []
    clusters_by_eps: dict[float, list[FixedPointCluster]] = {}

    for eps in args.eps:
        eps_samples: list[FixedPointSample] = []
        for sample_index in range(args.n_initial):
            sample = simulate_random_initial_condition(
                sample_index,
                L,
                eps,
                params,
                cfg,
                rng,
                x_range,
                y_range,
                u=u,
            )
            eps_samples.append(sample)
        clusters = cluster_fixed_point_samples(eps_samples, args.cluster_tol, params)
        clusters_by_eps[float(eps)] = clusters
        all_samples.extend(eps_samples)

        counts = ", ".join(str(c.count) for c in clusters[:8])
        if len(clusters) > 8:
            counts += ", ..."
        print(
            f"  epsilon={eps:>8g}  fixed_points={len(clusters):>4d}  "
            f"stable={sum(c.stable for c in clusters):>4d}  "
            f"largest_basin={clusters[0].count / args.n_initial:.3f}  "
            f"basin_counts=[{counts}]"
        )

    rows = summarize_clusters(clusters_by_eps, args.n_initial)
    return all_samples, clusters_by_eps, rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["memory", "discover"],
        default="memory",
        help="memory: launch binary memories; discover: random initial conditions and cluster fixed points",
    )
    parser.add_argument("--N", type=int, default=8, help="number of NDR/FN memory motifs")
    parser.add_argument(
        "--graph",
        choices=["path", "ring", "complete", "star", "erdos"],
        default="ring",
        help="internal resistor-network topology",
    )
    parser.add_argument("--weight", type=float, default=1.0, help="edge weight")
    parser.add_argument("--p-edge", type=float, default=0.3, help="edge probability for erdos graph")
    parser.add_argument(
        "--eps",
        type=float,
        nargs="+",
        default=[0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
        help="coupling strengths epsilon to sweep",
    )
    parser.add_argument(
        "--patterns",
        choices=["all", "corners", "single-flip", "random"],
        default="all",
        help="which memory patterns to launch",
    )
    parser.add_argument("--random-patterns", type=int, default=128)
    parser.add_argument(
        "--n-initial",
        type=int,
        default=256,
        help="random initial conditions per epsilon for --mode discover",
    )
    parser.add_argument(
        "--cluster-tol",
        type=float,
        default=1e-5,
        help="max-norm tolerance for clustering terminal fixed points",
    )
    parser.add_argument(
        "--x-range",
        type=float,
        nargs=2,
        default=[-2.0, 2.0],
        metavar=("XMIN", "XMAX"),
        help="uniform random initial x range for --mode discover",
    )
    parser.add_argument(
        "--y-range",
        type=float,
        nargs=2,
        default=[-1.5, 1.5],
        metavar=("YMIN", "YMAX"),
        help="uniform random initial y range for --mode discover",
    )
    parser.add_argument("--a", type=float, default=0.0)
    parser.add_argument("--b", type=float, default=2.0)
    parser.add_argument("--c", type=float, default=5.0)
    parser.add_argument("--z", type=float, default=0.0)
    parser.add_argument("--t-final", type=float, default=250.0)
    parser.add_argument("--max-step", type=float, default=0.2)
    parser.add_argument("--rtol", type=float, default=1e-8)
    parser.add_argument("--atol", type=float, default=1e-10)
    parser.add_argument("--jitter", type=float, default=1e-3)
    parser.add_argument("--no-refine", action="store_true", help="skip final root refinement")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--out-dir", type=Path, default=Path("simulations/FN/output"))
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "memory":
        results, rows = run_sweep(args)

        summary_path = args.out_dir / "ndr_fn_memory_summary.csv"
        patterns_path = args.out_dir / "ndr_fn_memory_patterns.csv"
        write_summary_csv(rows, summary_path)
        write_patterns_csv(results, patterns_path)
        print(f"\nWrote {summary_path}")
        print(f"Wrote {patterns_path}")

        if args.plot:
            plot_path = args.out_dir / "ndr_fn_memory_summary.png"
            plot_summary(rows, plot_path)
            print(f"Wrote {plot_path}")
    else:
        samples, clusters_by_eps, rows = run_discovery(args)

        summary_path = args.out_dir / "ndr_fn_fixed_point_summary.csv"
        clusters_path = args.out_dir / "ndr_fn_fixed_point_clusters.csv"
        samples_path = args.out_dir / "ndr_fn_fixed_point_samples.csv"
        write_summary_csv(rows, summary_path)
        write_fixed_point_clusters_csv(clusters_by_eps, clusters_path)
        write_fixed_point_samples_csv(samples, samples_path)
        print(f"\nWrote {summary_path}")
        print(f"Wrote {clusters_path}")
        print(f"Wrote {samples_path}")

        if args.plot:
            plot_path = args.out_dir / "ndr_fn_fixed_point_summary.png"
            plot_discovery_summary(rows, plot_path)
            print(f"Wrote {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
