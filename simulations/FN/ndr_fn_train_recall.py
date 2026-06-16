"""
Train-and-recall simulator for Laplacian-coupled NDR/FitzHugh-Nagumo memories.

Model:

    dx/dt = c * (z + x - x^3/3 - y) - epsilon * L x + epsilon * u
    dy/dt = (x - a - b y) / c

The trainable object is the passive resistor mesh:

    L = B G B^T,   G = diag(g_e),   g_e >= 0.

Given target voltage patterns mu^p, training solves the fixed-point equations

    h(mu^p) - epsilon L mu^p + epsilon u = 0,

where h_i(mu) = c * (z + mu_i - mu_i^3/3 - (mu_i - a)/b).

Equivalently: epsilon L mu^p - epsilon u = h(mu^p),
which is linear in the conductances g_e (and optional shared bias u).

FIX (amplitude): the default amplitude is now sqrt(3/2) ≈ 1.2247, the exact
location of the natural stable equilibria of the isolated FN motif (for the
default a=0, b=2, c=5, z=0).  At this amplitude h(±sqrt(3/2)) = 0 exactly,
so a pure-passive mesh (no bias) can represent all-same-bit patterns with
zero residual.  You may still override with --amplitude.

FIX (Hurwitz): Hurwitz stability is now checked at the *network* fixed point
found by Newton refinement from mu^p, not at mu^p itself.

NEW: --plot now writes a comprehensive set of figures after the recall trials:
  pattern_grid.png         — all training patterns as a tiled image
  learned_adjacency.png    — heatmap of the learned conductance matrix
  conductance_histogram.png— histogram of edge conductances
  recall_hamming.png       — per-trial (input, final) Hamming scatter
  recall_heatmap.png       — exact-recall fraction per pattern × noise level
  recall_montage_page_*.png— target / corrupted / recalled / error image arrays
  recall_trajectories.png  — sample x(t) trajectories for one trial per pattern
  eigenvalue_spectra.png   — Jacobian eigenvalue scatter at network fixed points

Usage examples
--------------
# 4x4 grid, demo patterns, natural amplitude, no-bias:
python ndr_fn_train_recall.py \\
    --rows 4 --cols 4 \\
    --pattern-set demo \\
    --graph grid \\
    --epsilon 0.25 \\
    --flip-prob 0.10 \\
    --trials-per-pattern 25 \\
    --out-dir output_demo \\
    --plot

# Same but with trainable bias:
python ndr_fn_train_recall.py \\
    --rows 4 --cols 4 \\
    --pattern-set demo \\
    --graph grid \\
    --epsilon 0.25 \\
    --train-bias \\
    --flip-prob 0.10 \\
    --trials-per-pattern 25 \\
    --out-dir output_demo_bias \\
    --plot

# 8x8 digit prototypes, complete graph, with bias:
python ndr_fn_train_recall.py \\
    --rows 8 --cols 8 \\
    --pattern-set digits \\
    --digit-dataset auto \\
    --digit-source prototype \\
    --graph complete \\
    --epsilon 0.25 \\
    --train-bias \\
    --flip-prob 0.05 \\
    --trials-per-pattern 10 \\
    --out-dir output_digits \\
    --plot
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import lsq_linear, root


# ──────────────────────────────────────────────────────────────────────────────
# Parameter containers
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FNParams:
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
    refine: bool = True


@dataclass
class TrainingResult:
    patterns_bits: list[tuple[int, ...]]
    pattern_labels: list[str]
    epsilon: float
    params: FNParams
    target_voltages: np.ndarray
    edges: list[tuple[int, int]]
    conductances: np.ndarray
    adjacency: np.ndarray
    laplacian: np.ndarray
    bias: np.ndarray
    relative_residual: float
    max_pattern_residual: float
    pattern_residuals: np.ndarray       # residual at mu^p (training target)
    network_fp_residuals: np.ndarray    # residual at Newton-refined network FP
    max_real_eigs: np.ndarray           # evaluated at Newton-refined network FP
    hurwitz: np.ndarray
    network_fixed_points: np.ndarray    # shape (P, N) x-components
    success: bool


# ──────────────────────────────────────────────────────────────────────────────
# FitzHugh–Nagumo equilibrium analysis
# ──────────────────────────────────────────────────────────────────────────────

def natural_amplitude(params: FNParams) -> float:
    """Return the positive stable fixed-point x* of the isolated FN motif.

    For a=0, b=2, z=0 this is sqrt(3/2). In general it is the largest real
    root of h(x) = c*(z + x - x^3/3 - (x-a)/b) = 0.
    """
    eqs = isolated_equilibria(params)
    stable = [eq for eq in eqs if eq["stable"]]
    if not stable:
        raise ValueError("No stable isolated equilibria found.")
    return max(eq["x"] for eq in stable)


def isolated_equilibria(params: FNParams) -> list[dict]:
    if params.b == 0:
        raise ValueError("b must be nonzero.")
    coeffs = np.array(
        [-1.0 / 3.0, 0.0, 1.0 - 1.0 / params.b, params.z + params.a / params.b],
        dtype=float,
    )
    roots = np.roots(coeffs)
    real_roots = sorted(float(r.real) for r in roots if abs(r.imag) < 1e-9)
    eqs = []
    for x in real_roots:
        y = (x - params.a) / params.b
        J = isolated_jacobian(x, params)
        eig = np.linalg.eigvals(J)
        eqs.append({"x": x, "y": y, "eig": eig, "stable": bool(np.max(eig.real) < 0.0)})
    return eqs


def isolated_jacobian(x: float, params: FNParams) -> np.ndarray:
    return np.array(
        [[params.c * (1.0 - x * x), -params.c],
         [1.0 / params.c, -params.b / params.c]],
        dtype=float,
    )


def stable_memory_equilibria(params: FNParams) -> tuple[dict, dict, dict]:
    eq = isolated_equilibria(params)
    if len(eq) != 3:
        raise ValueError(
            "Parameters do not give three isolated equilibria. "
            "Use a bistable regime, e.g. a=0, b=2, z=0."
        )
    left, middle, right = eq
    if not (left["stable"] and right["stable"] and not middle["stable"]):
        raise ValueError("Expected stable–saddle–stable equilibria.")
    return left, middle, right


# ──────────────────────────────────────────────────────────────────────────────
# Graph constructors
# ──────────────────────────────────────────────────────────────────────────────

def grid_adjacency(rows: int, cols: int, weight: float = 1.0) -> np.ndarray:
    N = rows * cols
    A = np.zeros((N, N), dtype=float)
    def idx(r, c): return r * cols + c
    for r in range(rows):
        for c in range(cols):
            i = idx(r, c)
            if r + 1 < rows:
                j = idx(r + 1, c); A[i, j] = A[j, i] = weight
            if c + 1 < cols:
                j = idx(r, c + 1); A[i, j] = A[j, i] = weight
    return A


def complete_adjacency(N: int, weight: float = 1.0) -> np.ndarray:
    return weight * (np.ones((N, N)) - np.eye(N))


def ring_adjacency(N: int, weight: float = 1.0) -> np.ndarray:
    A = np.zeros((N, N), dtype=float)
    for i in range(N):
        A[i, (i + 1) % N] = weight
        A[(i + 1) % N, i] = weight
    return A


def path_adjacency(N: int, weight: float = 1.0) -> np.ndarray:
    A = np.zeros((N, N), dtype=float)
    for i in range(N - 1):
        A[i, i + 1] = weight; A[i + 1, i] = weight
    return A


def laplacian_from_adjacency(A: np.ndarray) -> np.ndarray:
    return np.diag(A.sum(axis=1)) - A


def edges_from_adjacency(A: np.ndarray) -> list[tuple[int, int]]:
    N = A.shape[0]
    return [(i, j) for i in range(N) for j in range(i + 1, N) if abs(A[i, j]) > 0.0]


def incidence_vector(N: int, edge: tuple[int, int]) -> np.ndarray:
    b = np.zeros(N, dtype=float)
    b[edge[0]] = 1.0; b[edge[1]] = -1.0
    return b


def build_template_adjacency(graph: str, rows: int, cols: int, weight: float) -> np.ndarray:
    N = rows * cols
    if graph == "grid":    return grid_adjacency(rows, cols, weight)
    if graph == "complete": return complete_adjacency(N, weight)
    if graph == "ring":    return ring_adjacency(N, weight)
    if graph == "path":    return path_adjacency(N, weight)
    raise ValueError(f"Unknown graph: {graph}")


# ──────────────────────────────────────────────────────────────────────────────
# Bit / voltage helpers
# ──────────────────────────────────────────────────────────────────────────────

def bits_to_voltage(bits: tuple[int, ...], amplitude: float) -> np.ndarray:
    return amplitude * (2.0 * np.array(bits, dtype=float) - 1.0)


def voltage_to_bits(x: np.ndarray) -> tuple[int, ...]:
    return tuple(int(v > 0.0) for v in x)


def hamming(a: tuple[int, ...], b: tuple[int, ...]) -> int:
    return sum(x != y for x, y in zip(a, b))


# ──────────────────────────────────────────────────────────────────────────────
# Network dynamics
# ──────────────────────────────────────────────────────────────────────────────

def local_equilibrium_force_x(x: np.ndarray, params: FNParams) -> np.ndarray:
    """h_i(x) = c*(z + x_i - x_i^3/3 - (x_i - a)/b)."""
    y_eq = (x - params.a) / params.b
    return params.c * (params.z + x - x ** 3 / 3.0 - y_eq)


def rhs(
    _t: float,
    state: np.ndarray,
    L: np.ndarray,
    epsilon: float,
    params: FNParams,
    u: np.ndarray | None = None,
) -> np.ndarray:
    N = L.shape[0]
    x, y = state[:N], state[N:]
    if u is None:
        u = np.zeros(N, dtype=float)
    xdot = params.c * (params.z + x - x ** 3 / 3.0 - y) - epsilon * (L @ x) + epsilon * u
    ydot = (x - params.a - params.b * y) / params.c
    return np.concatenate([xdot, ydot])


def network_jacobian(x: np.ndarray, L: np.ndarray, epsilon: float, params: FNParams) -> np.ndarray:
    N = L.shape[0]
    Jxx = np.diag(params.c * (1.0 - x * x)) - epsilon * L
    Jxy = -params.c * np.eye(N)
    Jyx = (1.0 / params.c) * np.eye(N)
    Jyy = -(params.b / params.c) * np.eye(N)
    return np.vstack([np.hstack([Jxx, Jxy]), np.hstack([Jyx, Jyy])])


def refine_equilibrium(
    state: np.ndarray,
    L: np.ndarray,
    epsilon: float,
    params: FNParams,
    u: np.ndarray | None,
) -> np.ndarray:
    fun = lambda s: rhs(0.0, s, L, epsilon, params, u)
    sol = root(fun, state, method="hybr", tol=1e-12)
    if sol.success and np.linalg.norm(fun(sol.x)) < np.linalg.norm(fun(state)):
        return np.asarray(sol.x, dtype=float)
    return state


def integrate_to_equilibrium(
    state0: np.ndarray,
    L: np.ndarray,
    epsilon: float,
    params: FNParams,
    cfg: SimulationConfig,
    u: np.ndarray | None = None,
    record_times: np.ndarray | None = None,
) -> tuple[np.ndarray, float, float, bool, np.ndarray | None]:
    """Integrate the network ODE to (near-)equilibrium.

    Returns
    -------
    state       : final state vector (2N,)
    residual    : ||F(state)||
    max_real_eig: largest real part of Jacobian eigenvalue at final state
    stable      : max_real_eig < 0
    trajectory  : array (len(record_times), 2N) if record_times is provided, else None
    """
    fun = lambda t, s: rhs(t, s, L, epsilon, params, u)
    traj = None

    if record_times is not None:
        t_eval = np.unique(np.concatenate([[0.0], record_times, [cfg.t_final]]))
        sol = solve_ivp(fun, (0.0, cfg.t_final), state0, method="LSODA",
                        t_eval=t_eval, max_step=cfg.max_step, rtol=cfg.rtol, atol=cfg.atol)
        # Interpolate back to requested times
        from scipy.interpolate import interp1d
        interp = interp1d(sol.t, sol.y, kind="linear", fill_value="extrapolate")
        traj = interp(record_times).T  # shape (len(record_times), 2N)
    else:
        sol = solve_ivp(fun, (0.0, cfg.t_final), state0, method="LSODA",
                        max_step=cfg.max_step, rtol=cfg.rtol, atol=cfg.atol)

    if not sol.success:
        raise RuntimeError(f"integration failed: {sol.message}")

    state = sol.y[:, -1]
    if cfg.refine:
        state = refine_equilibrium(state, L, epsilon, params, u)

    N = L.shape[0]
    x = state[:N]
    residual_norm = float(np.linalg.norm(rhs(0.0, state, L, epsilon, params, u)))
    eig = np.linalg.eigvals(network_jacobian(x, L, epsilon, params))
    max_real_eig = float(np.max(eig.real))
    return state, residual_norm, max_real_eig, max_real_eig < 0.0, traj


# ──────────────────────────────────────────────────────────────────────────────
# Pattern generators
# ──────────────────────────────────────────────────────────────────────────────

def grid_pattern_horizontal(rows, cols, row):
    bits = np.zeros((rows, cols), dtype=int); bits[row, :] = 1
    return tuple(int(v) for v in bits.ravel())

def grid_pattern_vertical(rows, cols, col):
    bits = np.zeros((rows, cols), dtype=int); bits[:, col] = 1
    return tuple(int(v) for v in bits.ravel())

def grid_pattern_checkerboard(rows, cols):
    bits = np.fromfunction(lambda i, j: (i + j) % 2, (rows, cols), dtype=int)
    return tuple(int(v) for v in bits.ravel())

def grid_pattern_inverse_checkerboard(rows, cols):
    bits = np.fromfunction(lambda i, j: 1 - ((i + j) % 2), (rows, cols), dtype=int)
    return tuple(int(v) for v in bits.ravel())

def grid_pattern_center_block(rows, cols):
    bits = np.zeros((rows, cols), dtype=int)
    bits[max(0, rows//4):min(rows, rows - rows//4),
         max(0, cols//4):min(cols, cols - cols//4)] = 1
    return tuple(int(v) for v in bits.ravel())

def grid_pattern_border(rows, cols):
    bits = np.zeros((rows, cols), dtype=int)
    bits[0, :] = bits[-1, :] = bits[:, 0] = bits[:, -1] = 1
    return tuple(int(v) for v in bits.ravel())


def make_patterns(rows: int, cols: int, pattern_set: str) -> list[tuple[int, ...]]:
    N = rows * cols
    if pattern_set == "demo":
        patterns = [grid_pattern_horizontal(rows, cols, rows // 2),
                    grid_pattern_vertical(rows, cols, cols // 2),
                    grid_pattern_center_block(rows, cols),
                    grid_pattern_border(rows, cols)]
    elif pattern_set == "smooth":
        patterns = [grid_pattern_horizontal(rows, cols, rows // 2),
                    grid_pattern_vertical(rows, cols, cols // 2),
                    grid_pattern_center_block(rows, cols)]
    elif pattern_set == "checker":
        patterns = [grid_pattern_checkerboard(rows, cols),
                    grid_pattern_inverse_checkerboard(rows, cols)]
    elif pattern_set == "corners":
        patterns = [tuple(0 for _ in range(N)), tuple(1 for _ in range(N))]
    elif pattern_set == "digits":
        raise ValueError("Use load_digit_patterns for pattern_set='digits'.")
    else:
        raise ValueError(f"Unknown pattern set: {pattern_set}")
    return sorted(set(patterns))


def load_digit_patterns(
    rows, cols, digit_classes, dataset, source, samples_per_class, threshold, seed
) -> tuple[list[tuple[int, ...]], list[str]]:
    if rows != 8 or cols != 8:
        raise ValueError("--pattern-set digits uses 8x8 digits; set --rows 8 --cols 8.")
    rng = np.random.default_rng(seed)
    if dataset in ("auto", "sklearn"):
        try:
            from sklearn.datasets import load_digits
            digits = load_digits()
            images, labels = digits.images.astype(float), digits.target.astype(int)
            source_name = "sklearn"
        except ImportError as exc:
            if dataset == "sklearn":
                raise RuntimeError("scikit-learn not installed.") from exc
            images, labels = synthetic_digit_dataset(max(1, samples_per_class), seed)
            source_name = "synthetic"
    elif dataset == "synthetic":
        images, labels = synthetic_digit_dataset(max(1, samples_per_class), seed)
        source_name = "synthetic"
    else:
        raise ValueError(f"Unknown digit dataset: {dataset}")

    patterns, pattern_labels = [], []
    for digit in digit_classes:
        idx = np.flatnonzero(labels == digit)
        if idx.size == 0:
            raise ValueError(f"Digit class {digit} not present in dataset.")
        if source == "prototype":
            image = images[idx].mean(axis=0)
            bits = tuple(int(v >= threshold) for v in image.ravel())
            patterns.append(bits)
            pattern_labels.append(f"digit_{digit}_{source_name}_prototype")
        elif source == "samples":
            n_choose = idx.size if samples_per_class <= 0 else samples_per_class
            if n_choose > idx.size:
                raise ValueError(f"Not enough samples for digit {digit}.")
            chosen = rng.choice(idx, size=n_choose, replace=False)
            for k, image_idx in enumerate(chosen):
                bits = tuple(int(v >= threshold) for v in images[image_idx].ravel())
                patterns.append(bits)
                pattern_labels.append(f"digit_{digit}_{source_name}_sample_{k}")
        else:
            raise ValueError(f"Unknown digit source: {source}")

    seen: set[tuple[int, ...]] = set()
    unique_patterns, unique_labels = [], []
    for bits, label in zip(patterns, pattern_labels):
        if bits not in seen:
            seen.add(bits); unique_patterns.append(bits); unique_labels.append(label)
    return unique_patterns, unique_labels


def synthetic_digit_dataset(samples_per_class: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    templates = {
        0: ["00111100","01100110","01101110","01110110","01100110","01100110","00111100","00000000"],
        1: ["00011000","00111000","00011000","00011000","00011000","00011000","00111100","00000000"],
        2: ["00111100","01100110","00000110","00001100","00110000","01100000","01111110","00000000"],
        3: ["00111100","01100110","00000110","00011100","00000110","01100110","00111100","00000000"],
        4: ["00001100","00011100","00101100","01001100","01111110","00001100","00011110","00000000"],
        5: ["01111110","01100000","01111100","00000110","00000110","01100110","00111100","00000000"],
        6: ["00111100","01100110","01100000","01111100","01100110","01100110","00111100","00000000"],
        7: ["01111110","01100110","00000110","00001100","00011000","00011000","00011000","00000000"],
        8: ["00111100","01100110","01100110","00111100","01100110","01100110","00111100","00000000"],
        9: ["00111100","01100110","01100110","00111110","00000110","01100110","00111100","00000000"],
    }
    rng = np.random.default_rng(seed)
    images, labels = [], []
    for digit, rows in templates.items():
        base = np.array([[16.0 if ch == "1" else 0.0 for ch in row] for row in rows], dtype=float)
        for k in range(samples_per_class):
            img = base.copy()
            if k > 0:
                flips = rng.random(img.shape) < 0.035
                img = np.where(flips, 16.0 - img, img)
            images.append(img); labels.append(digit)
    return np.stack(images), np.array(labels, dtype=int)


def load_patterns_csv(path: Path, expected_N: int) -> list[tuple[int, ...]]:
    patterns = []
    with path.open("r", newline="") as fh:
        for row in csv.reader(fh):
            if not row: continue
            if len(row) == 1 and set(row[0].strip()) <= {"0", "1"}:
                bits = tuple(int(ch) for ch in row[0].strip())
            else:
                bits = tuple(int(float(v)) for v in row)
            if len(bits) != expected_N:
                raise ValueError(f"Pattern length {len(bits)} != N={expected_N}.")
            patterns.append(bits)
    return patterns


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def build_training_system(
    target_voltages: np.ndarray,
    edges: list[tuple[int, int]],
    epsilon: float,
    params: FNParams,
    train_bias: bool,
) -> tuple[np.ndarray, np.ndarray]:
    P, N = target_voltages.shape
    E = len(edges)
    bs = [incidence_vector(N, edge) for edge in edges]
    n_unknowns = E + (N if train_bias else 0)
    A_train = np.zeros((P * N, n_unknowns), dtype=float)
    rhs_train = np.zeros(P * N, dtype=float)
    row = 0
    for p in range(P):
        mu = target_voltages[p]
        h = local_equilibrium_force_x(mu, params)
        for i in range(N):
            for e_idx, bvec in enumerate(bs):
                A_train[row, e_idx] = epsilon * bvec[i] * float(bvec @ mu)
            if train_bias:
                A_train[row, E + i] = -epsilon
            rhs_train[row] = h[i]
            row += 1
    return A_train, rhs_train


def train_resistive_mesh(
    patterns_bits: list[tuple[int, ...]],
    pattern_labels: list[str],
    A_template: np.ndarray,
    epsilon: float,
    params: FNParams,
    amplitude: float,
    train_bias: bool,
    conductance_upper: float | None,
) -> TrainingResult:
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive.")
    edges = edges_from_adjacency(A_template)
    if not edges:
        raise ValueError("Template graph has no trainable edges.")

    target_voltages = np.vstack([bits_to_voltage(b, amplitude) for b in patterns_bits])
    P, N = target_voltages.shape
    E = len(edges)

    A_train, rhs_train = build_training_system(target_voltages, edges, epsilon, params, train_bias)

    g_upper = np.inf if conductance_upper is None else float(conductance_upper)
    if train_bias:
        lower = np.concatenate([np.zeros(E), -np.inf * np.ones(N)])
        upper = np.concatenate([g_upper * np.ones(E), np.inf * np.ones(N)])
    else:
        lower, upper = np.zeros(E), g_upper * np.ones(E)

    sol = lsq_linear(A_train, rhs_train, bounds=(lower, upper),
                     tol=1e-12, lsmr_tol="auto", max_iter=10000)
    theta = sol.x
    conductances = theta[:E]
    bias = theta[E:] if train_bias else np.zeros(N, dtype=float)

    # Reconstruct adjacency and Laplacian
    A_learned = np.zeros_like(A_template, dtype=float)
    L_learned = np.zeros_like(A_template, dtype=float)
    for ge, edge in zip(conductances, edges):
        i, j = edge
        A_learned[i, j] = A_learned[j, i] = ge
        bvec = incidence_vector(N, edge)
        L_learned += ge * np.outer(bvec, bvec)

    # ── FIX: evaluate Hurwitz at Newton-refined network fixed points ──────────
    pattern_residuals = np.zeros(P)
    network_fp_residuals = np.zeros(P)
    max_real_eigs = np.zeros(P)
    hurwitz = np.zeros(P, dtype=bool)
    network_fixed_points = np.zeros((P, N))

    for p_idx, mu in enumerate(target_voltages):
        # Residual at the training target mu^p
        y_mu = (mu - params.a) / params.b
        fp_res = (params.c * (params.z + mu - mu ** 3 / 3.0 - y_mu)
                  - epsilon * (L_learned @ mu) + epsilon * bias)
        pattern_residuals[p_idx] = float(np.linalg.norm(fp_res))

        # Newton-refine to the true network fixed point nearest to mu^p
        state0 = np.concatenate([mu, y_mu])
        refined = refine_equilibrium(state0, L_learned, epsilon, params, bias)
        x_fp = refined[:N]
        network_fixed_points[p_idx] = x_fp

        fp_res2 = rhs(0.0, refined, L_learned, epsilon, params, bias)
        network_fp_residuals[p_idx] = float(np.linalg.norm(fp_res2))

        # Jacobian evaluated at the *refined* fixed point
        J = network_jacobian(x_fp, L_learned, epsilon, params)
        eig = np.linalg.eigvals(J)
        max_real_eigs[p_idx] = float(np.max(eig.real))
        hurwitz[p_idx] = max_real_eigs[p_idx] < 0.0

    relative_residual = float(
        np.linalg.norm(A_train @ theta - rhs_train) / max(1.0, np.linalg.norm(rhs_train))
    )

    return TrainingResult(
        patterns_bits=patterns_bits,
        pattern_labels=pattern_labels,
        epsilon=epsilon,
        params=params,
        target_voltages=target_voltages,
        edges=edges,
        conductances=conductances,
        adjacency=A_learned,
        laplacian=L_learned,
        bias=bias,
        relative_residual=relative_residual,
        max_pattern_residual=float(np.max(pattern_residuals)),
        pattern_residuals=pattern_residuals,
        network_fp_residuals=network_fp_residuals,
        max_real_eigs=max_real_eigs,
        hurwitz=hurwitz,
        network_fixed_points=network_fixed_points,
        success=bool(sol.success),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Recall trials
# ──────────────────────────────────────────────────────────────────────────────

def corrupt_bits(bits: tuple[int, ...], flip_prob: float, rng: np.random.Generator) -> tuple[int, ...]:
    arr = np.array(bits, dtype=int)
    flips = rng.random(arr.size) < flip_prob
    arr[flips] = 1 - arr[flips]
    return tuple(int(v) for v in arr)


def initial_state_from_bits(
    bits: tuple[int, ...], amplitude: float, params: FNParams,
    jitter: float, rng: np.random.Generator,
) -> np.ndarray:
    x = bits_to_voltage(bits, amplitude)
    y = (x - params.a) / params.b
    if jitter > 0.0:
        x += jitter * rng.standard_normal(x.shape)
        y += jitter * rng.standard_normal(y.shape)
    return np.concatenate([x, y])


def run_recall_trials(
    train: TrainingResult,
    epsilon: float,
    params: FNParams,
    cfg: SimulationConfig,
    amplitude: float,
    flip_prob: float,
    trials_per_pattern: int,
    seed: int,
    jitter: float,
    record_times: np.ndarray | None = None,
) -> tuple[list[dict], list[np.ndarray | None]]:
    """Run recall trials and optionally record trajectories.

    Returns
    -------
    rows        : list of per-trial result dicts
    trajectories: list parallel to rows; each entry is array (T, 2N) or None
    """
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    trajectories: list[np.ndarray | None] = []

    for p_idx, bits in enumerate(train.patterns_bits):
        for trial in range(trials_per_pattern):
            noisy_bits = corrupt_bits(bits, flip_prob, rng)
            state0 = initial_state_from_bits(noisy_bits, amplitude, params, jitter, rng)

            state, residual_norm, max_real_eig, stable, traj = integrate_to_equilibrium(
                state0=state0,
                L=train.laplacian,
                epsilon=epsilon,
                params=params,
                cfg=cfg,
                u=train.bias,
                record_times=record_times,
            )

            N = len(bits)
            x_final = state[:N]
            final_bits = voltage_to_bits(x_final)
            ham_to_target = hamming(bits, final_bits)
            ham_from_noisy = hamming(bits, noisy_bits)

            rows.append({
                "pattern_id": p_idx,
                "pattern_label": train.pattern_labels[p_idx],
                "trial": trial,
                "target_bits":  "".join(map(str, bits)),
                "noisy_bits":   "".join(map(str, noisy_bits)),
                "final_bits":   "".join(map(str, final_bits)),
                "input_hamming":  ham_from_noisy,
                "final_hamming":  ham_to_target,
                "exact_recall":   int(ham_to_target == 0),
                "residual_norm":  residual_norm,
                "max_real_eig":   max_real_eig,
                "stable":         int(stable),
                "x_final": " ".join(f"{v:.8g}" for v in x_final),
                "y_final": " ".join(f"{v:.8g}" for v in state[N:]),
            })
            trajectories.append(traj)

    return rows, trajectories


# ──────────────────────────────────────────────────────────────────────────────
# CSV / npy output
# ──────────────────────────────────────────────────────────────────────────────

def write_training_outputs(train: TrainingResult, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "learned_laplacian.npy", train.laplacian)
    np.save(out_dir / "learned_adjacency.npy", train.adjacency)
    np.save(out_dir / "learned_bias.npy", train.bias)
    np.save(out_dir / "target_voltages.npy", train.target_voltages)
    np.save(out_dir / "network_fixed_points.npy", train.network_fixed_points)

    with (out_dir / "learned_edges.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["edge_id","i","j","conductance"])
        w.writeheader()
        for e_idx, ((i, j), g) in enumerate(zip(train.edges, train.conductances)):
            w.writerow({"edge_id": e_idx, "i": i, "j": j, "conductance": f"{g:.12g}"})

    with (out_dir / "training_patterns.csv").open("w", newline="") as fh:
        fields = ["pattern_id","pattern_label","bits","fp_residual_at_mu",
                  "network_fp_residual","max_real_eig","hurwitz","target_voltage"]
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for idx, bits in enumerate(train.patterns_bits):
            w.writerow({
                "pattern_id": idx,
                "pattern_label": train.pattern_labels[idx],
                "bits": "".join(map(str, bits)),
                "fp_residual_at_mu": f"{train.pattern_residuals[idx]:.12g}",
                "network_fp_residual": f"{train.network_fp_residuals[idx]:.12g}",
                "max_real_eig": f"{train.max_real_eigs[idx]:.12g}",
                "hurwitz": int(train.hurwitz[idx]),
                "target_voltage": " ".join(f"{v:.8g}" for v in train.target_voltages[idx]),
            })

    with (out_dir / "training_summary.csv").open("w", newline="") as fh:
        fields = ["success","relative_residual","max_pattern_residual",
                  "max_network_fp_residual","n_patterns","n_edges",
                  "n_nonzero_edges","sum_conductance","all_hurwitz"]
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerow({
            "success": int(train.success),
            "relative_residual": f"{train.relative_residual:.12g}",
            "max_pattern_residual": f"{train.max_pattern_residual:.12g}",
            "max_network_fp_residual": f"{float(np.max(train.network_fp_residuals)):.12g}",
            "n_patterns": len(train.patterns_bits),
            "n_edges": len(train.edges),
            "n_nonzero_edges": int(np.sum(train.conductances > 1e-10)),
            "sum_conductance": f"{float(np.sum(train.conductances)):.12g}",
            "all_hurwitz": int(np.all(train.hurwitz)),
        })


def write_recall_csv(rows: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with (out_dir / "recall_trials.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    pattern_ids = sorted(set(r["pattern_id"] for r in rows))
    summary_rows = []
    for pid in pattern_ids:
        group = [r for r in rows if r["pattern_id"] == pid]
        summary_rows.append({
            "pattern_id": pid,
            "pattern_label": group[0].get("pattern_label", str(pid)),
            "n_trials": len(group),
            "exact_recall_fraction": float(np.mean([r["exact_recall"] for r in group])),
            "mean_input_hamming":    float(np.mean([r["input_hamming"] for r in group])),
            "mean_final_hamming":    float(np.mean([r["final_hamming"] for r in group])),
            "max_final_hamming":     int(max(r["final_hamming"] for r in group)),
            "stable_fraction":       float(np.mean([r["stable"] for r in group])),
            "max_residual":          float(max(r["residual_norm"] for r in group)),
            "worst_max_real_eig":    float(max(r["max_real_eig"] for r in group)),
        })
    with (out_dir / "recall_summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
        w.writeheader(); w.writerows(summary_rows)


# ──────────────────────────────────────────────────────────────────────────────
# Figures
# ──────────────────────────────────────────────────────────────────────────────

def _setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 150,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
    })
    return plt


def fig_pattern_grid(train: TrainingResult, rows: int, cols: int, out_dir: Path,
                     page_size: int = 80) -> list[Path]:
    """Paged tiles of all training patterns."""
    plt = _setup_mpl()
    P = len(train.patterns_bits)
    page_size = max(1, int(page_size))
    written: list[Path] = []
    for page_idx, start in enumerate(range(0, P, page_size), start=1):
        page_bits = train.patterns_bits[start:start + page_size]
        ncols = min(len(page_bits), 8)
        nrows = (len(page_bits) + ncols - 1) // ncols
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(1.75 * ncols, 1.9 * nrows), squeeze=False
        )
        for ax in axes.ravel():
            ax.axis("off")
        for local_idx, bits in enumerate(page_bits):
            idx = start + local_idx
            ax = axes[local_idx // ncols][local_idx % ncols]
            arr = np.array(bits, dtype=float).reshape(rows, cols)
            ax.imshow(arr, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
            label = train.pattern_labels[idx]
            ax.set_title(label if len(label) < 20 else label[:18] + "...", fontsize=7)
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"Training patterns page {page_idx}", fontweight="bold")
        fig.tight_layout()
        out_path = out_dir / ("pattern_grid.png" if P <= page_size else f"pattern_grid_page_{page_idx:03d}.png")
        fig.savefig(out_path)
        plt.close(fig)
        written.append(out_path)
    return written


def fig_learned_adjacency(train: TrainingResult, out_dir: Path) -> None:
    """Heatmap of the learned conductance adjacency matrix."""
    plt = _setup_mpl()
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(train.adjacency, cmap="viridis", interpolation="nearest")
    fig.colorbar(im, ax=ax, label="Conductance g_e")
    ax.set_title("Learned conductance adjacency matrix")
    ax.set_xlabel("Node j"); ax.set_ylabel("Node i")
    fig.tight_layout()
    fig.savefig(out_dir / "learned_adjacency.png")
    plt.close(fig)


def fig_conductance_histogram(train: TrainingResult, out_dir: Path) -> None:
    """Histogram of nonzero edge conductances."""
    plt = _setup_mpl()
    g = train.conductances
    nonzero = g[g > 1e-10]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

    axes[0].hist(g, bins=30, color="steelblue", edgecolor="white", linewidth=0.4)
    axes[0].set_title("All edge conductances")
    axes[0].set_xlabel("g_e"); axes[0].set_ylabel("Count")
    axes[0].axvline(0, color="k", linewidth=0.8, linestyle="--")

    if nonzero.size:
        axes[1].hist(nonzero, bins=20, color="darkorange", edgecolor="white", linewidth=0.4)
        axes[1].set_title(f"Non-zero conductances (>{len(nonzero)}/{len(g)})")
        axes[1].set_xlabel("g_e"); axes[1].set_ylabel("Count")
    else:
        axes[1].text(0.5, 0.5, "No nonzero conductances", ha="center", va="center")

    fig.suptitle("Learned conductance distribution", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "conductance_histogram.png")
    plt.close(fig)


def fig_recall_hamming(rows_data: list[dict], train: TrainingResult, out_dir: Path) -> None:
    """Scatter: input Hamming vs final Hamming, coloured by pattern, with jitter."""
    plt = _setup_mpl()
    import matplotlib.cm as cm

    P = len(train.patterns_bits)
    colors = cm.tab10(np.linspace(0, 1, max(P, 1)))
    rng = np.random.default_rng(0)

    fig, ax = plt.subplots(figsize=(6, 5))
    handles = []
    for p_idx in range(P):
        group = [r for r in rows_data if r["pattern_id"] == p_idx]
        x_vals = np.array([r["input_hamming"] for r in group], dtype=float)
        y_vals = np.array([r["final_hamming"] for r in group], dtype=float)
        x_jit = x_vals + rng.uniform(-0.25, 0.25, x_vals.shape)
        y_jit = y_vals + rng.uniform(-0.25, 0.25, y_vals.shape)
        sc = ax.scatter(x_jit, y_jit, c=[colors[p_idx % 10]], alpha=0.6, s=25, label=train.pattern_labels[p_idx])
        handles.append(sc)

    max_ham = max((r["input_hamming"] for r in rows_data), default=1)
    ax.plot([0, max_ham], [0, max_ham], "k--", linewidth=0.8, label="no correction")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    ax.set_xlabel("Input Hamming distance (noisy → target)")
    ax.set_ylabel("Final Hamming distance (recalled → target)")
    ax.set_title("Recall quality: input vs output Hamming distance")
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / "recall_hamming.png")
    plt.close(fig)


def fig_recall_heatmap(rows_data: list[dict], train: TrainingResult, out_dir: Path) -> None:
    """Grid: exact-recall fraction per pattern (rows) × input Hamming (cols)."""
    plt = _setup_mpl()

    P = len(train.patterns_bits)
    all_ham = sorted(set(r["input_hamming"] for r in rows_data))
    if not all_ham:
        return

    matrix = np.full((P, len(all_ham)), np.nan)
    for p_idx in range(P):
        for h_idx, h in enumerate(all_ham):
            group = [r for r in rows_data if r["pattern_id"] == p_idx and r["input_hamming"] == h]
            if group:
                matrix[p_idx, h_idx] = float(np.mean([r["exact_recall"] for r in group]))

    fig, ax = plt.subplots(figsize=(max(5, len(all_ham) * 0.5 + 1.5), max(3, P * 0.55 + 1.2)))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    fig.colorbar(im, ax=ax, label="Exact recall fraction")
    ax.set_xticks(range(len(all_ham))); ax.set_xticklabels(all_ham)
    ax.set_yticks(range(P))
    ylabels = [lb if len(lb) < 22 else lb[:20] + "…" for lb in train.pattern_labels]
    ax.set_yticklabels(ylabels, fontsize=7)
    ax.set_xlabel("Input Hamming distance"); ax.set_ylabel("Pattern")
    ax.set_title("Exact recall fraction (green=perfect, red=failed)")
    fig.tight_layout()
    fig.savefig(out_dir / "recall_heatmap.png")
    plt.close(fig)


def fig_recall_bar(rows_data: list[dict], train: TrainingResult, out_dir: Path) -> None:
    """Bar chart of exact-recall fraction per pattern."""
    plt = _setup_mpl()

    P = len(train.patterns_bits)
    fractions = []
    for p_idx in range(P):
        group = [r for r in rows_data if r["pattern_id"] == p_idx]
        fractions.append(float(np.mean([r["exact_recall"] for r in group])) if group else 0.0)

    fig, ax = plt.subplots(figsize=(max(4, P * 0.7 + 1.5), 3.5))
    bars = ax.bar(range(P), fractions, color="steelblue", edgecolor="white")
    for bar, frac in zip(bars, fractions):
        ax.text(bar.get_x() + bar.get_width() / 2, frac + 0.02,
                f"{frac:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(P))
    xlabels = [lb if len(lb) < 16 else lb[:14] + "…" for lb in train.pattern_labels]
    ax.set_xticklabels(xlabels, rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Exact recall fraction")
    ax.set_title("Exact recall fraction per pattern")
    ax.axhline(1.0, color="green", linewidth=0.8, linestyle="--")
    fig.tight_layout()
    fig.savefig(out_dir / "recall_bar.png")
    plt.close(fig)


def _bits_from_string(bits: str) -> tuple[int, ...]:
    return tuple(int(ch) for ch in bits.strip())


def fig_recall_montage(
    rows_data: list[dict],
    train: TrainingResult,
    rows: int,
    cols: int,
    out_dir: Path,
    page_size: int,
    montage_cols: int,
) -> list[Path]:
    """Paged target/input/output/error image arrays for all trained patterns.

    One representative recall trial is shown per pattern. The representative is
    the first trial for that pattern, so for large digit training sets use
    --trials-per-pattern 1 to make the montage exactly one row per memory.
    """
    if not rows_data:
        return []

    plt = _setup_mpl()
    rows_by_pattern: dict[int, list[dict]] = {}
    for row in rows_data:
        rows_by_pattern.setdefault(int(row["pattern_id"]), []).append(row)

    representative_rows: list[dict] = []
    for pattern_id in range(len(train.patterns_bits)):
        group = rows_by_pattern.get(pattern_id, [])
        if group:
            representative_rows.append(group[0])

    if not representative_rows:
        return []

    page_size = max(1, int(page_size))
    montage_cols = max(1, int(montage_cols))
    written: list[Path] = []

    for page_idx, start in enumerate(range(0, len(representative_rows), page_size), start=1):
        page = representative_rows[start:start + page_size]
        item_rows = int(np.ceil(len(page) / montage_cols))
        fig, axes = plt.subplots(
            item_rows * 4,
            montage_cols,
            figsize=(2.05 * montage_cols, 1.95 * item_rows * 4),
            squeeze=False,
        )
        for ax in axes.ravel():
            ax.axis("off")

        for local_idx, row in enumerate(page):
            grid_c = local_idx % montage_cols
            block_r = (local_idx // montage_cols) * 4
            pattern_id = int(row["pattern_id"])
            label = str(row.get("pattern_label", f"pattern_{pattern_id}"))
            if len(label) > 24:
                label = label[:22] + "..."

            target = np.array(_bits_from_string(row["target_bits"]), dtype=float).reshape(rows, cols)
            noisy = np.array(_bits_from_string(row["noisy_bits"]), dtype=float).reshape(rows, cols)
            final = np.array(_bits_from_string(row["final_bits"]), dtype=float).reshape(rows, cols)
            error = np.abs(final - target)

            panels = [
                ("target", target, "Blues", 0.0, 1.0),
                ("input", noisy, "Blues", 0.0, 1.0),
                ("recall", final, "Blues", 0.0, 1.0),
                ("error", error, "Reds", 0.0, 1.0),
            ]

            for panel_idx, (panel_name, image, cmap, vmin, vmax) in enumerate(panels):
                ax = axes[block_r + panel_idx][grid_c]
                ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
                ax.set_xticks([])
                ax.set_yticks([])
                if panel_idx == 0:
                    ax.set_title(
                        f"{pattern_id}: {label}\n"
                        f"in={row['input_hamming']} out={row['final_hamming']}",
                        fontsize=7,
                    )
                else:
                    ax.text(
                        0.02,
                        0.98,
                        panel_name,
                        transform=ax.transAxes,
                        ha="left",
                        va="top",
                        fontsize=6,
                        color="black",
                        bbox=dict(facecolor="white", alpha=0.65, edgecolor="none", pad=1.0),
                    )

        fig.suptitle(
            f"Recall montage page {page_idx}: target / corrupted input / recalled / error",
            fontweight="bold",
        )
        fig.tight_layout()
        out_path = out_dir / f"recall_montage_page_{page_idx:03d}.png"
        fig.savefig(out_path, bbox_inches="tight", dpi=170)
        plt.close(fig)
        written.append(out_path)

    return written


def fig_recall_trajectories(
    rows_data: list[dict],
    trajectories: list[np.ndarray | None],
    record_times: np.ndarray,
    train: TrainingResult,
    rows: int,
    cols: int,
    out_dir: Path,
) -> None:
    """For each pattern, plot x_i(t) for the first successful and first failed trial.

    Left column: trajectories coloured by node index.
    Right column: final recalled pattern as a grid image.
    """
    plt = _setup_mpl()
    import matplotlib.cm as cm

    P = len(train.patterns_bits)
    N = rows * cols
    node_colors = cm.tab20(np.linspace(0, 1, N))

    fig, axes = plt.subplots(P, 3,
                             figsize=(11, 2.8 * P),
                             gridspec_kw={"width_ratios": [4, 1.2, 1.2]},
                             squeeze=False)

    trial_cursor = 0
    for p_idx in range(P):
        # Collect trial indices for this pattern
        p_rows = [(i, r) for i, r in enumerate(rows_data) if r["pattern_id"] == p_idx]
        successes = [(i, r) for i, r in p_rows if r["exact_recall"] == 1]
        failures  = [(i, r) for i, r in p_rows if r["exact_recall"] == 0]

        ax_traj = axes[p_idx][0]
        ax_succ = axes[p_idx][1]
        ax_fail = axes[p_idx][2]

        # Target pattern background line (dashed)
        mu = train.target_voltages[p_idx]

        plotted = False
        for label_str, candidates, linestyle in [
            ("success", successes, "-"),
            ("failure", failures, "--"),
        ]:
            if not candidates:
                continue
            i_trial, r = candidates[0]
            traj = trajectories[i_trial]
            if traj is None:
                continue
            x_traj = traj[:, :N]   # shape (T, N)
            alpha = 0.85 if label_str == "success" else 0.55
            lw    = 1.2 if label_str == "success" else 1.0
            for n in range(N):
                ax_traj.plot(record_times, x_traj[:, n],
                             color=node_colors[n], alpha=alpha,
                             linewidth=lw, linestyle=linestyle)
            plotted = True

        # Reference horizontal lines for ±amplitude
        amp = float(np.max(np.abs(train.target_voltages)))
        ax_traj.axhline(amp,  color="gray", linewidth=0.6, linestyle=":")
        ax_traj.axhline(-amp, color="gray", linewidth=0.6, linestyle=":")
        ax_traj.axhline(0,    color="gray", linewidth=0.4, linestyle=":")
        ax_traj.set_xlabel("t")
        ax_traj.set_ylabel("x_i(t)")
        short_label = train.pattern_labels[p_idx]
        if len(short_label) > 28: short_label = short_label[:26] + "…"
        ax_traj.set_title(f"Pattern {p_idx}: {short_label}  (solid=success, dashed=fail)")

        # Target pattern grid
        target_arr = np.array(train.patterns_bits[p_idx], dtype=float).reshape(rows, cols)
        ax_succ.imshow(target_arr, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
        ax_succ.set_title("Target", fontsize=8)
        ax_succ.set_xticks([]); ax_succ.set_yticks([])

        # Best recalled pattern (from first success, else first trial)
        shown = successes[0] if successes else p_rows[0]
        i_trial, r = shown
        final_bits = tuple(int(ch) for ch in r["final_bits"])
        final_arr = np.array(final_bits, dtype=float).reshape(rows, cols)
        ax_fail.imshow(final_arr, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
        recall_label = "Recalled\n(success)" if successes else "Recalled\n(failed)"
        ax_fail.set_title(recall_label, fontsize=8)
        ax_fail.set_xticks([]); ax_fail.set_yticks([])

    fig.suptitle("Recall trajectories per pattern", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "recall_trajectories.png", bbox_inches="tight")
    plt.close(fig)


def fig_eigenvalue_spectra(train: TrainingResult, out_dir: Path) -> None:
    """Scatter of Jacobian eigenvalues at Newton-refined network fixed points.

    Each pattern gets its own colour.  The dashed vertical line marks Re=0.
    """
    plt = _setup_mpl()
    import matplotlib.cm as cm

    P = len(train.patterns_bits)
    colors = cm.tab10(np.linspace(0, 1, max(P, 1)))
    N = train.laplacian.shape[0]

    fig, ax = plt.subplots(figsize=(7, 4))
    for p_idx, x_fp in enumerate(train.network_fixed_points):
        J = network_jacobian(x_fp, train.laplacian, train.epsilon, train.params)
        eig = np.linalg.eigvals(J)
        ax.scatter(eig.real, eig.imag, color=colors[p_idx % 10], alpha=0.7,
                   s=18, label=train.pattern_labels[p_idx])

    ax.axvline(0, color="k", linewidth=0.8, linestyle="--")
    ax.axhline(0, color="gray", linewidth=0.4)
    ax.set_xlabel("Re(λ)")
    ax.set_ylabel("Im(λ)")
    ax.set_title("Jacobian eigenvalue spectra at network fixed points")
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "eigenvalue_spectra.png")
    plt.close(fig)


def fig_convergence_residuals(rows_data: list[dict], train: TrainingResult, out_dir: Path) -> None:
    """Box plot of ODE residual norms across trials per pattern."""
    plt = _setup_mpl()

    P = len(train.patterns_bits)
    data = []
    for p_idx in range(P):
        group = [r for r in rows_data if r["pattern_id"] == p_idx]
        data.append([r["residual_norm"] for r in group])

    fig, ax = plt.subplots(figsize=(max(4, P * 0.8 + 1.5), 3.5))
    bp = ax.boxplot(data, patch_artist=True,
                    boxprops=dict(facecolor="lightsteelblue", color="steelblue"),
                    medianprops=dict(color="darkblue", linewidth=1.5),
                    whiskerprops=dict(color="steelblue"),
                    capprops=dict(color="steelblue"),
                    flierprops=dict(marker="o", markersize=3, color="gray", alpha=0.5))
    ax.set_yscale("log")
    ax.set_xticks(range(1, P + 1))
    xlabels = [lb if len(lb) < 16 else lb[:14] + "…" for lb in train.pattern_labels]
    ax.set_xticklabels(xlabels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("||F(x_final)|| (log scale)")
    ax.set_title("ODE fixed-point residual norms across recall trials")
    ax.axhline(1e-6, color="green", linewidth=0.8, linestyle="--", label="1e-6 ref")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "convergence_residuals.png")
    plt.close(fig)


def make_all_plots(
    train: TrainingResult,
    rows_data: list[dict],
    trajectories: list[np.ndarray | None],
    record_times: np.ndarray,
    grid_rows: int,
    grid_cols: int,
    out_dir: Path,
    montage_page_size: int,
    montage_cols: int,
) -> None:
    print("  Writing figures …")
    pattern_paths = fig_pattern_grid(train, grid_rows, grid_cols, out_dir)
    for path in pattern_paths:
        print(f"    {path.name}")
    fig_learned_adjacency(train, out_dir)
    print("    learned_adjacency.png")
    fig_conductance_histogram(train, out_dir)
    print("    conductance_histogram.png")
    fig_recall_hamming(rows_data, train, out_dir)
    print("    recall_hamming.png")
    fig_recall_heatmap(rows_data, train, out_dir)
    print("    recall_heatmap.png")
    fig_recall_bar(rows_data, train, out_dir)
    print("    recall_bar.png")
    montage_paths = fig_recall_montage(
        rows_data,
        train,
        grid_rows,
        grid_cols,
        out_dir,
        page_size=montage_page_size,
        montage_cols=montage_cols,
    )
    for path in montage_paths:
        print(f"    {path.name}")
    if any(t is not None for t in trajectories):
        fig_recall_trajectories(rows_data, trajectories, record_times, train,
                                grid_rows, grid_cols, out_dir)
        print("    recall_trajectories.png")
    fig_eigenvalue_spectra(train, out_dir)
    print("    eigenvalue_spectra.png")
    fig_convergence_residuals(rows_data, train, out_dir)
    print("    convergence_residuals.png")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # Grid
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--cols", type=int, default=8)
    parser.add_argument("--graph", choices=["grid","complete","ring","path"], default="complete")
    parser.add_argument("--weight", type=float, default=1.0)
    # Patterns
    parser.add_argument("--pattern-set",
                        choices=["demo","smooth","checker","corners","digits"], default="digits")
    parser.add_argument("--patterns-csv", type=Path, default=None)
    parser.add_argument("--digit-source", choices=["prototype","samples"], default="prototype")
    parser.add_argument("--digit-dataset", choices=["auto","sklearn","synthetic"], default="auto")
    parser.add_argument("--digit-classes", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--digit-samples-per-class", type=int, default=1)
    parser.add_argument("--digit-threshold", type=float, default=7.5)
    # Physics
    parser.add_argument("--epsilon", type=float, default=0.25)
    parser.add_argument(
        "--amplitude", type=float, default=None,
        help="Target voltage amplitude.  Default: sqrt(3/2) ≈ 1.2247 (natural FN equilibrium). "
             "Set explicitly only if you have changed a, b, c, z.",
    )
    parser.add_argument("--train-bias", action="store_true")
    parser.add_argument("--conductance-upper", type=float, default=None)
    parser.add_argument("--a", type=float, default=0.0)
    parser.add_argument("--b", type=float, default=2.0)
    parser.add_argument("--c", type=float, default=5.0)
    parser.add_argument("--z", type=float, default=0.0)
    # Recall
    parser.add_argument("--flip-prob", type=float, default=0.05)
    parser.add_argument("--trials-per-pattern", type=int, default=1)
    parser.add_argument("--jitter", type=float, default=0.0)
    # Simulation
    parser.add_argument("--t-final", type=float, default=250.0)
    parser.add_argument("--max-step", type=float, default=0.2)
    parser.add_argument("--rtol", type=float, default=1e-8)
    parser.add_argument("--atol", type=float, default=1e-10)
    parser.add_argument("--no-refine", action="store_true")
    parser.add_argument(
        "--n-traj-points", type=int, default=300,
        help="Number of time points recorded for trajectory figures when --record-trajectories is set.",
    )
    parser.add_argument(
        "--record-trajectories",
        action="store_true",
        help="Record x(t) traces for recall_trajectories.png. Slower for 64-node digit runs.",
    )
    parser.add_argument(
        "--montage-page-size",
        type=int,
        default=40,
        help="Number of pattern recalls per recall_montage page.",
    )
    parser.add_argument(
        "--montage-cols",
        type=int,
        default=5,
        help="Number of pattern columns in recall_montage pages.",
    )
    # Output
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--out-dir", type=Path, default=Path("simulations/FN/output_learning"))
    parser.add_argument("--plot", action="store_true")

    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    N = args.rows * args.cols
    params = FNParams(a=args.a, b=args.b, c=args.c, z=args.z)
    cfg = SimulationConfig(
        t_final=args.t_final, max_step=args.max_step,
        rtol=args.rtol, atol=args.atol, refine=not args.no_refine,
    )

    # ── Determine amplitude ───────────────────────────────────────────────────
    left, middle, right = stable_memory_equilibria(params)
    if args.amplitude is None:
        amplitude = natural_amplitude(params)
        print(f"Using natural amplitude sqrt(3/2) = {amplitude:.10f}  "
              f"(h(±A) = {local_equilibrium_force_x(np.array([amplitude]), params)[0]:.2e})")
    else:
        amplitude = args.amplitude
        print(f"Using user-specified amplitude = {amplitude}")

    print("\nIsolated motif equilibria:")
    for name, eq in [("left", left), ("middle", middle), ("right", right)]:
        print(f"  {name:>6s}: x={eq['x']:+.8f}, y={eq['y']:+.8f}, "
              f"stable={eq['stable']}, max_Re={np.max(eq['eig'].real):+.6g}")

    # ── Load patterns ─────────────────────────────────────────────────────────
    A_template = build_template_adjacency(args.graph, args.rows, args.cols, args.weight)
    if args.patterns_csv is not None:
        patterns = load_patterns_csv(args.patterns_csv, expected_N=N)
        pattern_labels = [f"csv_{idx}" for idx in range(len(patterns))]
    elif args.pattern_set == "digits":
        patterns, pattern_labels = load_digit_patterns(
            rows=args.rows, cols=args.cols, digit_classes=args.digit_classes,
            dataset=args.digit_dataset, source=args.digit_source,
            samples_per_class=args.digit_samples_per_class,
            threshold=args.digit_threshold, seed=args.seed,
        )
    else:
        patterns = make_patterns(args.rows, args.cols, args.pattern_set)
        pattern_labels = [f"{args.pattern_set}_{idx}" for idx in range(len(patterns))]

    print(f"\nTraining N={N} nodes on {len(patterns)} patterns.")
    if args.pattern_set == "digits" and args.patterns_csv is None:
        digit_source_used = "sklearn" if any("_sklearn_" in label for label in pattern_labels) else "synthetic"
        print(
            f"Digit setup: {args.rows}x{args.cols} = {N} nodes, "
            f"dataset={digit_source_used}, mode={args.digit_source}, "
            f"classes={args.digit_classes}, threshold={args.digit_threshold}"
        )
    print(f"Graph={args.graph}, available edges={len(edges_from_adjacency(A_template))}")
    print(f"epsilon={args.epsilon}, amplitude={amplitude:.6g}, train_bias={args.train_bias}")

    # ── Train ─────────────────────────────────────────────────────────────────
    train = train_resistive_mesh(
        patterns_bits=patterns, pattern_labels=pattern_labels,
        A_template=A_template, epsilon=args.epsilon, params=params,
        amplitude=amplitude, train_bias=args.train_bias,
        conductance_upper=args.conductance_upper,
    )

    print("\nTraining result:")
    print(f"  optimizer success       : {train.success}")
    print(f"  relative residual       : {train.relative_residual:.6e}")
    print(f"  max |h(mu^p) - ε L mu^p + ε u|   : {train.max_pattern_residual:.6e}  ← residual at mu^p")
    print(f"  max |F(x*)|  (at network FP)      : {float(np.max(train.network_fp_residuals)):.6e}  ← after Newton refinement")
    print(f"  nonzero conductances    : {np.sum(train.conductances > 1e-10)} / {len(train.conductances)}")
    print(f"  total conductance       : {float(np.sum(train.conductances)):.6g}")
    print(f"  all Jacobians Hurwitz   : {bool(np.all(train.hurwitz))}  (evaluated at Newton-refined FPs)")
    print(f"  worst max_Re(λ) at FP   : {float(np.max(train.max_real_eigs)):+.6g}")

    if train.max_pattern_residual > 1e-4:
        print("\n  ⚠  Training residual at mu^p is large.")
        print("     Try --train-bias, --graph complete, or fewer patterns.")
        print(f"     Residual at Newton-refined FP: {float(np.max(train.network_fp_residuals)):.2e}")
        print("     (If this is small, the patterns ARE stored — just shifted off mu^p.)")

    write_training_outputs(train, args.out_dir)

    # ── Recall ────────────────────────────────────────────────────────────────
    record_times = (
        np.linspace(0.0, cfg.t_final, args.n_traj_points)
        if args.plot and args.record_trajectories
        else None
    )

    recall_rows, trajectories = run_recall_trials(
        train=train, epsilon=args.epsilon, params=params, cfg=cfg,
        amplitude=amplitude, flip_prob=args.flip_prob,
        trials_per_pattern=args.trials_per_pattern,
        seed=args.seed, jitter=args.jitter,
        record_times=record_times,
    )

    write_recall_csv(recall_rows, args.out_dir)

    exact_fraction    = float(np.mean([r["exact_recall"]   for r in recall_rows])) if recall_rows else 0.0
    mean_input_hamming = float(np.mean([r["input_hamming"]  for r in recall_rows])) if recall_rows else 0.0
    mean_final_hamming = float(np.mean([r["final_hamming"]  for r in recall_rows])) if recall_rows else 0.0

    print("\nRecall result:")
    print(f"  trials                  : {len(recall_rows)}")
    print(f"  exact recall fraction   : {exact_fraction:.3f}")
    print(f"  mean input Hamming      : {mean_input_hamming:.3f}")
    print(f"  mean final Hamming      : {mean_final_hamming:.3f}")

    # ── Figures ───────────────────────────────────────────────────────────────
    if args.plot:
        make_all_plots(
            train=train, rows_data=recall_rows, trajectories=trajectories,
            record_times=record_times,
            grid_rows=args.rows, grid_cols=args.cols, out_dir=args.out_dir,
            montage_page_size=args.montage_page_size,
            montage_cols=args.montage_cols,
        )

    print(f"\nOutputs written to: {args.out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
