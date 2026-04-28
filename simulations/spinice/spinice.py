"""
Non-reciprocal colloidal artificial spin ice on a square lattice
================================================================

Model
-----
Square L x L lattice with periodic boundaries. Each *edge* hosts one colloid
trapped in a bistable (double-well) potential along the edge. The colloid
position is encoded as an Ising-like variable s_e in {-1, +1} that points
toward one of the two endpoint vertices of edge e.

For each vertex v we count the number of incident "in"-pointing spins n_v.
The square ice rule selects 2-in-2-out vertices as the ground state. We use
the standard vertex penalty

    H_vertex = J * sum_v (n_v - 2)^2

so that 2-in-2-out costs 0, 3-1 (or 1-3) costs J, and 4-in/4-out costs 4J.
This is the same family of Hamiltonians used in colloidal ice studies
(Reichhardt et al.) where vertex penalties enforce the ice rule.

Non-reciprocal rotational shaking
---------------------------------
A uniform external force is applied to every colloid:

    F(t) = F0 * ( cos(omega t),  sin(omega t) ) 

It is "rotational" because the force vector rotates with frequency omega.
Its projection along an edge tilts that edge's double well. Calling
e_hat the edge unit vector pointing from the "minus" endpoint to the
"plus" endpoint, the bistable tilt becomes a linear bias on s_e:

    H_drive(t) = - sum_e h_e(t) * s_e ,   h_e(t) = F0 * (F_hat(t) . e_hat)

For horizontal edges only F_x tilts the well; for vertical edges only F_y.
Since F_x = F0 cos(wt) and F_y = F0 sin(wt) are 90 degrees out of phase,
horizontal and vertical edges feel a quarter-period-shifted tilt --- this
is the non-reciprocal/"circularly polarized" character of the drive that
breaks detailed balance and pushes the system out of equilibrium.

Total instantaneous Hamiltonian:

    H(t) = J * sum_v (n_v(s) - 2)^2  -  sum_e h_e(t) * s_e

Dynamics
--------
Single-spin-flip Metropolis using the *instantaneous* tilted H(t). Because
h_e depends on time and the protocol is time-periodic but not symmetric
under time reversal, the dynamics generically violate detailed balance:
the system reaches a non-equilibrium steady state (NESS) that depends on
omega and F0.

The relevant intrinsic timescale is the Kramers / Metropolis flip time at
F=0; stochastic-resonance-like effects appear when omega is comparable to
the inverse of that timescale.

Observable
----------
Defect density:
    rho_def = (# vertices not 2-in-2-out) / L^2

We sweep (omega, F0) and map rho_def averaged over a stroboscopic NESS
window. We also output an "ice rule violation" weighted by charge.

Outputs
-------
  - phase_diagram.png : heatmap of <rho_def> in (omega, F0) plane
  - line_cut.png      : <rho_def> vs omega at fixed F0 (resonance curve)
  - lattice_snap.png  : a snapshot of the lattice at low omega vs high omega
  - results.npz       : raw arrays for further analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from dataclasses import dataclass
from pathlib import Path
import time


# -----------------------------------------------------------------------------
# Lattice bookkeeping
# -----------------------------------------------------------------------------
#
# We label edges by (x, y, d) where (x,y) is the lower-left vertex of the
# edge, d=0 means the horizontal edge going from (x,y) to (x+1,y),
# d=1 means the vertical edge going from (x,y) to (x,y+1).
#
# Convention for s_e:
#   d=0 (horizontal):  s = +1  <=>  colloid sits closer to (x+1, y)  i.e.
#                                   spin "points to the right vertex"
#   d=1 (vertical):    s = +1  <=>  colloid sits closer to (x, y+1)  i.e.
#                                   spin "points up"
#
# For vertex (x,y) we collect its 4 incident edges and the sign that makes
# "spin points INTO this vertex":
#   left  edge  : (x-1, y, 0)   incoming if s = +1  --> sign +1
#   right edge  : (x,   y, 0)   incoming if s = -1  --> sign -1
#   below edge  : (x,   y-1, 1) incoming if s = +1  --> sign +1
#   above edge  : (x,   y, 1)   incoming if s = -1  --> sign -1
#
# Then n_in(v) = sum over those 4 edges of  (1 + sigma * s) / 2
# where sigma is +/- 1 from the table above.


def build_lattice(L):
    """Precompute incidence tables for an LxL periodic square lattice.

    Returns
    -------
    edges : (Ne, 3) int array      -- (x, y, d) for each edge, Ne = 2 L^2
    edge_unit : (Ne, 2) float      -- unit vector along edge (e_hat)
    vertex_edges : (L*L, 4) int    -- indices of the 4 edges around each vertex
    vertex_signs : (L*L, 4) int    -- +/- 1 sign so that incoming-iff (sigma*s == +1)
    """
    N = L * L
    Ne = 2 * N
    edges = np.zeros((Ne, 3), dtype=np.int64)
    edge_unit = np.zeros((Ne, 2), dtype=np.float64)
    edge_index = {}
    k = 0
    for y in range(L):
        for x in range(L):
            edges[k] = (x, y, 0)
            edge_unit[k] = (1.0, 0.0)
            edge_index[(x, y, 0)] = k
            k += 1
            edges[k] = (x, y, 1)
            edge_unit[k] = (0.0, 1.0)
            edge_index[(x, y, 1)] = k
            k += 1

    vertex_edges = np.zeros((N, 4), dtype=np.int64)
    vertex_signs = np.zeros((N, 4), dtype=np.int64)
    for y in range(L):
        for x in range(L):
            v = y * L + x
            xm = (x - 1) % L
            ym = (y - 1) % L
            # left edge:  (xm, y, 0), incoming if s = +1
            vertex_edges[v, 0] = edge_index[(xm, y, 0)]
            vertex_signs[v, 0] = +1
            # right edge: (x, y, 0), incoming if s = -1
            vertex_edges[v, 1] = edge_index[(x, y, 0)]
            vertex_signs[v, 1] = -1
            # below edge: (x, ym, 1), incoming if s = +1
            vertex_edges[v, 2] = edge_index[(x, ym, 1)]
            vertex_signs[v, 2] = +1
            # above edge: (x, y, 1), incoming if s = -1
            vertex_edges[v, 3] = edge_index[(x, y, 1)]
            vertex_signs[v, 3] = -1

    # For each edge, the two vertices it touches and the sign convention
    # for "this edge incoming to vertex" --- needed for fast local energy.
    edge_vertices = np.zeros((Ne, 2), dtype=np.int64)
    edge_vsign = np.zeros((Ne, 2), dtype=np.int64)
    for k in range(Ne):
        x, y, d = edges[k]
        if d == 0:
            v_minus = y * L + x            # left endpoint (s=+1 means spin away from here)
            v_plus = y * L + ((x + 1) % L) # right endpoint (s=+1 means spin into here)
            # at v_minus (left), "incoming" = s == -1, so sigma = -1
            # at v_plus  (right), "incoming" = s == +1, so sigma = +1
            edge_vertices[k] = (v_minus, v_plus)
            edge_vsign[k] = (-1, +1)
        else:
            v_minus = y * L + x
            v_plus = ((y + 1) % L) * L + x
            edge_vertices[k] = (v_minus, v_plus)
            edge_vsign[k] = (-1, +1)

    return edges, edge_unit, vertex_edges, vertex_signs, edge_vertices, edge_vsign


def vertex_charges(s, vertex_edges, vertex_signs):
    """Return n_in - 2 for every vertex (the topological charge / deviation)."""
    # incoming indicator = (1 + sigma*s)/2  for each (vertex, slot)
    s_at_slots = s[vertex_edges] * vertex_signs   # shape (N, 4)
    n_in = np.sum((1 + s_at_slots) // 2, axis=1)  # integer 0..4
    return n_in - 2


# -----------------------------------------------------------------------------
# Energetics
# -----------------------------------------------------------------------------
def total_vertex_energy(s, vertex_edges, vertex_signs, J):
    q = vertex_charges(s, vertex_edges, vertex_signs)
    return J * np.sum(q * q)


def local_dE_flip(k, s, edge_vertices, edge_vsign, vertex_edges, vertex_signs,
                  J, h_e):
    """Energy change if we flip s[k] -> -s[k], at instantaneous tilt h_e.

    Vertex term: each of the two endpoints v of edge k has charge q_v that
    changes by Delta_v = - sigma_{k,v} * s[k] when the spin flips
    (because n_in changes by sigma*Delta_s/2 where Delta_s = -2 s[k]).
    Penalty change at vertex v: J*((q + dq)^2 - q^2) = J*(2 q dq + dq^2).
    Drive term: -h_e[k] * (-2 s[k]) = 2 h_e[k] s[k].
    """
    # current charges at the two endpoints
    v0 = edge_vertices[k, 0]
    v1 = edge_vertices[k, 1]
    sig0 = edge_vsign[k, 0]
    sig1 = edge_vsign[k, 1]
    sk = s[k]

    # compute current charges at v0 and v1 from scratch (fast, only 8 lookups)
    q0 = -2
    for j in range(4):
        q0 += (1 + s[vertex_edges[v0, j]] * vertex_signs[v0, j]) // 2
    q1 = -2
    for j in range(4):
        q1 += (1 + s[vertex_edges[v1, j]] * vertex_signs[v1, j]) // 2

    dq0 = -sig0 * sk
    dq1 = -sig1 * sk

    dE_vertex = J * ((2 * q0 * dq0 + dq0 * dq0) + (2 * q1 * dq1 + dq1 * dq1))
    dE_drive = 2.0 * h_e * sk
    return dE_vertex + dE_drive


# -----------------------------------------------------------------------------
# Monte Carlo driver
# -----------------------------------------------------------------------------
@dataclass
class SimParams:
    L: int = 16            # linear lattice size
    J: float = 2.0         # vertex penalty (units of kT)
    T: float = 1.0         # temperature (sets units; kT=1)
    F0: float = 1.0        # drive amplitude
    omega: float = 0.05    # drive angular frequency, in units of 1/MC-sweep
    n_warmup: int = 200    # MC sweeps to reach NESS
    n_measure: int = 800   # MC sweeps to average
    seed: int = 0


def run_simulation(params: SimParams, lattice_data, return_snapshots=False):
    L = params.L
    J = params.J
    beta = 1.0 / params.T
    F0 = params.F0
    omega = params.omega

    edges, edge_unit, vertex_edges, vertex_signs, edge_vertices, edge_vsign = lattice_data
    Ne = edges.shape[0]
    N = L * L
    x_edges = edges[:, 2] == 0
    y_edges = edges[:, 2] == 1

    rng = np.random.default_rng(params.seed)
    # initial state: random Ising configuration
    s = rng.choice(np.array([-1, 1], dtype=np.int64), size=Ne)

    # ----------- warmup -----------
    # During warmup we already drive with the time-dependent field so the
    # NESS is reached.
    t_mc = 0.0
    dt = 1.0  # one MC sweep increments time by 1 unit
    for sweep in range(params.n_warmup):
        # tilt at start of this sweep (we update tilt once per sweep)
        Fx = F0 * np.cos(omega * t_mc)
        Fy = F0 * np.sin(omega * t_mc)
        # h_e = F . e_hat
        h_e_array = Fx * edge_unit[:, 0] + Fy * edge_unit[:, 1]

        # Ne single-spin-flip attempts = 1 sweep
        ks = rng.integers(0, Ne, size=Ne)
        rs = rng.random(Ne)
        for i in range(Ne):
            k = ks[i]
            dE = local_dE_flip(k, s, edge_vertices, edge_vsign,
                               vertex_edges, vertex_signs, J, h_e_array[k])
            if dE <= 0.0 or rs[i] < np.exp(-beta * dE):
                s[k] = -s[k]
        t_mc += dt

    # ----------- measurement -----------
    rho_def_trace = np.zeros(params.n_measure)
    q2_trace = np.zeros(params.n_measure)  # mean q^2 (charge density squared)
    mx_trace = np.zeros(params.n_measure)
    my_trace = np.zeros(params.n_measure)
    drive_t_trace = params.n_warmup + np.arange(params.n_measure, dtype=np.float64)

    for m in range(params.n_measure):
        Fx = F0 * np.cos(omega * t_mc)
        Fy = F0 * np.sin(omega * t_mc)
        h_e_array = Fx * edge_unit[:, 0] + Fy * edge_unit[:, 1]

        ks = rng.integers(0, Ne, size=Ne)
        rs = rng.random(Ne)
        for i in range(Ne):
            k = ks[i]
            dE = local_dE_flip(k, s, edge_vertices, edge_vsign,
                               vertex_edges, vertex_signs, J, h_e_array[k])
            if dE <= 0.0 or rs[i] < np.exp(-beta * dE):
                s[k] = -s[k]
        t_mc += dt

        q = vertex_charges(s, vertex_edges, vertex_signs)
        rho_def_trace[m] = np.mean(q != 0)
        q2_trace[m] = np.mean(q * q)
        mx_trace[m] = np.mean(s[x_edges])
        my_trace[m] = np.mean(s[y_edges])

    out = dict(
        rho_def_mean=float(np.mean(rho_def_trace)),
        rho_def_std=float(np.std(rho_def_trace)),
        q2_mean=float(np.mean(q2_trace)),
        rho_def_trace=rho_def_trace,
        q2_trace=q2_trace,
        mx_trace=mx_trace,
        my_trace=my_trace,
        t_trace=params.n_warmup + np.arange(1, params.n_measure + 1, dtype=np.float64),
        drive_t_trace=drive_t_trace,
        s_final=s.copy(),
    )
    return out


# -----------------------------------------------------------------------------
# Numba acceleration of the inner loop (optional but a big win)
# -----------------------------------------------------------------------------
try:
    from numba import njit

    @njit(cache=True, fastmath=True)
    def _sweep_numba(s, edge_unit, edge_vertices, edge_vsign,
                     vertex_edges, vertex_signs, J, beta,
                     Fx, Fy, ks, rs):
        Ne = s.shape[0]
        for i in range(Ne):
            k = ks[i]
            sk = s[k]
            # h_e for this edge
            h_e = Fx * edge_unit[k, 0] + Fy * edge_unit[k, 1]
            # charges at the two endpoint vertices
            v0 = edge_vertices[k, 0]
            v1 = edge_vertices[k, 1]
            sig0 = edge_vsign[k, 0]
            sig1 = edge_vsign[k, 1]
            q0 = -2
            for j in range(4):
                q0 += (1 + s[vertex_edges[v0, j]] * vertex_signs[v0, j]) // 2
            q1 = -2
            for j in range(4):
                q1 += (1 + s[vertex_edges[v1, j]] * vertex_signs[v1, j]) // 2
            dq0 = -sig0 * sk
            dq1 = -sig1 * sk
            dE_vertex = J * ((2 * q0 * dq0 + dq0 * dq0) + (2 * q1 * dq1 + dq1 * dq1))
            dE_drive = 2.0 * h_e * sk
            dE = dE_vertex + dE_drive
            if dE <= 0.0 or rs[i] < np.exp(-beta * dE):
                s[k] = -sk

    @njit(cache=True, fastmath=True)
    def _vertex_charges_numba(s, vertex_edges, vertex_signs):
        N = vertex_edges.shape[0]
        q = np.zeros(N, dtype=np.int64)
        for v in range(N):
            n_in = 0
            for j in range(4):
                n_in += (1 + s[vertex_edges[v, j]] * vertex_signs[v, j]) // 2
            q[v] = n_in - 2
        return q

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


def run_simulation_fast(params: SimParams, lattice_data):
    """Numba-accelerated version. Falls back to pure-numpy if numba missing."""
    if not HAS_NUMBA:
        return run_simulation(params, lattice_data)

    L = params.L
    J = params.J
    beta = 1.0 / params.T
    F0 = params.F0
    omega = params.omega

    edges, edge_unit, vertex_edges, vertex_signs, edge_vertices, edge_vsign = lattice_data
    Ne = edges.shape[0]
    x_edges = edges[:, 2] == 0
    y_edges = edges[:, 2] == 1

    rng = np.random.default_rng(params.seed)
    s = rng.choice(np.array([-1, 1], dtype=np.int64), size=Ne)

    t_mc = 0.0
    for sweep in range(params.n_warmup):
        Fx = F0 * np.cos(omega * t_mc)
        Fy = F0 * np.sin(omega * t_mc)
        ks = rng.integers(0, Ne, size=Ne).astype(np.int64)
        rs = rng.random(Ne)
        _sweep_numba(s, edge_unit, edge_vertices, edge_vsign,
                     vertex_edges, vertex_signs, float(J), float(beta),
                     float(Fx), float(Fy), ks, rs)
        t_mc += 1.0

    rho_def_trace = np.zeros(params.n_measure)
    q2_trace = np.zeros(params.n_measure)
    mx_trace = np.zeros(params.n_measure)
    my_trace = np.zeros(params.n_measure)
    drive_t_trace = params.n_warmup + np.arange(params.n_measure, dtype=np.float64)
    for m in range(params.n_measure):
        Fx = F0 * np.cos(omega * t_mc)
        Fy = F0 * np.sin(omega * t_mc)
        ks = rng.integers(0, Ne, size=Ne).astype(np.int64)
        rs = rng.random(Ne)
        _sweep_numba(s, edge_unit, edge_vertices, edge_vsign,
                     vertex_edges, vertex_signs, float(J), float(beta),
                     float(Fx), float(Fy), ks, rs)
        t_mc += 1.0

        q = _vertex_charges_numba(s, vertex_edges, vertex_signs)
        rho_def_trace[m] = np.mean(q != 0)
        q2_trace[m] = np.mean(q * q)
        mx_trace[m] = np.mean(s[x_edges])
        my_trace[m] = np.mean(s[y_edges])

    return dict(
        rho_def_mean=float(np.mean(rho_def_trace)),
        rho_def_std=float(np.std(rho_def_trace)),
        q2_mean=float(np.mean(q2_trace)),
        rho_def_trace=rho_def_trace,
        q2_trace=q2_trace,
        mx_trace=mx_trace,
        my_trace=my_trace,
        t_trace=params.n_warmup + np.arange(1, params.n_measure + 1, dtype=np.float64),
        drive_t_trace=drive_t_trace,
        s_final=s.copy(),
    )


# -----------------------------------------------------------------------------
# Phase-diagram sweeps
# -----------------------------------------------------------------------------
def sweep_phase_diagram(L, J, T, omegas, F0s, n_warmup, n_measure, seed=42):
    lattice_data = build_lattice(L)
    rho = np.zeros((len(F0s), len(omegas)))
    rho_err = np.zeros_like(rho)
    q2 = np.zeros_like(rho)

    total = len(F0s) * len(omegas)
    done = 0
    t0 = time.time()
    for i, F0 in enumerate(F0s):
        for j, omega in enumerate(omegas):
            params = SimParams(L=L, J=J, T=T, F0=F0, omega=omega,
                               n_warmup=n_warmup, n_measure=n_measure,
                               seed=seed + 1000 * i + j)
            out = run_simulation_fast(params, lattice_data)
            rho[i, j] = out['rho_def_mean']
            rho_err[i, j] = out['rho_def_std'] / np.sqrt(n_measure)
            q2[i, j] = out['q2_mean']
            done += 1
            elapsed = time.time() - t0
            print(f"  [{done:3d}/{total}] F0={F0:.3f} omega={omega:.4f}  "
                  f"<rho_def>={rho[i,j]:.4f}  ({elapsed:.1f}s)")
    return rho, rho_err, q2


def line_cut_omega(L, J, T, F0, omegas, n_warmup, n_measure, seed=42):
    lattice_data = build_lattice(L)
    rho = np.zeros(len(omegas))
    err = np.zeros(len(omegas))
    q2 = np.zeros(len(omegas))
    for j, omega in enumerate(omegas):
        params = SimParams(L=L, J=J, T=T, F0=F0, omega=omega,
                           n_warmup=n_warmup, n_measure=n_measure,
                           seed=seed + j)
        out = run_simulation_fast(params, lattice_data)
        rho[j] = out['rho_def_mean']
        err[j] = out['rho_def_std'] / np.sqrt(n_measure)
        q2[j] = out['q2_mean']
        print(f"  omega={omega:.4f}  <rho_def>={rho[j]:.4f} +/- {err[j]:.4f}")
    return rho, err, q2


def make_phase_diagram_grid(omega_log10_min=-3.0, omega_log10_max=0.0,
                            F0_min=0.2, F0_max=2.5,
                            base_omega_points=12, base_F0_points=9,
                            density=2.0):
    """Build a phase-diagram grid with a single density control."""
    if density <= 0:
        raise ValueError("phase-diagram density must be positive")

    n_omega = max(2, int(np.ceil(base_omega_points * density)))
    n_F0 = max(2, int(np.ceil(base_F0_points * density)))
    omegas = np.logspace(omega_log10_min, omega_log10_max, n_omega)
    F0s = np.linspace(F0_min, F0_max, n_F0)
    return omegas, F0s


def format_parameter_text(L, J, T, omega=None, F0=None,
                          omega_range=None, F0_range=None,
                          phase_density=None):
    """Build a compact multi-line parameter label for figure annotations."""
    lines = [fr"$L={L}$", fr"$J={J:.2f}$", fr"$T={T:.2f}$"]

    if omega is not None:
        lines.append(fr"$\omega={omega:.4g}$")
    elif omega_range is not None:
        lines.append(fr"$\omega \in [{omega_range[0]:.3g},\,{omega_range[1]:.3g}]$")

    if F0 is not None:
        lines.append(fr"$F_0={F0:.3g}$")
    elif F0_range is not None:
        lines.append(fr"$F_0 \in [{F0_range[0]:.3g},\,{F0_range[1]:.3g}]$")

    if phase_density is not None:
        lines.append(fr"density$={phase_density:.2f}$")

    return "\n".join(lines)


def add_parameter_box(ax, text, loc="upper left", is_3d=False):
    """Add a consistent parameter annotation box to a figure."""
    positions = {
        "upper left": (0.02, 0.98),
        "upper right": (0.98, 0.98),
        "lower left": (0.02, 0.02),
        "lower right": (0.98, 0.02),
    }
    x, y = positions[loc]
    va = "top" if "upper" in loc else "bottom"
    ha = "left" if "left" in loc else "right"
    bbox = dict(boxstyle="round,pad=0.35", facecolor="white",
                edgecolor="#444444", alpha=0.88)

    if is_3d:
        ax.text2D(x, y, text, transform=ax.transAxes, fontsize=9,
                  va=va, ha=ha, bbox=bbox)
    else:
        ax.text(x, y, text, transform=ax.transAxes, fontsize=9,
                va=va, ha=ha, bbox=bbox)


def format_param_value(value):
    """Format a numeric parameter for use in filenames."""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.3g}".replace("-", "m").replace(".", "p")


def protocol_components(F0, omega, drive_times):
    """Return the drive components and phase for the supplied protocol times."""
    Fx = F0 * np.cos(omega * drive_times)
    Fy = F0 * np.sin(omega * drive_times)
    phase = np.mod(omega * drive_times, 2.0 * np.pi)
    return Fx, Fy, phase


def compute_cycle_work(mx_trace, my_trace, F0, omega, drive_times, L):
    """Discrete protocol work using the configuration held fixed during field updates."""
    Fx, Fy, phase = protocol_components(F0, omega, drive_times)
    if drive_times.size < 2:
        work_steps = np.zeros(0)
        cycle_index = np.zeros(0, dtype=np.int64)
        return Fx, Fy, phase, work_steps, cycle_index, np.zeros(0), np.zeros(0)

    dFx = np.diff(Fx)
    dFy = np.diff(Fy)
    sector_size = float(L * L)
    work_steps = -sector_size * (mx_trace[:-1] * dFx + my_trace[:-1] * dFy)
    cycle_index = np.floor(np.abs(omega) * drive_times[:-1] / (2.0 * np.pi)).astype(np.int64)
    n_cycles = int(cycle_index.max()) + 1 if cycle_index.size else 0
    cycle_work = np.zeros(n_cycles)
    for idx, work in zip(cycle_index, work_steps):
        cycle_work[idx] += work
    cumulative_work = np.cumsum(work_steps)
    return Fx, Fy, phase, work_steps, cycle_index, cycle_work, cumulative_work


# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------
def plot_phase_diagram(omegas, F0s, rho, savepath, parameter_text=None):
    Path(savepath).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    cmap = LinearSegmentedColormap.from_list(
        "iceflame", ["#0a1640", "#2b5fa3", "#7fb6d8", "#f4d35e", "#ee6c4d", "#9b1d20"])
    # use log-x for omega axis to see the resonance
    extent = [np.log10(omegas[0]), np.log10(omegas[-1]), F0s[0], F0s[-1]]
    im = ax.imshow(rho, origin='lower', extent=extent, aspect='auto',
                   cmap=cmap, vmin=0, vmax=max(rho.max(), 0.2))
    cb = fig.colorbar(im, ax=ax, label=r"defect density $\langle\rho_{\rm def}\rangle$")
    ax.set_xlabel(r"$\log_{10}\,\omega$  (drive frequency, 1/MC-sweep)")
    ax.set_ylabel(r"$F_0$  (drive amplitude, $k_B T$ / lattice)")
    ax.set_title("Non-equilibrium phase diagram\nrotating drive on colloidal square ice")
    if parameter_text is not None:
        add_parameter_box(ax, parameter_text, loc="upper left")
    fig.tight_layout()
    fig.savefig(savepath, dpi=160)
    plt.close(fig)


def plot_phase_diagram_3d(omegas, F0s, rho, savepath, parameter_text=None):
    Path(savepath).parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8.2, 6.2))
    ax = fig.add_subplot(111, projection='3d')
    cmap = LinearSegmentedColormap.from_list(
        "iceflame", ["#0a1640", "#2b5fa3", "#7fb6d8", "#f4d35e", "#ee6c4d", "#9b1d20"])

    log_omegas = np.log10(omegas)
    X, Y = np.meshgrid(log_omegas, F0s)
    surf = ax.plot_surface(X, Y, rho, cmap=cmap, linewidth=0,
                           antialiased=True, alpha=0.95)
    ax.contour(X, Y, rho, zdir='z', offset=0.0, levels=10,
               colors="#15304d", linewidths=0.7, alpha=0.6)

    cb = fig.colorbar(surf, ax=ax, shrink=0.72, pad=0.1)
    cb.set_label(r"defect density $\langle\rho_{\rm def}\rangle$")
    ax.set_xlabel(r"$\log_{10}\,\omega$")
    ax.set_ylabel(r"$F_0$")
    ax.set_zlabel(r"$\langle\rho_{\rm def}\rangle$")
    ax.set_zlim(0.0, max(float(rho.max()) * 1.05, 0.2))
    ax.view_init(elev=28, azim=-128)
    ax.set_title("Non-equilibrium phase diagram (3D surface)")
    if parameter_text is not None:
        add_parameter_box(ax, parameter_text, loc="upper left", is_3d=True)
    fig.tight_layout()
    fig.savefig(savepath, dpi=180)
    plt.close(fig)


def plot_line_cut(omegas, rho, err, F0, savepath, parameter_text=None):
    Path(savepath).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(omegas, rho, yerr=err, fmt='o-', color="#2b5fa3",
                ecolor="#9bbcdb", capsize=3, lw=1.5, ms=5,
                label=fr"$F_0 = {F0:.2f}$")
    ax.set_xscale('log')
    ax.set_xlabel(r"drive frequency $\omega$  (1/MC-sweep)")
    ax.set_ylabel(r"$\langle\rho_{\rm def}\rangle$")
    ax.set_title("Defect density vs drive frequency (line cut)")
    ax.grid(True, which='both', alpha=0.3)
    ax.legend()
    if parameter_text is not None:
        add_parameter_box(ax, parameter_text, loc="upper left")
    fig.tight_layout()
    fig.savefig(savepath, dpi=160)
    plt.close(fig)


def plot_defect_time_trace(times, rho_trace, F0, omega, savepath,
                           parameter_text=None, q2_trace=None,
                           drive_times=None):
    Path(savepath).parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 6.2), sharex=True,
                             gridspec_kw={"height_ratios": [2.3, 1.0]})
    ax_rho, ax_drive = axes

    ax_rho.plot(times, rho_trace, color="#b23a48", lw=1.35,
                label=r"$\rho_{\rm def}(t)$")
    if q2_trace is not None:
        ax_rho.plot(times, q2_trace, color="#355070", lw=1.0, alpha=0.8,
                    label=r"$\langle q^2\rangle(t)$")
    ax_rho.set_ylabel("defect observables")
    ax_rho.set_title("Time-resolved nonequilibrium defect response")
    ax_rho.grid(True, alpha=0.25)
    ax_rho.legend(loc="upper right", fontsize=8, framealpha=0.9)
    if parameter_text is not None:
        add_parameter_box(ax_rho, parameter_text, loc="upper left")

    if drive_times is None:
        drive_times = times
    Fx, Fy, _ = protocol_components(F0, omega, drive_times)
    ax_drive.plot(drive_times, Fx, color="#2b5fa3", lw=1.15, label=r"$F_x(t)$")
    ax_drive.plot(drive_times, Fy, color="#2a9d8f", lw=1.15, label=r"$F_y(t)$")
    ax_drive.axhline(0.0, color="#666666", lw=0.8, alpha=0.7)
    ax_drive.set_xlabel(r"Monte Carlo time $t_{\rm MC}$ (sweeps)")
    ax_drive.set_ylabel("drive")
    ax_drive.grid(True, alpha=0.25)
    ax_drive.legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(savepath, dpi=170)
    plt.close(fig)


def plot_phase_resolved_hysteresis(mx_trace, my_trace, rho_trace, F0, omega, drive_times,
                                   savepath, parameter_text=None):
    Path(savepath).parent.mkdir(parents=True, exist_ok=True)
    Fx, Fy, phase = protocol_components(F0, omega, drive_times)
    phase_color = phase / (2.0 * np.pi)

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.4))
    cmap = plt.cm.twilight
    markersize = 12

    axes[0].scatter(Fx, mx_trace, c=phase_color, cmap=cmap, s=markersize, alpha=0.9)
    axes[0].plot(Fx, mx_trace, color="#355070", lw=0.8, alpha=0.55)
    axes[0].set_xlabel(r"$F_x$")
    axes[0].set_ylabel(r"$m_x$")
    axes[0].set_title(r"Horizontal Hysteresis: $m_x$ vs $F_x$")
    axes[0].grid(True, alpha=0.25)

    axes[1].scatter(Fy, my_trace, c=phase_color, cmap=cmap, s=markersize, alpha=0.9)
    axes[1].plot(Fy, my_trace, color="#2a9d8f", lw=0.8, alpha=0.55)
    axes[1].set_xlabel(r"$F_y$")
    axes[1].set_ylabel(r"$m_y$")
    axes[1].set_title(r"Vertical Hysteresis: $m_y$ vs $F_y$")
    axes[1].grid(True, alpha=0.25)

    sc = axes[2].scatter(Fx, rho_trace, c=phase_color, cmap=cmap, s=markersize, alpha=0.9)
    axes[2].plot(Fx, rho_trace, color="#b23a48", lw=0.8, alpha=0.55)
    axes[2].set_xlabel(r"$F_x$")
    axes[2].set_ylabel(r"$\rho_{\rm def}$")
    axes[2].set_title(r"Defect Response: $\rho_{\rm def}$ vs $F_x$")
    axes[2].grid(True, alpha=0.25)

    cb = fig.colorbar(sc, ax=axes, shrink=0.86, pad=0.02)
    cb.set_label(r"drive phase $\phi/2\pi$")
    if parameter_text is not None:
        add_parameter_box(axes[0], parameter_text, loc="upper left")

    fig.tight_layout()
    fig.savefig(savepath, dpi=170)
    plt.close(fig)


def plot_cycle_work(mx_trace, my_trace, F0, omega, drive_times, L, savepath,
                    parameter_text=None):
    Path(savepath).parent.mkdir(parents=True, exist_ok=True)
    Fx, Fy, phase, work_steps, cycle_index, cycle_work, cumulative_work = compute_cycle_work(
        mx_trace, my_trace, F0, omega, drive_times, L
    )

    fig, axes = plt.subplots(2, 1, figsize=(8.4, 6.4),
                             gridspec_kw={"height_ratios": [1.7, 1.2]})
    ax_work, ax_cycle = axes

    if work_steps.size:
        step_times = drive_times[:-1]
        ax_work.plot(step_times, cumulative_work, color="#7b2cbf", lw=1.4,
                     label=r"cumulative work")
        ax_work.bar(step_times, work_steps, width=0.8, color="#c77dff", alpha=0.4,
                    label=r"$\delta W$ per sweep")
    ax_work.set_ylabel("work")
    ax_work.set_title("Cycle work and cumulative injected work")
    ax_work.grid(True, alpha=0.25)
    ax_work.legend(loc="upper left", fontsize=8, framealpha=0.9)
    if parameter_text is not None:
        add_parameter_box(ax_work, parameter_text, loc="upper right")

    if cycle_work.size:
        cycle_axis = np.arange(cycle_work.size)
        ax_cycle.plot(cycle_axis, cycle_work, marker='o', color="#3a86ff", lw=1.2)
        ax_cycle.axhline(np.mean(cycle_work), color="#e63946", lw=1.0, alpha=0.9,
                         label=fr"mean = {np.mean(cycle_work):.3g}")
        ax_cycle.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax_cycle.set_xlabel("cycle index")
    ax_cycle.set_ylabel(r"$W_{\rm cycle}$")
    ax_cycle.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(savepath, dpi=170)
    plt.close(fig)


def plot_lattice_snapshot(s, L, lattice_data, title, savepath, parameter_text=None):
    """Draw the lattice with arrows on each edge and color vertices by charge."""
    Path(savepath).parent.mkdir(parents=True, exist_ok=True)
    edges, edge_unit, vertex_edges, vertex_signs, edge_vertices, edge_vsign = lattice_data
    q = vertex_charges(s, vertex_edges, vertex_signs)
    colloid_radius = 0.085
    colloid_fill = "#f7fbff"
    colloid_edge = "#274c77"

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.set_aspect('equal')
    ax.set_xlim(-0.6, L - 0.4)
    ax.set_ylim(-0.6, L - 0.4)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)

    # vertex coloring: 0 -> grey, +1 -> red, -1 -> blue, +-2 -> deeper
    color_map = {-2: "#08306b", -1: "#3182bd", 0: "#dddddd",
                 1: "#e6550d", 2: "#67000d"}
    for v in range(L * L):
        x = v % L
        y = v // L
        ax.add_patch(plt.Circle((x, y), 0.13,
                                color=color_map.get(int(q[v]), "#555555"),
                                zorder=3))

    # edge arrows
    for k in range(edges.shape[0]):
        x, y, d = edges[k]
        sk = s[k]
        if d == 0:
            x0, y0 = x, y
            x1, y1 = x + 1, y
        else:
            x0, y0 = x, y
            x1, y1 = x, y + 1
        # spin direction
        if sk > 0:
            xa, ya = x0, y0
            xb, yb = x1, y1
        else:
            xa, ya = x1, y1
            xb, yb = x0, y0
        ax.annotate("",
                    xy=(0.7 * xb + 0.3 * xa, 0.7 * yb + 0.3 * ya),
                    xytext=(0.3 * xb + 0.7 * xa, 0.3 * yb + 0.7 * ya),
                    arrowprops=dict(arrowstyle="->", color="#444444",
                                    lw=1.2, alpha=0.85),
                    zorder=2)

        # Draw the colloid closer to the endpoint selected by the spin.
        xc = 0.72 * xb + 0.28 * xa
        yc = 0.72 * yb + 0.28 * ya
        ax.add_patch(plt.Circle((xc, yc), colloid_radius,
                                facecolor=colloid_fill, edgecolor=colloid_edge,
                                linewidth=0.9, alpha=0.55, zorder=4))

    # legend
    handles = [
        plt.Line2D([], [], marker='o', color='w', markerfacecolor=color_map[0],
                   markersize=10, label='ground state (2-in-2-out)'),
        plt.Line2D([], [], marker='o', color='w', markerfacecolor=color_map[1],
                   markersize=10, label='+1 monopole (3-in-1-out)'),
        plt.Line2D([], [], marker='o', color='w', markerfacecolor=color_map[-1],
                   markersize=10, label='-1 monopole (1-in-3-out)'),
        plt.Line2D([], [], marker='o', color='w', markerfacecolor=colloid_fill,
                   markeredgecolor=colloid_edge, alpha=0.55,
                   markersize=9, label='colloid position'),
    ]
    ax.legend(handles=handles, loc='upper right', fontsize=8, framealpha=0.9)
    if parameter_text is not None:
        add_parameter_box(ax, parameter_text, loc="lower left")
    fig.tight_layout()
    fig.savefig(savepath, dpi=160)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("Non-reciprocal colloidal artificial spin ice")
    print("Rotating shake on a square lattice -- Monte Carlo NESS sweep")
    print("Numba acceleration:", HAS_NUMBA)
    print("=" * 70)

    # ---- 1. line-cut: defect density vs frequency at one drive amplitude ----
    L = 30
    J = 2.0
    T = 5
    F0_line = 1.5
    phase_diagram_density = 3.0
    output_dir = Path(__file__).resolve().parent / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    base_tag = (
        f"L{format_param_value(L)}_"
        f"J{format_param_value(J)}_"
        f"T{format_param_value(T)}"
    )

    omegas_line = np.logspace(-3.5, 0.0, 18)   # span four decades of frequency
    print("\n[1] Line cut: rho_def(omega) at F0 =", F0_line)
    rho_line, err_line, q2_line = line_cut_omega(
        L=L, J=J, T=T, F0=F0_line, omegas=omegas_line,
        n_warmup=300, n_measure=600, seed=12345)
    line_params = format_parameter_text(
        L=L, J=J, T=T, F0=F0_line,
        omega_range=(float(omegas_line[0]), float(omegas_line[-1]))
    )
    line_cut_name = (
        f"line_cut_{base_tag}_"
        f"F0_{format_param_value(F0_line)}_"
        f"omega_{format_param_value(omegas_line[0])}_to_{format_param_value(omegas_line[-1])}.png"
    )
    plot_line_cut(omegas_line, rho_line, err_line, F0_line,
                  output_dir / line_cut_name,
                  parameter_text=line_params)

    # ---- 2. snapshots at low and high omega ----
    lattice_data = build_lattice(L)
    print("\n[2] Snapshots at low and high omega")
    p_lo = SimParams(L=L, J=J, T=T, F0=F0_line, omega=1e-3,
                     n_warmup=400, n_measure=400, seed=7)
    out_lo = run_simulation_fast(p_lo, lattice_data)
    low_snapshot_params = format_parameter_text(
        L=L, J=J, T=T, omega=p_lo.omega, F0=p_lo.F0
    )
    snap_low_name = (
        f"snap_low_omega_{base_tag}_"
        f"omega_{format_param_value(p_lo.omega)}_"
        f"F0_{format_param_value(p_lo.F0)}.png"
    )
    trace_low_name = (
        f"defect_trace_low_omega_{base_tag}_"
        f"omega_{format_param_value(p_lo.omega)}_"
        f"F0_{format_param_value(p_lo.F0)}.png"
    )
    hysteresis_low_name = (
        f"hysteresis_low_omega_{base_tag}_"
        f"omega_{format_param_value(p_lo.omega)}_"
        f"F0_{format_param_value(p_lo.F0)}.png"
    )
    work_low_name = (
        f"cycle_work_low_omega_{base_tag}_"
        f"omega_{format_param_value(p_lo.omega)}_"
        f"F0_{format_param_value(p_lo.F0)}.png"
    )
    plot_lattice_snapshot(out_lo['s_final'], L, lattice_data,
                          fr"Quasi-adiabatic drive  ($\omega=10^{{-3}}$, $F_0={F0_line}$): "
                          fr"$\rho_{{\rm def}}={out_lo['rho_def_mean']:.3f}$",
                          output_dir / snap_low_name,
                          parameter_text=low_snapshot_params)
    plot_defect_time_trace(out_lo['t_trace'], out_lo['rho_def_trace'],
                           p_lo.F0, p_lo.omega, output_dir / trace_low_name,
                           parameter_text=low_snapshot_params,
                           q2_trace=out_lo['q2_trace'],
                           drive_times=out_lo['drive_t_trace'])
    plot_phase_resolved_hysteresis(out_lo['mx_trace'], out_lo['my_trace'],
                                   out_lo['rho_def_trace'], p_lo.F0, p_lo.omega,
                                   out_lo['drive_t_trace'], output_dir / hysteresis_low_name,
                                   parameter_text=low_snapshot_params)
    plot_cycle_work(out_lo['mx_trace'], out_lo['my_trace'], p_lo.F0, p_lo.omega,
                    out_lo['drive_t_trace'], L, output_dir / work_low_name,
                    parameter_text=low_snapshot_params)

    p_hi = SimParams(L=L, J=J, T=T, F0=F0_line, omega=0.5,
                     n_warmup=400, n_measure=400, seed=7)
    out_hi = run_simulation_fast(p_hi, lattice_data)
    high_snapshot_params = format_parameter_text(
        L=L, J=J, T=T, omega=p_hi.omega, F0=p_hi.F0
    )
    snap_high_name = (
        f"snap_high_omega_{base_tag}_"
        f"omega_{format_param_value(p_hi.omega)}_"
        f"F0_{format_param_value(p_hi.F0)}.png"
    )
    trace_high_name = (
        f"defect_trace_high_omega_{base_tag}_"
        f"omega_{format_param_value(p_hi.omega)}_"
        f"F0_{format_param_value(p_hi.F0)}.png"
    )
    hysteresis_high_name = (
        f"hysteresis_high_omega_{base_tag}_"
        f"omega_{format_param_value(p_hi.omega)}_"
        f"F0_{format_param_value(p_hi.F0)}.png"
    )
    work_high_name = (
        f"cycle_work_high_omega_{base_tag}_"
        f"omega_{format_param_value(p_hi.omega)}_"
        f"F0_{format_param_value(p_hi.F0)}.png"
    )
    plot_lattice_snapshot(out_hi['s_final'], L, lattice_data,
                          fr"Fast drive  ($\omega=0.5$, $F_0={F0_line}$): "
                          fr"$\rho_{{\rm def}}={out_hi['rho_def_mean']:.3f}$",
                          output_dir / snap_high_name,
                          parameter_text=high_snapshot_params)
    plot_defect_time_trace(out_hi['t_trace'], out_hi['rho_def_trace'],
                           p_hi.F0, p_hi.omega, output_dir / trace_high_name,
                           parameter_text=high_snapshot_params,
                           q2_trace=out_hi['q2_trace'],
                           drive_times=out_hi['drive_t_trace'])
    plot_phase_resolved_hysteresis(out_hi['mx_trace'], out_hi['my_trace'],
                                   out_hi['rho_def_trace'], p_hi.F0, p_hi.omega,
                                   out_hi['drive_t_trace'], output_dir / hysteresis_high_name,
                                   parameter_text=high_snapshot_params)
    plot_cycle_work(out_hi['mx_trace'], out_hi['my_trace'], p_hi.F0, p_hi.omega,
                    out_hi['drive_t_trace'], L, output_dir / work_high_name,
                    parameter_text=high_snapshot_params)

    # ---- 3. full phase diagram (omega, F0) ----
    print("\n[3] Phase diagram sweep over (omega, F0)")
    omegas_pd, F0s_pd = make_phase_diagram_grid(density=phase_diagram_density)
    print(f"    density={phase_diagram_density:.2f} -> "
          f"{len(omegas_pd)} omega points x {len(F0s_pd)} F0 points")
    rho_pd, err_pd, q2_pd = sweep_phase_diagram(
        L=L, J=J, T=T, omegas=omegas_pd, F0s=F0s_pd,
        n_warmup=200, n_measure=300, seed=99)
    phase_params = format_parameter_text(
        L=L, J=J, T=T,
        omega_range=(float(omegas_pd[0]), float(omegas_pd[-1])),
        F0_range=(float(F0s_pd[0]), float(F0s_pd[-1])),
        phase_density=phase_diagram_density
    )
    phase_tag = (
        f"{base_tag}_"
        f"omega_{format_param_value(omegas_pd[0])}_to_{format_param_value(omegas_pd[-1])}_"
        f"F0_{format_param_value(F0s_pd[0])}_to_{format_param_value(F0s_pd[-1])}_"
        f"dens_{format_param_value(phase_diagram_density)}"
    )
    plot_phase_diagram(omegas_pd, F0s_pd, rho_pd,
                       output_dir / f"phase_diagram_{phase_tag}.png",
                       parameter_text=phase_params)
    plot_phase_diagram_3d(omegas_pd, F0s_pd, rho_pd,
                          output_dir / f"phase_diagram_3d_{phase_tag}.png",
                          parameter_text=phase_params)

    # ---- 4. save raw data ----
    np.savez(output_dir / f"results_{phase_tag}.npz",
             omegas_line=omegas_line, rho_line=rho_line, err_line=err_line,
             q2_line=q2_line,
             omegas_pd=omegas_pd, F0s_pd=F0s_pd,
             rho_pd=rho_pd, err_pd=err_pd, q2_pd=q2_pd,
             L=L, J=J, T=T, F0_line=F0_line,
             phase_diagram_density=phase_diagram_density)
    print(f"\nDone. Results saved to {output_dir}/")
