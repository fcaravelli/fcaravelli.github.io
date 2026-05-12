import numpy as np
import matplotlib.pyplot as plt


def haar_orthogonal(n, rng):
    """
    Sample Haar-random O(n) using the QR method.
    """
    X = rng.normal(size=(n, n))
    Q, R = np.linalg.qr(X)
    d = np.sign(np.diag(R))
    d[d == 0] = 1.0
    return Q @ np.diag(d)


def make_admissible_B(Nr, Nu, sigma_y, sigma_z, rho, rng):
    """
    Generate B satisfying the Schur admissibility condition

        sigma_z I - sigma_y^{-1} B^T B >= 0.

    We enforce ||B||_2 <= rho * sqrt(sigma_y sigma_z), with rho < 1.
    """
    B = rng.normal(size=(Nr, Nu))
    op_norm = np.linalg.norm(B, 2)
    if op_norm == 0:
        return B
    B *= rho * np.sqrt(sigma_y * sigma_z) / op_norm
    return B


def make_orthogonal_direction(B, rng):
    """
    Construct P with Tr(P^T B) = 0 and ||P||_F = ||B||_F.
    """
    P = rng.normal(size=B.shape)

    inner_PB = np.sum(P * B)
    norm_B2 = np.sum(B * B)

    if norm_B2 > 0:
        P = P - (inner_PB / norm_B2) * B

    norm_P = np.linalg.norm(P, "fro")
    norm_B = np.linalg.norm(B, "fro")

    if norm_P > 0:
        P *= norm_B / norm_P

    return P


def equal_time_theory(Nr, sigma_y, lambdas, A_ru, B):
    """
    Stationary equal-time delta-supported effective variance coefficient:

        V_eq = Nr sigma_y
             + (2 ||A_uu^{-1}||_1 / Nu) Tr(A_ru^T B).

    Here lambdas are stable, i.e. lambdas < 0.
    """
    Nu = len(lambdas)
    Ainv_1 = np.sum(np.abs(1.0 / lambdas))
    cross = np.trace(A_ru.T @ B)
    return Nr * sigma_y + (2.0 * Ainv_1 / Nu) * cross


def equal_time_haar_mc(Nr, sigma_y, lambdas, A_ru, B, n_haar, rng):
    """
    Monte Carlo estimate of the Haar-averaged stationary equal-time coefficient.

    For a fixed Haar rotation O, the stationary cross term is

        2 Tr[ O^T diag(-1/lambda_i) O A_ru^T B ].

    Averaging over O should reproduce

        (2 ||A_uu^{-1}||_1 / Nu) Tr(A_ru^T B).
    """
    Nu = len(lambdas)
    Dinv = np.diag(-1.0 / lambdas)
    C = A_ru.T @ B

    vals = []
    for _ in range(n_haar):
        O = haar_orthogonal(Nu, rng)
        eta_O = 2.0 * np.trace(O.T @ Dinv @ O @ C)
        vals.append(Nr * sigma_y + eta_O)

    vals = np.array(vals)
    return vals.mean(), vals.std(ddof=1) / np.sqrt(n_haar)


def run_experiment(
    Nr=5,
    Nu_values=range(10, 151, 10),
    sigma_y=1.0,
    sigma_z=1.0,
    rho_B=0.6,
    n_haar=200,
    seed=7,
):
    rng = np.random.default_rng(seed)

    native = []
    anti_theory = []
    anti_mc = []
    anti_mc_err = []

    half_anti_theory = []
    aligned_theory = []
    orth_theory = []

    alpha_star_values = []

    for Nu in Nu_values:
        Nu = int(Nu)

        # Stable bath spectrum.
        # More negative means faster relaxation.
        lambdas = -rng.uniform(0.5, 2.0, size=Nu)

        # Cross-correlation matrix B, scaled to satisfy admissibility.
        B = make_admissible_B(Nr, Nu, sigma_y, sigma_z, rho_B, rng)

        norm_B2 = np.sum(B * B)
        Ainv_1 = np.sum(np.abs(1.0 / lambdas))

        # Coupling strength that cancels the leading equal-time contribution
        # for A_ru = -alpha B.
        alpha_star = (Nr * Nu * sigma_y) / (2.0 * Ainv_1 * norm_B2)
        alpha_star_values.append(alpha_star)

        # Coupling choices.
        A_zero = np.zeros_like(B)
        A_anti = -alpha_star * B
        A_half_anti = -0.5 * alpha_star * B
        A_aligned = alpha_star * B

        P_perp = make_orthogonal_direction(B, rng)
        A_orth = alpha_star * P_perp

        native.append(equal_time_theory(Nr, sigma_y, lambdas, A_zero, B))
        anti_theory.append(equal_time_theory(Nr, sigma_y, lambdas, A_anti, B))
        half_anti_theory.append(equal_time_theory(Nr, sigma_y, lambdas, A_half_anti, B))
        aligned_theory.append(equal_time_theory(Nr, sigma_y, lambdas, A_aligned, B))
        orth_theory.append(equal_time_theory(Nr, sigma_y, lambdas, A_orth, B))

        mc_mean, mc_err = equal_time_haar_mc(
            Nr, sigma_y, lambdas, A_anti, B, n_haar, rng
        )
        anti_mc.append(mc_mean)
        anti_mc_err.append(mc_err)

    return {
        "Nu": np.array(list(Nu_values), dtype=float),
        "native": np.array(native),
        "anti_theory": np.array(anti_theory),
        "anti_mc": np.array(anti_mc),
        "anti_mc_err": np.array(anti_mc_err),
        "half_anti_theory": np.array(half_anti_theory),
        "aligned_theory": np.array(aligned_theory),
        "orth_theory": np.array(orth_theory),
        "alpha_star": np.array(alpha_star_values),
    }


if __name__ == "__main__":
    results = run_experiment(
        Nr=5,
        Nu_values=range(10, 151, 10),
        sigma_y=1.0,
        sigma_z=1.0,
        rho_B=0.6,
        n_haar=300,
        seed=11,
    )

    Nu = results["Nu"]

    plt.figure(figsize=(7.2, 4.8))

    plt.plot(Nu, results["native"], marker="o", label=r"No coupling, $A_{ru}=0$")
    plt.plot(
        Nu,
        results["half_anti_theory"],
        marker="o",
        label=r"Half anti-aligned, $A_{ru}=-\alpha_\star B/2$",
    )
    plt.plot(
        Nu,
        results["anti_theory"],
        marker="o",
        label=r"Anti-aligned theory, $A_{ru}=-\alpha_\star B$",
    )
    plt.errorbar(
        Nu,
        results["anti_mc"],
        yerr=results["anti_mc_err"],
        fmt="s",
        capsize=3,
        label=r"Anti-aligned Haar MC",
    )
    plt.plot(
        Nu,
        results["orth_theory"],
        marker="o",
        label=r"Orthogonal direction, $\mathrm{Tr}(A_{ru}^{T}B)=0$",
    )
    plt.plot(
        Nu,
        results["aligned_theory"],
        marker="o",
        label=r"Aligned, $A_{ru}=+\alpha_\star B$",
    )

    plt.axhline(0.0, linestyle="--", linewidth=1)

    plt.xlabel(r"Bath size $N_u$")
    plt.ylabel(r"Stationary equal-time variance coefficient")
    plt.title(r"Coupling-induced noise suppression vs bath size")
    plt.legend(fontsize=8)
    plt.tight_layout()

    plt.savefig("noise_suppression_vs_bath_size.pdf")
    plt.savefig("noise_suppression_vs_bath_size.png", dpi=300)
    plt.show()

    plt.figure(figsize=(7.2, 4.0))
    plt.plot(Nu, results["alpha_star"], marker="o")
    plt.xlabel(r"Bath size $N_u$")
    plt.ylabel(r"Cancellation coupling $\alpha_\star$")
    plt.title(r"Coupling strength required for equal-time cancellation")
    plt.tight_layout()
    plt.savefig("alpha_star_vs_bath_size.pdf")
    plt.savefig("alpha_star_vs_bath_size.png", dpi=300)
    plt.show()