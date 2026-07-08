# =========================================================
# Dark-dark analytical solution helpers
# =========================================================

import numpy as np


def two_dark_dark_solution(X, T, q_plus, zeta1, zeta2, gamma1, gamma2, q0):
    """
    Analytical two dark-dark soliton solution.

    Returns
    -------
    q : array, shape (2, Nx, Nt)
        Two-component solution q = (q1, q2).

    D : array, shape (Nx, Nt)
        Denominator D(x,t).
    """
    kappa1 = np.real(zeta1)
    kappa2 = np.real(zeta2)

    nu1 = np.imag(zeta1)
    nu2 = np.imag(zeta2)

    xi1 = X - 2.0 * kappa1 * T
    xi2 = X - 2.0 * kappa2 * T

    E1 = np.exp(2.0 * nu1 * xi1)
    E2 = np.exp(2.0 * nu2 * xi2)
    E12 = E1 * E2

    z1 = zeta1
    z2 = zeta2

    z1c = np.conj(z1)
    z2c = np.conj(z2)

    D = (
        1.0
        + gamma1 / (z1 - z1c) * E1
        + gamma2 / (z2 - z2c) * E2
        + (
            gamma1
            * gamma2
            * np.abs(z1 - z2) ** 2
            / (
                (z1 - z1c)
                * (z2 - z2c)
                * np.abs(z1c - z2) ** 2
            )
        )
        * E12
    )

    cross_coeff = (
        gamma1
        * gamma2
        * np.abs(z1 - z2) ** 2
        * (z1 * z2 - z1c * z2c)
        / (
            z1
            * z2
            * (z1 - z1c)
            * (z2 - z2c)
            * np.abs(z1 - z2c) ** 2
        )
    )

    correction = (
        gamma1 / z1 * E1
        + gamma2 / z2 * E2
        + cross_coeff * E12
    )

    scalar_factor = np.exp(2.0j * q0**2 * T) * (1.0 - correction / D)

    q = q_plus[:, np.newaxis, np.newaxis] * scalar_factor[np.newaxis, :, :]

    return q, D


def build_symmetric_dark_dark_parameters(
    kappa_abs=0.60,
    x_sep=5.0,
    q_plus_scalar=None,
):
    """
    Build the scalar-equivalent symmetric dark-dark collision parameters.

    The two solitons have:
        kappa1 = +kappa_abs
        kappa2 = -kappa_abs

    Approximate initial positions:
        x1(0) = -x_sep
        x2(0) = +x_sep
    """
    if q_plus_scalar is None:
        q_plus_scalar = np.array([1.25 + 0.0j, 0.0 + 0.0j], dtype=complex)
    else:
        q_plus_scalar = np.asarray(q_plus_scalar, dtype=complex)

    q0_scalar = np.linalg.norm(q_plus_scalar)

    kappa1 = +kappa_abs
    kappa2 = -kappa_abs

    nu1_scalar = np.sqrt(q0_scalar**2 - kappa1**2)
    nu2_scalar = np.sqrt(q0_scalar**2 - kappa2**2)

    zeta1_scalar = kappa1 + 1j * nu1_scalar
    zeta2_scalar = kappa2 + 1j * nu2_scalar

    x1_initial = -x_sep
    x2_initial = +x_sep

    gamma1 = 1j * 2.0 * nu1_scalar * np.exp(-2.0 * nu1_scalar * x1_initial)
    gamma2 = 1j * 2.0 * nu2_scalar * np.exp(-2.0 * nu2_scalar * x2_initial)

    t_collision_est = x_sep / (2.0 * kappa_abs)

    return {
        "q_plus_scalar": q_plus_scalar,
        "q0_scalar": q0_scalar,
        "kappa_abs": kappa_abs,
        "kappa1": kappa1,
        "kappa2": kappa2,
        "nu1_scalar": nu1_scalar,
        "nu2_scalar": nu2_scalar,
        "zeta1_scalar": zeta1_scalar,
        "zeta2_scalar": zeta2_scalar,
        "x_sep": x_sep,
        "x1_initial": x1_initial,
        "x2_initial": x2_initial,
        "gamma1": gamma1,
        "gamma2": gamma2,
        "t_collision_est": t_collision_est,
    }


def scalar_dark_dark_solution_on_grid(
    x_grid,
    t_grid,
    q_plus_scalar,
    zeta1_scalar,
    zeta2_scalar,
    gamma1,
    gamma2,
    q0_scalar,
):
    """
    Evaluate the scalar-equivalent dark-dark analytical solution.

    Returns
    -------
    psi_exact_xt : array, shape (Nx, Nt)
        Scalar-equivalent wavefunction.

    D_eval : array, shape (Nx, Nt)
        Denominator of the analytical solution.
    """
    x_grid = np.asarray(x_grid, dtype=float)
    t_grid = np.asarray(t_grid, dtype=float)

    X_eval, T_eval = np.meshgrid(
        x_grid,
        t_grid,
        indexing="ij",
    )

    q_eval, D_eval = two_dark_dark_solution(
        X_eval,
        T_eval,
        q_plus=q_plus_scalar,
        zeta1=zeta1_scalar,
        zeta2=zeta2_scalar,
        gamma1=gamma1,
        gamma2=gamma2,
        q0=q0_scalar,
    )

    psi_exact_xt = q_eval[0]

    return psi_exact_xt, D_eval


def compare_density_nqs_vs_exact(
    tau_grid,
    rho_nqs,
    x_grid,
    t_exact_initial,
    t_exact_final,
    q_plus_scalar,
    zeta1_scalar,
    zeta2_scalar,
    gamma1,
    gamma2,
    q0_scalar,
    inner_window=(-25.0, 25.0),
):
    """
    Compare NQS density with the analytical dark-dark density using directly:

        t_exact = t_exact_initial + 0.5 * tau

    No time-scale scan is performed.
    """
    tau_grid = np.asarray(tau_grid, dtype=float)
    rho_nqs = np.asarray(rho_nqs, dtype=float)
    x_grid = np.asarray(x_grid, dtype=float)

    t_exact = t_exact_initial + 0.5 * tau_grid

    valid_t_mask = (
        (t_exact >= t_exact_initial)
        & (t_exact <= t_exact_final)
    )

    tau_cmp = tau_grid[valid_t_mask]
    t_exact_cmp = t_exact[valid_t_mask]
    rho_nqs_cmp = rho_nqs[valid_t_mask, :]

    psi_exact_xt, _ = scalar_dark_dark_solution_on_grid(
        x_grid,
        t_exact_cmp,
        q_plus_scalar=q_plus_scalar,
        zeta1_scalar=zeta1_scalar,
        zeta2_scalar=zeta2_scalar,
        gamma1=gamma1,
        gamma2=gamma2,
        q0_scalar=q0_scalar,
    )

    rho_exact_cmp = np.abs(psi_exact_xt) ** 2
    rho_exact_cmp = rho_exact_cmp.T

    Nt_cmp = min(rho_nqs_cmp.shape[0], rho_exact_cmp.shape[0])
    Nx_cmp = min(rho_nqs_cmp.shape[1], rho_exact_cmp.shape[1])

    tau_cmp = tau_cmp[:Nt_cmp]
    t_exact_cmp = t_exact_cmp[:Nt_cmp]
    x_cmp = x_grid[:Nx_cmp]

    rho_nqs_cmp = rho_nqs_cmp[:Nt_cmp, :Nx_cmp]
    rho_exact_cmp = rho_exact_cmp[:Nt_cmp, :Nx_cmp]

    rho_error = rho_nqs_cmp - rho_exact_cmp

    l2 = np.sqrt(
        np.trapezoid(
            np.trapezoid(rho_error**2, x_cmp, axis=1),
            tau_cmp,
        )
    )

    l2_exact = np.sqrt(
        np.trapezoid(
            np.trapezoid(rho_exact_cmp**2, x_cmp, axis=1),
            tau_cmp,
        )
    )

    rel_l2 = l2 / l2_exact

    x_left, x_right = inner_window
    mask_inner = (x_cmp > x_left) & (x_cmp < x_right)

    rho_error_inner = rho_error[:, mask_inner]

    l2_inner = np.sqrt(
        np.trapezoid(
            np.trapezoid(
                rho_error_inner**2,
                x_cmp[mask_inner],
                axis=1,
            ),
            tau_cmp,
        )
    )

    l2_exact_inner = np.sqrt(
        np.trapezoid(
            np.trapezoid(
                rho_exact_cmp[:, mask_inner] ** 2,
                x_cmp[mask_inner],
                axis=1,
            ),
            tau_cmp,
        )
    )

    rel_l2_inner = l2_inner / l2_exact_inner

    max_abs_error = np.max(np.abs(rho_error))
    max_abs_inner_error = np.max(np.abs(rho_error_inner))

    norm_nqs = np.array([
        np.trapezoid(rho_nqs_cmp[it, :], x_cmp)
        for it in range(Nt_cmp)
    ])

    norm_exact = np.array([
        np.trapezoid(rho_exact_cmp[it, :], x_cmp)
        for it in range(Nt_cmp)
    ])

    norm_rel_drift_nqs = (
        np.max(np.abs(norm_nqs - norm_nqs[0]))
        / abs(norm_nqs[0])
    )

    return {
        "x": x_cmp,
        "tau": tau_cmp,
        "t_exact": t_exact_cmp,
        "rho_nqs": rho_nqs_cmp,
        "rho_exact": rho_exact_cmp,
        "rho_error": rho_error,
        "norm_nqs": norm_nqs,
        "norm_exact": norm_exact,
        "rel_l2": rel_l2,
        "rel_l2_inner": rel_l2_inner,
        "max_abs_error": max_abs_error,
        "max_abs_inner_error": max_abs_inner_error,
        "norm_rel_drift_nqs": norm_rel_drift_nqs,
    }