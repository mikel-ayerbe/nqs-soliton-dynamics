import numpy as np
from scipy.linalg import expm
from scipy.signal import find_peaks


# =========================================================
# 1. Basic matrix helpers
# =========================================================

def _dagger(M):
    return np.conjugate(M.T)


def _as_complex_matrix(M, name="matrix"):
    M = np.asarray(M, dtype=np.complex128)
    if M.ndim != 2:
        raise ValueError(f"{name} must be 2D")
    return M


def solve_sylvester_right_dagger(A, RHS):
    """
    Solve:
        A X + X A^† = RHS

    by vectorization:
        vec(A X + X A^†) = (I ⊗ A + conj(A) ⊗ I) vec(X)
    """
    A = _as_complex_matrix(A, "A")
    RHS = _as_complex_matrix(RHS, "RHS")

    n = A.shape[0]

    if A.shape != (n, n):
        raise ValueError("A must be square")

    if RHS.shape != (n, n):
        raise ValueError("RHS must have same shape as A")

    K = (
        np.kron(np.eye(n, dtype=np.complex128), A)
        + np.kron(np.conjugate(A), np.eye(n, dtype=np.complex128))
    )

    rhs = RHS.reshape(n * n, order="F")
    X = np.linalg.solve(K, rhs).reshape((n, n), order="F")

    return X


# =========================================================
# 2. General ABC formalism from section 6
# =========================================================

def q0_n0_from_lyapunov(A, B, C, symmetrize=True):
    """
    Compute Q(0,0) and N(0) from section 6:

        Q0 A + A^† Q0 = C^† C
        A N0 + N0 A^† = B B^†

    Parameters
    ----------
    A : (n,n)
    B : (n,1)
    C : (1,n)

    Returns
    -------
    Q0, N0 : (n,n)
    """
    A = _as_complex_matrix(A, "A")
    B = _as_complex_matrix(B, "B")
    C = _as_complex_matrix(C, "C")

    n = A.shape[0]

    if A.shape != (n, n):
        raise ValueError("A must be square")

    if B.shape != (n, 1):
        raise ValueError("B must have shape (n,1)")

    if C.shape != (1, n):
        raise ValueError("C must have shape (1,n)")

    RHS_Q = _dagger(C) @ C
    RHS_N = B @ _dagger(B)

    Q0 = solve_sylvester_right_dagger(_dagger(A), RHS_Q)
    N0 = solve_sylvester_right_dagger(A, RHS_N)

    if symmetrize:
        Q0 = 0.5 * (Q0 + _dagger(Q0))
        N0 = 0.5 * (N0 + _dagger(N0))

    return Q0, N0


def p_matrix_general(x, t, A):
    """
    P(x,t) from section 6:
        P(x,t) = exp(-2 A x - 4 i A^2 t)
    """
    A = _as_complex_matrix(A, "A")
    return expm(-2.0 * A * x - 4.0j * (A @ A) * t)


def omega_matrix_general(x, t, A, Q0, N0):
    """
    Omega(x,t):
        Omega = I + P^† Q0 P N0
    """
    A = _as_complex_matrix(A, "A")
    Q0 = _as_complex_matrix(Q0, "Q0")
    N0 = _as_complex_matrix(N0, "N0")

    P = p_matrix_general(x, t, A)

    return (
        np.eye(A.shape[0], dtype=np.complex128)
        + _dagger(P) @ Q0 @ P @ N0
    )


def exact_nls_abc_general(
    x,
    t,
    A,
    B,
    C,
    check_invertibility=True,
    return_debug=False,
):
    """
    General section-6 exact solution:

        u(x,t) = -2 B^† Omega(x,t)^(-1) P(x,t)^† C^†
    """
    x_arr = np.asarray(x, dtype=np.float64)
    x_flat = x_arr.ravel()

    A = _as_complex_matrix(A, "A")
    B = _as_complex_matrix(B, "B")
    C = _as_complex_matrix(C, "C")

    Q0, N0 = q0_n0_from_lyapunov(A, B, C, symmetrize=True)

    out = np.empty_like(x_flat, dtype=np.complex128)
    omega_min_abs = np.empty_like(x_flat, dtype=np.float64)

    for k, xx in enumerate(x_flat):
        P = p_matrix_general(xx, t, A)

        Omega = (
            np.eye(A.shape[0], dtype=np.complex128)
            + _dagger(P) @ Q0 @ P @ N0
        )

        det_omega = np.linalg.det(Omega)
        omega_min_abs[k] = np.min(np.abs(np.linalg.eigvals(Omega)))

        if check_invertibility and not np.isfinite(det_omega):
            raise RuntimeError(f"Non-finite det(Omega) at x={xx}, t={t}")

        rhs = _dagger(P) @ _dagger(C)
        sol = np.linalg.solve(Omega, rhs)

        out[k] = -2.0 * (_dagger(B) @ sol)[0, 0]

    psi = out.reshape(x_arr.shape)

    if return_debug:
        return psi, {
            "Q0": Q0,
            "N0": N0,
            "omega_min_abs": omega_min_abs.reshape(x_arr.shape),
        }

    return psi


def exact_nls_abc_general_on_time_grid(
    x,
    t_grid,
    A,
    B,
    C,
    check_invertibility=True,
):
    """
    Evaluate the general ABC exact solution on a time grid.

    Returns
    -------
    psi_tx : ndarray, shape (Nt, Nx)
    """
    x = np.asarray(x, dtype=np.float64)
    t_grid = np.asarray(t_grid, dtype=np.float64)

    psi_tx = np.empty((len(t_grid), len(x)), dtype=np.complex128)

    for i, tt in enumerate(t_grid):
        psi_tx[i] = exact_nls_abc_general(
            x,
            tt,
            A,
            B,
            C,
            check_invertibility=check_invertibility,
            return_debug=False,
        )

    return psi_tx


def exact_nls_abc_general_on_time_grid_scaled(
    x,
    t_grid,
    A,
    B,
    C,
    time_scale=1.0,
    check_invertibility=True,
    progress=True,
    progress_chunk=50,
):
    """
    Evaluate the general ABC exact solution on a physical time grid,
    using t_internal = time_scale * t_grid.

    For your NQS comparison you were using:
        exact_nls_abc_general(x, t / 2, A, B, C)

    so here use:
        time_scale = 0.5
    """
    x = np.asarray(x, dtype=np.float64)
    t_grid = np.asarray(t_grid, dtype=np.float64)

    psi_tx = np.empty((len(t_grid), len(x)), dtype=np.complex128)

    if progress:
        try:
            from tqdm.auto import tqdm
            iterator = tqdm(
                range(0, len(t_grid), progress_chunk),
                desc="Exact evolution",
                mininterval=0.2,
            )
        except Exception:
            iterator = range(0, len(t_grid), progress_chunk)
    else:
        iterator = range(0, len(t_grid), progress_chunk)

    for start in iterator:
        end = min(start + progress_chunk, len(t_grid))

        for it in range(start, end):
            psi_tx[it] = exact_nls_abc_general(
                x,
                time_scale * t_grid[it],
                A,
                B,
                C,
                check_invertibility=check_invertibility,
                return_debug=False,
            )

    return psi_tx


# =========================================================
# 3. Collision-specific helpers
# =========================================================

def detect_two_main_peaks(rho, x):
    rho = np.asarray(rho, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)

    peaks, props = find_peaks(
        rho,
        height=0.08 * np.max(rho),
        distance=max(10, len(x) // 30),
    )

    if len(peaks) < 2:
        return None

    heights = props["peak_heights"]
    idx_sorted = np.argsort(heights)[::-1][:2]

    main_peaks = peaks[idx_sorted]
    order = np.argsort(x[main_peaks])

    return main_peaks[order]


def weighted_local_phase_demodulated(
    x,
    psi,
    rho,
    x_center,
    slope,
    window=2.0,
):
    x = np.asarray(x, dtype=np.float64)
    psi = np.asarray(psi, dtype=np.complex128)
    rho = np.asarray(rho, dtype=np.float64)

    mask = np.abs(x - x_center) < window

    if np.sum(mask) < 5:
        return np.nan

    psi_demod = psi[mask] * np.exp(-1j * slope * x[mask])

    z = np.trapezoid(
        rho[mask] * psi_demod / np.maximum(np.abs(psi_demod), 1e-14),
        x[mask],
    )

    return np.angle(z)


def wrap_to_pi(phi):
    return np.angle(np.exp(1j * phi))


def build_abc_two_solitons(
    eta1,
    eta2,
    k1,
    k2,
    x01,
    x02,
    phi_input,
):
    """
    Build the ABC matrices for the two-bright-soliton collision.

    Conventions match your original main block:

        xi_j = -k_j / 2
        lambda_j = xi_j + i eta_j
        c1 = 2 eta1 exp(2 eta1 x01)
        c2 = 2 eta2 exp(2 eta2 x02) s12^2 exp(i phi_input)
    """
    xi1 = -k1 / 2.0
    xi2 = -k2 / 2.0

    lam1 = xi1 + 1j * eta1
    lam2 = xi2 + 1j * eta2

    s12 = (lam1 - np.conjugate(lam2)) / (lam1 - lam2)
    interaction_factor_complex = s12**2

    c1 = 2.0 * eta1 * np.exp(2.0 * eta1 * x01)

    c2 = (
        2.0
        * eta2
        * np.exp(2.0 * eta2 * x02)
        * interaction_factor_complex
        * np.exp(1j * phi_input)
    )

    A = np.array(
        [
            [-1j * lam1, 0.0 + 0.0j],
            [0.0 + 0.0j, -1j * lam2],
        ],
        dtype=np.complex128,
    )

    B = np.array(
        [
            [1.0 + 0.0j],
            [1.0 + 0.0j],
        ],
        dtype=np.complex128,
    )

    C = np.array(
        [
            [c1, c2],
        ],
        dtype=np.complex128,
    )

    return {
        "A": A,
        "B": B,
        "C": C,
        "lam1": lam1,
        "lam2": lam2,
        "xi1": xi1,
        "xi2": xi2,
        "s12": s12,
        "interaction_factor_complex": interaction_factor_complex,
        "c1": c1,
        "c2": c2,
    }


def intrinsic_relative_phase_from_exact(
    x,
    eta1,
    eta2,
    k1,
    k2,
    x01,
    x02,
    phi_input,
    window=2.0,
):
    """
    Measure the intrinsic relative phase between the two separated peaks
    at t = 0, after demodulating the local carrier slopes.
    """
    data = build_abc_two_solitons(
        eta1=eta1,
        eta2=eta2,
        k1=k1,
        k2=k2,
        x01=x01,
        x02=x02,
        phi_input=phi_input,
    )

    psi_t0 = exact_nls_abc_general(
        x,
        0.0,
        data["A"],
        data["B"],
        data["C"],
        check_invertibility=True,
        return_debug=False,
    )

    rho_t0 = np.abs(psi_t0) ** 2

    peaks = detect_two_main_peaks(rho_t0, x)

    if peaks is None:
        raise RuntimeError(
            "Could not detect two peaks during automatic phase calibration."
        )

    x_left = x[peaks[0]]
    x_right = x[peaks[1]]

    slope_right = -2.0 * np.real(data["lam1"])
    slope_left = -2.0 * np.real(data["lam2"])

    phi_left_intr = weighted_local_phase_demodulated(
        x,
        psi_t0,
        rho_t0,
        x_left,
        slope_left,
        window=window,
    )

    phi_right_intr = weighted_local_phase_demodulated(
        x,
        psi_t0,
        rho_t0,
        x_right,
        slope_right,
        window=window,
    )

    return wrap_to_pi(phi_right_intr - phi_left_intr)


def calibrate_phi_input(
    x,
    eta1,
    eta2,
    k1,
    k2,
    x01,
    x02,
    phi_desired,
    window=2.0,
):
    """
    Calibrate phi_input so that the measured intrinsic relative phase
    at t=0 matches phi_desired.
    """
    phi_offset = intrinsic_relative_phase_from_exact(
        x=x,
        eta1=eta1,
        eta2=eta2,
        k1=k1,
        k2=k2,
        x01=x01,
        x02=x02,
        phi_input=0.0,
        window=window,
    )

    phi_input = wrap_to_pi(phi_desired - phi_offset)

    return phi_input, phi_offset


def prepare_two_soliton_collision_abc(
    x,
    eta1,
    eta2,
    k1,
    k2,
    x01,
    x02,
    phi_desired,
    calibration_window=2.0,
):
    """
    High-level helper used by main.py.

    It:
      1. calibrates phi_input,
      2. builds A, B, C,
      3. returns all relevant exact parameters in one dictionary.
    """
    x = np.asarray(x, dtype=np.float64)

    phi_input, phi_offset = calibrate_phi_input(
        x=x,
        eta1=eta1,
        eta2=eta2,
        k1=k1,
        k2=k2,
        x01=x01,
        x02=x02,
        phi_desired=phi_desired,
        window=calibration_window,
    )

    data = build_abc_two_solitons(
        eta1=eta1,
        eta2=eta2,
        k1=k1,
        k2=k2,
        x01=x01,
        x02=x02,
        phi_input=phi_input,
    )

    data.update(
        {
            "eta1": eta1,
            "eta2": eta2,
            "k1": k1,
            "k2": k2,
            "x01": x01,
            "x02": x02,
            "phi_desired": phi_desired,
            "phi_offset": phi_offset,
            "phi_input": phi_input,
        }
    )

    return data


def density_from_psi(psi):
    return np.abs(np.asarray(psi)) ** 2


def norm_from_density(rho, x):
    return np.trapezoid(np.asarray(rho), np.asarray(x, dtype=np.float64))


def norms_on_time_grid(rho_tx, x):
    rho_tx = np.asarray(rho_tx)
    x = np.asarray(x, dtype=np.float64)

    return np.array([
        norm_from_density(rho_tx[it], x)
        for it in range(len(rho_tx))
    ])