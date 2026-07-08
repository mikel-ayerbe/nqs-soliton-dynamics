import numpy as np

def ho_ground_state_density(x, omega, x0=0.0):
    """
    Exact ground-state density of the 1D harmonic oscillator:
        rho(x) = sqrt(omega/pi) * exp[-omega (x-x0)^2]
    """
    x = np.asarray(x, dtype=np.float64)
    return np.sqrt(omega / np.pi) * np.exp(-omega * (x - x0) ** 2)

def is_ho_linear_benchmark(g, k, gauss_amplitude, wall, tol=1e-12):
    """
    Returns True only if the Hamiltonian corresponds to the
    linear harmonic-oscillator benchmark:
      - no interaction
      - no kick
      - no Gaussian barrier
      - no wall
    """
    return (
        np.isclose(g, 0.0, atol=tol) and
        np.isclose(k, 0.0, atol=tol) and
        np.isclose(gauss_amplitude, 0.0, atol=tol) and
        np.isclose(wall, 0.0, atol=tol)
    )

def is_ho_kick_benchmark(g, w, k, gauss_amplitude, wall, tol=1e-12):
    """
    Returns True only if the case corresponds to the kicked
    linear harmonic oscillator benchmark:
      - no interaction
      - harmonic trap present
      - nonzero kick
      - no Gaussian barrier
      - no wall
    """
    return (
        np.isclose(g, 0.0, atol=tol) and
        (w > tol) and
        (abs(k) > tol) and
        np.isclose(gauss_amplitude, 0.0, atol=tol) and
        np.isclose(wall, 0.0, atol=tol)
    )

def relative_l2_error_density(rho_num, rho_exact, x):
    """
    Relative L2 error between two 1D densities.
    """
    x = np.asarray(x, dtype=np.float64)
    rho_num = np.asarray(rho_num, dtype=np.float64)
    rho_exact = np.asarray(rho_exact, dtype=np.float64)

    num = np.sqrt(np.trapezoid((rho_num - rho_exact) ** 2, x))
    den = np.sqrt(np.trapezoid(rho_exact ** 2, x))
    return num / den

def ho_kick_x_mean(t, k, omega, x0=0.0):
    """
    Exact center-of-mass motion for a kicked 1D harmonic oscillator:
        <x>(t) = x0 + (k/omega) * sin(omega t)
    """
    t = np.asarray(t, dtype=np.float64)
    return x0 + (k / omega) * np.sin(omega * t)

def ho_kick_variance(omega):
    """
    Exact variance of the kicked HO ground state.
    The kick changes the center-of-mass motion, not the width.
    """
    return 1.0 / (2.0 * omega)

def ho_kick_energy_code(omega, k):
    """
    Expected energy with the CURRENT code convention,
    including the +1 offset in hamiltonian().
    """
    return omega / 2.0 + k**2 / 2.0 + 1.0
def is_ho_quench_benchmark(g, k, wi, wf, gauss_amplitude, wall, tol=1e-12):
    """
    Returns True only if the case corresponds to the linear HO quench benchmark:
      - no interaction
      - no kick
      - harmonic trap present before and after the quench
      - actual quench: wi != wf
      - no Gaussian barrier
      - no wall
    """
    return (
        np.isclose(g, 0.0, atol=tol) and
        np.isclose(k, 0.0, atol=tol) and
        (wi > tol) and
        (wf > tol) and
        (not np.isclose(wi, wf, atol=tol)) and
        np.isclose(gauss_amplitude, 0.0, atol=tol) and
        np.isclose(wall, 0.0, atol=tol)
    )

def ho_quench_x2(t, wi, wf):
    """
    Exact <x^2>(t) for a sudden quench wi -> wf in the 1D harmonic oscillator,
    starting from the ground state of wi.
    """
    t = np.asarray(t, dtype=np.float64)
    return (
        (1.0 / (4.0 * wi)) * (1.0 + (wi**2 / wf**2))
        + (1.0 / (4.0 * wi)) * (1.0 - (wi**2 / wf**2)) * np.cos(2.0 * wf * t)
    )

def ho_quench_variance(t, wi, wf):
    """
    For x0 = 0 and k = 0, the variance equals <x^2>(t).
    """
    return ho_quench_x2(t, wi, wf)

def ho_quench_energy_code(wi, wf):
    """
    Expected energy after the quench with the CURRENT code convention,
    including the +1 offset in hamiltonian().
    """
    return 1.0 + wi / 4.0 + (wf**2) / (4.0 * wi)
import numpy as np

def estimate_frequency_from_peaks(t, y):
    """
    Estimate the angular frequency from consecutive local maxima.

    Parameters
    ----------
    t : array-like
        Time grid.
    y : array-like
        Signal sampled on t.

    Returns
    -------
    omega_num : float
        Estimated angular frequency.
    peak_times : ndarray
        Times of detected peaks.
    periods : ndarray
        Consecutive peak-to-peak periods.
    """
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    peak_indices = []
    for i in range(1, len(y) - 1):
        if y[i] > y[i - 1] and y[i] > y[i + 1]:
            peak_indices.append(i)

    peak_indices = np.array(peak_indices, dtype=int)

    if len(peak_indices) < 2:
        return np.nan, t[peak_indices], np.array([])

    peak_times = t[peak_indices]
    periods = np.diff(peak_times)
    T_num = np.mean(periods)
    omega_num = 2.0 * np.pi / T_num

    return omega_num, peak_times, periods
def bright_soliton_density(x, rho_max, x0=0.0):
    """
    Normalized bright-soliton density written in terms of the maximum density rho_max:
        rho(x) = rho_max * sech^2( 2 rho_max (x - x0) )
    so that ∫ rho(x) dx = 1.
    """
    x = np.asarray(x, dtype=np.float64)
    arg = 2.0 * rho_max * (x - x0)
    return rho_max / (np.cosh(arg) ** 2)

def is_bright_soliton_benchmark(g, w, k, gauss_amplitude, wall, tol=1e-12):
    """
    Returns True only if the case corresponds to the free bright soliton benchmark:
      - attractive interaction
      - no trap
      - no kick
      - no Gaussian barrier
      - no wall
    """
    return (
        (g < -tol) and
        np.isclose(w, 0.0, atol=tol) and
        np.isclose(k, 0.0, atol=tol) and
        np.isclose(gauss_amplitude, 0.0, atol=tol) and
        np.isclose(wall, 0.0, atol=tol)
    )

def bright_soliton_energy_code_rhomax(rho_max):
    """
    Expected energy with the CURRENT code convention,
    written in terms of the soliton peak density rho_max.
    """
    return 1.0 + (2.0 / 3.0) * rho_max**2-(2.0 / 3.0) * rho_max
def is_bright_soliton_kick_benchmark(g, w, k, gauss_amplitude, wall, tol=1e-12):
    """
    Returns True only if the case corresponds to the free bright soliton
    benchmark with a nonzero kick:
      - attractive interaction
      - no trap
      - nonzero kick
      - no Gaussian barrier
      - no wall
    """
    return (
        (g < -tol) and
        np.isclose(w, 0.0, atol=tol) and
        (abs(k) > tol) and
        np.isclose(gauss_amplitude, 0.0, atol=tol) and
        np.isclose(wall, 0.0, atol=tol)
    )

def bright_soliton_xmax(t, k, x0=0.0):
    """
    Expected trajectory of the density maximum for a free kicked bright soliton:
        x_max(t) = x0 + k t
    """
    t = np.asarray(t, dtype=np.float64)
    return x0 + k * t

def density_peak_position(rho_t, x):
    """
    Compute the position of the density maximum at each time step.

    Parameters
    ----------
    rho_t : array-like, shape (Nt, Nx)
        Time-dependent density.
    x : array-like, shape (Nx,)
        Spatial grid.

    Returns
    -------
    x_max_t : ndarray, shape (Nt,)
        Position of the density maximum as a function of time.
    """
    rho_t = np.asarray(rho_t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    return np.array([x[np.argmax(rho)] for rho in rho_t])

def density_peak_height(rho_t):
    """
    Compute the maximum density at each time step.
    """
    rho_t = np.asarray(rho_t, dtype=np.float64)
    return np.max(rho_t, axis=1)

def relative_l2_error_signal(y_num, y_exact, t):
    """
    Relative L2 error between two time-dependent signals.
    """
    y_num = np.asarray(y_num, dtype=np.float64)
    y_exact = np.asarray(y_exact, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)

    num = np.sqrt(np.trapezoid((y_num - y_exact) ** 2, t))
    den = np.sqrt(np.trapezoid(y_exact ** 2, t))

    if np.isclose(den, 0.0):
        return np.nan

    return num / den

def max_relative_drift(y):
    """
    Maximum relative drift with respect to the initial value.
    """
    y = np.asarray(y, dtype=np.float64)
    y0 = y[0]

    if np.isclose(y0, 0.0):
        return np.nan

    return np.max(np.abs(y - y0)) / np.abs(y0)
def normalize_wavefunction(psi, x):
    """
    Normaliza una función de onda 1D con la medida continua:
        integral |psi|^2 dx = 1
    """
    psi = np.asarray(psi, dtype=np.complex128)
    x = np.asarray(x, dtype=np.float64)

    norm = np.trapezoid(np.abs(psi)**2, x)
    return psi / np.sqrt(norm)


def overlap_wavefunctions(psi_target, psi_fit, x):
    """
    Solapamiento complejo:
        <psi_target | psi_fit> = integral psi_target^*(x) psi_fit(x) dx
    usando ambas funciones normalizadas.
    """
    psi_target = normalize_wavefunction(psi_target, x)
    psi_fit = normalize_wavefunction(psi_fit, x)

    return np.trapezoid(np.conjugate(psi_target) * psi_fit, x)


def fidelity_wavefunctions(psi_target, psi_fit, x):
    """
    Fidelidad:
        |<psi_target | psi_fit>|^2
    """
    ov = overlap_wavefunctions(psi_target, psi_fit, x)
    return np.abs(ov)**2
def relative_l2_error_wavefunction(psi_num, psi_exact, x):
    """
    Relative L2 error between two 1D wavefunctions.
    Both wavefunctions are normalized before comparison.
    """
    x = np.asarray(x, dtype=np.float64)

    psi_num = normalize_wavefunction(psi_num, x)
    psi_exact = normalize_wavefunction(psi_exact, x)

    num = np.sqrt(np.trapezoid(np.abs(psi_num - psi_exact) ** 2, x))
    den = np.sqrt(np.trapezoid(np.abs(psi_exact) ** 2, x))
    return num / den
def relative_l2_error_wavefunction_phase_aligned(psi_num, psi_exact, x):
    """
    Relative L2 error between two 1D wavefunctions after removing
    the best global phase difference.
    """
    x = np.asarray(x, dtype=np.float64)

    psi_num = normalize_wavefunction(psi_num, x)
    psi_exact = normalize_wavefunction(psi_exact, x)

    ov = np.trapezoid(np.conjugate(psi_exact) * psi_num, x)
    if np.abs(ov) > 0:
        psi_num = psi_num * np.exp(-1j * np.angle(ov))

    num = np.sqrt(np.trapezoid(np.abs(psi_num - psi_exact) ** 2, x))
    den = np.sqrt(np.trapezoid(np.abs(psi_exact) ** 2, x))
    return num / den
def ho_ground_state_wavefunction(x, omega, x0=0.0):
    """
    Exact ground-state wavefunction of the 1D harmonic oscillator:
        psi(x) = (omega/pi)^(1/4) * exp[-omega (x-x0)^2 / 2]
    chosen real and positive.
    """
    x = np.asarray(x, dtype=np.float64)
    psi = (omega / np.pi) ** 0.25 * np.exp(-0.5 * omega * (x - x0) ** 2)
    return psi.astype(np.complex128)
def bright_soliton_wavefunction(x, N, x0=0.0, k=0.0, g=-1.0):
    A = N * abs(g) / 2.0                     # = N/2 para g=-1
    psi = A / np.cosh(A * (x - x0))          # sqrt(A²) = A, argumento correcto
    psi = psi * np.exp(1j * k * x)
    return psi.astype(np.complex128)