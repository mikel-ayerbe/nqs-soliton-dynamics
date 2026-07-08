import numpy as np
from scipy.optimize import curve_fit

def is_dark_wall_benchmark(g, w, k, gauss_amplitude, wall, tol=1e-12):
    """
    Caso benchmark para una solución tipo healing-wall:
      - interacción repulsiva
      - sin trampa
      - sin kick
      - sin barrera gaussiana
      - pared presente
    """
    return (
        (g > tol) and
        np.isclose(w, 0.0, atol=tol) and
        np.isclose(k, 0.0, atol=tol) and
        np.isclose(gauss_amplitude, 0.0, atol=tol) and
        (wall > tol)
    )


def healing_length(mu):
    """
    Healing length para la GPE 1D en las unidades del código:
        xi = 1 / sqrt(2 mu)
    """
    return 1.0 / np.sqrt(2.0 * mu)


def bulk_amplitude(mu, g):
    """
    Amplitud de bulk:
        psi_0 = sqrt(mu / g)
    """
    return np.sqrt(mu / g)


def dark_wall_profile(s, mu, g):
    """
    Perfil analítico de healing junto a una pared:
        psi(s) = psi_0 * tanh( s / (sqrt(2) xi) )
    donde s >= 0 es la distancia a la pared.
    """
    s = np.asarray(s, dtype=np.float64)
    psi0 = bulk_amplitude(mu, g)
    xi = healing_length(mu)
    return psi0 * np.tanh(s / (np.sqrt(2.0) * xi))


def dark_wall_density_profile(s, mu, g):
    """
    Densidad analítica asociada al perfil tanh.
    """
    psi = dark_wall_profile(s, mu, g)
    return psi**2


def extract_allowed_half(x, y, wall_position=0.0, allowed_side='left'):
    """
    Extrae la mitad permitida y devuelve:
      s = distancia a la pared (>= 0)
      y_half = valores correspondientes
    Si allowed_side='left', toma x < wall_position.
    Si allowed_side='right', toma x > wall_position.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if allowed_side == 'left':
        mask = x < wall_position
        s = wall_position - x[mask]
    elif allowed_side == 'right':
        mask = x > wall_position
        s = x[mask] - wall_position
    else:
        raise ValueError("allowed_side must be 'left' or 'right'")

    y_half = y[mask]

    order = np.argsort(s)
    return s[order], y_half[order]


def tanh_profile_for_fit(s, A, xi):
    """
    Perfil tanh con amplitud y healing length libres.
    """
    s = np.asarray(s, dtype=np.float64)
    return A * np.tanh(s / (np.sqrt(2.0) * xi))


def fit_dark_wall_profile(s, psi_abs, A0=None, xi0=None):
    """
    Ajuste del perfil |psi| a A * tanh(s / (sqrt(2) xi)).
    """
    s = np.asarray(s, dtype=np.float64)
    psi_abs = np.asarray(psi_abs, dtype=np.float64)

    if A0 is None:
        A0 = np.max(psi_abs)
    if xi0 is None:
        xi0 = 1.0

    popt, pcov = curve_fit(
        tanh_profile_for_fit,
        s,
        psi_abs,
        p0=[A0, xi0],
        bounds=(0.0, np.inf)
    )
    return popt, pcov


def relative_l2_error_profile(y_num, y_exact, x):
    """
    Relative L2 error entre dos perfiles 1D.
    """
    x = np.asarray(x, dtype=np.float64)
    y_num = np.asarray(y_num, dtype=np.float64)
    y_exact = np.asarray(y_exact, dtype=np.float64)

    num = np.sqrt(np.trapezoid((y_num - y_exact) ** 2, x))
    den = np.sqrt(np.trapezoid(y_exact ** 2, x))

    if np.isclose(den, 0.0):
        return np.nan
    return num / den