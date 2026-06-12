"""
potential.py
============
SNAIL (Superconducting Nonlinear Asymmetric Inductive eLement) potential
and derived circuit parameters.

Reference: Chapman et al., PRX Quantum 4, 020355 (2023), Appendix B.

SNAIL circuit: a small Josephson junction (energy beta*E_J) shunted by an
array of M larger junctions (each energy E_J).

Potential (Eq. B4):
    U_s(phi_s) / E_J = -beta * cos(phi_s - phi_e) - M * cos(phi_s / M)

Parameters
----------
beta   : float   ratio of small-to-large junction energies, 0 < beta < 1/M
M      : int     number of large junctions in the shunting array
phi_e  : float   external flux bias in radians (phi_e = 2*pi * Phi_e/Phi_0)
E_J    : float   Josephson energy of each large junction (GHz, i.e. E_J/h)
E_C    : float   charging energy (GHz, i.e. E_C/h)
E_L    : float   linear inductive energy of external inductance (GHz).
                 Set to 0 (default) when there is no external inductance
                 (i.e. a bare SNAIL, participation p = 1).

All energies / frequencies are in GHz (i.e., E/h, with h = Planck constant).
"""

from __future__ import annotations

import warnings
import numpy as np
from scipy.optimize import brentq, minimize_scalar


# ---------------------------------------------------------------------------
# Potential
# ---------------------------------------------------------------------------

def snail_potential(
    phi: np.ndarray,
    *,
    beta: float,
    M: int,
    phi_e: float,
    E_J: float = 1.0,
) -> np.ndarray:
    """
    SNAIL potential energy as a function of phase (Eq. B4).

        U_s(phi) = E_J * [-beta * cos(phi - phi_e) - M * cos(phi / M)]

    Parameters
    ----------
    phi : array-like
        Phase variable (dimensionless, in radians).
    beta : float
        Junction asymmetry: ratio of small-junction to large-junction
        Josephson energies.  Must satisfy 0 < beta < 1/M.
    M : int
        Number of large junctions in the shunting array.
    phi_e : float
        External flux bias in radians.
    E_J : float
        Josephson energy of each large junction in GHz.  Default 1
        (normalised units).

    Returns
    -------
    U : ndarray
        Potential in GHz (same shape as phi).

    Example
    -------
    >>> phi = np.linspace(-2*np.pi, 2*np.pi, 500)
    >>> U = snail_potential(phi, beta=0.1, M=3, phi_e=0.4*2*np.pi)
    """
    phi = np.asarray(phi, dtype=float)
    return E_J * (-beta * np.cos(phi - phi_e) - M * np.cos(phi / M))


# ---------------------------------------------------------------------------
# Potential minimum
# ---------------------------------------------------------------------------

def _potential_deriv(phi: float, beta: float, M: int, phi_e: float) -> float:
    """dU/dphi / E_J = beta*sin(phi - phi_e) + sin(phi/M) (Eq. B5)."""
    return beta * np.sin(phi - phi_e) + np.sin(phi / M)


def find_snail_minimum(
    beta: float,
    M: int,
    phi_e: float,
    phi_search_center: float = 0.0,
) -> float:
    """
    Find the principal potential minimum phi_m by solving Eq. B5:

        beta * sin(phi_m - phi_e) + sin(phi_m / M) = 0

    The SNAIL has a unique minimum per 2*pi*M interval for beta < 1/M.
    We search in [phi_search_center - pi*M, phi_search_center + pi*M].

    Parameters
    ----------
    beta : float
    M : int
    phi_e : float
        External flux in radians.
    phi_search_center : float
        Centre of the search interval.  Default 0.

    Returns
    -------
    phi_m : float
        Phase at the potential minimum.

    Example
    -------
    >>> phi_m = find_snail_minimum(beta=0.1, M=3, phi_e=0.4*2*np.pi)
    """
    if beta >= 1.0 / M:
        warnings.warn(
            f"beta = {beta:.4f} >= 1/M = {1/M:.4f}. "
            "Multiple potential minima may exist; result may be unreliable.",
            stacklevel=2,
        )

    lo = phi_search_center - np.pi * M
    hi = phi_search_center + np.pi * M

    result = minimize_scalar(
        lambda p: -beta * np.cos(p - phi_e) - M * np.cos(p / M),
        bounds=(lo, hi),
        method="bounded",
        options={"xatol": 1e-12},
    )
    return float(result.x)


# ---------------------------------------------------------------------------
# Taylor coefficients c_2 ... c_5  (Eq. B7)
# ---------------------------------------------------------------------------

def snail_taylor_coefficients(
    beta: float,
    M: int,
    phi_e: float,
    phi_m: float | None = None,
) -> dict[str, float]:
    """
    Taylor expansion coefficients of U_s/E_J around the potential minimum
    (Eq. B6-B7):

        U_s/E_J ≈ c_2/2! * dphi^2 + c_3/3! * dphi^3
                 + c_4/4! * dphi^4 + c_5/5! * dphi^5 + ...

    where dphi = phi - phi_m.

    Parameters
    ----------
    beta, M, phi_e : as defined in module docstring.
    phi_m : float, optional
        Pre-computed potential minimum.  Computed via find_snail_minimum
        if not provided.

    Returns
    -------
    dict with keys 'c2', 'c3', 'c4', 'c5', 'phi_m'.

    Example
    -------
    >>> coeffs = snail_taylor_coefficients(beta=0.1, M=3, phi_e=0.4*2*np.pi)
    >>> print(coeffs['c3'])  # positive for phi_e near pi
    """
    if phi_m is None:
        phi_m = find_snail_minimum(beta, M, phi_e)

    cos_small = np.cos(phi_m - phi_e)
    sin_array = np.sin(phi_m / M)
    cos_array = np.cos(phi_m / M)

    c2 = beta * cos_small + (1.0 / M) * cos_array
    c3 = (M**2 - 1) / M**2 * sin_array
    c4 = -beta * cos_small - (1.0 / M**3) * cos_array
    c5 = (1.0 - M**4) / M**4 * sin_array

    return {"c2": c2, "c3": c3, "c4": c4, "c5": c5, "phi_m": phi_m}


# ---------------------------------------------------------------------------
# Renormalized coefficients c̃_j  (Eq. B9)
# ---------------------------------------------------------------------------

def snail_renormalized_coefficients(
    beta: float,
    M: int,
    phi_e: float,
    E_J: float,
    E_L: float = 0.0,
    phi_m: float | None = None,
) -> dict[str, float]:
    """
    Renormalized potential coefficients c̃_j accounting for a linear
    (geometric) inductance E_L in parallel with the SNAIL (Eq. B9).

    The inductive participation ratio of the Josephson junctions is:
        p = c_2 * E_J / (E_L + c_2 * E_J)

    For a bare SNAIL (E_L = 0) the participation p = 1 and c̃_j = c_j.

    Parameters
    ----------
    beta, M, phi_e, E_J : as defined in module docstring.
    E_L : float
        Linear inductive energy in GHz.  Default 0 (no external inductor).
    phi_m : float, optional
        Pre-computed potential minimum.

    Returns
    -------
    dict with keys 'c2t', 'c3t', 'c4t', 'c5t', 'p', plus all bare 'c*' keys.
    """
    bare = snail_taylor_coefficients(beta, M, phi_e, phi_m=phi_m)
    c2, c3, c4, c5 = bare["c2"], bare["c3"], bare["c4"], bare["c5"]

    if E_L == 0.0:
        p = 1.0
    else:
        p = c2 * E_J / (E_L + c2 * E_J)

    q = 1.0 - p

    c2t = p * c2
    c3t = p**3 * c3
    c4t = p**4 * (c4 - 3.0 * c3**2 / c2 * q)
    c5t = p**5 * (c5 - 10.0 * c4 * c3 / c2 * q + 15.0 * c3**2 / c2**2 * q**2)

    return {**bare, "c2t": c2t, "c3t": c3t, "c4t": c4t, "c5t": c5t, "p": p}


# ---------------------------------------------------------------------------
# Circuit parameters  (Eq. B8)
# ---------------------------------------------------------------------------

def snail_circuit_params(
    beta: float,
    M: int,
    phi_e: float,
    E_J: float,
    E_C: float,
    E_L: float = 0.0,
    phi_m: float | None = None,
) -> dict[str, float]:
    """
    Compute the dressed circuit parameters of the SNAIL mode (Eq. B8):

        omega_c  = sqrt(8 * c̃_2 * E_C * E_J)   [GHz]
        phi_c    = (2 * E_C / (c̃_2 * E_J))^(1/4)   [dimensionless ZPF]
        g_3      = E_J * phi_c^3 * c̃_3 / 6         [GHz]
        g_4      = E_J * phi_c^4 * c̃_4 / 24        [GHz]
        g_5      = E_J * phi_c^5 * c̃_5 / 120       [GHz]
        alpha_c  = 12 * (g_4 - 5*g_3^2 / omega_c)  [GHz]  (Eq. 3)
        xi_crit  = 3*pi / (2*phi_c)                        (Eq. F11)

    All energies E_J, E_C, E_L and returned frequencies are in GHz (= E/h).

    Parameters
    ----------
    beta, M, phi_e, E_J, E_C, E_L : as defined in module docstring.
    phi_m : float, optional
        Pre-computed potential minimum.

    Returns
    -------
    dict with keys: 'omega_c', 'phi_c', 'g3', 'g4', 'g5', 'alpha_c', 'xi_crit', 'p',
    plus all 'c*' / 'c*t' keys from snail_renormalized_coefficients.

    Example
    -------
    >>> params = snail_circuit_params(beta=0.1, M=3, phi_e=0.4*2*np.pi,
    ...                               E_J=10.0, E_C=0.2)
    >>> print(f"omega_c = {params['omega_c']:.3f} GHz")
    >>> print(f"g3 = {params['g3']*1e3:.1f} MHz")
    """
    coeffs = snail_renormalized_coefficients(
        beta, M, phi_e, E_J, E_L=E_L, phi_m=phi_m
    )
    c2t = coeffs["c2t"]
    c3t = coeffs["c3t"]
    c4t = coeffs["c4t"]
    c5t = coeffs["c5t"]

    omega_c = np.sqrt(8.0 * c2t * E_C * E_J)
    phi_c   = (2.0 * E_C / (c2t * E_J)) ** 0.25

    g3 = E_J * phi_c**3 * c3t / 6.0
    g4 = E_J * phi_c**4 * c4t / 24.0
    g5 = E_J * phi_c**5 * c5t / 120.0

    alpha_c = 12.0 * (g4 - 5.0 * g3**2 / omega_c)

    xi_crit = 3.0 * np.pi / (2.0 * phi_c)

    return {
        **coeffs,
        "omega_c": omega_c,
        "phi_c": phi_c,
        "g3": g3,
        "g4": g4,
        "g5": g5,
        "alpha_c": alpha_c,
        "xi_crit": xi_crit,
    }


# ---------------------------------------------------------------------------
# Beam-splitter rate vs pump amplitude
# ---------------------------------------------------------------------------

def gbs_linear(
    xi: np.ndarray,
    g3: float,
    ga_over_delta_a: float,
    gb_over_delta_b: float,
) -> np.ndarray:
    """
    Lowest-order (perturbative) beam-splitter rate (Eq. F9):

        g_bs = 6 * (g_a / Delta_a) * (g_b / Delta_b) * g_3 * |xi|

    Parameters
    ----------
    xi : array-like  dimensionless pump amplitude
    g3 : float       cubic nonlinearity [GHz]
    ga_over_delta_a : float   hybridisation g_a/Delta_a of cavity Alice
    gb_over_delta_b : float   hybridisation g_b/Delta_b of cavity Bob

    Returns
    -------
    g_bs : ndarray in GHz
    """
    xi = np.asarray(xi, dtype=float)
    return 6.0 * ga_over_delta_a * gb_over_delta_b * g3 * np.abs(xi)


def gbs_rwa(
    xi: np.ndarray,
    g_odd: list[float],
    ga_over_delta_a: float,
    gb_over_delta_b: float,
    n_terms: int | None = None,
) -> np.ndarray:
    """
    Beam-splitter rate including all odd nonlinearities via RWA (Eq. F10):

        g_bs = (g_a/Delta_a) * (g_b/Delta_b) * |xi|
               * sum_{m=1}^{n_terms} [(2m+1)! / (m! (m-1)!)] * g_{2m+1} * |xi|^{2m-2}

    Parameters
    ----------
    xi : array-like
        Dimensionless pump amplitude.
    g_odd : list of float
        Odd nonlinearities [g3, g5, g7, ...] in GHz.
        g_odd[0] = g3 (m=1), g_odd[1] = g5 (m=2), etc.
    ga_over_delta_a : float
    gb_over_delta_b : float
    n_terms : int, optional
        Number of terms to include.  Defaults to len(g_odd).

    Returns
    -------
    g_bs : ndarray in GHz
    """
    import math

    xi = np.asarray(xi, dtype=float)
    xi_abs = np.abs(xi)

    if n_terms is None:
        n_terms = len(g_odd)

    result = np.zeros_like(xi_abs)
    for m, g2m1 in enumerate(g_odd[:n_terms], start=1):
        coeff = math.factorial(2 * m + 1) / (math.factorial(m) * math.factorial(m - 1))
        result += coeff * g2m1 * xi_abs ** (2 * m - 2)

    return ga_over_delta_a * gb_over_delta_b * xi_abs * result
