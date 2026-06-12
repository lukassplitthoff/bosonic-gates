"""
transmon.py
===========
Transmon qubit Hamiltonian in the harmonic oscillator basis.

All energies are in GHz (hbar = 1).

Reference: Koch et al., PRA 76, 042319 (2007).
"""

import math
import warnings

import numpy as np
import qutip as qtp

from .harmonic_oscillator import _ho_ladder_ops


_EJ_EC_MIN = 10.0  # Minimum Ej/Ec ratio for valid HO-basis representation


def transmon_hamiltonian(
    Ej: float,
    Ec: float,
    N: int,
    ng: float = 0.0,
) -> qtp.Qobj:
    """
    Build the transmon Hamiltonian in the harmonic oscillator basis.

    The Hamiltonian is:

        H = 4*Ec*(n_op - ng*I)^2 - Ej*cos(phi_op)

    The oscillator length uses Ej as the effective inductive energy scale:

        phi_naught = (8*Ec/Ej)^(1/4)
        phi_op = (phi_naught/sqrt(2)) * (c + cdag)
        n_op   = -i/(sqrt(2)*phi_naught) * (c - cdag)

    The cosine is computed exactly via matrix exponentiation.

    Parameters
    ----------
    Ej : float
        Josephson energy in GHz.
    Ec : float
        Capacitive (charging) energy in GHz.
    N : int
        Hilbert space truncation dimension.
    ng : float, optional
        Dimensionless gate charge offset. Default 0.0 (sweet spot).
        At ng = 0.5 the charge degeneracy point is reached.

    Returns
    -------
    H : qtp.Qobj  (dimension N, Hermitian)

    Notes
    -----
    Valid in the transmon regime (Ej/Ec >> 1, roughly >= 20). For
    Cooper-pair-box parameters (Ej/Ec ~ 1) a charge-basis representation
    is more accurate.

    For a flux-tunable (SQUID) transmon, compute the effective Josephson
    energy with squid_ej_eff() and pass it as Ej.

    Example
    -------
    >>> H = transmon_hamiltonian(Ej=20.0, Ec=0.3, N=10)
    """
    if Ej <= 0:
        raise ValueError(
            f"Ej must be positive (got {Ej}). "
            "For a symmetric SQUID at phi=pi, Ej_eff=0 and the transmon "
            "HO basis is undefined. Use asymmetric junctions (Ej1 != Ej2) "
            "to maintain a non-zero minimum Ej_eff = |Ej1 - Ej2|."
        )
    if Ej / Ec < _EJ_EC_MIN:
        warnings.warn(
            f"Ej/Ec = {Ej/Ec:.1f} < {_EJ_EC_MIN}. "
            "The HO basis may be inaccurate outside the transmon regime. "
            "Consider using a charge-basis representation for Ej/Ec < 20.",
            stacklevel=2,
        )

    c, cdag = _ho_ladder_ops(N)
    phi_naught = (8.0 * Ec / Ej) ** 0.25

    phi_op = (phi_naught / math.sqrt(2)) * (c + cdag)
    n_op   = (-1j / (math.sqrt(2) * phi_naught)) * (c - cdag)

    H_C = 4.0 * Ec * (n_op - ng * qtp.qeye(N)) ** 2

    expm_plus  = (1j * phi_op).expm()
    expm_minus = (-1j * phi_op).expm()
    H_J = -Ej * (expm_plus + expm_minus) / 2.0

    return H_C + H_J
