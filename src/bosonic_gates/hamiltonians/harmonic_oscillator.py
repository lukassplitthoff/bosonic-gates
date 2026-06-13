"""
harmonic_oscillator.py
======================
Harmonic resonator Hamiltonian and coupled qubit-resonator system.

All energies are in GHz (hbar = 1).
Tensor ordering convention: qubit ⊗ resonator.
"""

import numpy as np
import qutip as qtp


# ---------------------------------------------------------------------------
# Internal helper — shared by all hamiltonians modules
# ---------------------------------------------------------------------------

def _ho_ladder_ops(N: int):
    """Return (annihilation, creation) operators for an N-dimensional Fock space."""
    return qtp.destroy(N), qtp.create(N)


# ---------------------------------------------------------------------------
# Resonator
# ---------------------------------------------------------------------------

def resonator_hamiltonian(w: float, M: int) -> qtp.Qobj:
    """
    Build the harmonic resonator Hamiltonian.

        H_res = w * (adag*a + 1/2)

    The zero-point energy term (w/2) is included, so eigenvalues are
    w/2, 3w/2, 5w/2, ... rather than 0, w, 2w, ...  This does not affect
    dynamics or transition frequencies, but will shift absolute energies.
    To drop the constant, subtract ``w/2 * qt.qeye(M)`` from the result.

    Parameters
    ----------
    w : float
        Angular frequency in GHz (ħ = 1 convention; same as Ej/Ec in transmon).
    M : int
        Hilbert space truncation dimension.

    Returns
    -------
    H : qtp.Qobj  (dimension M)
    """
    a, adag = _ho_ladder_ops(M)
    return w * (adag * a + 0.5 * qtp.qeye(M))


def resonator_number_operator(N: int, M: int) -> qtp.Qobj:
    """
    Return the resonator photon number operator in the full qubit-resonator space.

        n_res = I_qubit ⊗ (adag * a)

    Tensor order: qubit left, resonator right.

    Parameters
    ----------
    N : int  Qubit Hilbert space dimension.
    M : int  Resonator Hilbert space dimension.

    Returns
    -------
    n_res : qtp.Qobj  (dimension N*M)
    """
    a, adag = _ho_ladder_ops(M)
    return qtp.tensor(qtp.qeye(N), adag * a)


# ---------------------------------------------------------------------------
# Coupled qubit-resonator system
# ---------------------------------------------------------------------------

def coupled_system_hamiltonian(
    H_qubit: qtp.Qobj,
    H_resonator: qtp.Qobj,
    N: int,
    M: int,
    g: float,
) -> qtp.Qobj:
    """
    Assemble the full qubit-resonator Hamiltonian with Jaynes-Cummings coupling.

        H = H_qubit ⊗ I_res + I_qubit ⊗ H_res + g*(c ⊗ adag + cdag ⊗ a)

    The coupling term is in the rotating-wave approximation (RWA), retaining
    only energy-conserving processes.

    Tensor order: qubit left, resonator right.

    Parameters
    ----------
    H_qubit : qtp.Qobj
        Qubit Hamiltonian of dimension N.
    H_resonator : qtp.Qobj
        Resonator Hamiltonian of dimension M.
    N : int
        Qubit Hilbert space dimension (must equal H_qubit.shape[0]).
    M : int
        Resonator Hilbert space dimension (must equal H_resonator.shape[0]).
    g : float
        Qubit-resonator coupling strength in GHz.

    Returns
    -------
    H_sys : qtp.Qobj  (dimension N*M, Hermitian)

    Raises
    ------
    ValueError
        If H_qubit or H_resonator dimensions do not match N and M.
    """
    if H_qubit.shape[0] != N:
        raise ValueError(
            f"H_qubit has dimension {H_qubit.shape[0]}, expected N={N}."
        )
    if H_resonator.shape[0] != M:
        raise ValueError(
            f"H_resonator has dimension {H_resonator.shape[0]}, expected M={M}."
        )

    c, cdag = _ho_ladder_ops(N)
    a, adag = _ho_ladder_ops(M)

    H_sys = (
        qtp.tensor(H_qubit, qtp.qeye(M))
        + qtp.tensor(qtp.qeye(N), H_resonator)
        + g * (qtp.tensor(c, adag) + qtp.tensor(cdag, a))
    )
    return H_sys
