"""
SNAP gates — Selective Number-dependent Arbitrary Phase gates.

A SNAP gate applies independent phases to each Fock number state:

    SNAP(θ_0, θ_1, ..., θ_{N-1}) = Σ_n exp(i θ_n) |n⟩⟨n|

Physical realization: in a transmon-cavity system, SNAP gates are implemented
by applying photon-number-selective qubit drives. Because the dispersive
shift χ makes the qubit frequency depend on the cavity photon number, drives
at frequency ω_ge + n·χ exclusively rotate the qubit conditioned on n photons,
implementing a controlled-phase on the cavity.

Reference: Heeres et al., PRL 115, 137002 (2015)
           Krastanov et al., PRA 92, 040303 (2015) — universality of SNAP+D

The functions in this module work with the cavity as a single-mode Fock space
of dimension N, returning QuTiP Qobj operators.
"""

import numpy as np
import qutip as qt


def snap_operator(N: int, thetas: np.ndarray) -> qt.Qobj:
    """
    Construct the SNAP unitary operator in Fock space.

    SNAP(θ) = Σ_n exp(i θ_n) |n⟩⟨n|

    This is a diagonal unitary in the Fock basis.

    Parameters
    ----------
    N : int
        Hilbert space dimension (Fock truncation).
    thetas : array-like of float
        Phase angles θ_n for each Fock state |n⟩.
        len(thetas) must equal N; excess angles are ignored if shorter,
        and zero-padded if shorter.

    Returns
    -------
    U : qt.Qobj  (N×N unitary)

    Example
    -------
    >>> import numpy as np
    >>> U = snap_operator(5, thetas=[0, np.pi, 0, 0, 0])
    >>> # Applies a π phase to |1⟩, leaves all other states unchanged
    """
    thetas = np.asarray(thetas, dtype=float)
    if len(thetas) < N:
        thetas = np.pad(thetas, (0, N - len(thetas)))
    elif len(thetas) > N:
        thetas = thetas[:N]

    phases = np.exp(1j * thetas)
    U = qt.Qobj(np.diag(phases))
    U.dims = [[N], [N]]
    return U


def snap_unitary_ideal(N: int, thetas: np.ndarray) -> qt.Qobj:
    """Alias for snap_operator (explicit name for benchmarking contexts)."""
    return snap_operator(N, thetas)


def apply_snap(state: qt.Qobj, thetas: np.ndarray) -> qt.Qobj:
    """
    Apply a SNAP gate to a quantum state.

    Works for both pure states (ket) and density matrices.

    Parameters
    ----------
    state : qt.Qobj
        Input state — ket (shape [N, 1]) or density matrix (shape [N, N]).
    thetas : array-like of float
        Phase angles for each Fock state.

    Returns
    -------
    qt.Qobj
        Transformed state (same type as input).

    Example
    -------
    >>> import qutip as qt, numpy as np
    >>> psi = qt.coherent(10, 2.0)
    >>> psi_out = apply_snap(psi, thetas=[0, np.pi, 0, 0, 0, 0, 0, 0, 0, 0])
    """
    N = state.shape[0]
    U = snap_operator(N, thetas)
    if state.type == "ket":
        return U * state
    else:  # density matrix
        return U * state * U.dag()


def snap_phase_gradient(N: int, k: int, theta_k: float = np.pi) -> qt.Qobj:
    """
    SNAP gate that applies a phase to only one Fock level |k⟩.

    This is the elementary building block. Any SNAP gate decomposes into
    a product of at most N such elementary SNAPs.

    Parameters
    ----------
    N : int
        Hilbert space dimension.
    k : int
        Fock level to phase.
    theta_k : float
        Phase applied to |k⟩ (default: π).

    Returns
    -------
    U : qt.Qobj  (N×N unitary)

    Example
    -------
    >>> U = snap_phase_gradient(10, k=3, theta_k=np.pi/2)
    """
    thetas = np.zeros(N)
    thetas[k] = theta_k
    return snap_operator(N, thetas)
