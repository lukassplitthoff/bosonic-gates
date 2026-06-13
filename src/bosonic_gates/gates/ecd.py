"""
ECD gates — Echoed Conditional Displacement gates.

An ECD gate is defined on the joint qubit ⊗ cavity Hilbert space as:

    ECD(β) = |e⟩⟨e| ⊗ D(+β/2) + |g⟩⟨g| ⊗ D(−β/2)

Tensor ordering: qubit ⊗ cavity throughout (qubit is the left factor).
In matrix form this is qt.tensor(|e⟩⟨e|, D(+β/2)) + qt.tensor(|g⟩⟨g|, D(−β/2)).

The "echo" in the name refers to the fact that the ECD is implemented as
a sequence of two conditional displacements separated by a qubit π pulse:

    D(β) Z D(-β) / 2  →  (up to global phase)

where the π pulse echoes the qubit state, converting the sequence
into a conditional displacement that depends on the qubit state.

Physical motivation: In a dispersive qubit-cavity system, a displacement drive
at frequency ω_r shifts the cavity by different amounts depending on the qubit
state. The echo removes the unconditioned rotation acquired by the qubit,
leaving only the conditional displacement on the cavity.

Reference: Campagne-Ibarcq et al., Nature 584, 368 (2020)
           Eickbusch et al., Nature Physics 18, 1464 (2022)

Tensor ordering: qubit ⊗ cavity (qubit is the left factor).
"""

from __future__ import annotations
import numpy as np
import qutip as qt


def _qubit_projectors() -> tuple[qt.Qobj, qt.Qobj]:
    """Return |g⟩⟨g| and |e⟩⟨e| for the qubit (2-level system)."""
    g = qt.basis(2, 0)  # ground state
    e = qt.basis(2, 1)  # excited state
    return g * g.dag(), e * e.dag()


def _displacement(N: int, beta: complex) -> qt.Qobj:
    """Displacement operator D(beta) on an N-dimensional Fock space."""
    return qt.displace(N, beta)


def ecd_operator(N: int, beta: complex) -> qt.Qobj:
    """
    Construct the ECD gate on the joint 2N-dimensional qubit ⊗ cavity space.

    ECD(β) = D(+β/2) ⊗ |e⟩⟨e| + D(−β/2) ⊗ |g⟩⟨g|

    Tensor order: qubit ⊗ cavity (qubit dims first).

    Parameters
    ----------
    N : int
        Cavity Hilbert space dimension (Fock truncation).
    beta : complex
        Displacement amplitude.  The conditional displacement is ±β/2.

    Returns
    -------
    U_ecd : qt.Qobj  (2N × 2N unitary)
        Acts on the joint qubit ⊗ cavity Hilbert space.

    Example
    -------
    >>> N = 20
    >>> U = ecd_operator(N, beta=1.5)
    >>> # Apply to |g, 0⟩:
    >>> psi0 = qt.tensor(qt.basis(2, 0), qt.basis(N, 0))
    >>> psi_out = U * psi0
    """
    Pg, Pe = _qubit_projectors()
    D_plus  = _displacement(N, beta / 2)
    D_minus = _displacement(N, -beta / 2)

    # ECD = |e⟩⟨e| ⊗ D(+β/2) + |g⟩⟨g| ⊗ D(-β/2)
    U_ecd = qt.tensor(Pe, D_plus) + qt.tensor(Pg, D_minus)
    return U_ecd


def ecd_circuit(
    N: int,
    betas: list[complex],
    alphas: list[complex] | None = None,
) -> qt.Qobj:
    """
    Build a multi-ECD gate sequence, optionally interleaved with cavity displacements.

    The circuit is:
        U = [D(α_n)] ECD(β_n) ... [D(α_1)] ECD(β_1) [D(α_0)]

    where D(α_k) are optional unconditional cavity displacements applied
    between ECD gates.  This is the GKP state preparation protocol from
    Campagne-Ibarcq et al. (2020).

    Parameters
    ----------
    N : int
        Cavity Hilbert space dimension.
    betas : list of complex
        Conditional displacement amplitudes for each ECD gate.
    alphas : list of complex, optional
        Unconditional displacement amplitudes interleaved between ECDs.
        If provided, len(alphas) must equal len(betas) + 1
        (one before, one between each ECD, one after).
        If None, no unconditional displacements are applied.

    Returns
    -------
    U_circuit : qt.Qobj  (2N × 2N unitary)

    Example
    -------
    >>> # Two ECD gates with interleaved displacements
    >>> N = 20
    >>> betas = [1.0 + 0j, 0.5 + 0.5j]
    >>> alphas = [0.2, -0.1, 0.0]
    >>> U = ecd_circuit(N, betas, alphas)
    """
    I_qubit = qt.qeye(2)

    if alphas is not None and len(alphas) != len(betas) + 1:
        raise ValueError(
            f"len(alphas)={len(alphas)} must equal len(betas)+1={len(betas)+1}"
        )

    U = qt.tensor(I_qubit, qt.qeye(N))  # identity

    for k, beta in enumerate(betas):
        if alphas is not None:
            D_alpha = qt.tensor(I_qubit, _displacement(N, alphas[k]))
            U = D_alpha * U
        U = ecd_operator(N, beta) * U

    if alphas is not None:
        D_alpha_last = qt.tensor(I_qubit, _displacement(N, alphas[-1]))
        U = D_alpha_last * U

    return U


def ecd_qubit_reset_sequence(N: int, beta: complex) -> list[qt.Qobj]:
    """
    Decompose ECD(β) into physical operations for simulation.

    The ECD gate is physically implemented as:
        1. Conditional displacement D_cond(+β/2) (drive at ω_r + χ/2)
        2. Qubit π pulse (X gate)
        3. Conditional displacement D_cond(-β/2) (drive at ω_r + χ/2 again)
        4. Qubit π pulse (X gate) — to reset qubit rotation

    This decomposition preserves the qubit state after the gate.

    Returns a list of [U_1, U_2, U_3, U_4] to be applied in order.

    Parameters
    ----------
    N : int
        Cavity Hilbert space dimension.
    beta : complex
        ECD amplitude.

    Returns
    -------
    list of qt.Qobj
        Sequence of unitaries implementing ECD(β).
    """
    Pg, Pe = _qubit_projectors()
    X = qt.sigmax()  # Pauli X = qubit π pulse

    # Conditional displacement (same for both steps when frame-rotated)
    D_plus  = qt.tensor(Pe, _displacement(N, beta / 2)) + qt.tensor(Pg, qt.qeye(N))
    X_qubit = qt.tensor(X, qt.qeye(N))
    D_minus = qt.tensor(Pe, _displacement(N, -beta / 2)) + qt.tensor(Pg, qt.qeye(N))

    return [D_plus, X_qubit, D_minus, X_qubit]
