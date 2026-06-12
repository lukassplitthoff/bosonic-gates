"""Tests for bosonic_gates.hamiltonians."""

import numpy as np
import pytest
import qutip as qt

from bosonic_gates.hamiltonians import (
    resonator_hamiltonian,
    resonator_number_operator,
    coupled_system_hamiltonian,
    transmon_hamiltonian,
)


def test_resonator_hamiltonian_hermitian():
    H = resonator_hamiltonian(w=5.0, M=10)
    assert H.isherm


def test_resonator_hamiltonian_eigenvalues():
    """Eigenvalues of H_res = w*(n + 1/2) should be w*(0.5, 1.5, 2.5, ...)."""
    w = 5.0
    M = 6
    H = resonator_hamiltonian(w=w, M=M)
    evals = sorted(H.eigenenergies())
    expected = [w * (n + 0.5) for n in range(M)]
    for ev, ex in zip(evals, expected):
        assert abs(ev - ex) < 1e-10


def test_resonator_number_operator_dimensions():
    N, M = 3, 5
    n_res = resonator_number_operator(N, M)
    assert n_res.shape == (N * M, N * M)


def test_coupled_system_hamiltonian_hermitian():
    N, M = 3, 5
    H_q = transmon_hamiltonian(Ej=20.0, Ec=0.3, N=N)
    H_r = resonator_hamiltonian(w=5.0, M=M)
    H_sys = coupled_system_hamiltonian(H_q, H_r, N, M, g=0.1)
    assert H_sys.isherm


def test_transmon_hamiltonian_hermitian():
    H = transmon_hamiltonian(Ej=20.0, Ec=0.3, N=10)
    assert H.isherm


def test_transmon_hamiltonian_anharmonicity():
    """Transmon 0→1 frequency should be close to sqrt(8*Ej*Ec) - Ec (leading order)."""
    Ej, Ec, N = 20.0, 0.25, 15
    H = transmon_hamiltonian(Ej=Ej, Ec=Ec, N=N)
    evals = sorted(H.eigenenergies())
    omega_01 = evals[1] - evals[0]
    omega_12 = evals[2] - evals[1]
    # Anharmonicity alpha = omega_12 - omega_01 should be ≈ -Ec (negative for transmon)
    alpha = omega_12 - omega_01
    assert alpha < 0  # transmon is always sub-anharmonic
    assert abs(alpha + Ec) / Ec < 0.15  # within 15% of leading-order Ec


def test_transmon_invalid_ej():
    with pytest.raises(ValueError):
        transmon_hamiltonian(Ej=-1.0, Ec=0.3, N=10)


def test_coupled_system_dimension_mismatch():
    N, M = 3, 5
    H_q = transmon_hamiltonian(Ej=20.0, Ec=0.3, N=N)
    H_r = resonator_hamiltonian(w=5.0, M=M)
    with pytest.raises(ValueError):
        coupled_system_hamiltonian(H_q, H_r, N=4, M=M, g=0.1)
