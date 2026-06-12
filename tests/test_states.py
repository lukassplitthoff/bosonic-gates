"""Tests for bosonic_gates.states."""

import math
import numpy as np
import pytest
import qutip as qt

from bosonic_gates.states import (
    coherent_state,
    fock_state,
    squeezed_state,
    thermal_state,
    cat_state,
    displaced_squeezed_state,
    binomial_state,
    fock_superposition,
    BosonicState,
)


def test_coherent_state_eigenstate():
    """Coherent state |α⟩ is an eigenstate of a with eigenvalue α."""
    N = 30
    alpha = 1.5 + 0.5j
    s = coherent_state(alpha, N=N)
    a = qt.destroy(N)
    residual = (a * s.state - alpha * s.state).norm()
    assert residual < 1e-6


def test_fock_state_orthonormality():
    N = 20
    s0 = fock_state(0, N=N)
    s3 = fock_state(3, N=N)
    assert abs(s0.state.dag() * s3.state) < 1e-12
    assert abs(abs(complex(s3.state.dag() * s3.state)) - 1.0) < 1e-12


def test_fock_state_photon_number():
    for n in [0, 1, 5, 10]:
        s = fock_state(n, N=n + 20)
        assert abs(s.photon_number() - n) < 1e-10


def test_fock_state_dimension_check():
    with pytest.raises(ValueError):
        fock_state(10, N=5)


def test_squeezed_state_variance():
    """Squeezed state has variance < 1/4 in the squeezed quadrature."""
    N = 50
    r = 1.0
    s = squeezed_state(r, phi=0, N=N)
    a = qt.destroy(N)
    X = (a + a.dag()) / 2
    var_X = float(qt.variance(X, s.state))
    assert var_X < 0.25  # below vacuum noise


def test_thermal_state_purity():
    """Thermal state is mixed (purity < 1)."""
    s = thermal_state(n_mean=2.0, N=30)
    assert s.purity() < 1.0


def test_thermal_state_mean_photon():
    n_mean = 3.0
    s = thermal_state(n_mean=n_mean, N=50)
    assert abs(s.photon_number() - n_mean) < 0.05


def test_cat_state_even_parity():
    """Even cat state has even-parity Fock distribution (only even n)."""
    N = 40
    s = cat_state(2.0, N=N, phase=0)
    dist = s.fock_distribution()
    # Odd Fock components should be near zero for even cat
    for n in range(1, N, 2):
        assert abs(dist[n]) < 1e-8


def test_cat_state_odd_parity():
    """Odd cat state has only odd Fock components."""
    N = 40
    s = cat_state(2.0, N=N, phase=np.pi)
    dist = s.fock_distribution()
    for n in range(0, N, 2):
        assert abs(dist[n]) < 1e-8


def test_binomial_state_normalization():
    s = binomial_state(N=30, theta=np.pi / 4, n_max=5)
    assert abs(s.state.norm() - 1.0) < 1e-10


def test_fock_superposition_normalization():
    s = fock_superposition([0, 2, 4], [1, 1, 1])
    assert abs(s.state.norm() - 1.0) < 1e-10


def test_bosonic_state_density_matrix():
    N = 20
    s = fock_state(3, N=N)
    rho = s.density_matrix()
    assert rho.type == "oper"
    assert abs(rho.tr() - 1.0) < 1e-10


def test_bosonic_state_purity_pure():
    s = coherent_state(1.0, N=30)
    assert abs(s.purity() - 1.0) < 1e-6
