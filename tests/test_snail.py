"""Tests for bosonic_gates.snail."""

import numpy as np
import pytest

from bosonic_gates.snail import (
    snail_potential,
    find_snail_minimum,
    snail_taylor_coefficients,
    snail_circuit_params,
    gbs_linear,
    gbs_rwa,
)


# Typical SNAIL parameters (3-junction, beta~0.1, near optimal flux)
BETA = 0.1
M = 3
PHI_E = 0.4 * 2 * np.pi   # flux near Φ_0/2 for large g3
E_J = 10.0
E_C = 0.2


def test_potential_minimum_is_local_minimum():
    """find_snail_minimum returns a true local minimum."""
    phi_m = find_snail_minimum(BETA, M, PHI_E)
    phi = np.linspace(phi_m - 0.1, phi_m + 0.1, 200)
    U = snail_potential(phi, beta=BETA, M=M, phi_e=PHI_E)
    U_min = snail_potential(np.array([phi_m]), beta=BETA, M=M, phi_e=PHI_E)[0]
    assert np.all(U >= U_min - 1e-10), "phi_m is not a global minimum in neighbourhood"


def test_taylor_coefficients_c2_positive():
    """c2 > 0 ensures the potential is locally stable (quadratic well)."""
    coeffs = snail_taylor_coefficients(BETA, M, PHI_E)
    assert coeffs["c2"] > 0


def test_taylor_coefficients_c3_sign():
    """For phi_e near pi, c3 should be non-zero (that's the point of SNAIL)."""
    coeffs = snail_taylor_coefficients(BETA, M, PHI_E)
    assert abs(coeffs["c3"]) > 1e-4


def test_circuit_params_frequency_positive():
    params = snail_circuit_params(BETA, M, PHI_E, E_J=E_J, E_C=E_C)
    assert params["omega_c"] > 0


def test_circuit_params_zpf_positive():
    params = snail_circuit_params(BETA, M, PHI_E, E_J=E_J, E_C=E_C)
    assert params["phi_c"] > 0


def test_circuit_params_returns_all_keys():
    params = snail_circuit_params(BETA, M, PHI_E, E_J=E_J, E_C=E_C)
    for key in ("omega_c", "phi_c", "g3", "g4", "g5", "alpha_c", "xi_crit", "p"):
        assert key in params


def test_gbs_linear_scaling():
    """Linear beam-splitter rate scales as |xi|."""
    xi = np.array([0.0, 0.5, 1.0])
    g3 = 0.01
    gbs = gbs_linear(xi, g3=g3, ga_over_delta_a=0.1, gb_over_delta_b=0.1)
    assert np.all(gbs >= 0)
    assert gbs[0] == 0.0
    np.testing.assert_allclose(gbs[2] / gbs[1], 2.0, rtol=1e-10)


def test_gbs_rwa_reduces_to_linear_at_low_xi():
    """At low xi, gbs_rwa should match gbs_linear (leading-order term only)."""
    params = snail_circuit_params(BETA, M, PHI_E, E_J=E_J, E_C=E_C)
    g3, g5 = params["g3"], params["g5"]
    ga, gb = 0.05, 0.05

    xi = np.array([0.01])  # very small amplitude
    gbs_lin = gbs_linear(xi, g3=g3, ga_over_delta_a=ga, gb_over_delta_b=gb)
    gbs_full = gbs_rwa(xi, [g3, g5], ga_over_delta_a=ga, gb_over_delta_b=gb)
    np.testing.assert_allclose(gbs_full, gbs_lin, rtol=1e-2)
