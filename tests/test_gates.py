"""Tests for bosonic_gates.gates — SNAP and ECD gates."""

import numpy as np
import pytest
import qutip as qt

from bosonic_gates.gates import snap_operator, apply_snap, snap_unitary_ideal
from bosonic_gates.gates.ecd import ecd_operator, ecd_circuit
from bosonic_gates.gates.snap import snap_phase_gradient


class TestSNAP:
    def test_snap_is_unitary(self):
        N = 8
        thetas = np.random.default_rng(0).uniform(0, 2 * np.pi, N)
        U = snap_operator(N, thetas)
        diff = (U * U.dag() - qt.qeye(N)).norm()
        assert diff < 1e-10

    def test_snap_diagonal_in_fock_basis(self):
        N = 6
        thetas = [0, np.pi / 2, np.pi, 0, 0, 0]
        U = snap_operator(N, thetas)
        U_np = U.full()
        # Off-diagonal elements should be zero
        assert np.allclose(np.abs(U_np - np.diag(np.diag(U_np))), 0, atol=1e-12)

    def test_snap_phase_on_fock_state(self):
        N = 8
        thetas = np.zeros(N)
        thetas[3] = np.pi
        psi_in = qt.basis(N, 3)
        psi_out = apply_snap(psi_in, thetas)
        overlap = psi_in.dag() * psi_out
        assert abs(abs(overlap[0][0][0]) - 1.0) < 1e-10
        assert abs(np.angle(overlap[0][0][0]) - np.pi) < 1e-10

    def test_snap_identity(self):
        N = 6
        thetas = np.zeros(N)
        U = snap_operator(N, thetas)
        diff = (U - qt.qeye(N)).norm()
        assert diff < 1e-10

    def test_snap_phase_gradient(self):
        N = 8
        U = snap_phase_gradient(N, k=2, theta_k=np.pi / 3)
        assert U.isherm is False  # generally not Hermitian
        diff = (U * U.dag() - qt.qeye(N)).norm()
        assert diff < 1e-10

    def test_apply_snap_density_matrix(self):
        N = 6
        rho = qt.ket2dm(qt.basis(N, 2))
        thetas = [0, 0, np.pi, 0, 0, 0]
        rho_out = apply_snap(rho, thetas)
        assert rho_out.type == "oper"
        assert abs(rho_out.tr() - 1.0) < 1e-10

    def test_snap_short_thetas_zero_padded(self):
        """Providing fewer thetas than N should zero-pad."""
        N = 8
        thetas_short = [np.pi, 0, 0]  # length 3 < N=8
        U = snap_operator(N, thetas_short)
        diff = (U * U.dag() - qt.qeye(N)).norm()
        assert diff < 1e-10


class TestECD:
    def test_ecd_is_unitary(self):
        N = 15
        beta = 1.0 + 0.5j
        U = ecd_operator(N, beta)
        diff = (U * U.dag() - qt.tensor(qt.qeye(2), qt.qeye(N))).norm()
        assert diff < 1e-8

    def test_ecd_dimensions(self):
        N = 10
        U = ecd_operator(N, beta=1.0)
        assert U.shape == (2 * N, 2 * N)

    def test_ecd_displaces_by_half_beta(self):
        """ECD(β) applied to |g,0⟩ gives |g, D(-β/2)|0⟩⟩."""
        N = 20
        beta = 1.5
        U = ecd_operator(N, beta)
        psi_g0 = qt.tensor(qt.basis(2, 0), qt.basis(N, 0))
        psi_out = U * psi_g0

        # Trace out qubit to get cavity state
        rho_full = qt.ket2dm(psi_out)
        rho_cavity = rho_full.ptrace(1)  # cavity is second subsystem

        # Expected: D(-β/2)|0⟩
        psi_expected_cavity = qt.displace(N, -beta / 2) * qt.basis(N, 0)
        fid = qt.fidelity(rho_cavity, qt.ket2dm(psi_expected_cavity))
        assert fid > 0.99

    def test_ecd_circuit_single_gate(self):
        """ecd_circuit with one beta should equal ecd_operator."""
        N = 15
        beta = 0.8 + 0.3j
        U_single = ecd_circuit(N, [beta])
        U_direct = ecd_operator(N, beta)
        diff = (U_single - U_direct).norm()
        assert diff < 1e-10

    def test_ecd_circuit_alpha_length_check(self):
        N = 10
        with pytest.raises(ValueError):
            ecd_circuit(N, betas=[1.0, 0.5], alphas=[0.1])  # wrong length
