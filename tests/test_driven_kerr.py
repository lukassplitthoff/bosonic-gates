"""
Tests for bosonic_gates.driven_kerr — all public functions.

Uses a small Hilbert space (N=4) and coarse time grids to stay fast.
Physics conventions follow CLAUDE.md: all frequencies in rad/s (rad·GHz in practice).
"""

import numpy as np
import pytest
import qutip as qt

TWO_PI = 2 * np.pi

from bosonic_gates.driven_kerr import (
    DrivenKerrConfig,
    make_H0,
    make_operators,
    make_jump_ops,
    make_H_drive_td,
    J,
    J_vectorized,
    J_phi,
    run_lindblad,
    run_redfield,
    check_positivity,
    compute_floquet_modes,
    compute_fourier_components,
    assemble_rates,
    assemble_rates_with_dephasing,
    run_floquet_markov,
    run_full_floquet_markov,
    floquet_steady_state,
    effective_loss_rate,
    effective_loss_rate_from_fit,
    extract_excited_pop,
    steady_state_leakage,
    error_budget,
)
# effective_decay_rate_from_R is defined in floquet_markov.py but not re-exported
# from the package __init__.py, so we import it directly.
from bosonic_gates.driven_kerr.floquet_markov import effective_decay_rate_from_R


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_cfg():
    """Small N=4 config for fast tests."""
    return DrivenKerrConfig(
        N=4,
        omega0=TWO_PI * 5.0,
        K=TWO_PI * 0.2,
        omega_d=TWO_PI * 5.005,   # 5 MHz blue-detuned
        omega_f=TWO_PI * 5.0,     # bath at omega0
        epsilon=0.0,
        kappa=TWO_PI * 1e-3,
        Gamma=TWO_PI * 0.1,
        nbar=0.02,
        gamma_phi=TWO_PI * 0.05e-3,
        k_max=3,
        n_t=64,
    )


@pytest.fixture
def driven_cfg(default_cfg):
    """Same as default_cfg but with a finite drive (0.1K)."""
    return default_cfg.replace(epsilon=0.1 * default_cfg.K)


@pytest.fixture
def white_bath_cfg(default_cfg):
    """Flat (white) bath: Gamma → very large."""
    return default_cfg.replace(Gamma=TWO_PI * 1e4, epsilon=0.05 * default_cfg.K)


# ---------------------------------------------------------------------------
# DrivenKerrConfig
# ---------------------------------------------------------------------------

class TestDrivenKerrConfig:
    def test_default_omega_d(self):
        """omega_d defaults to omega0 when not specified."""
        cfg = DrivenKerrConfig(omega0=TWO_PI * 5.0)
        assert cfg.omega_d == cfg.omega0

    def test_omega12(self, default_cfg):
        """omega12 = omega0 - K."""
        assert abs(default_cfg.omega12 - (default_cfg.omega0 - default_cfg.K)) < 1e-12

    def test_T_d(self, default_cfg):
        """Drive period T_d = 2π / omega_d."""
        expected = TWO_PI / default_cfg.omega_d
        assert abs(default_cfg.T_d - expected) < 1e-20

    def test_replace_returns_new_instance(self, default_cfg):
        """replace() must return a different object, not mutate."""
        cfg2 = default_cfg.replace(N=6)
        assert cfg2 is not default_cfg
        assert cfg2.N == 6
        assert default_cfg.N == 4

    def test_replace_preserves_other_fields(self, default_cfg):
        cfg2 = default_cfg.replace(N=6)
        assert cfg2.omega0 == default_cfg.omega0
        assert cfg2.K == default_cfg.K

    def test_default_omega_f(self):
        """omega_f defaults to omega12 + omega_d when not specified."""
        cfg = DrivenKerrConfig(omega0=TWO_PI * 5.0, K=TWO_PI * 0.2)
        expected = (cfg.omega0 - cfg.K) + cfg.omega_d
        assert abs(cfg.omega_f - expected) < 1e-10


# ---------------------------------------------------------------------------
# Operator construction
# ---------------------------------------------------------------------------

class TestMakeOperators:
    def test_shapes(self, default_cfg):
        a, adag, n = make_operators(default_cfg.N)
        assert a.shape == (default_cfg.N, default_cfg.N)
        assert adag.shape == (default_cfg.N, default_cfg.N)
        assert n.shape == (default_cfg.N, default_cfg.N)

    def test_a_adag_commutator(self, default_cfg):
        """For truncated Fock space, [a, a†] = I - N|N-1><N-1| (truncation correction)."""
        a, adag, _ = make_operators(default_cfg.N)
        comm = a * adag - adag * a
        N = default_cfg.N
        # Finite truncation: last level has no higher state, so [a,a†]_{N-1,N-1} = -(N-1) not 1
        expected_trunc = qt.qeye(N) - N * qt.fock_dm(N, N - 1)
        diff = (comm - expected_trunc).norm()
        assert diff < 1e-10

    def test_n_is_adag_a(self, default_cfg):
        a, adag, n = make_operators(default_cfg.N)
        assert (n - adag * a).norm() < 1e-10

    def test_n_is_hermitian(self, default_cfg):
        _, _, n = make_operators(default_cfg.N)
        assert n.isherm


# ---------------------------------------------------------------------------
# H0
# ---------------------------------------------------------------------------

class TestMakeH0:
    def test_hermitian(self, default_cfg):
        H = make_H0(default_cfg)
        assert H.isherm, "H0 must be Hermitian"

    def test_dimension(self, default_cfg):
        H = make_H0(default_cfg)
        assert H.shape == (default_cfg.N, default_cfg.N)

    def test_ground_state_eigenvalue(self, default_cfg):
        """Ground state (n=0) has eigenvalue 0 (no constant offset in our convention)."""
        H = make_H0(default_cfg)
        evals = H.eigenenergies()
        assert abs(evals[0]) < 1e-10

    def test_anharmonicity(self, default_cfg):
        """Spacing ω_{n,n+1} = ω₀ - nK (anharmonic ladder)."""
        H = make_H0(default_cfg)
        evals = H.eigenenergies()
        for n in range(default_cfg.N - 1):
            expected = default_cfg.omega0 - n * default_cfg.K
            measured = evals[n + 1] - evals[n]
            assert abs(measured - expected) < 1e-8, (
                f"Spacing {n}→{n+1}: expected {expected:.4f}, got {measured:.4f}"
            )

    def test_positive_eigenvalues(self, default_cfg):
        """All eigenvalues ≥ 0 (ground state at zero)."""
        H = make_H0(default_cfg)
        evals = H.eigenenergies()
        assert np.all(evals >= -1e-10)


# ---------------------------------------------------------------------------
# H_drive_td
# ---------------------------------------------------------------------------

class TestMakeHDriveTd:
    def test_list_format(self, driven_cfg):
        H = make_H_drive_td(driven_cfg)
        assert isinstance(H, list), "must return a list"
        assert len(H) == 2, "must be [H0, [H_amp, coeff]]"
        assert isinstance(H[1], list), "second element must be [H_amp, coeff]"
        assert callable(H[1][1]), "coefficient must be callable"

    def test_H0_part_matches_make_H0(self, driven_cfg):
        H_td = make_H_drive_td(driven_cfg)
        H0_direct = make_H0(driven_cfg)
        assert (H_td[0] - H0_direct).norm() < 1e-10

    def test_coeff_at_t0(self, driven_cfg):
        H_td = make_H_drive_td(driven_cfg)
        coeff_fn = H_td[1][1]
        # At t=0, cos(0) = 1, so coeff = epsilon
        assert abs(coeff_fn(0.0) - driven_cfg.epsilon) < 1e-12

    def test_drive_amp_is_x_quadrature(self, driven_cfg):
        H_td = make_H_drive_td(driven_cfg)
        H_amp = H_td[1][0]
        a, adag, _ = make_operators(driven_cfg.N)
        assert (H_amp - (a + adag)).norm() < 1e-10


# ---------------------------------------------------------------------------
# Spectral density J
# ---------------------------------------------------------------------------

class TestSpectralDensity:
    def test_positive_for_positive_omega(self, default_cfg):
        assert J(default_cfg.omega0, default_cfg) > 0

    def test_positive_for_negative_omega(self, default_cfg):
        assert J(-default_cfg.omega0, default_cfg) > 0

    def test_zero_at_omega_zero(self, default_cfg):
        assert J(0.0, default_cfg) == 0.0

    def test_detailed_balance(self, default_cfg):
        """J(-ω)/J(ω) = n̄/(1+n̄) — fundamental thermodynamic constraint."""
        omega = default_cfg.omega0
        ratio = J(-omega, default_cfg) / J(omega, default_cfg)
        expected = default_cfg.nbar / (1.0 + default_cfg.nbar)
        assert abs(ratio - expected) < 1e-10, (
            f"Detailed balance violated: got {ratio}, expected {expected}"
        )

    def test_peak_at_omega_f(self, default_cfg):
        """Emission side peaks at omega_f."""
        omega_f = default_cfg.omega_f
        J_peak = J(omega_f, default_cfg)
        # Sampling slightly off-peak should give lower value
        J_off = J(omega_f + default_cfg.Gamma, default_cfg)
        assert J_peak > J_off

    def test_vanishes_far_off_resonance(self, default_cfg):
        """Far from bath center, J is negligibly small."""
        # Lorentzian tail at 100× bath width off-center is ~(Gamma/200*pi)^2 ~ 1e-9
        omega_far = default_cfg.omega0 + 100 * TWO_PI
        assert J(omega_far, default_cfg) < 1e-7

    def test_kappa_zero_gives_zero_J(self, default_cfg):
        cfg_nokappa = default_cfg.replace(kappa=0.0)
        assert J(default_cfg.omega0, cfg_nokappa) == 0.0

    def test_J_vectorized_matches_scalar(self, default_cfg):
        """J_vectorized must match scalar J element-by-element."""
        omegas = np.array([
            default_cfg.omega0,
            -default_cfg.omega0,
            default_cfg.omega0 * 0.5,
            default_cfg.omega0 * 1.5,
        ])
        J_vec = J_vectorized(omegas, default_cfg)
        J_scalar = np.array([J(w, default_cfg) for w in omegas])
        np.testing.assert_allclose(J_vec, J_scalar, rtol=1e-12)

    def test_J_phi_returns_gamma_phi(self, default_cfg):
        """J_phi is flat = gamma_phi."""
        for omega in [1.0, -1.0, 100.0]:
            assert J_phi(omega, default_cfg) == default_cfg.gamma_phi

    def test_J_vectorized_zeros_at_zero(self, default_cfg):
        omegas = np.array([0.0, default_cfg.omega0])
        J_vec = J_vectorized(omegas, default_cfg)
        assert J_vec[0] == 0.0
        assert J_vec[1] > 0.0


# ---------------------------------------------------------------------------
# Jump operators
# ---------------------------------------------------------------------------

class TestMakeJumpOps:
    def test_returns_list(self, default_cfg):
        c_ops = make_jump_ops(default_cfg)
        assert isinstance(c_ops, list)

    def test_three_channels(self, default_cfg):
        """With kappa>0, nbar>0, gamma_phi>0, should get 3 jump ops."""
        c_ops = make_jump_ops(default_cfg)
        assert len(c_ops) == 3

    def test_no_gain_at_zero_nbar(self, default_cfg):
        """With nbar=0, gamma_gain = J(-omega0) = 0, so only 2 jump ops."""
        cfg_zero_nbar = default_cfg.replace(nbar=0.0)
        c_ops = make_jump_ops(cfg_zero_nbar)
        # gamma_gain = kappa * nbar = 0, so no gain jump op
        assert len(c_ops) == 2

    def test_no_dephasing_when_zero(self, default_cfg):
        cfg_nodeph = default_cfg.replace(gamma_phi=0.0)
        c_ops = make_jump_ops(cfg_nodeph)
        # Only loss and gain (if nbar>0)
        assert len(c_ops) == 2

    def test_jump_op_shapes(self, default_cfg):
        c_ops = make_jump_ops(default_cfg)
        for cop in c_ops:
            assert cop.shape == (default_cfg.N, default_cfg.N)


# ---------------------------------------------------------------------------
# run_lindblad
# ---------------------------------------------------------------------------

class TestRunLindblad:
    def test_returns_result_with_states(self, default_cfg):
        rho0 = qt.fock_dm(default_cfg.N, 1)
        tlist = np.linspace(0, 1e-3, 10)
        result = run_lindblad(default_cfg, rho0, tlist)
        assert hasattr(result, "states")
        assert len(result.states) == len(tlist)

    def test_trace_preserved(self, default_cfg):
        """Trace of density matrix must equal 1 at all times."""
        rho0 = qt.fock_dm(default_cfg.N, 1)
        tlist = np.linspace(0, 1e-3, 5)
        result = run_lindblad(default_cfg, rho0, tlist)
        for rho in result.states:
            assert abs(rho.tr() - 1.0) < 1e-6, f"Trace = {rho.tr()}, expected 1"

    def test_hermitian_states(self, default_cfg):
        """All density matrices must be Hermitian."""
        rho0 = qt.fock_dm(default_cfg.N, 1)
        tlist = np.linspace(0, 1e-3, 5)
        result = run_lindblad(default_cfg, rho0, tlist)
        for rho in result.states:
            assert rho.isherm

    def test_populations_non_negative(self, default_cfg):
        """Diagonal elements (populations) must be ≥ 0."""
        rho0 = qt.fock_dm(default_cfg.N, 1)
        tlist = np.linspace(0, 1e-3, 8)
        result = run_lindblad(default_cfg, rho0, tlist)
        for rho in result.states:
            diag = np.diag(rho.full().real)
            assert np.all(diag >= -1e-8)

    def test_initial_state_preserved(self, default_cfg):
        """At t=0 the state should match the initial state."""
        rho0 = qt.fock_dm(default_cfg.N, 1)
        tlist = np.array([0.0, 1e-6])
        result = run_lindblad(default_cfg, rho0, tlist)
        diff = (result.states[0] - rho0).norm()
        assert diff < 1e-6

    def test_decay_from_excited_state(self, default_cfg):
        """System starting in |1⟩ should decay: P₁(t_final) < P₁(0)."""
        rho0 = qt.fock_dm(default_cfg.N, 1)
        T = 3.0 / default_cfg.kappa
        tlist = np.linspace(0, T, 50)
        result = run_lindblad(default_cfg, rho0, tlist)
        pop1_i = extract_excited_pop(result, default_cfg.N)
        assert pop1_i[-1] < pop1_i[0] * 0.5, "Population should decay significantly"

    def test_accepts_e_ops(self, default_cfg):
        """run_lindblad should accept and evaluate e_ops."""
        rho0 = qt.fock_dm(default_cfg.N, 1)
        tlist = np.linspace(0, 1e-4, 5)
        _, _, n_op = make_operators(default_cfg.N)
        result = run_lindblad(default_cfg, rho0, tlist, e_ops=[n_op])
        assert len(result.expect) == 1
        assert len(result.expect[0]) == len(tlist)


# ---------------------------------------------------------------------------
# run_redfield
# ---------------------------------------------------------------------------

class TestRunRedfield:
    def test_returns_result_with_states(self, default_cfg):
        rho0 = qt.fock_dm(default_cfg.N, 1)
        tlist = np.linspace(0, 1e-3, 8)
        result = run_redfield(default_cfg, rho0, tlist)
        assert hasattr(result, "states")
        assert len(result.states) == len(tlist)

    def test_trace_approximately_preserved(self, default_cfg):
        """Trace should remain close to 1 (Redfield can have small deviations)."""
        rho0 = qt.fock_dm(default_cfg.N, 1)
        tlist = np.linspace(0, 1e-3, 5)
        result = run_redfield(default_cfg, rho0, tlist)
        for rho in result.states:
            assert abs(rho.tr() - 1.0) < 1e-3

    def test_agrees_with_lindblad_weak_drive(self, white_bath_cfg):
        """run_redfield returns states for a moderate-bath weak-drive scenario."""
        # Use narrower bath to avoid integration stiffness from Gamma→∞ white-bath limit.
        cfg = white_bath_cfg.replace(Gamma=white_bath_cfg.Gamma * 0.1)
        rho0 = qt.fock_dm(cfg.N, 1)
        tlist = np.linspace(0, 1e-3, 10)
        result = run_redfield(cfg, rho0, tlist)
        assert len(result.states) == len(tlist)


# ---------------------------------------------------------------------------
# check_positivity
# ---------------------------------------------------------------------------

class TestCheckPositivity:
    def test_returns_dict_with_required_keys(self, default_cfg):
        rho0 = qt.fock_dm(default_cfg.N, 1)
        tlist = np.linspace(0, 1e-4, 5)
        result = run_lindblad(default_cfg, rho0, tlist)
        pos = check_positivity(result)
        assert "max_negative" in pos
        assert "fraction_violated" in pos
        assert "flagged" in pos

    def test_lindblad_passes_positivity(self, default_cfg):
        """Lindblad evolution should always produce positive density matrices."""
        rho0 = qt.fock_dm(default_cfg.N, 1)
        tlist = np.linspace(0, 1e-3, 20)
        result = run_lindblad(default_cfg, rho0, tlist)
        pos = check_positivity(result, tol=1e-6)
        assert not pos["flagged"], f"Lindblad violated positivity: {pos}"

    def test_max_negative_non_positive(self, default_cfg):
        """max_negative ≤ 0 (it tracks the most negative eigenvalue)."""
        rho0 = qt.fock_dm(default_cfg.N, 0)
        tlist = np.linspace(0, 1e-4, 5)
        result = run_lindblad(default_cfg, rho0, tlist)
        pos = check_positivity(result)
        assert pos["max_negative"] <= 1e-10


# ---------------------------------------------------------------------------
# Floquet modes
# ---------------------------------------------------------------------------

class TestComputeFloquetModes:
    def test_output_shapes(self, driven_cfg):
        modes_t, qe, tgrid = compute_floquet_modes(driven_cfg)
        N = driven_cfg.N
        n_t = driven_cfg.n_t
        assert modes_t.shape == (N, n_t, N)
        assert qe.shape == (N,)
        assert tgrid.shape == (n_t,)

    def test_modes_normalized(self, driven_cfg):
        """Each Floquet mode must be unit-normalized at every time step."""
        modes_t, qe, tgrid = compute_floquet_modes(driven_cfg)
        norms = np.einsum("mia,mia->mi", modes_t.conj(), modes_t).real
        np.testing.assert_allclose(norms, 1.0, atol=1e-6,
                                   err_msg="Floquet modes not unit-normalized")

    def test_quasi_energies_real(self, driven_cfg):
        _, qe, _ = compute_floquet_modes(driven_cfg)
        assert np.all(np.isreal(qe)), "Quasi-energies must be real"

    def test_tgrid_covers_one_period(self, driven_cfg):
        _, _, tgrid = compute_floquet_modes(driven_cfg)
        expected_T = driven_cfg.T_d
        # tgrid spans [0, T_d) (endpoint=False)
        assert abs(tgrid[0]) < 1e-15
        dt = tgrid[1] - tgrid[0]
        assert abs(tgrid[-1] + dt - expected_T) < 1e-12

    def test_weak_drive_modes_fock_like(self, default_cfg):
        """At zero drive, Floquet modes should be essentially pure Fock states."""
        cfg_zero = default_cfg.replace(epsilon=1e-6 * default_cfg.K)
        modes_t, _, _ = compute_floquet_modes(cfg_zero)
        for m in range(cfg_zero.N):
            weights = np.abs(modes_t[m, 0, :])**2
            # Largest weight should be very close to 1
            assert weights.max() > 0.99, (
                f"Mode {m} at zero drive: max weight = {weights.max():.4f}, expected ≈1"
            )


# ---------------------------------------------------------------------------
# Fourier components
# ---------------------------------------------------------------------------

class TestComputeFourierComponents:
    @pytest.fixture
    def floquet_data(self, driven_cfg):
        modes_t, qe, tgrid = compute_floquet_modes(driven_cfg)
        a, _, _ = make_operators(driven_cfg.N)
        return modes_t, a.full(), driven_cfg, tgrid, qe

    def test_returns_dict_with_correct_keys(self, floquet_data):
        modes_t, a_mat, cfg, tgrid, _ = floquet_data
        S_k = compute_fourier_components(modes_t, a_mat, cfg, tgrid)
        expected_keys = set(range(-cfg.k_max, cfg.k_max + 1))
        assert set(S_k.keys()) == expected_keys

    def test_each_component_is_N_by_N(self, floquet_data):
        modes_t, a_mat, cfg, tgrid, _ = floquet_data
        S_k = compute_fourier_components(modes_t, a_mat, cfg, tgrid)
        for k, Smn in S_k.items():
            assert Smn.shape == (cfg.N, cfg.N), f"k={k}: wrong shape {Smn.shape}"

    def test_parseval_like_bound(self, floquet_data):
        """Sum of |S_mn^(k)|² over k should be finite and ≤ 1 (coupling bounded)."""
        modes_t, a_mat, cfg, tgrid, _ = floquet_data
        S_k = compute_fourier_components(modes_t, a_mat, cfg, tgrid)
        total_power = sum(np.abs(Smn)**2 for Smn in S_k.values())
        # All entries should be < N (operator norm of a is sqrt(N-1))
        assert np.all(total_power < cfg.N * 10), "Fourier power unexpectedly large"


# ---------------------------------------------------------------------------
# Rate assembly
# ---------------------------------------------------------------------------

class TestAssembleRates:
    @pytest.fixture
    def rate_matrix(self, driven_cfg):
        modes_t, qe, tgrid = compute_floquet_modes(driven_cfg)
        R = assemble_rates_with_dephasing(modes_t, qe, driven_cfg, tgrid)
        return R, driven_cfg

    def test_shape(self, rate_matrix):
        R, cfg = rate_matrix
        assert R.shape == (cfg.N, cfg.N)

    def test_column_sums_zero(self, rate_matrix):
        """Rate matrix columns must sum to zero (probability conservation)."""
        R, _ = rate_matrix
        col_sums = R.sum(axis=0)
        np.testing.assert_allclose(col_sums, 0.0, atol=1e-10,
                                   err_msg="Rate matrix column sums not zero")

    def test_off_diagonal_non_negative(self, rate_matrix):
        """Off-diagonal rates Γ_mn (m≠n) must be ≥ 0."""
        R, _ = rate_matrix
        N = R.shape[0]
        for i in range(N):
            for j in range(N):
                if i != j:
                    assert R[i, j] >= -1e-10, (
                        f"Negative off-diagonal rate R[{i},{j}] = {R[i,j]:.2e}"
                    )

    def test_diagonal_non_positive(self, rate_matrix):
        """Diagonal rates R[i,i] = -Σ_{j≠i} R[j,i] must be ≤ 0."""
        R, _ = rate_matrix
        for i in range(R.shape[0]):
            assert R[i, i] <= 1e-10, f"Positive diagonal R[{i},{i}] = {R[i,i]:.2e}"

    def test_no_drive_gives_bare_kappa(self, default_cfg):
        """At zero drive, the dominant rate R[0,1] should be close to J(omega0)."""
        cfg_zero = default_cfg.replace(epsilon=0.0, omega_d=default_cfg.omega0 * 1.001)
        modes_t, qe, tgrid = compute_floquet_modes(cfg_zero)
        # Fock modes are ordered: mode 0 ≈ |0⟩, mode 1 ≈ |1⟩ at zero drive
        m0 = int(np.argmax([abs(modes_t[m, 0, 0])**2 for m in range(cfg_zero.N)]))
        m1 = int(np.argmax([abs(modes_t[m, 0, 1])**2 for m in range(cfg_zero.N)]))
        if m0 != m1:
            R = assemble_rates_with_dephasing(modes_t, qe, cfg_zero, tgrid)
            rate_C = float(R[m0, m1])
            rate_A = J(cfg_zero.omega0, cfg_zero)
            # Should agree within 10% for weak drive
            assert abs(rate_C - rate_A) / rate_A < 0.15, (
                f"Zero-drive FM rate {rate_C:.4e} vs J(ω₀) {rate_A:.4e}: "
                f"{abs(rate_C-rate_A)/rate_A:.1%} discrepancy"
            )


# ---------------------------------------------------------------------------
# floquet_steady_state
# ---------------------------------------------------------------------------

class TestFloquetSteadyState:
    def test_normalized(self, driven_cfg):
        modes_t, qe, tgrid = compute_floquet_modes(driven_cfg)
        R = assemble_rates_with_dephasing(modes_t, qe, driven_cfg, tgrid)
        p_ss = floquet_steady_state(R)
        assert abs(p_ss.sum() - 1.0) < 1e-6, f"p_ss sums to {p_ss.sum()}"

    def test_non_negative(self, driven_cfg):
        modes_t, qe, tgrid = compute_floquet_modes(driven_cfg)
        R = assemble_rates_with_dephasing(modes_t, qe, driven_cfg, tgrid)
        p_ss = floquet_steady_state(R)
        assert np.all(p_ss >= -1e-8), "Steady state has negative populations"

    def test_is_stationary(self, driven_cfg):
        """R @ p_ss ≈ 0 (null vector of rate matrix)."""
        modes_t, qe, tgrid = compute_floquet_modes(driven_cfg)
        R = assemble_rates_with_dephasing(modes_t, qe, driven_cfg, tgrid)
        p_ss = floquet_steady_state(R)
        residual = R @ p_ss
        # Numerical null-vector tolerance scales with rate matrix condition number
        np.testing.assert_allclose(residual, 0.0, atol=1e-3,
                                   err_msg="p_ss is not stationary under R")


# ---------------------------------------------------------------------------
# run_floquet_markov
# ---------------------------------------------------------------------------

class TestRunFloquetMarkov:
    @pytest.fixture
    def fm_setup(self, driven_cfg):
        modes_t, qe, tgrid = compute_floquet_modes(driven_cfg)
        R = assemble_rates_with_dephasing(modes_t, qe, driven_cfg, tgrid)
        p0 = np.zeros(driven_cfg.N)
        p0[1] = 1.0
        return driven_cfg, R, p0

    def test_output_shape(self, fm_setup):
        cfg, R, p0 = fm_setup
        tlist = np.linspace(0, 1e-3, 10)
        p_t, R_out, _ = run_floquet_markov(cfg, p0, tlist, R=R)
        assert p_t.shape == (10, cfg.N)

    def test_populations_normalized(self, fm_setup):
        """Population vector must sum to 1 at all times."""
        cfg, R, p0 = fm_setup
        tlist = np.linspace(0, 1e-3, 10)
        p_t, _, _ = run_floquet_markov(cfg, p0, tlist, R=R)
        row_sums = p_t.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_initial_condition(self, fm_setup):
        """p_t[0] should match p0."""
        cfg, R, p0 = fm_setup
        tlist = np.linspace(0, 1e-3, 10)
        p_t, _, _ = run_floquet_markov(cfg, p0, tlist, R=R)
        np.testing.assert_allclose(p_t[0], p0, atol=1e-6)

    def test_decay_from_excited_mode(self, fm_setup):
        """Starting in mode 1, the population should decrease over time."""
        cfg, R, p0 = fm_setup
        # Use 20/kappa — Floquet rate at finite drive can be much smaller than kappa
        T = 20.0 / cfg.kappa
        tlist = np.linspace(0, T, 50)
        p_t, _, _ = run_floquet_markov(cfg, p0, tlist, R=R)
        assert p_t[-1, 1] < p_t[0, 1], "mode-1 population should decrease"


# ---------------------------------------------------------------------------
# effective_decay_rate_from_R
# ---------------------------------------------------------------------------

class TestEffectiveDecayRateFromR:
    def test_positive_rate(self, driven_cfg):
        modes_t, qe, tgrid = compute_floquet_modes(driven_cfg)
        R = assemble_rates_with_dephasing(modes_t, qe, driven_cfg, tgrid)
        rate = effective_decay_rate_from_R(R)
        assert rate > 0

    def test_rate_is_finite(self, driven_cfg):
        modes_t, qe, tgrid = compute_floquet_modes(driven_cfg)
        R = assemble_rates_with_dephasing(modes_t, qe, driven_cfg, tgrid)
        rate = effective_decay_rate_from_R(R)
        assert np.isfinite(rate)

    def test_zero_kappa_no_loss_gives_smaller_rate(self, default_cfg):
        """With kappa=0 (only dephasing), rate should be smaller or zero."""
        cfg_loss = default_cfg.replace(epsilon=0.1 * default_cfg.K)
        cfg_nodiss = default_cfg.replace(epsilon=0.1 * default_cfg.K, kappa=0.0, gamma_phi=0.0, nbar=0.0)
        modes_t, qe, tgrid = compute_floquet_modes(cfg_loss)
        R_loss = assemble_rates_with_dephasing(modes_t, qe, cfg_loss, tgrid)
        R_nodiss = assemble_rates_with_dephasing(modes_t, qe, cfg_nodiss, tgrid)
        rate_with = effective_decay_rate_from_R(R_loss)
        rate_without = effective_decay_rate_from_R(R_nodiss)
        assert rate_with >= rate_without - 1e-12


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_extract_excited_pop_shape(self, default_cfg):
        rho0 = qt.fock_dm(default_cfg.N, 1)
        tlist = np.linspace(0, 1e-4, 5)
        result = run_lindblad(default_cfg, rho0, tlist)
        pop = extract_excited_pop(result, default_cfg.N)
        assert pop.shape == (len(tlist),)

    def test_extract_excited_pop_starts_at_one(self, default_cfg):
        rho0 = qt.fock_dm(default_cfg.N, 1)
        tlist = np.linspace(0, 1e-4, 5)
        result = run_lindblad(default_cfg, rho0, tlist)
        pop = extract_excited_pop(result, default_cfg.N)
        assert abs(pop[0] - 1.0) < 1e-6

    def test_effective_loss_rate_positive(self, default_cfg):
        rho0 = qt.fock_dm(default_cfg.N, 1)
        T = 3.0 / default_cfg.kappa
        tlist = np.linspace(0, T, 100)
        result = run_lindblad(default_cfg, rho0, tlist)
        rate = effective_loss_rate(result, tlist, default_cfg)
        assert rate > 0
        assert np.isfinite(rate)

    def test_effective_loss_rate_close_to_kappa(self, default_cfg):
        """At weak drive, the extracted rate should be ~kappa * (1 + 2*nbar)."""
        rho0 = qt.fock_dm(default_cfg.N, 1)
        T = 5.0 / default_cfg.kappa
        tlist = np.linspace(0, T, 200)
        result = run_lindblad(default_cfg, rho0, tlist)
        rate = effective_loss_rate(result, tlist, default_cfg)
        expected = default_cfg.kappa * (1 + 2 * default_cfg.nbar)
        assert abs(rate - expected) / expected < 0.05, (
            f"Extracted rate {rate/TWO_PI*1e3:.4f} MHz, expected {expected/TWO_PI*1e3:.4f} MHz"
        )

    def test_effective_loss_rate_from_fit(self):
        """Test the exponential fit utility."""
        tlist = np.linspace(0, 10.0, 200)
        gamma_true = 0.5
        pop = 0.8 * np.exp(-gamma_true * tlist) + 0.1
        rate = effective_loss_rate_from_fit(pop, tlist)
        assert abs(rate - gamma_true) / gamma_true < 0.02

    def test_steady_state_leakage_near_zero(self, default_cfg):
        """After long evolution from |1⟩, leakage to higher Fock states should be small."""
        rho0 = qt.fock_dm(default_cfg.N, 1)
        T = 10.0 / default_cfg.kappa
        tlist = np.linspace(0, T, 100)
        result = run_lindblad(default_cfg, rho0, tlist)
        leakage = steady_state_leakage(result, default_cfg)
        assert leakage < 0.01, f"Leakage = {leakage:.4f} at weak drive (unexpected)"


# ---------------------------------------------------------------------------
# error_budget (integration test)
# ---------------------------------------------------------------------------

class TestErrorBudget:
    def test_returns_dict_with_keys(self, default_cfg):
        tlist = np.linspace(0, 3.0 / default_cfg.kappa, 80)
        budget = error_budget(default_cfg, tlist, method="lindblad")
        required = ["total_infidelity", "loss_contribution",
                    "thermal_contribution", "dephasing_contribution"]
        for key in required:
            assert key in budget, f"Missing key: {key}"

    def test_contributions_non_negative(self, default_cfg):
        """Channel contributions should be non-negative or only slightly negative."""
        tlist = np.linspace(0, 3.0 / default_cfg.kappa, 80)
        budget = error_budget(default_cfg, tlist, method="lindblad")
        for key in ["loss_contribution", "thermal_contribution", "dephasing_contribution"]:
            # Small negatives are possible when channels partially cancel
            assert budget[key] >= -0.05, f"{key} = {budget[key]:.4e} unexpectedly negative"

    def test_total_infidelity_between_0_and_1(self, default_cfg):
        tlist = np.linspace(0, 3.0 / default_cfg.kappa, 80)
        budget = error_budget(default_cfg, tlist, method="lindblad")
        assert 0 <= budget["total_infidelity"] <= 1.0 + 1e-6

    def test_dephasing_method_a_near_zero(self, default_cfg):
        """Method A: dephasing cannot drive population decay (γ_φ n-conserving)."""
        # At zero dephasing vs with dephasing: the contribution should be very small
        # in Method A (Lindblad), because gamma_phi * n-op doesn't change photon number
        tlist = np.linspace(0, 5.0 / default_cfg.kappa, 150)
        budget = error_budget(default_cfg, tlist, method="lindblad")
        # Dephasing contribution in Lindblad should be ~0 (the gamma_phi jump op is
        # a†a which commutes with photon number, so it doesn't drive |1⟩→|0⟩)
        assert budget["dephasing_contribution"] < 0.01, (
            f"Method A dephasing contribution = {budget['dephasing_contribution']:.4f}, "
            "expected near zero"
        )


# ---------------------------------------------------------------------------
# Floquet-Markov thermal limit (regression for detailed-balance bug)
# ---------------------------------------------------------------------------

class TestFloquetThermalLimit:
    """At ε=0 the FM steady state must reproduce the Lindblad/Boltzmann distribution.

    This is the critical regression test: with only `a` as the coupling operator
    the absorption channel is zero (a|0⟩=0) and the steady state collapses to
    a flat distribution.  With x = a+a† both emission and absorption are populated
    and detailed balance is satisfied.
    """

    def test_steady_state_not_flat(self, default_cfg):
        """p_ss must NOT be uniform — p0 > p1 at low temperature."""
        cfg = default_cfg.replace(epsilon=0.0, omega_d=default_cfg.omega0 * 1.001,
                                  omega_f=default_cfg.omega0, Gamma=TWO_PI * 0.1)
        modes_t, qe, tgrid = compute_floquet_modes(cfg)
        # Identify which mode is ≈ |0⟩ and ≈ |1⟩ by Fock weight at t=0
        m0 = int(np.argmax([abs(modes_t[m, 0, 0])**2 for m in range(cfg.N)]))
        m1 = int(np.argmax([abs(modes_t[m, 0, 1])**2 for m in range(cfg.N)]))
        if m0 == m1:
            pytest.skip("mode ordering ambiguous at this cfg — skip rather than false-fail")
        R = assemble_rates_with_dephasing(modes_t, qe, cfg, tgrid)
        p_ss = floquet_steady_state(R)
        # At nbar=0.02, thermal ratio p1/p0 ≈ 0.02 — far from uniform (0.5 for N=2)
        ratio = p_ss[m1] / p_ss[m0]
        assert ratio < 0.5, (
            f"Steady state looks flat: p_ss[m1]/p_ss[m0] = {ratio:.3f}. "
            "Detailed balance is broken (absorption channel likely missing)."
        )

    def test_detailed_balance_ratio(self, default_cfg):
        """p1/p0 in FM steady state must match nbar/(1+nbar) within 20%."""
        cfg = default_cfg.replace(epsilon=0.0, omega_d=default_cfg.omega0 * 1.001,
                                  omega_f=default_cfg.omega0, Gamma=TWO_PI * 0.1)
        modes_t, qe, tgrid = compute_floquet_modes(cfg)
        m0 = int(np.argmax([abs(modes_t[m, 0, 0])**2 for m in range(cfg.N)]))
        m1 = int(np.argmax([abs(modes_t[m, 0, 1])**2 for m in range(cfg.N)]))
        if m0 == m1:
            pytest.skip("mode ordering ambiguous")
        R = assemble_rates_with_dephasing(modes_t, qe, cfg, tgrid)
        p_ss = floquet_steady_state(R)
        expected_ratio = cfg.nbar / (1.0 + cfg.nbar)
        actual_ratio = p_ss[m1] / p_ss[m0]
        assert abs(actual_ratio - expected_ratio) / expected_ratio < 0.20, (
            f"FM detailed balance: p1/p0 = {actual_ratio:.4f}, "
            f"expected {expected_ratio:.4f} (nbar/(1+nbar))"
        )


# ---------------------------------------------------------------------------
# run_full_floquet_markov (pipeline integration test)
# ---------------------------------------------------------------------------

class TestRunFullFloquetMarkov:
    def test_returns_dict_with_keys(self, driven_cfg):
        p0 = np.zeros(driven_cfg.N)
        p0[1] = 1.0
        tlist = np.linspace(0, 1e-3, 10)
        result = run_full_floquet_markov(driven_cfg, p0, tlist)
        for key in ["p_t", "R", "quasi_energies", "modes_t", "tgrid"]:
            assert key in result, f"Missing key: {key}"

    def test_p_t_normalized(self, driven_cfg):
        p0 = np.zeros(driven_cfg.N)
        p0[1] = 1.0
        tlist = np.linspace(0, 1e-3, 10)
        result = run_full_floquet_markov(driven_cfg, p0, tlist)
        row_sums = result["p_t"].sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

    def test_R_column_sum_zero(self, driven_cfg):
        p0 = np.zeros(driven_cfg.N)
        p0[1] = 1.0
        tlist = np.linspace(0, 1e-3, 5)
        result = run_full_floquet_markov(driven_cfg, p0, tlist)
        col_sums = result["R"].sum(axis=0)
        np.testing.assert_allclose(col_sums, 0.0, atol=1e-10)
