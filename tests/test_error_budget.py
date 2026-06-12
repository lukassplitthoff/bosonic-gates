"""
Tests for bosonic_gates.error_budget — compute_error_budget and ErrorBudget.

Uses a small Hilbert space (N=4) and coarse time grids to stay fast.
Follows the same fixture and test-class conventions as test_driven_kerr.py.
"""

import numpy as np
import pytest

TWO_PI = 2 * np.pi

from bosonic_gates.driven_kerr import DrivenKerrConfig
from bosonic_gates.error_budget import compute_error_budget, ErrorBudget
from bosonic_gates.error_budget.budget import compute_error_budget_sweep


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_cfg():
    """Small N=4 configuration for fast tests."""
    return DrivenKerrConfig(
        N=4,
        omega0=TWO_PI * 5.0,
        K=TWO_PI * 0.2,
        omega_d=TWO_PI * 4.995,
        kappa=TWO_PI * 1e-3,
        Gamma=TWO_PI * 0.1,
        nbar=0.02,
        gamma_phi=TWO_PI * 0.05e-3,
        k_max=3,
        n_t=64,
    )


@pytest.fixture
def short_tlist(small_cfg):
    """Time array covering 2 decay times at κ (epsilon=0 only — lab-frame H is static)."""
    return np.linspace(0, 2.0 / small_cfg.kappa, 40)


@pytest.fixture
def tiny_tlist():
    """Very short time array for sweep tests that use nonzero ε.

    When ε > 0 the time-dependent lab-frame H oscillates at ω₀ ≈ 31 rad/GHz.
    A long tlist forces the ODE solver through thousands of oscillation cycles
    and triggers IntegratorException. Using T << 1/ω₀ avoids this: the tests
    only check return-value structure, not physics values.
    """
    return np.linspace(0, 1e-4, 5)


# ---------------------------------------------------------------------------
# TestComputeErrorBudget
# ---------------------------------------------------------------------------

class TestComputeErrorBudget:
    def test_returns_error_budget_instance(self, small_cfg, short_tlist):
        budget = compute_error_budget(small_cfg, short_tlist)
        assert isinstance(budget, ErrorBudget)

    def test_required_keys_in_channels(self, small_cfg, short_tlist):
        budget = compute_error_budget(small_cfg, short_tlist)
        for key in ["photon_loss", "thermal", "dephasing"]:
            assert key in budget.channels

    def test_total_infidelity_between_0_and_1(self, small_cfg, short_tlist):
        budget = compute_error_budget(small_cfg, short_tlist)
        assert 0.0 <= budget.total_infidelity <= 1.0 + 1e-6

    def test_total_dominates_individual_channels(self, small_cfg, short_tlist):
        """Each per-channel contribution should be ≤ total (up to small tolerance)."""
        budget = compute_error_budget(small_cfg, short_tlist)
        for key, val in budget.channels.items():
            assert val <= budget.total_infidelity + 0.05, (
                f"{key} contribution {val:.4e} exceeds total {budget.total_infidelity:.4e}"
            )

    def test_method_stored_in_result(self, small_cfg, short_tlist):
        budget = compute_error_budget(small_cfg, short_tlist, method="lindblad")
        assert budget.method == "lindblad"

    def test_cfg_stored_in_result(self, small_cfg, short_tlist):
        budget = compute_error_budget(small_cfg, short_tlist)
        assert budget.cfg is small_cfg

    def test_as_dict_returns_all_keys(self, small_cfg, short_tlist):
        budget = compute_error_budget(small_cfg, short_tlist)
        d = budget.as_dict()
        assert "total" in d
        for key in ["photon_loss", "thermal", "dephasing"]:
            assert key in d

    def test_as_dict_total_matches_total_infidelity(self, small_cfg, short_tlist):
        budget = compute_error_budget(small_cfg, short_tlist)
        d = budget.as_dict()
        assert abs(d["total"] - budget.total_infidelity) < 1e-14

    def test_zero_kappa_reduces_loss(self, small_cfg, short_tlist):
        """Setting kappa=0 should reduce total infidelity (loss channel gone)."""
        budget_full = compute_error_budget(small_cfg, short_tlist)
        cfg_no_loss = small_cfg.replace(kappa=0.0)
        budget_no_loss = compute_error_budget(cfg_no_loss, short_tlist)
        assert budget_no_loss.total_infidelity <= budget_full.total_infidelity + 0.02

    def test_invalid_method_raises(self, small_cfg, short_tlist):
        with pytest.raises(ValueError):
            compute_error_budget(small_cfg, short_tlist, method="invalid")

    def test_repr_contains_method(self, small_cfg, short_tlist):
        budget = compute_error_budget(small_cfg, short_tlist)
        assert "lindblad" in repr(budget).lower()

    def test_repr_contains_total(self, small_cfg, short_tlist):
        """__repr__ should report the total infidelity."""
        budget = compute_error_budget(small_cfg, short_tlist)
        r = repr(budget)
        assert "total" in r.lower()

    def test_redfield_method_accepted(self, small_cfg, short_tlist):
        """Method 'redfield' should not raise and return an ErrorBudget."""
        budget = compute_error_budget(small_cfg, short_tlist, method="redfield")
        assert isinstance(budget, ErrorBudget)
        assert budget.method == "redfield"

    def test_channels_non_negative_at_weak_drive(self, small_cfg, short_tlist):
        """At weak drive all channel contributions should be non-negative."""
        # small_cfg has epsilon=0 so all contributions should be non-negative
        budget = compute_error_budget(small_cfg, short_tlist)
        for key, val in budget.channels.items():
            assert val >= -0.01, (
                f"Channel {key!r} = {val:.4e} unexpectedly negative at weak drive"
            )

    def test_zero_nbar_removes_thermal(self, small_cfg, short_tlist):
        """Setting nbar=0 should make the thermal contribution very small."""
        cfg_zero_nbar = small_cfg.replace(nbar=0.0)
        budget = compute_error_budget(cfg_zero_nbar, short_tlist)
        assert abs(budget.channels["thermal"]) < 0.01, (
            f"Thermal contribution = {budget.channels['thermal']:.4e} with nbar=0"
        )

    def test_zero_gamma_phi_removes_dephasing(self, small_cfg, short_tlist):
        """Setting gamma_phi=0 should make the dephasing contribution very small."""
        cfg_zero_deph = small_cfg.replace(gamma_phi=0.0)
        budget = compute_error_budget(cfg_zero_deph, short_tlist)
        # Lindblad dephasing (n-op) does not drive population decay → near zero
        assert abs(budget.channels["dephasing"]) < 0.01, (
            f"Dephasing contribution = {budget.channels['dephasing']:.4e} with gamma_phi=0"
        )

    def test_dephasing_method_a_near_zero(self, small_cfg, short_tlist):
        """Method A: dephasing operator n̂ commutes with photon number — no population decay."""
        tlist_long = np.linspace(0, 5.0 / small_cfg.kappa, 100)
        budget = compute_error_budget(small_cfg, tlist_long, method="lindblad")
        assert budget.channels["dephasing"] < 0.01, (
            f"Method A dephasing contribution = {budget.channels['dephasing']:.4f}, "
            "expected near zero (n̂ is number-conserving in Lindblad)"
        )

    def test_loss_dominates_at_high_kappa(self, short_tlist):
        """At high κ, loss should be the dominant channel."""
        cfg_high_kappa = DrivenKerrConfig(
            N=4,
            omega0=TWO_PI * 5.0,
            K=TWO_PI * 0.2,
            omega_d=TWO_PI * 4.995,
            kappa=TWO_PI * 10e-3,      # 10× higher loss rate
            Gamma=TWO_PI * 0.1,
            nbar=0.001,                # very small thermal
            gamma_phi=TWO_PI * 0.01e-3,  # very small dephasing
            k_max=3,
            n_t=64,
        )
        budget = compute_error_budget(cfg_high_kappa, short_tlist)
        # Loss must be the largest channel
        loss = budget.channels["photon_loss"]
        for key, val in budget.channels.items():
            if key != "photon_loss":
                assert loss >= val - 1e-4, (
                    f"Loss {loss:.4e} should dominate over {key} = {val:.4e}"
                )


# ---------------------------------------------------------------------------
# TestComputeErrorBudgetSweep
# ---------------------------------------------------------------------------

class TestComputeErrorBudgetSweep:
    def test_returns_dict_with_required_keys(self, small_cfg, tiny_tlist):
        eps_arr = np.array([TWO_PI * 0.01, TWO_PI * 0.05])
        result = compute_error_budget_sweep(small_cfg, eps_arr, tiny_tlist)
        for key in ["epsilon", "total", "photon_loss", "thermal", "dephasing"]:
            assert key in result

    def test_arrays_have_correct_length(self, small_cfg, tiny_tlist):
        eps_arr = np.linspace(TWO_PI * 0.01, TWO_PI * 0.1, 3)
        result = compute_error_budget_sweep(small_cfg, eps_arr, tiny_tlist)
        for key in ["total", "photon_loss", "thermal", "dephasing"]:
            assert len(result[key]) == len(eps_arr), (
                f"Key {key!r}: expected length {len(eps_arr)}, got {len(result[key])}"
            )

    def test_epsilon_array_preserved(self, small_cfg, tiny_tlist):
        """The returned 'epsilon' array must be identical to the input."""
        eps_arr = np.array([TWO_PI * 0.01, TWO_PI * 0.05])
        result = compute_error_budget_sweep(small_cfg, eps_arr, tiny_tlist)
        np.testing.assert_array_equal(result["epsilon"], eps_arr)

    def test_totals_are_non_negative(self, small_cfg, tiny_tlist):
        eps_arr = np.array([TWO_PI * 0.01, TWO_PI * 0.05])
        result = compute_error_budget_sweep(small_cfg, eps_arr, tiny_tlist)
        assert np.all(result["total"] >= -1e-8)

    def test_totals_at_most_one(self, small_cfg, tiny_tlist):
        eps_arr = np.array([TWO_PI * 0.01, TWO_PI * 0.05])
        result = compute_error_budget_sweep(small_cfg, eps_arr, tiny_tlist)
        assert np.all(result["total"] <= 1.0 + 1e-6)

    def test_single_epsilon_matches_compute_error_budget(self, small_cfg, tiny_tlist):
        """Single-point sweep must agree with compute_error_budget at the same ε."""
        eps = TWO_PI * 0.02
        sweep = compute_error_budget_sweep(small_cfg, np.array([eps]), tiny_tlist)
        cfg_eps = small_cfg.replace(epsilon=eps)
        single = compute_error_budget(cfg_eps, tiny_tlist)
        assert abs(sweep["total"][0] - single.total_infidelity) < 1e-10
        for key in ["photon_loss", "thermal", "dephasing"]:
            assert abs(sweep[key][0] - single.channels[key]) < 1e-10

    def test_redfield_method_accepted(self, small_cfg, tiny_tlist):
        eps_arr = np.array([TWO_PI * 0.01])
        result = compute_error_budget_sweep(
            small_cfg, eps_arr, tiny_tlist, method="redfield"
        )
        assert "total" in result
        assert len(result["total"]) == 1

    def test_three_eps_points_returns_three_values(self, small_cfg, tiny_tlist):
        eps_arr = np.geomspace(TWO_PI * 0.005, TWO_PI * 0.05, 3)
        result = compute_error_budget_sweep(small_cfg, eps_arr, tiny_tlist)
        for key in ["total", "photon_loss", "thermal", "dephasing"]:
            assert result[key].shape == (3,)
