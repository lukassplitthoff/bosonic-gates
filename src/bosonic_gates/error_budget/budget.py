"""
Error budget computation for bosonic gate simulations.

Implements the "turn-one-on-at-a-time" error budget methodology:
  1. Simulate the gate with ALL decoherence channels active → baseline infidelity.
  2. For each channel, run the simulation with that channel disabled.
  3. The per-channel contribution is (full - disabled) infidelity.

This additive decomposition is accurate when infidelities are small (< ~10%).

Channels tracked:
  - photon_loss    : κ (energy decay)
  - thermal        : thermal photons (nbar > 0)
  - dephasing      : γ_φ (pure dephasing)
  - structured_bath: Lorentzian structure of J(ω) vs white bath
"""

from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np
import qutip as qt

from ..driven_kerr.config import DrivenKerrConfig
from ..driven_kerr.lindblad import run_lindblad
from ..driven_kerr.redfield import run_redfield
from ..driven_kerr.floquet_markov import run_full_floquet_markov, floquet_steady_state


@dataclass
class ErrorBudget:
    """Per-channel infidelity contributions.

    Attributes
    ----------
    total_infidelity : float
        Total 1 - F_avg from the full simulation.
    channels : dict[str, float]
        Per-channel infidelity contributions (sum ≈ total_infidelity).
    method : str
        Simulation method used ('lindblad', 'redfield', 'floquet_markov').
    cfg : DrivenKerrConfig
        Configuration used for the simulation.
    """
    total_infidelity: float
    channels: dict[str, float]
    method: str
    cfg: DrivenKerrConfig

    def __repr__(self):
        lines = [f"ErrorBudget (method={self.method!r}, total={self.total_infidelity:.4e})"]
        for name, val in self.channels.items():
            lines.append(f"  {name:<20}: {val:.4e}  ({val/max(self.total_infidelity, 1e-15)*100:.1f}%)")
        return "\n".join(lines)

    def as_dict(self) -> dict[str, float]:
        """Return a flat dict of all budget entries (total + per-channel)."""
        return {"total": self.total_infidelity, **self.channels}


def _run_sim(cfg: DrivenKerrConfig, rho0: qt.Qobj, tlist: np.ndarray, method: str) -> qt.solver.Result:
    """Dispatch to the correct simulation backend."""
    if method == "lindblad":
        return run_lindblad(cfg, rho0, tlist)
    elif method == "redfield":
        return run_redfield(cfg, rho0, tlist)
    else:
        raise ValueError(f"method must be 'lindblad' or 'redfield', got {method!r}")


def _pop1_final(result: qt.solver.Result, N: int) -> float:
    """Return P(|1⟩) at the final time step."""
    return float(qt.expect(qt.ket2dm(qt.basis(N, 1)), result.states[-1]))


def compute_error_budget(
    cfg: DrivenKerrConfig,
    tlist: np.ndarray,
    method: str = "lindblad",
) -> ErrorBudget:
    """
    Compute the per-channel error budget for a driven-Kerr gate simulation.

    Runs four simulations:
      1. All channels active (baseline).
      2. Loss disabled (kappa=0).
      3. Thermal photons disabled (nbar=0).
      4. Dephasing disabled (gamma_phi=0).

    Per-channel infidelity = infidelity(all) - infidelity(channel_off).

    Parameters
    ----------
    cfg : DrivenKerrConfig
        Configuration with all channels active.
    tlist : np.ndarray
        Simulation time array.
    method : str
        'lindblad' or 'redfield'.

    Returns
    -------
    ErrorBudget

    Example
    -------
    >>> from bosonic_gates.driven_kerr import DrivenKerrConfig
    >>> from bosonic_gates.error_budget import compute_error_budget
    >>> import numpy as np
    >>> cfg = DrivenKerrConfig(epsilon=2*np.pi*0.05e9)
    >>> tlist = np.linspace(0, 5e-6, 300)
    >>> budget = compute_error_budget(cfg, tlist)
    >>> print(budget)
    """
    rho0 = qt.ket2dm(qt.basis(cfg.N, 1))

    # Baseline: all channels
    result_full = _run_sim(cfg, rho0, tlist, method)
    f_full = _pop1_final(result_full, cfg.N)
    infidelity_full = 1.0 - f_full

    # Turn off loss (kappa → 0)
    result_no_loss = _run_sim(cfg.replace(kappa=0.0), rho0, tlist, method)
    f_no_loss = _pop1_final(result_no_loss, cfg.N)

    # Turn off thermal (nbar → 0)
    result_no_thermal = _run_sim(cfg.replace(nbar=0.0), rho0, tlist, method)
    f_no_thermal = _pop1_final(result_no_thermal, cfg.N)

    # Turn off dephasing (gamma_phi → 0)
    result_no_deph = _run_sim(cfg.replace(gamma_phi=0.0), rho0, tlist, method)
    f_no_deph = _pop1_final(result_no_deph, cfg.N)

    channels = {
        "photon_loss": f_no_loss - f_full,        # loss contribution to infidelity
        "thermal":     f_no_thermal - f_full,     # thermal contribution
        "dephasing":   f_no_deph - f_full,        # dephasing contribution
    }

    return ErrorBudget(
        total_infidelity=infidelity_full,
        channels=channels,
        method=method,
        cfg=cfg,
    )


def compute_error_budget_sweep(
    cfg_base: DrivenKerrConfig,
    epsilon_values: np.ndarray,
    tlist: np.ndarray,
    method: str = "lindblad",
) -> dict[str, np.ndarray]:
    """
    Compute the error budget over a sweep of drive amplitudes.

    Parameters
    ----------
    cfg_base : DrivenKerrConfig
        Base configuration (epsilon will be overridden).
    epsilon_values : np.ndarray
        Array of drive amplitudes ε to sweep over.
    tlist : np.ndarray
        Simulation time array.
    method : str
        'lindblad' or 'redfield'.

    Returns
    -------
    dict with keys 'epsilon', 'total', 'photon_loss', 'thermal', 'dephasing',
    each mapped to a 1D array of the same length as epsilon_values.
    """
    results = {
        "epsilon": epsilon_values,
        "total": np.zeros(len(epsilon_values)),
        "photon_loss": np.zeros(len(epsilon_values)),
        "thermal": np.zeros(len(epsilon_values)),
        "dephasing": np.zeros(len(epsilon_values)),
    }

    for i, eps in enumerate(epsilon_values):
        cfg = cfg_base.replace(epsilon=eps)
        budget = compute_error_budget(cfg, tlist, method)
        results["total"][i] = budget.total_infidelity
        for key in ("photon_loss", "thermal", "dephasing"):
            results[key][i] = budget.channels[key]

    return results
