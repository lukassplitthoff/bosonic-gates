"""
Shared physics for the driven-Kerr open-system simulation.

This module is the single source of truth for:
  - Operator construction (a, a†, n)
  - Hamiltonian (static H0 + time-dependent drive)
  - Bath spectral density J(ω) with thermal factors
  - Jump operators for the Lindblad baseline

All three methods (Lindblad, Redfield, Floquet–Markov) import from here.

Spectral-density convention
---------------------------
J(ω) is the *two-sided* spectral density with thermal factors built in:

    ω > 0  (emission/loss):   J(ω)  = κ_Lor(ω)  × (1 + nbar)
    ω < 0  (absorption/gain): J(ω)  = κ_Lor(−ω) × nbar
    ω = 0:                    J(ω)  = 0

where κ_Lor(ω) = κ × (Γ/2)² / [(ω − ω_f)² + (Γ/2)²]  (Lorentzian, positive ω only).

Detailed balance is satisfied by construction:
    J(−ω) / J(ω) = nbar / (1 + nbar) = exp(−ℏω / k_B T)

The dephasing channel uses a flat (white) spectral density independent of ω:
    J_phi(ω) = γ_φ     for all ω ≠ 0
"""

import numpy as np
import qutip as qt
from .config import DrivenKerrConfig


def make_operators(N: int) -> tuple[qt.Qobj, qt.Qobj, qt.Qobj]:
    """Return (a, a†, n) for an N-level Fock space."""
    a = qt.destroy(N)
    return a, a.dag(), a.dag() * a


def make_H0(cfg: DrivenKerrConfig) -> qt.Qobj:
    """Static (undriven) Hamiltonian: H₀ = ω₀ a†a − (K/2) a†a†aa."""
    a, adag, num = make_operators(cfg.N)
    return cfg.omega0 * num - (cfg.K / 2) * adag * adag * a * a


def make_H_drive_td(cfg: DrivenKerrConfig) -> list:
    """Return the time-dependent Hamiltonian in QuTiP list format for mesolve.

    Format: [H0, [H_amp, coeff_func]]
    where H_amp = a + a† (X quadrature coupling) and coeff_func encodes ε cos(ω_d t).

    The drive is kept in full lab-frame form — no RWA, no rotating frame.
    Floquet periodicity H(t) = H(t + T_d) is preserved exactly.
    """
    a, adag, num = make_operators(cfg.N)
    H0 = make_H0(cfg)
    H_amp = a + adag
    epsilon = cfg.epsilon
    omega_d = cfg.omega_d

    def coeff(t):
        return epsilon * np.cos(omega_d * t)

    return [H0, [H_amp, coeff]]


# ---------------------------------------------------------------------------
# Spectral density
# ---------------------------------------------------------------------------

def J(omega: float, cfg: DrivenKerrConfig) -> float:
    """Two-sided bath spectral density with thermal factors.

    Convention (see module docstring):
        ω > 0 → loss/emission weight ~ (1 + nbar)
        ω < 0 → gain/absorption weight ~ nbar
        ω = 0 → 0

    The bare Lorentzian κ_Lor is centred at ω_f (positive frequency).
    Dephasing is *not* included here; it uses J_phi separately.
    """
    if omega == 0.0:
        return 0.0
    half_Gamma = cfg.Gamma / 2.0
    if omega > 0:
        kappa_lor = cfg.kappa * half_Gamma**2 / ((omega - cfg.omega_f)**2 + half_Gamma**2)
        return kappa_lor * (1.0 + cfg.nbar)
    else:
        kappa_lor = cfg.kappa * half_Gamma**2 / ((-omega - cfg.omega_f)**2 + half_Gamma**2)
        return kappa_lor * cfg.nbar


def J_vectorized(omega: np.ndarray, cfg: DrivenKerrConfig) -> np.ndarray:
    """Vectorized version of J for use in rate summations."""
    omega = np.asarray(omega, dtype=float)
    out = np.zeros_like(omega)
    half_Gamma = cfg.Gamma / 2.0
    pos = omega > 0
    neg = omega < 0
    out[pos] = cfg.kappa * half_Gamma**2 / ((omega[pos] - cfg.omega_f)**2 + half_Gamma**2) * (1.0 + cfg.nbar)
    out[neg] = cfg.kappa * half_Gamma**2 / ((-omega[neg] - cfg.omega_f)**2 + half_Gamma**2) * cfg.nbar
    return out


def J_phi(omega: float, cfg: DrivenKerrConfig) -> float:
    """Flat dephasing spectral density (independent of ω).

    Returns γ_φ as the one-sided rate.
    """
    return cfg.gamma_phi


# ---------------------------------------------------------------------------
# Jump operators for Lindblad
# ---------------------------------------------------------------------------

def make_jump_ops(cfg: DrivenKerrConfig) -> list[qt.Qobj]:
    """Return Lindblad collapse operators.

    Rates are fixed by sampling J at the bare transition frequency ω₀.
    They do NOT depend on the drive amplitude ε.

    Returns: [√γ_loss · a,  √γ_gain · a†,  √γ_φ · a†a]
    """
    a, adag, num = make_operators(cfg.N)
    gamma_loss = J(cfg.omega0, cfg)
    gamma_gain = J(-cfg.omega0, cfg)
    c_ops = []
    if gamma_loss > 0:
        c_ops.append(np.sqrt(gamma_loss) * a)
    if gamma_gain > 0:
        c_ops.append(np.sqrt(gamma_gain) * adag)
    if cfg.gamma_phi > 0:
        c_ops.append(np.sqrt(cfg.gamma_phi) * num)
    return c_ops
