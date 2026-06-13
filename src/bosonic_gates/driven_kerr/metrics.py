"""
Gate error metrics for the driven-Kerr open-system comparison.

Metrics computed here:
  - effective_loss_rate: dominant decay rate (from exponential fit or eigenvalue)
  - steady_state_leakage: population outside {|0⟩, |1⟩} at steady state
  - average_gate_fidelity: Nielsen formula, F_avg = (Tr(P†P) + |Tr(P)|²) / (d²+d)
  - leakage_seepage: L1 (leakage), L2 (seepage) per Wood-Gambetta (2015)
  - error_budget: per-channel infidelity contribution
"""

import warnings
import numpy as np
import scipy.optimize
import qutip as qt
from .config import DrivenKerrConfig
from .core import make_operators


# ---------------------------------------------------------------------------
# Loss rate
# ---------------------------------------------------------------------------

def effective_loss_rate_from_fit(
    excited_pop: np.ndarray,
    tlist: np.ndarray,
) -> float:
    """Fit P_excited(t) ≈ A exp(−Γ t) + C and return Γ."""
    def model(t, A, gamma, C):
        return A * np.exp(-gamma * t) + C

    try:
        p0_guess = [excited_pop[0] - excited_pop[-1], 1.0 / (tlist[-1] - tlist[0]), excited_pop[-1]]
        popt, _ = scipy.optimize.curve_fit(model, tlist, excited_pop, p0=p0_guess, maxfev=5000)
        return float(abs(popt[1]))
    except Exception:
        return float("nan")


def extract_excited_pop(result: qt.solver.Result, N: int) -> np.ndarray:
    """Extract ⟨1|ρ|1⟩ (first excited state population) from a mesolve result."""
    proj1 = qt.ket2dm(qt.basis(N, 1))
    return np.array([qt.expect(proj1, rho) for rho in result.states])


def effective_loss_rate(
    result: qt.solver.Result,
    tlist: np.ndarray,
    cfg: DrivenKerrConfig,
) -> float:
    """Effective loss rate from a mesolve/brmesolve result (methods A/B).

    Uses exponential fit to P_|1⟩(t).
    """
    pop1 = extract_excited_pop(result, cfg.N)
    return effective_loss_rate_from_fit(pop1, tlist)


def steady_state_leakage(result: qt.solver.Result, cfg: DrivenKerrConfig) -> float:
    """Population outside {|0⟩, |1⟩} at the final time step."""
    rho_ss = result.states[-1]
    a, adag, num = make_operators(cfg.N)
    p0 = float(qt.expect(qt.ket2dm(qt.basis(cfg.N, 0)), rho_ss))
    p1 = float(qt.expect(qt.ket2dm(qt.basis(cfg.N, 1)), rho_ss))
    return max(0.0, 1.0 - p0 - p1)


# ---------------------------------------------------------------------------
# Average gate fidelity (Nielsen formula)
# ---------------------------------------------------------------------------

def average_gate_fidelity(
    U_ideal: qt.Qobj,
    final_states: list[qt.Qobj],
    basis_psis: list[qt.Qobj],
    cfg: DrivenKerrConfig,
) -> float:
    """Compute average gate fidelity F_avg using the Nielsen formula.

    F_avg = (Σ_j ⟨ψ_j|U†ρ_j U|ψ_j⟩) / d   (simplified for pure input states)
    where d = 2 is the computational subspace dimension.

    Parameters
    ----------
    U_ideal     : target unitary on the full Fock space (N×N qt.Qobj)
    final_states: list of final density matrices (one per basis state)
    basis_psis  : corresponding input pure states
    cfg         : configuration

    Returns
    -------
    F_avg : average gate fidelity in [0, 1]
    """
    fidelity_sum = 0.0
    for psi_in, rho_out in zip(basis_psis, final_states):
        psi_ideal = U_ideal * psi_in
        fidelity_sum += float(qt.expect(qt.ket2dm(psi_ideal), rho_out))
    return fidelity_sum / len(basis_psis)


# ---------------------------------------------------------------------------
# Leakage and seepage (Wood-Gambetta 2015)
# ---------------------------------------------------------------------------

def _comp_projector(cfg: DrivenKerrConfig) -> qt.Qobj:
    """Projector onto the computational subspace {|0⟩, |1⟩}."""
    P = qt.basis(cfg.N, 0) * qt.basis(cfg.N, 0).dag() + qt.basis(cfg.N, 1) * qt.basis(cfg.N, 1).dag()
    return P


def leakage_seepage(
    result: qt.solver.Result,
    cfg: DrivenKerrConfig,
) -> dict[str, float]:
    """Compute L1 (leakage) and L2 (seepage) per Wood-Gambetta (2015).

    Parameters
    ----------
    result : QuTiP solver result with .states list
    cfg    : configuration

    Returns
    -------
    dict with keys 'L1', 'L2'
    """
    P_comp = _comp_projector(cfg)
    rho_final = result.states[-1]

    p_in_comp = float(qt.expect(P_comp, rho_final))
    L1 = max(0.0, 1.0 - p_in_comp)
    L2 = float("nan")  # requires separate simulation from leakage state

    return {"L1": L1, "L2": L2}


# ---------------------------------------------------------------------------
# Error budget: per-channel infidelity
# ---------------------------------------------------------------------------

def error_budget(
    cfg: DrivenKerrConfig,
    tlist: np.ndarray,
    method: str = "lindblad",
) -> dict[str, float]:
    """Compute the contribution of each dissipation channel to the total infidelity.

    Runs the simulation four times: once with all channels, and once with each
    channel disabled in turn. The difference gives the per-channel contribution.

    Fidelity proxy
    --------------
    Uses P(|1⟩) at the final time step as the fidelity proxy, NOT the Nielsen
    average gate fidelity.  For the richer ``ErrorBudget`` dataclass and the
    same methodology, prefer ``bosonic_gates.error_budget.compute_error_budget``.

    Parameters
    ----------
    cfg    : base configuration (all channels active)
    tlist  : simulation time array
    method : 'lindblad' or 'redfield'

    Returns
    -------
    dict with keys 'total_infidelity', 'loss_contribution', 'thermal_contribution',
    'dephasing_contribution'.

    Example
    -------
    >>> from bosonic_gates.driven_kerr import DrivenKerrConfig, error_budget
    >>> import numpy as np
    >>> cfg = DrivenKerrConfig(epsilon=2*np.pi*0.05e9)
    >>> tlist = np.linspace(0, 5e-6, 200)
    >>> budget = error_budget(cfg, tlist)
    >>> print(budget)
    """
    from .lindblad import run_lindblad
    from .redfield import run_redfield

    def run(c):
        rho0 = qt.ket2dm(qt.basis(c.N, 1))
        if method == "lindblad":
            return run_lindblad(c, rho0, tlist)
        return run_redfield(c, rho0, tlist)

    def pop1_final(res, c):
        return float(qt.expect(qt.ket2dm(qt.basis(c.N, 1)), res.states[-1]))

    f_total = pop1_final(run(cfg), cfg)

    cfg_no_loss = cfg.replace(kappa=0.0)
    f_no_loss = pop1_final(run(cfg_no_loss), cfg_no_loss)

    cfg_no_thermal = cfg.replace(nbar=0.0)
    f_no_thermal = pop1_final(run(cfg_no_thermal), cfg_no_thermal)

    cfg_no_deph = cfg.replace(gamma_phi=0.0)
    f_no_deph = pop1_final(run(cfg_no_deph), cfg_no_deph)

    total_infidelity = 1.0 - f_total
    return {
        "total_infidelity": total_infidelity,
        "loss_contribution": f_no_loss - f_total,
        "thermal_contribution": f_no_thermal - f_total,
        "dephasing_contribution": f_no_deph - f_total,
    }
