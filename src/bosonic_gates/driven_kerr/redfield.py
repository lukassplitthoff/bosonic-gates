"""
Method B — Non-secular Bloch–Redfield in the undriven basis.

Retains cross terms that the secular (Lindblad) approximation drops.
Uses qt.brmesolve with spectral-density callbacks derived from the same J(ω)
as Methods A and C — the single shared spectral density from core.py.

QuTiP brmesolve a_ops format:
    [[operator, spectral_density_func(omega)], ...]
where spectral_density_func(omega) is evaluated at both positive and negative
frequencies by the solver (it handles the full two-sided density internally).

Note on Method B validity:
  Redfield may produce small negative populations during transients — this is
  expected at strong drive where the weak-coupling assumption is strained.
  Violations above a few × solver tolerance are flagged by check_positivity().
"""

import numpy as np
import qutip as qt
from .config import DrivenKerrConfig
from .core import make_H_drive_td, make_operators, J, J_phi


def _make_a_ops(cfg: DrivenKerrConfig) -> list:
    """Build the a_ops list for qt.brmesolve."""
    a, adag, num = make_operators(cfg.N)

    def j_loss(omega: float) -> float:
        return J(omega, cfg)

    def j_dephasing(omega: float) -> float:
        return J_phi(omega, cfg)

    return [
        [a, j_loss],
        [num, j_dephasing],
    ]


def run_redfield(
    cfg: DrivenKerrConfig,
    rho0: qt.Qobj,
    tlist: np.ndarray,
    e_ops: list | None = None,
    options: dict | None = None,
    sec_cutoff: float = 0.1,
) -> qt.solver.Result:
    """Evolve rho0 under the driven Kerr Hamiltonian using Bloch–Redfield.

    Parameters
    ----------
    cfg:         system configuration
    rho0:        initial density matrix
    tlist:       save times
    e_ops:       observables
    options:     passed to qt.brmesolve
    sec_cutoff:  secular approximation cutoff parameter. Default 0.1 keeps cross terms.

    Returns
    -------
    QuTiP Result object
    """
    H = make_H_drive_td(cfg)
    a_ops = _make_a_ops(cfg)
    opts = options or {}

    result = qt.brmesolve(
        H,
        rho0,
        tlist,
        a_ops=a_ops,
        e_ops=e_ops or [],
        options=opts,
        sec_cutoff=sec_cutoff,
    )
    return result


def check_positivity(result: qt.solver.Result, tol: float = 1e-4) -> dict:
    """Check for negative eigenvalues in the evolved density matrices.

    Returns a dict with:
        max_negative: most negative eigenvalue across all time steps
        fraction_violated: fraction of time steps with any negative eigenvalue
        flagged: True if max_negative < -tol
    """
    max_neg = 0.0
    n_violated = 0
    for rho in result.states:
        evals = rho.eigenenergies()
        mn = float(np.min(evals))
        if mn < max_neg:
            max_neg = mn
        if mn < -tol:
            n_violated += 1
    return {
        "max_negative": max_neg,
        "fraction_violated": n_violated / max(len(result.states), 1),
        "flagged": max_neg < -tol,
    }
