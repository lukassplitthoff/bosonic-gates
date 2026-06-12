"""
Method A — Standard (secular) Lindblad in the undriven basis.

Jump operators and rates are fixed once from the bare transition frequency ω₀.
The drive enters only the coherent part H(t); the dissipator is static.
Implemented via qt.mesolve with a time-dependent Hamiltonian.
"""

import numpy as np
import qutip as qt
from .config import DrivenKerrConfig
from .core import make_H_drive_td, make_jump_ops


def run_lindblad(
    cfg: DrivenKerrConfig,
    rho0: qt.Qobj,
    tlist: np.ndarray,
    e_ops: list | None = None,
    options: dict | None = None,
) -> qt.solver.Result:
    """Evolve rho0 under the driven Kerr Hamiltonian using standard Lindblad.

    Parameters
    ----------
    cfg:     system configuration
    rho0:    initial density matrix (N×N qt.Qobj)
    tlist:   array of times at which to save the state
    e_ops:   list of observables to evaluate at each time step
    options: passed to qt.mesolve (e.g. {"nsteps": 10000, "rtol": 1e-8})

    Returns
    -------
    QuTiP Result object with .states or .expect

    Example
    -------
    >>> from bosonic_gates.driven_kerr import DrivenKerrConfig, run_lindblad
    >>> import qutip as qt, numpy as np
    >>> cfg = DrivenKerrConfig(epsilon=2*np.pi*0.05e9)
    >>> rho0 = qt.ket2dm(qt.basis(cfg.N, 1))
    >>> tlist = np.linspace(0, 5e-6, 200)
    >>> result = run_lindblad(cfg, rho0, tlist)
    """
    H = make_H_drive_td(cfg)
    c_ops = make_jump_ops(cfg)
    opts = options or {}
    return qt.mesolve(H, rho0, tlist, c_ops=c_ops, e_ops=e_ops or [], options=opts)
