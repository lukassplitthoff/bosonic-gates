"""
Method C — Floquet–Markov master equation (hand-built rate assembly).

Overview of the four-step procedure:
  1. compute_floquet_modes: integrate one-period propagator U(T_d, 0),
     diagonalize to get quasi-energies ε_m and modes |φ_m(t)⟩.
  2. compute_fourier_components: FFT of ⟨φ_m(t)|S|φ_n(t)⟩ → S_mn^(k).
  3. assemble_rates: Γ_mn = Σ_k |S_mn^(k)|² · J(Δ_mn + k·ω_d).
  4. run_floquet_markov: solve dp/dt = R·p (Pauli/rate equation).

Thermal convention (same as core.J):
    Γ_mn = Σ_k |S_mn^(k)|² · J(Δ_mn + k·ω_d, cfg)
where J is the two-sided density from core.py (absorbs thermal factors).
Detailed balance is guaranteed by J(−ω)/J(ω) = nbar/(1+nbar).
"""

import warnings
import numpy as np
import scipy.linalg
import scipy.integrate
import qutip as qt
from .config import DrivenKerrConfig
from .core import make_H_drive_td, make_operators, J, J_vectorized


# ---------------------------------------------------------------------------
# Step 1: Floquet modes
# ---------------------------------------------------------------------------

def compute_floquet_modes(
    cfg: DrivenKerrConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Floquet quasi-energies and modes over one drive period.

    Returns
    -------
    modes_t : ndarray, shape (N, n_t, N)
        modes_t[m, i, :] is the state vector of Floquet mode m at time t[i].
    quasi_energies : ndarray, shape (N,)
        Quasi-energies ε_m defined mod ω_d.
    tgrid : ndarray, shape (n_t,)
        Time grid over one period [0, T_d].
    """
    N = cfg.N
    n_t = cfg.n_t
    H = make_H_drive_td(cfg)
    tgrid = np.linspace(0.0, cfg.T_d, n_t, endpoint=False)

    # Build the one-period propagator column by column
    U_mat = np.zeros((N, N), dtype=complex)
    for j in range(N):
        psi0 = qt.basis(N, j)
        result = qt.sesolve(H, psi0, [0.0, cfg.T_d], options={"nsteps": 50000, "rtol": 1e-10, "atol": 1e-12})
        psi_T = result.states[-1].full().ravel()
        U_mat[:, j] = psi_T

    # Diagonalize: U_mat @ v = exp(-i ε_m T_d) v
    evals, evecs = np.linalg.eig(U_mat)

    # Extract quasi-energies from eigenvalues exp(-i ε_m T_d)
    quasi_energies = -np.angle(evals) / cfg.T_d

    order = np.argsort(quasi_energies)
    quasi_energies = quasi_energies[order]
    evecs = evecs[:, order]

    # Propagate each Floquet mode over one full period
    modes_t = np.zeros((N, n_t, N), dtype=complex)
    for m in range(N):
        psi0_m = qt.Qobj(evecs[:, m])
        result = qt.sesolve(H, psi0_m, tgrid, options={"nsteps": 50000, "rtol": 1e-10, "atol": 1e-12})
        for i, state in enumerate(result.states):
            modes_t[m, i, :] = state.full().ravel()

    return modes_t, quasi_energies, tgrid


# ---------------------------------------------------------------------------
# Step 2: Fourier components
# ---------------------------------------------------------------------------

def compute_fourier_components(
    modes_t: np.ndarray,
    op_matrix: np.ndarray,
    cfg: DrivenKerrConfig,
    tgrid: np.ndarray,
) -> dict[int, np.ndarray]:
    """Compute S_mn^(k) = (1/T_d) ∫₀^{T_d} e^{−i k ω_d t} ⟨φ_m(t)|S|φ_n(t)⟩ dt.

    Uses the FFT along the time axis for efficiency.

    Parameters
    ----------
    modes_t  : shape (N, n_t, N) — Floquet modes over one period
    op_matrix: shape (N, N) dense complex array — the coupling operator S
    cfg      : configuration
    tgrid    : shape (n_t,) time grid

    Returns
    -------
    dict mapping integer k (−k_max … +k_max) → complex matrix (N×N)
    """
    N_modes, n_t, N_hilbert = modes_t.shape

    Smn_t = np.einsum(
        "mia,ab,nib->imn",
        modes_t.conj(),
        op_matrix,
        modes_t,
        optimize=True,
    )

    Smn_fft = np.fft.fft(Smn_t, axis=0) / n_t

    result = {}
    for k in range(-cfg.k_max, cfg.k_max + 1):
        bin_idx = k % n_t
        result[k] = Smn_fft[bin_idx]
    return result


# ---------------------------------------------------------------------------
# Step 3: Rate assembly
# ---------------------------------------------------------------------------

def assemble_rates(
    fourier_components_list: list[dict[int, np.ndarray]],
    quasi_energies: np.ndarray,
    cfg: DrivenKerrConfig,
) -> np.ndarray:
    """Build the Pauli rate matrix R for the Floquet–Markov equation.

    Γ_mn = Σ_{ops} Σ_k |S_mn^(k)|² · J(Δ_mn + k·ω_d, cfg)
    """
    N = len(quasi_energies)
    Gamma = np.zeros((N, N))
    Delta = quasi_energies[:, None] - quasi_energies[None, :]

    for S_k in fourier_components_list:
        for k, Smn_k in S_k.items():
            freqs = Delta - k * cfg.omega_d
            J_vals = J_vectorized(freqs.ravel(), cfg).reshape(N, N)
            Gamma += np.abs(Smn_k)**2 * J_vals

    R = Gamma.copy()
    np.fill_diagonal(R, 0.0)
    np.fill_diagonal(R, -np.sum(R, axis=0))
    return R


def assemble_rates_with_dephasing(
    modes_t: np.ndarray,
    quasi_energies: np.ndarray,
    cfg: DrivenKerrConfig,
    tgrid: np.ndarray,
) -> np.ndarray:
    """Convenience wrapper: assembles rates for loss and dephasing channels.

    Loss channel: coupling operator a (annihilation).
    Dephasing channel: coupling operator n = a†a, spectral density = γ_φ (flat).
    """
    a, adag, num_op = make_operators(cfg.N)
    a_mat = a.full()
    n_mat = num_op.full()

    S_k_loss = compute_fourier_components(modes_t, a_mat, cfg, tgrid)
    S_k_deph = compute_fourier_components(modes_t, n_mat, cfg, tgrid)

    N = len(quasi_energies)
    Delta = quasi_energies[:, None] - quasi_energies[None, :]
    Gamma = np.zeros((N, N))

    for k, Smn_k in S_k_loss.items():
        freqs = Delta - k * cfg.omega_d
        J_vals = J_vectorized(freqs.ravel(), cfg).reshape(N, N)
        Gamma += np.abs(Smn_k)**2 * J_vals

    for k, Smn_k in S_k_deph.items():
        Gamma += np.abs(Smn_k)**2 * cfg.gamma_phi

    R = Gamma.copy()
    np.fill_diagonal(R, 0.0)
    np.fill_diagonal(R, -np.sum(R, axis=0))
    return R


# ---------------------------------------------------------------------------
# Step 4: Run the Floquet–Markov rate equation
# ---------------------------------------------------------------------------

def run_floquet_markov(
    cfg: DrivenKerrConfig,
    p0: np.ndarray,
    tlist: np.ndarray,
    R: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve dp/dt = R·p in the Floquet basis.

    Returns
    -------
    p_t           : populations over time, shape (len(tlist), N)
    R             : rate matrix used
    quasi_energies: quasi-energies from the Floquet decomposition
    """
    if R is None:
        modes_t, quasi_energies, tgrid = compute_floquet_modes(cfg)
        R = assemble_rates_with_dephasing(modes_t, quasi_energies, cfg, tgrid)
    else:
        quasi_energies = None

    def rhs(t, p):
        return R @ p

    sol = scipy.integrate.solve_ivp(
        rhs,
        (tlist[0], tlist[-1]),
        p0,
        t_eval=tlist,
        method="RK45",
        rtol=1e-10,
        atol=1e-12,
    )
    p_t = sol.y.T
    return p_t, R, quasi_energies


def floquet_steady_state(R: np.ndarray) -> np.ndarray:
    """Compute the steady-state population vector (null vector of R)."""
    N = R.shape[0]
    A = R.T.copy()
    A[-1, :] = 1.0
    b = np.zeros(N)
    b[-1] = 1.0
    p_ss = np.linalg.solve(A, b)
    p_ss = np.abs(p_ss)
    p_ss /= p_ss.sum()
    return p_ss


# ---------------------------------------------------------------------------
# Full pipeline: compute modes + rates + evolve
# ---------------------------------------------------------------------------

def run_full_floquet_markov(
    cfg: DrivenKerrConfig,
    p0: np.ndarray,
    tlist: np.ndarray,
) -> dict:
    """Run the complete Floquet–Markov pipeline.

    Returns a dict with keys:
        p_t, R, quasi_energies, modes_t, tgrid
    """
    modes_t, quasi_energies, tgrid = compute_floquet_modes(cfg)
    R = assemble_rates_with_dephasing(modes_t, quasi_energies, cfg, tgrid)
    p_t, _, _ = run_floquet_markov(cfg, p0, tlist, R=R)
    return {
        "p_t": p_t,
        "R": R,
        "quasi_energies": quasi_energies,
        "modes_t": modes_t,
        "tgrid": tgrid,
    }


# ---------------------------------------------------------------------------
# Cross-check against QuTiP fmmesolve
# ---------------------------------------------------------------------------

def crosscheck_fmmesolve(
    cfg: DrivenKerrConfig,
    rho0: qt.Qobj,
    tlist: np.ndarray,
    e_ops: list | None = None,
) -> dict:
    """Cross-check Method C against QuTiP's built-in fmmesolve.

    Returns a dict with 'available' key indicating whether the
    cross-check could be performed.
    """
    H = make_H_drive_td(cfg)
    a, adag, num_op = make_operators(cfg.N)

    result = {"available": False, "qutip_result": None}

    try:
        from qutip import fmmesolve, FloquetBasis

        def j_loss(omega):
            return J(omega, cfg)

        def j_dephasing(omega):
            return cfg.gamma_phi

        fbasis = FloquetBasis(H, cfg.T_d, options={"nsteps": 50000})
        qt_result = fmmesolve(
            H,
            rho0,
            tlist,
            T=cfg.T_d,
            a_ops=[[a, j_loss], [num_op, j_dephasing]],
            e_ops=e_ops or [],
            options={"nsteps": 50000},
        )
        result["available"] = True
        result["qutip_result"] = qt_result
        result["floquet_basis"] = fbasis
    except ImportError:
        pass
    except Exception as exc:
        warnings.warn(f"QuTiP fmmesolve cross-check failed: {exc}")

    if not result["available"]:
        try:
            qt_result = qt.fmmesolve(
                H,
                rho0,
                tlist,
                c_ops=[],
                a_ops=[[a, lambda w: J(w, cfg)]],
                e_ops=e_ops or [],
                T=cfg.T_d,
                options=qt.Options(nsteps=50000),
            )
            result["available"] = True
            result["qutip_result"] = qt_result
        except Exception as exc:
            warnings.warn(f"QuTiP 4.x fmmesolve also failed: {exc}")

    return result


def effective_decay_rate_from_R(R: np.ndarray) -> float:
    """Extract the dominant relaxation rate from the rate matrix."""
    evals = np.linalg.eigvals(R)
    evals_real = np.real(evals)
    nonzero = evals_real[np.abs(evals_real) > 1e-15 * np.abs(evals_real).max()]
    if len(nonzero) == 0:
        return 0.0
    return float(np.min(np.abs(nonzero)))
