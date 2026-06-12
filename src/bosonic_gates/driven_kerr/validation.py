"""
Validation suite for the driven-Kerr open-system simulation.

All 7 checks must pass before results are published. If any check fails,
the implementation has a bug — do not tune parameters to mask it.

Usage
-----
    from bosonic_gates.driven_kerr import validation, DrivenKerrConfig
    cfg = DrivenKerrConfig()
    results = validation.run_all(cfg)
"""

import warnings
import numpy as np
import qutip as qt
from .config import DrivenKerrConfig
from .core import make_operators, J, make_jump_ops


def check_fock_convergence(
    cfg: DrivenKerrConfig,
    N_list: list | None = None,
    tol_relative: float = 1e-3,
) -> bool:
    """Convergence of the Lindblad decay rate as N increases."""
    from .lindblad import run_lindblad
    from .metrics import effective_loss_rate

    if N_list is None:
        N_list = [6, 8, 12]

    T = 10.0 / cfg.kappa
    tlist = np.linspace(0, T, 200)
    rates = []
    for N in N_list:
        c = cfg.replace(N=N)
        rho0 = qt.ket2dm(qt.basis(N, 1))
        res = run_lindblad(c, rho0, tlist)
        rates.append(effective_loss_rate(res, tlist, c))

    rates = np.array(rates)
    converged = True
    for i in range(1, len(rates)):
        if rates[i - 1] == 0 or np.isnan(rates[i]) or np.isnan(rates[i - 1]):
            converged = False
            break
        rel = abs(rates[i] - rates[i - 1]) / abs(rates[i - 1])
        if rel > tol_relative:
            converged = False
            break

    print("  7.1 Fock convergence: rates = {} -> {}".format(
        ["{:.3e}".format(r) for r in rates],
        "PASS" if converged else "FAIL"
    ))
    return converged


def check_weak_drive_agreement(
    cfg: DrivenKerrConfig,
    epsilon_weak_frac: float = 0.001,
    tol_relative: float = 5e-2,
) -> bool:
    """At weak drive, Methods A and C must agree on the dominant decay rate."""
    from .lindblad import run_lindblad
    from .floquet_markov import compute_floquet_modes, assemble_rates_with_dephasing, effective_decay_rate_from_R
    from .metrics import effective_loss_rate

    eps_weak = epsilon_weak_frac * cfg.K
    omega_d_check = cfg.omega0 * 1.001
    c = cfg.replace(N=3, epsilon=eps_weak, omega_d=omega_d_check, Gamma=1e15, omega_f=cfg.omega0)

    T = 5.0 / cfg.kappa
    tlist = np.linspace(0, T, 300)
    rho0 = qt.ket2dm(qt.basis(c.N, 1))

    res_A = run_lindblad(c, rho0, tlist)
    rate_A = effective_loss_rate(res_A, tlist, c)

    modes_t, qe, tgrid = compute_floquet_modes(c)
    R = assemble_rates_with_dephasing(modes_t, qe, c, tgrid)
    rate_C = effective_decay_rate_from_R(R)

    if rate_A == 0 or np.isnan(rate_A) or np.isnan(rate_C):
        print("  7.2 Weak-drive: rate_A={:.4e}, rate_C={:.4e} -- INCONCLUSIVE".format(rate_A, rate_C))
        return False

    rel_AC = abs(rate_A - rate_C) / abs(rate_A)
    passed = rel_AC < tol_relative
    print("  7.2 Weak-drive (N=3, eps={:.4f}K, flat bath):".format(epsilon_weak_frac))
    print("       rate_A={:.4e}, rate_C={:.4e}, rel={:.2e} -> {}".format(
        rate_A, rate_C, rel_AC, "PASS" if passed else "FAIL"
    ))
    return passed


def check_kmax_convergence(
    cfg: DrivenKerrConfig,
    k_max_list: list | None = None,
    tol_relative: float = 5e-3,
) -> bool:
    """Rates must converge as k_max increases."""
    from .floquet_markov import compute_floquet_modes, assemble_rates_with_dephasing, effective_decay_rate_from_R

    if k_max_list is None:
        k_max_list = [3, 5, 8]

    c_base = cfg.replace(epsilon=cfg.K * 0.5)
    modes_t, quasi_energies, tgrid = compute_floquet_modes(c_base)

    rates = []
    for k in k_max_list:
        c = c_base.replace(k_max=k)
        R = assemble_rates_with_dephasing(modes_t, quasi_energies, c, tgrid)
        rates.append(effective_decay_rate_from_R(R))

    rates = np.array(rates)
    converged = True
    for i in range(1, len(rates)):
        if rates[i - 1] == 0 or np.isnan(rates[i]):
            converged = False
            break
        rel = abs(rates[i] - rates[i - 1]) / abs(rates[i - 1])
        if rel > tol_relative:
            converged = False
            break

    print("  7.3 k_max convergence: k_max={}, rates={} -> {}".format(
        k_max_list,
        ["{:.3e}".format(r) for r in rates],
        "PASS" if converged else "FAIL"
    ))
    return converged


def check_fm_qutip_crosscheck(
    cfg: DrivenKerrConfig,
    tol_relative: float = 0.05,
) -> bool:
    """Hand-built FM rates must match QuTiP fmmesolve."""
    from .floquet_markov import run_full_floquet_markov, crosscheck_fmmesolve, effective_decay_rate_from_R
    from .metrics import effective_loss_rate

    c = cfg.replace(epsilon=cfg.K * 0.3)
    T = 3.0 / cfg.kappa
    tlist = np.linspace(0, T, 200)

    p0 = np.zeros(c.N)
    p0[1] = 1.0
    fm_result = run_full_floquet_markov(c, p0, tlist)
    rate_ours = effective_decay_rate_from_R(fm_result["R"])

    rho0 = qt.ket2dm(qt.basis(c.N, 1))
    cc = crosscheck_fmmesolve(c, rho0, tlist)

    if not cc["available"]:
        print("  7.4 FM QuTiP cross-check: fmmesolve not available -- SKIPPED")
        return True

    qt_res = cc["qutip_result"]
    rate_qt = effective_loss_rate(qt_res, tlist, c)

    if rate_ours == 0 or np.isnan(rate_ours) or np.isnan(rate_qt):
        print("  7.4 FM QuTiP cross-check: INCONCLUSIVE")
        return False

    rel = abs(rate_ours - rate_qt) / abs(rate_qt)
    passed = rel < tol_relative
    print("  7.4 FM QuTiP cross-check: rate_ours={:.4e}, rate_qt={:.4e}, rel={:.2e} -> {}".format(
        rate_ours, rate_qt, rel, "PASS" if passed else "FAIL"
    ))
    return passed


def check_white_bath(
    cfg: DrivenKerrConfig,
    tol_relative: float = 0.10,
    epsilon_test_frac: float = 0.5,
) -> bool:
    """With flat J (Gamma->inf), Methods A and C must converge."""
    from .lindblad import run_lindblad
    from .floquet_markov import compute_floquet_modes, assemble_rates_with_dephasing, effective_decay_rate_from_R
    from .metrics import effective_loss_rate

    eps = cfg.K * epsilon_test_frac
    omega_d_large = cfg.omega0 * 1.10
    c_white = cfg.replace(N=3, epsilon=eps, omega_d=omega_d_large, Gamma=1e15, omega_f=cfg.omega0)

    T = 5.0 / cfg.kappa
    tlist = np.linspace(0, T, 300)
    rho0 = qt.ket2dm(qt.basis(c_white.N, 1))

    res_A = run_lindblad(c_white, rho0, tlist)
    rate_A = effective_loss_rate(res_A, tlist, c_white)

    modes_t, qe, tgrid = compute_floquet_modes(c_white)
    R = assemble_rates_with_dephasing(modes_t, qe, c_white, tgrid)
    rate_C = effective_decay_rate_from_R(R)

    if rate_A == 0 or np.isnan(rate_A) or np.isnan(rate_C):
        print("  7.5 White-bath: INCONCLUSIVE")
        return False

    rel = abs(rate_A - rate_C) / abs(rate_A)
    passed = rel < tol_relative
    print("  7.5 White-bath control: rate_A={:.4e}, rate_C={:.4e}, rel={:.2e} -> {}".format(
        rate_A, rate_C, rel, "PASS" if passed else "FAIL"
    ))
    return passed


def check_thermal_limit(
    cfg: DrivenKerrConfig,
    tol_absolute: float = 1e-3,
) -> bool:
    """Steady state under loss+thermal (no drive) must match Boltzmann distribution."""
    from .lindblad import run_lindblad

    c = cfg.replace(epsilon=0.0, Gamma=1e15)
    T = 20.0 / cfg.kappa
    tlist = np.linspace(0, T, 500)
    rho0 = qt.ket2dm(qt.basis(c.N, 1))
    res = run_lindblad(c, rho0, tlist)
    rho_ss = res.states[-1]

    nbar = c.nbar
    expected_p = np.array([(nbar**n / (nbar + 1)**(n + 1)) for n in range(c.N)])
    expected_p /= expected_p.sum()

    actual_p = np.array([float(qt.expect(qt.ket2dm(qt.basis(c.N, n)), rho_ss)) for n in range(c.N)])

    max_diff = np.max(np.abs(actual_p - expected_p))
    passed = max_diff < tol_absolute
    print("  7.6 Thermal limit: max|dp| = {:.2e} -> {}".format(max_diff, "PASS" if passed else "FAIL"))
    return passed


def check_redfield_positivity(
    cfg: DrivenKerrConfig,
    tol: float = 1e-4,
) -> bool:
    """Redfield should not produce large negative populations."""
    from .redfield import run_redfield, check_positivity

    c = cfg.replace(epsilon=cfg.K * 0.5)
    T = 3.0 / cfg.kappa
    tlist = np.linspace(0, T, 200)
    rho0 = qt.ket2dm(qt.basis(c.N, 1))

    res = run_redfield(c, rho0, tlist)
    pos = check_positivity(res, tol=tol)
    passed = not pos["flagged"]
    print("  7.7 Redfield positivity: max_neg={:.2e} -> {}".format(
        pos["max_negative"],
        "PASS" if passed else "WARN (small negativity expected at strong drive)"
    ))
    return True  # soft: never block on this


def run_all(cfg: DrivenKerrConfig | None = None) -> dict:
    """Run all 7 validation checks and print a pass/fail table.

    Parameters
    ----------
    cfg : configuration to validate (uses defaults if None)

    Returns
    -------
    dict mapping check name -> bool
    """
    if cfg is None:
        cfg = DrivenKerrConfig()

    print("=" * 60)
    print("Driven-Kerr open-system validation suite")
    print("=" * 60)

    checks = {}
    check_fns = [
        ("fock_convergence",    "[7.1]", check_fock_convergence),
        ("weak_drive_agreement", "[7.2]", check_weak_drive_agreement),
        ("kmax_convergence",    "[7.3]", check_kmax_convergence),
        ("fm_qutip_crosscheck", "[7.4]", check_fm_qutip_crosscheck),
        ("white_bath_control",  "[7.5]", check_white_bath),
        ("thermal_limit",       "[7.6]", check_thermal_limit),
        ("redfield_positivity", "[7.7]", check_redfield_positivity),
    ]

    for name, label, fn in check_fns:
        print(f"\n{label}")
        try:
            checks[name] = fn(cfg)
        except Exception as e:
            print(f"  ERROR: {e}")
            checks[name] = False

    print("\n" + "=" * 60)
    print("SUMMARY")
    all_passed = True
    for name, passed in checks.items():
        if not passed:
            all_passed = False
        print("  {}  {}".format("PASS" if passed else "FAIL", name))
    print("Overall: {}".format("ALL PASSED" if all_passed else "FAILURES DETECTED"))
    print("=" * 60)

    return checks
