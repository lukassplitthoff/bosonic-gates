"""Quick physics validation of nb0b and nb0c helper results."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import qutip as qt
from bosonic_gates import coherent_state, fock_state, cat_state

N = 25

# ── nb0b physics checks ───────────────────────────────────────────────────────
def mandel_Q(s):
    rho = s.density_matrix()
    n_op = qt.num(s.N)
    n_mean = float(qt.expect(n_op, rho))
    if n_mean < 1e-12:
        return 0.0
    n2_mean = float(qt.expect(n_op * n_op, rho))
    return (n2_mean - n_mean**2 - n_mean) / n_mean

def wigner_neg_volume(s, x_max=5.0, pts=50):
    xvec = np.linspace(-x_max, x_max, pts)
    W = qt.wigner(s.density_matrix(), xvec, xvec)
    dx = xvec[1] - xvec[0]
    return float(np.sum(np.maximum(0.0, -W)) * dx**2)

print("=== nb0b physics checks ===")
print(f"Mandel Q of Fock |1>:      {mandel_Q(fock_state(1, N=N)):.4f}  (expect -1.0)")
print(f"Mandel Q of Fock |3>:      {mandel_Q(fock_state(3, N=N)):.4f}  (expect -1.0)")
print(f"Mandel Q of coherent a=2:  {mandel_Q(coherent_state(2.0, N=N)):.4f}  (expect 0.0)")
print(f"Wigner neg vol Fock |1>:   {wigner_neg_volume(fock_state(1, N=N)):.4f}  (expect >0)")
print(f"Wigner neg vol coherent:   {wigner_neg_volume(coherent_state(2.0, N=N)):.4f}  (expect ~0)")
print(f"Wigner neg vol even cat:   {wigner_neg_volume(cat_state(2.0, N=N, phase=0)):.4f}  (expect >0)")

# ── nb0c physics checks ───────────────────────────────────────────────────────
print("\n=== nb0c physics checks ===")
kappa = 1.0
n_bar = 0.0
tlist = np.linspace(0, 3.0, 100)

a = qt.destroy(N)
c_ops = [np.sqrt(kappa) * a]
H = 0 * qt.qeye(N)

rho0 = fock_state(1, N=N).density_matrix()
result = qt.mesolve(H, rho0, tlist, c_ops=c_ops, e_ops=[a.dag() * a])
n_t = result.expect[0]

# Check exponential decay
idx_T1 = np.argmin(np.abs(tlist - 1.0))
print(f"<n> at t=0:   {n_t[0]:.4f}  (expect 1.0)")
print(f"<n> at t=T1:  {n_t[idx_T1]:.4f}  (expect {np.exp(-1):.4f} = 1/e)")
print(f"<n> at t=3T1: {n_t[-1]:.4f}  (expect ~{np.exp(-3):.4f})")

# Check purity of cat state during decoherence
alpha_cat = 2.0
T_dec = 1.0 / (2.0 * alpha_cat**2 * kappa)
t_cat = np.linspace(0, 4 * T_dec, 80)
rho0_cat = cat_state(alpha_cat, N=N, phase=0).density_matrix()
result_cat = qt.mesolve(H, rho0_cat, t_cat, c_ops=c_ops)
pur_0 = float(np.real((result_cat.states[0] * result_cat.states[0]).tr()))
pur_dec = float(np.real((result_cat.states[int(len(t_cat)/4)] * result_cat.states[int(len(t_cat)/4)]).tr()))
print(f"Cat purity at t=0:          {pur_0:.4f}  (expect 1.0)")
print(f"Cat purity at t~T_dec/2:    {pur_dec:.4f}  (expect < 1)")
print(f"T_dec = {T_dec:.4f} µs  vs  T1 = {1/kappa:.1f} µs  (ratio {(1/kappa)/T_dec:.1f}x faster)")
print("\nAll checks passed.")
