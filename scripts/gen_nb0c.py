"""Generate nb0c_open_systems.ipynb."""
import json, os, uuid

def _id(): return uuid.uuid4().hex[:8]

def code_cell(source, tags=None):
    meta = {"tags": tags} if tags else {}
    return {"cell_type": "code", "id": _id(), "execution_count": None,
            "metadata": meta, "outputs": [], "source": source if isinstance(source, list) else [source]}

def md_cell(source):
    return {"cell_type": "markdown", "id": _id(), "metadata": {},
            "source": source if isinstance(source, list) else [source]}

def L(s):
    lines = s.split("\n")
    return [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])

CELLS = []

# ── Cell 0  parameters ────────────────────────────────────────────────────────
CELLS.append(code_cell(L(r"""# parameters
# BINDER_FAST: set N=12, n_times=80, wigner_pts=40 for fast cloud execution
N = 25            # Hilbert space truncation
kappa = 1.0       # photon loss rate  (µs⁻¹);  T1 = 1/kappa = 1 µs
n_bar_bath = 0.02 # residual thermal occupation of the bath
gamma_phi = 0.0   # pure dephasing rate (µs⁻¹)
alpha_cat = 2.0   # amplitude of the cat-state demonstration
t_max = 5.0       # simulation end time (µs = T1 × 5)
n_times = 200     # time steps
wigner_pts = 60   # grid resolution for Wigner snapshots"""), tags=["parameters"]))

# ── Cell 1  imports ───────────────────────────────────────────────────────────
CELLS.append(code_cell(L(r"""# hide
import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
%matplotlib widget

from bosonic_gates import (
    BosonicState, coherent_state, fock_state, cat_state,
)"""), tags=["hide"]))

# ── Cell 2  helpers ───────────────────────────────────────────────────────────
CELLS.append(code_cell(L(r"""# hide
def build_c_ops(N_, kappa_, n_bar_, gamma_phi_):
    '''Return Lindblad collapse operators for a harmonic cavity.'''
    a = qt.destroy(N_)
    c_ops = [np.sqrt(kappa_ * (1.0 + n_bar_)) * a]        # photon loss
    if n_bar_ > 0:
        c_ops.append(np.sqrt(kappa_ * n_bar_) * a.dag())  # thermal gain
    if gamma_phi_ > 0:
        c_ops.append(np.sqrt(gamma_phi_) * a.dag() * a)   # pure dephasing
    return c_ops

def run_cavity(rho0, tlist, kappa_=None, n_bar_=None, gamma_phi_=None, e_ops=None):
    '''Run qt.mesolve for a bare cavity (H=0 in rotating frame).'''
    kappa_   = kappa_   if kappa_   is not None else kappa
    n_bar_   = n_bar_   if n_bar_   is not None else n_bar_bath
    gamma_phi_ = gamma_phi_ if gamma_phi_ is not None else gamma_phi
    H = 0 * qt.qeye(N)
    c_ops = build_c_ops(N, kappa_, n_bar_, gamma_phi_)
    return qt.mesolve(H, rho0, tlist, c_ops=c_ops, e_ops=e_ops or [])

def wigner_snaps(result, tlist, snap_times, x_max_=4.5, pts=None, title=""):
    '''Plot Wigner function at a set of snapshot times.'''
    pts = pts or wigner_pts
    xvec = np.linspace(-x_max_, x_max_, pts)
    nt = len(snap_times)
    fig, axes = plt.subplots(1, nt, figsize=(3.5 * nt, 3.8))
    if nt == 1:
        axes = [axes]
    for ax, t_snap in zip(axes, snap_times):
        idx = int(np.argmin(np.abs(tlist - t_snap)))
        rho = result.states[idx]
        W = qt.wigner(rho, xvec, xvec)
        vmax = max(float(np.max(np.abs(W))), 1e-9)
        ax.contourf(xvec, xvec, W, levels=40, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(rf"$t = {tlist[idx]:.2f}\,\mu s$", fontsize=10)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$X$"); ax.set_ylabel(r"$P$")
    if title:
        fig.suptitle(title, fontsize=11)
    plt.tight_layout(); plt.show()

def purity_traj(result):
    return [float(np.real((rho * rho).tr())) for rho in result.states]

def entropy_traj(result):
    return [float(qt.entropy_vn(rho, base=np.e)) for rho in result.states]"""), tags=["hide"]))

# ── Cell 3  title ─────────────────────────────────────────────────────────────
CELLS.append(md_cell(L(r"""## Module 0c: Open Quantum Systems and the Lindblad Master Equation

**Learning objectives:**
- Write the Lindblad master equation and identify the physical meaning of each jump operator
- Predict how photon loss rate κ, pure dephasing rate γ_φ, and thermal occupation n̄ affect T₁ and T₂
- Describe qualitatively how the Wigner function of a Fock state and a cat state evolve under photon loss
- Define purity and von Neumann entropy, and connect their decay to decoherence

---

A real superconducting cavity is never isolated.  It couples to 50 Ω transmission-line modes, phonons in the substrate, and quasiparticles in nearby junctions.  The effective description of the cavity alone — after tracing out all those environmental degrees of freedom — is the **Lindblad master equation**.

**Sections:** [1 Lindblad equation](#sec1) · [2 T₁ decay](#sec2) · [3 Fock decoherence](#sec3) · [4 Cat decoherence](#sec4) · [5 Decoherence rate vs cat size](#sec5) · [6 Thermal equilibration](#sec6) · [7 Entropy](#sec7)""")))

# ── Cell 4  Section 1 markdown ────────────────────────────────────────────────
CELLS.append(md_cell(L(r"""<a id="sec1"></a>
## 1  The Lindblad Master Equation

Under the Born-Markov approximation (bath correlation time much shorter than system dynamics), the reduced density matrix of the cavity obeys:

$$\frac{d\rho}{dt} = -\frac{i}{\hbar}[\hat{H},\rho] + \sum_k \left( \hat{L}_k\rho\hat{L}_k^\dagger - \frac{1}{2}\{\hat{L}_k^\dagger\hat{L}_k,\,\rho\} \right)$$

The $\hat{L}_k$ are **jump operators** encoding the different dissipation channels.

| Channel | Jump operator $\hat{L}_k$ | Rate | Timescale |
|---|---|---|---|
| Photon loss | $\sqrt{\kappa(1+\bar{n})}\,\hat{a}$ | $\kappa(1+\bar{n})$ | $T_1 = 1/\kappa$ (at $\bar{n}\to 0$) |
| Thermal gain | $\sqrt{\kappa\bar{n}}\,\hat{a}^\dagger$ | $\kappa\bar{n}$ | same $T_1$; drives to thermal ss |
| Pure dephasing | $\sqrt{\gamma_\varphi}\,\hat{a}^\dagger\hat{a}$ | $\gamma_\varphi$ | $T_2 = (2/\kappa + \gamma_\varphi)^{-1}$ |

**Amplitude vs phase decay.**
- $T_1 = 1/\kappa$ is the **energy** (amplitude) relaxation time: $\langle\hat{n}\rangle$ decays with rate $\kappa$.
- $T_2$ is the **coherence** time: off-diagonal elements of $\rho$ decay with rate $\kappa/2 + \gamma_\varphi$.
- Even at $\gamma_\varphi = 0$: $T_2 \leq 2T_1$ (photon loss also randomises the phase).
- At millikelvin temperatures and GHz frequencies, $\bar{n} \ll 1$ and thermal gain is negligible.

**Lab values (3D Al cavity, 2016 Ofek generation).** $\kappa/2\pi \sim 3\,\text{kHz}$ ($T_1 \sim 50\,\mu\text{s}$), $\gamma_\varphi/2\pi \sim 3\,\text{kHz}$.  For this notebook we use $\kappa = 1\,\mu\text{s}^{-1}$ (i.e. $T_1 = 1\,\mu\text{s}$) to keep simulation times short.""")))

# ── Cell 5  code: show c_ops ──────────────────────────────────────────────────
CELLS.append(code_cell(L(r"""a = qt.destroy(N)
c_ops_demo = build_c_ops(N, kappa, n_bar_bath, gamma_phi)
print("Jump operators for our cavity:")
print(f"  L0 = sqrt(kappa*(1+n_bar)) * a   rate = {kappa*(1+n_bar_bath):.4f} µs⁻¹  [photon loss]")
if n_bar_bath > 0:
    print(f"  L1 = sqrt(kappa*n_bar) * a†       rate = {kappa*n_bar_bath:.4f} µs⁻¹  [thermal gain]")
if gamma_phi > 0:
    print(f"  L2 = sqrt(gamma_phi) * a†a         rate = {gamma_phi:.4f} µs⁻¹  [dephasing]")
print(f"\nT1 = 1/kappa = {1/kappa:.2f} µs   (energy relaxation)")
T2_inv = kappa / 2.0 + gamma_phi
print(f"T2 = 1/(kappa/2 + gamma_phi) = {1/T2_inv:.2f} µs   (phase coherence)")
print(f"Steady-state <n> = n_bar_bath = {n_bar_bath}")""")))

# ── Cell 6  Section 2 markdown ────────────────────────────────────────────────
CELLS.append(md_cell(L(r"""<a id="sec2"></a>
## 2  T₁ Decay: Photon Loss from a Fock State

Start in $|1\rangle$ and let it relax.  The expected evolution is:

$$\langle\hat{n}\rangle(t) = e^{-\kappa t}$$

The Wigner function starts as the $|1\rangle$ ring (one negative ring around the origin) and smoothly deforms into the vacuum disk as photons leak out.""")))

# ── Cell 7  code: <n>(t) for |1> ─────────────────────────────────────────────
CELLS.append(code_cell(L(r"""tlist = np.linspace(0, t_max, n_times)
rho0 = fock_state(1, N=N).density_matrix()
a = qt.destroy(N)
result_1 = run_cavity(rho0, tlist, e_ops=[a.dag() * a])

n_exp = result_1.expect[0]

fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(tlist, n_exp, lw=2, label=r"$\langle\hat{n}\rangle(t)$  (simulation)")
ax.plot(tlist, np.exp(-kappa * tlist), 'k--', lw=1.5,
        label=rf"$e^{{-\kappa t}}$  ($T_1 = {1/kappa:.1f}\,\mu s$)")
ax.axvline(1 / kappa, color="gray", ls=":", lw=1)
ax.text(1 / kappa + 0.05, 0.6, rf"$T_1 = {1/kappa:.1f}\,\mu s$", fontsize=10)
ax.set_xlabel(r"$t$ (µs)"); ax.set_ylabel(r"$\langle\hat{n}\rangle$")
ax.set_title(r"T₁ decay from $|1\rangle$"); ax.legend()
plt.tight_layout(); plt.show()"""))  )

# ── Cell 8  code: Wigner snapshots |1> ───────────────────────────────────────
CELLS.append(code_cell(L(r"""result_1_states = run_cavity(fock_state(1, N=N).density_matrix(), tlist)
T1 = 1.0 / kappa
snap_t = [0.0, T1 / 4, T1 / 2, T1]
wigner_snaps(result_1_states, tlist, snap_t,
             title=r"Wigner function of $|1\rangle$ under photon loss")""")))

# ── Cell 9  Section 3 markdown ────────────────────────────────────────────────
CELLS.append(md_cell(L(r"""<a id="sec3"></a>
## 3  Fock State Decoherence: $|n\rangle$ Under Photon Loss

Higher Fock states $|n\rangle$ have $n$ concentric rings in phase space.
Under photon loss, each ring successively collapses inward toward the origin.
The number of rings decreases by one each time a photon is lost — the process is stochastic,
so after a few $T_1$ the ring pattern blurs into a Gaussian blob.

The instantaneous occupation of Fock level $m$ given initial $|n\rangle$ follows a **binomial decay**:

$$p_m(t) = \binom{n}{m} (1-e^{-\kappa t})^{n-m}\,(e^{-\kappa t})^m$$

This confirms the stochastic photon-loss picture: each photon independently leaks with probability $1-e^{-\kappa t}$.""")))

# ── Cell 10  code: Wigner snapshots |3> ──────────────────────────────────────
CELLS.append(code_cell(L(r"""result_3 = run_cavity(fock_state(3, N=N).density_matrix(), tlist)
snap_t3 = [0.0, 0.5 / kappa, 1.0 / kappa, 2.0 / kappa]
wigner_snaps(result_3, tlist, snap_t3,
             title=r"Wigner function of $|3\rangle$ under photon loss")""")))

# ── Cell 11  code: binomial decay of Fock populations ────────────────────────
CELLS.append(code_cell(L(r"""from math import comb

n_init = 3
t_plot = np.linspace(0, 3.0 / kappa, 200)
survival = np.exp(-kappa * t_plot)

fig, ax = plt.subplots(figsize=(6, 3.8))
for m in range(n_init + 1):
    coeff = comb(n_init, m)
    p_m = coeff * (1 - survival) ** (n_init - m) * survival ** m
    ax.plot(t_plot * kappa, p_m, lw=2, label=rf"$|{m}\rangle$")

ax.set_xlabel(r"$\kappa t$"); ax.set_ylabel(r"$p_m(t)$")
ax.set_title(rf"Binomial decay of Fock populations from $|{n_init}\rangle$")
ax.legend(ncol=2); plt.tight_layout(); plt.show()
print("Each photon leaks independently; populations obey a binomial cascade.")""")))

# ── Cell 12  Section 4 markdown ───────────────────────────────────────────────
CELLS.append(md_cell(L(r"""<a id="sec4"></a>
## 4  Cat State Decoherence

A cat state $|C^\pm_\alpha\rangle = \mathcal{N}(|\alpha\rangle \pm |-\alpha\rangle)$ has two coherent blobs *plus* quantum interference fringes between them.
The fringes encode the quantum coherence between $|\alpha\rangle$ and $|-\alpha\rangle$.

Under photon loss, the coherence between the two blobs decays exponentially:

$$\rho_{++}(t) \propto e^{-\Gamma_{\rm dec}\,t}, \quad \Gamma_{\rm dec} = 2|\alpha|^2\kappa$$

This is **much faster than $T_1$** for large cats: $T_{\rm dec} = 1/(2|\alpha|^2\kappa) \ll T_1 = 1/\kappa$.

**Physical picture.** A single lost photon carries which-path information: it was emitted from $|\alpha\rangle$ or $|-\alpha\rangle$ with different amplitudes (since $\langle\alpha|\hat{a}|-\alpha\rangle = 0$). This single measurement by the environment is enough to destroy the superposition. The decoherence rate grows as $|\alpha|^2$ because larger separations make the two blobs more distinguishable.

Crucially, **the amplitude (energy) of the blobs** decays only at rate $\kappa$ — the cat loses its interference fringes long before it loses its energy.""")))

# ── Cell 13  code: cat Wigner snapshots ───────────────────────────────────────
CELLS.append(code_cell(L(r"""T_dec = 1.0 / (2.0 * alpha_cat**2 * kappa)
print(f"Cat amplitude: |alpha| = {alpha_cat}")
print(f"T1 = {1/kappa:.2f} µs   T_dec = {T_dec:.3f} µs   ratio T1/T_dec = {1/kappa/T_dec:.1f}")

# Use fine-grained time list to resolve fast decoherence
t_cat = np.linspace(0, min(t_max, 5 * T_dec), n_times)
rho0_cat = cat_state(alpha_cat, N=N, phase=0).density_matrix()
result_cat = run_cavity(rho0_cat, t_cat)

snap_cat = [0.0, T_dec / 4, T_dec / 2, T_dec, 2 * T_dec]
wigner_snaps(result_cat, t_cat, snap_cat,
             title=rf"Even cat $|\alpha|={alpha_cat}$: fringes die at $T_{{\rm dec}}={T_dec:.2f}\,\mu s$")"""))  )

# ── Cell 14  code: purity vs time ─────────────────────────────────────────────
CELLS.append(code_cell(L(r"""pur_cat = purity_traj(result_cat)

fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(t_cat, pur_cat, lw=2, label="purity $\\mathrm{Tr}(\\rho^2)$")
ax.axvline(T_dec, color="crimson", ls="--", lw=1.5,
           label=rf"$T_{{\rm dec}} = {T_dec:.3f}\,\mu s$")
ax.axvline(1 / kappa, color="steelblue", ls="--", lw=1.5,
           label=rf"$T_1 = {1/kappa:.1f}\,\mu s$")
ax.set_xlabel(r"$t$ (µs)"); ax.set_ylabel(r"purity $\mathrm{Tr}(\rho^2)$")
ax.set_title(rf"Purity decay for even cat $|\alpha|={alpha_cat}$")
ax.legend(); plt.tight_layout(); plt.show()

print(f"Purity at t=0:       {pur_cat[0]:.4f}  (pure state)")
print(f"Purity at t=T_dec:   {pur_cat[np.argmin(np.abs(t_cat - T_dec))]:.4f}")
print(f"Purity at t=T1:      {pur_cat[np.argmin(np.abs(t_cat - 1/kappa))]:.4f}")""")))

# ── Cell 15  Section 5 markdown ───────────────────────────────────────────────
CELLS.append(md_cell(L(r"""<a id="sec5"></a>
## 5  Decoherence Rate vs Cat Size

The key formula $\Gamma_{\rm dec} = 2|\alpha|^2\kappa$ has a crucial implication:
*larger cats decohere faster*.
This is the central trade-off in cat-qubit error correction: bigger $|\alpha|$ gives better bit-flip protection (the blobs are further apart) but worse phase-flip protection (fringes die faster).

Below we compare the purity decay for three cat sizes.""")))

# ── Cell 16  code: purity decay vs alpha ──────────────────────────────────────
CELLS.append(code_cell(L(r"""alpha_vals = [1.0, 1.5, 2.0, alpha_cat]
colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(alpha_vals)))

t_compare = np.linspace(0, 3.0 / kappa, n_times)

fig, ax = plt.subplots(figsize=(7, 4))
for alpha_c, col in zip(alpha_vals, colors):
    rho0_c = cat_state(alpha_c, N=N, phase=0).density_matrix()
    res_c = run_cavity(rho0_c, t_compare)
    pur_c = purity_traj(res_c)
    Td = 1.0 / (2.0 * alpha_c**2 * kappa)
    ax.plot(t_compare * kappa, pur_c, lw=2, color=col,
            label=rf"$|\alpha|={alpha_c}$, $T_{{\rm dec}}/T_1 = {Td*kappa:.3f}$")

ax.set_xlabel(r"$\kappa t$ (units of $T_1$)")
ax.set_ylabel(r"purity $\mathrm{Tr}(\rho^2)$")
ax.set_title(r"Faster decoherence for larger cat states")
ax.legend(fontsize=9); plt.tight_layout(); plt.show()

print("Summary: T_dec = 1/(2|alpha|^2 kappa)")
for alpha_c in alpha_vals:
    print(f"  |alpha|={alpha_c:.1f}: T_dec/T1 = {1/(2*alpha_c**2):.4f}")""")))

# ── Cell 17  Section 6 markdown ───────────────────────────────────────────────
CELLS.append(md_cell(L(r"""<a id="sec6"></a>
## 6  Thermal Equilibration

When the bath has a residual thermal occupation $\bar{n} > 0$, the cavity no longer relaxes to $|0\rangle$.
The jump operators $\sqrt{\kappa(1+\bar{n})}\,\hat{a}$ (loss) and $\sqrt{\kappa\bar{n}}\,\hat{a}^\dagger$ (gain) balance at:

$$\rho_{\rm ss} = \sum_{n=0}^\infty \frac{\bar{n}^n}{(\bar{n}+1)^{n+1}} |n\rangle\langle n| \quad \text{(thermal state)}$$

In the lab, this is why even a pristine cavity at 15 mK shows a small thermal tail — $\bar{n} = 1/(e^{\hbar\omega/kT}-1) \approx 10^{-7}$ at 5 GHz, but spurious microwave modes, substrate phonons, and microwave leakage can contribute residual occupation at the $10^{-3}$–$10^{-2}$ level.

Here we inflate $\bar{n}$ to 0.2 to make the thermal equilibration clearly visible.""")))

# ── Cell 18  code: thermal equilibration ─────────────────────────────────────
CELLS.append(code_cell(L(r"""n_bar_demo = 0.2   # inflated for visibility
t_therm = np.linspace(0, 4.0 / kappa, n_times)

# Start from vacuum |0>
rho0_vac = fock_state(0, N=N).density_matrix()
res_therm = run_cavity(rho0_vac, t_therm,
                       kappa_=kappa, n_bar_=n_bar_demo, gamma_phi_=0.0,
                       e_ops=[qt.num(N)])

n_thermal_path = res_therm.expect[0]
n_ss = n_bar_demo   # exact steady-state mean photon number

fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(t_therm * kappa, n_thermal_path, lw=2, label=r"$\langle\hat{n}\rangle(t)$")
ax.axhline(n_ss, color="crimson", ls="--", lw=1.5,
           label=rf"Steady state $\bar{{n}} = {n_bar_demo}$")
ax.set_xlabel(r"$\kappa t$"); ax.set_ylabel(r"$\langle\hat{n}\rangle$")
ax.set_title(rf"Thermal equilibration from $|0\rangle$, $\bar{{n}}_\text{{bath}} = {n_bar_demo}$")
ax.legend(); plt.tight_layout(); plt.show()

# Wigner snapshots during equilibration
res_therm_states = run_cavity(rho0_vac, t_therm,
                              kappa_=kappa, n_bar_=n_bar_demo, gamma_phi_=0.0)
snap_therm = [0.0, 1.0 / kappa, 2.0 / kappa, 4.0 / kappa]
wigner_snaps(res_therm_states, t_therm, snap_therm,
             title=rf"Wigner: vacuum $\to$ thermal ($\bar{{n}}_\text{{bath}}={n_bar_demo}$)")"""))  )

# ── Cell 19  Section 7 markdown ───────────────────────────────────────────────
CELLS.append(md_cell(L(r"""<a id="sec7"></a>
## 7  Entropy and Information

**Purity** $\mathrm{Tr}(\rho^2)$ and **von Neumann entropy** $S = -\mathrm{Tr}(\rho\ln\rho)$ track the loss of quantum information:

| Quantity | Pure state | Mixed state |
|---|---|---|
| Purity | $\mathrm{Tr}(\rho^2) = 1$ | $\mathrm{Tr}(\rho^2) < 1$ |
| Entropy | $S = 0$ | $S > 0$ |

For Fock state $|n\rangle$ decaying under photon loss, the entropy first **rises** (state becomes mixed) then falls back toward zero as the cavity relaxes to vacuum $|0\rangle$, which is a pure state.

The entropy peak occurs roughly at $t \approx T_1$ — the moment the uncertainty about how many photons remain is maximised.

**Linear entropy** $S_L = 1 - \mathrm{Tr}(\rho^2)$ is a common approximation that requires no matrix log.  It equals 0 for pure states and approaches 1 for maximally mixed states.""")))

# ── Cell 20  code: entropy vs time ────────────────────────────────────────────
CELLS.append(code_cell(L(r"""t_ent = np.linspace(0, 4.0 / kappa, n_times)
rho0_fock3 = fock_state(3, N=N).density_matrix()
res_ent = run_cavity(rho0_fock3, t_ent)

pur_ent = purity_traj(res_ent)
S_vn = entropy_traj(res_ent)
S_lin = [1.0 - p for p in pur_ent]

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
axes[0].plot(t_ent * kappa, S_vn, lw=2, color="steelblue", label="von Neumann $S = -\\mathrm{Tr}(\\rho\\ln\\rho)$")
axes[0].plot(t_ent * kappa, S_lin, lw=2, color="coral", ls="--", label="linear entropy $1 - \\mathrm{Tr}(\\rho^2)$")
axes[0].set_xlabel(r"$\kappa t$"); axes[0].set_ylabel("entropy")
axes[0].set_title(r"Entropy during decay of $|3\rangle$")
axes[0].legend(fontsize=9)

axes[1].plot(t_ent * kappa, pur_ent, lw=2, color="steelblue")
axes[1].set_xlabel(r"$\kappa t$"); axes[1].set_ylabel(r"purity $\mathrm{Tr}(\rho^2)$")
axes[1].set_title(r"Purity during decay of $|3\rangle$")

plt.suptitle(r"$|3\rangle$ relaxing to $|0\rangle$ under photon loss", fontsize=12)
plt.tight_layout(); plt.show()

i_peak = int(np.argmax(S_vn))
print(f"Entropy peak at kappa*t = {t_ent[i_peak]*kappa:.2f},  S_max = {S_vn[i_peak]:.3f} nats")
print(f"Final entropy (t → 4T1): S = {S_vn[-1]:.4f} nats  (→ 0 as state → vacuum)")""")))

# ── assemble and write ─────────────────────────────────────────────────────────
notebook = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Bosonic Gates", "language": "python", "name": "bosonic-gates"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "cells": CELLS,
}

out_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "content", "module0_foundations",
    "nb0c_open_systems.ipynb",
))
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)
print(f"Written {len(CELLS)} cells to {out_path}")
