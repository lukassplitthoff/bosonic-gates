"""Generate nb0a_quantum_harmonic_oscillator.ipynb from scratch."""
import json
import os
import uuid


def _id():
    return uuid.uuid4().hex[:8]


def code_cell(source, tags=None):
    meta = {}
    if tags:
        meta["tags"] = tags
    return {
        "cell_type": "code",
        "id": _id(),
        "execution_count": None,
        "metadata": meta,
        "outputs": [],
        "source": source if isinstance(source, list) else [source],
    }


def md_cell(source):
    return {
        "cell_type": "markdown",
        "id": _id(),
        "metadata": {},
        "source": source if isinstance(source, list) else [source],
    }


def L(s):
    """Split a multi-line string into a list of notebook source lines."""
    lines = s.split("\n")
    return [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])


# =============================================================================
CELLS = []

# Cell 0 — parameters
CELLS.append(code_cell(L(r"""# parameters
# BINDER_FAST: set N=6, xvec_pts=40, cf_pts=15 for fast cloud execution
N = 25            # Hilbert space truncation
omega_r = 5.0     # resonator frequency in GHz
T_mk = 15.0       # fridge temperature in millikelvin
alpha_coherent = 2.0
alpha_cat = 2.0
x_max = 5.0       # phase-space half-extent for Wigner / Q plots
xvec_pts = 80     # Wigner / Q grid resolution
cf_pts = 25       # CF grid resolution (coarser — inner loop)"""), tags=["parameters"]))

# Cell 1 — imports (hide)
CELLS.append(code_cell(L(r"""# hide
import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.constants import h, Boltzmann
%matplotlib widget

from bosonic_gates import (
    BosonicState, coherent_state, fock_state, squeezed_state,
    thermal_state, cat_state, fock_superposition, binomial_state,
)
from bosonic_gates.hamiltonians.harmonic_oscillator import resonator_hamiltonian""")))

# Cell 2 — plot_state_quartet helper (hide)
CELLS.append(code_cell(L(r"""# hide
def plot_state_quartet(s, title, _x_max=x_max, _xvec_pts=xvec_pts, _cf_pts=cf_pts):
    rho = s.density_matrix()
    xvec = np.linspace(-_x_max, _x_max, _xvec_pts)
    bvec = np.linspace(-_x_max, _x_max, _cf_pts)

    W = qt.wigner(rho, xvec, xvec)
    Q = qt.qfunc(rho, xvec, xvec)

    chi = np.zeros((_cf_pts, _cf_pts))
    for i, bx in enumerate(bvec):
        for j, by in enumerate(bvec):
            D = qt.displace(rho.shape[0], bx + 1j * by)
            chi[i, j] = abs((D * rho).tr())

    dist = s.fock_distribution()
    peak = int(np.argmax(dist))
    n_max_plot = min(s.N - 1, max(peak * 2 + 6, 12))

    fig = plt.figure(figsize=(13, 3.4))
    gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.45)
    fig.suptitle(title, fontsize=12)

    ax0 = fig.add_subplot(gs[0])
    ax0.bar(range(s.N), dist, color="steelblue", alpha=0.85)
    ax0.set_xlabel("$n$")
    ax0.set_ylabel("$P(n)$")
    ax0.set_title("Fock distribution")
    ax0.set_xlim(-0.5, n_max_plot + 0.5)

    ax1 = fig.add_subplot(gs[1])
    vmax = max(float(np.max(np.abs(W))), 1e-9)
    ax1.contourf(xvec, xvec, W, levels=50, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax1.set_title(r"Wigner $W(\alpha)$")
    ax1.set_aspect("equal")
    ax1.set_xlabel(r"Re$(\alpha)$")
    ax1.set_ylabel(r"Im$(\alpha)$")

    ax2 = fig.add_subplot(gs[2])
    ax2.contourf(bvec, bvec, chi, levels=50, cmap="viridis")
    ax2.set_title(r"CF $|\chi(\beta)|$")
    ax2.set_aspect("equal")
    ax2.set_xlabel(r"Re$(\beta)$")
    ax2.set_ylabel(r"Im$(\beta)$")

    ax3 = fig.add_subplot(gs[3])
    ax3.contourf(xvec, xvec, Q, levels=50, cmap="inferno")
    ax3.set_title("Q function")
    ax3.set_aspect("equal")
    ax3.set_xlabel(r"Re$(\alpha)$")
    ax3.set_ylabel(r"Im$(\alpha)$")

    plt.tight_layout()
    plt.show()

    print(f"  <n> = {s.photon_number():.3f}   purity = {s.purity():.4f}")""")))

# Cell 3 — title markdown
CELLS.append(md_cell(L(r"""## Module 0a: The Quantum Harmonic Oscillator as a Microwave Resonator

**Learning objectives:**
- Identify the quantum states you encounter in cavity QED experiments: Fock, coherent, squeezed, thermal, cat, binomial, and GKP
- Read a Fock distribution, Wigner function, characteristic function, and Q function and know what each tells you
- Estimate the thermal photon population in your resonator for a given temperature and frequency
- Understand why the linear resonator cannot distinguish photon numbers without an added nonlinearity

---

When you look at $S_{21}$ on your VNA you are probing a quantum harmonic oscillator.
This notebook builds the quantum vocabulary that connects what you measure to what is happening in your cavity.

**Sections:** [1 QHO Hamiltonian](#sec1) · [2 Fock](#sec2) · [3 Coherent](#sec3) · [4 Squeezed](#sec4) · [5 Thermal](#sec5) · [6 Cat](#sec6) · [7 Binomial](#sec7) · [8 GKP](#sec8) · [9 Zoo](#sec9)""")))

# Cell 4 — Section 1 markdown: QHO Hamiltonian
CELLS.append(md_cell(L(r"""<a id="sec1"></a>
## 1  The Microwave Resonator as a Quantum Harmonic Oscillator

A linear microwave resonator — coplanar waveguide, 3-D aluminium cavity, or lumped-element LC circuit — is described by a single resonance frequency $\omega_0 = 1/\sqrt{LC}$.
Promoting the charge $\hat{Q}$ and flux $\hat{\Phi}$ to conjugate quantum operators with $[\hat{Q}, \hat{\Phi}] = i\hbar$ gives the Hamiltonian

$$\hat{H} = \hbar\omega_0\!\left(\hat{a}^\dagger\hat{a} + \frac{1}{2}\right),$$

where the dimensionless ladder operators $\hat{a}$ and $\hat{a}^\dagger$ satisfy $[\hat{a}, \hat{a}^\dagger] = 1$ and

$$\hat{a}\,|n\rangle = \sqrt{n}\,|n-1\rangle, \qquad \hat{a}^\dagger|n\rangle = \sqrt{n+1}\,|n+1\rangle.$$

The *quadrature operators* $\hat{X} = (\hat{a}+\hat{a}^\dagger)/2$ and $\hat{P} = (\hat{a}-\hat{a}^\dagger)/2i$ are the I and Q channels of your homodyne detector.
Vacuum fluctuations set the shot-noise floor $\Delta X^2 = \Delta P^2 = 1/4$.

**Lab note.** You measure $\omega_0$ from the dip (or peak) in $S_{21}$.
The observable $\hat{n} = \hat{a}^\dagger\hat{a}$ — photon number — is what dispersive readout tracks via the qubit–cavity cross-Kerr.
The $1/2$ zero-point energy is constant and drops out in the rotating frame.""")))

# Cell 5 — Energy spectrum
CELLS.append(code_cell(L(r"""H = resonator_hamiltonian(w=2 * np.pi * omega_r, M=N)
evals = H.eigenenergies()

fig, ax = plt.subplots(figsize=(3.5, 5))
for n, E in enumerate(evals[:8]):
    ax.axhline(E, xmin=0.15, xmax=0.75, lw=2.5, color="steelblue")
    ax.text(0.78, E, rf"$|{n}\rangle$", va="center", fontsize=10,
            transform=ax.get_yaxis_transform())

ax.set_ylabel(r"Energy $/ \hbar$ (GHz)")
ax.set_xticks([])
ax.set_title(rf"$\omega_r/2\pi = {omega_r:.1f}$ GHz, $N = {N}$")
ax.spines[["top", "right", "bottom"]].set_visible(False)
plt.tight_layout()
plt.show()

print("Spacing between adjacent levels (GHz):")
for n in range(5):
    print(f"  E_{n+1} - E_{n} = {evals[n+1] - evals[n]:.4f} GHz")""")))

# Cell 6 — Section 2 markdown: Fock states
CELLS.append(md_cell(L(r"""<a id="sec2"></a>
## 2  Fock States

A Fock state $|n\rangle$ has *exactly* $n$ photons.
The energy eigenvalue is $E_n = \hbar\omega_0(n+\tfrac{1}{2})$.

**Ground state at millikelvin.** The mean thermal photon number follows the Bose–Einstein distribution:

$$\bar{n}_{\rm th} = \frac{1}{e^{\hbar\omega_0/k_B T} - 1}.$$

At $T = 15\,\text{mK}$ and $\omega_0/2\pi = 5\,\text{GHz}$, $\hbar\omega_0/k_B T \approx 16$, so $\bar{n}_{\rm th} \approx e^{-16} \approx 10^{-7}$.
Your resonator sits in $|0\rangle$ for all practical purposes.

**Why you can't drive a Fock state directly.** The ladder is *equally spaced* — every transition $|n\rangle \to |n+1\rangle$ has the same frequency $\omega_0$.
A linear drive cannot selectively address a single rung.
Preparing $|n > 0\rangle$ requires a nonlinearity (transmon qubit + SNAP gate, Module 4).

**Wigner function signature.** Fock states are maximally non-classical:
$W(0) = (2/\pi)(-1)^n$, alternating sign at the origin. The Wigner function of $|n\rangle$ has $n+1$ oscillating rings.""")))

# Cell 7 — Thermal population plot
CELLS.append(code_cell(L(r"""T_range = np.concatenate([np.linspace(5e-3, 0.1, 400), np.linspace(0.1, 300, 400)])
x = h * omega_r * 1e9 / (Boltzmann * T_range)
n_bar = np.where(x > 500, 0.0, 1.0 / (np.expm1(x)))

fig, ax = plt.subplots(figsize=(6, 4))
ax.semilogy(T_range * 1e3, n_bar, lw=2)
ax.axvline(T_mk, color="crimson", ls="--", label=rf"$T = {T_mk}$ mK")
ax.axhline(1e-2, color="gray", ls=":", lw=1, label=r"$\bar{n} = 10^{-2}$")
ax.set_xlabel("Temperature (mK)")
ax.set_ylabel(r"Mean thermal photon number $\bar{n}$")
ax.set_title(rf"Thermal population at $\omega_r/2\pi = {omega_r}$ GHz")
ax.legend()
ax.set_xlim(5, 1e3)
ax.set_ylim(1e-12, 1e3)
plt.tight_layout()
plt.show()

x_mk = h * omega_r * 1e9 / (Boltzmann * T_mk * 1e-3)
n_bar_mk = 1.0 / np.expm1(x_mk)
print(f"At T = {T_mk} mK, f = {omega_r} GHz:  n_bar = {n_bar_mk:.2e}")""")))

# Cell 8 — Fock state quartets
CELLS.append(code_cell(L(r"""for n in [0, 1, 3]:
    plot_state_quartet(fock_state(n, N=N), rf"Fock $|{n}\rangle$")""")))

# Cell 9 — Section 3 markdown: Coherent states
CELLS.append(md_cell(L(r"""<a id="sec3"></a>
## 3  Coherent States

A coherent state $|\alpha\rangle = \hat{D}(\alpha)|0\rangle$ is generated by the displacement operator

$$\hat{D}(\alpha) = \exp\!\left(\alpha \hat{a}^\dagger - \alpha^* \hat{a}\right).$$

Coherent states are the eigenstates of $\hat{a}$: $\hat{a}|\alpha\rangle = \alpha|\alpha\rangle$.
The complex number $\alpha$ encodes field amplitude ($|\alpha|^2 = \bar{n}$) and phase.

**Lab connection.**
- A CW microwave tone drives your resonator into a coherent state.
- The VNA measures $|\alpha|$ via the transmission amplitude.
- In homodyne readout: $\langle\hat{X}\rangle = \text{Re}(\alpha)$, $\langle\hat{P}\rangle = \text{Im}(\alpha)$ — this is literally your IQ point.
- The noise floor is shot noise: $\Delta X^2 = \Delta P^2 = 1/4$ regardless of $|\alpha|$.

**Fock distribution.** Poissonian: $P(n) = e^{-|\alpha|^2}|\alpha|^{2n}/n!$.

**Wigner function.** A displaced Gaussian, always $W \geq 0$ — coherent states are as classical as quantum mechanics allows.""")))

# Cell 10 — Coherent state sweep
CELLS.append(code_cell(L(r"""for alpha in [0.0, 1.0, alpha_coherent, 3.0]:
    label = rf"Coherent $|\alpha| = {alpha:.1f}$,  $\bar{{n}} = {alpha**2:.1f}$"
    plot_state_quartet(coherent_state(alpha, N=N), label)""")))

# Cell 11 — Poisson overlay
CELLS.append(code_cell(L(r"""from scipy.stats import poisson

s_coh = coherent_state(alpha_coherent, N=N)
dist = s_coh.fock_distribution()
ns = np.arange(N)
poisson_pmf = poisson.pmf(ns, mu=alpha_coherent**2)

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(ns, dist, alpha=0.7, label="QuTiP", color="steelblue")
ax.plot(ns, poisson_pmf, "ro--", ms=5,
        label=rf"Poisson($\bar{{n}}={alpha_coherent**2:.1f}$)")
ax.set_xlabel("Photon number $n$")
ax.set_ylabel("$P(n)$")
ax.set_title(rf"Coherent state Fock distribution vs Poisson, $|\alpha|={alpha_coherent}$")
ax.legend()
ax.set_xlim(-0.5, 14.5)
plt.tight_layout()
plt.show()""")))

# Cell 12 — Section 4: Squeezed states
CELLS.append(md_cell(L(r"""<a id="sec4"></a>
## 4  Squeezed States

A squeezed vacuum state $|r\rangle = \hat{S}(r)|0\rangle$ has reduced noise in one quadrature at the cost of amplified noise in the conjugate:

$$\Delta X = \frac{e^{-r}}{2}, \qquad \Delta P = \frac{e^{r}}{2}.$$

**Generation.** Parametric amplification (flux-modulated SQUID, Josephson parametric amplifier) at twice the resonance frequency generates squeezed states.

**Phase-space picture.**
The Wigner function is an elliptical Gaussian centred at the origin.
The squeezing angle $\phi$ (controlled by the pump phase) rotates the ellipse.
Unlike Fock and cat states, $W \geq 0$ — squeezed vacuum is not non-classical in the Wigner sense — but the CF and Q function ellipses are oriented differently, making them useful for calibrating the squeezing axis.""")))

# Cell 13 — Squeezed state code
CELLS.append(code_cell(L(r"""for r in [0.5, 1.0]:
    plot_state_quartet(squeezed_state(r, N=N), rf"Squeezed vacuum $r = {r}$")""")))

# Cell 14 — Section 5: Thermal states
CELLS.append(md_cell(L(r"""<a id="sec5"></a>
## 5  Thermal States

A thermal state is a *statistical mixture* of Fock states:

$$\rho_{\rm th} = \sum_{n=0}^\infty p_n |n\rangle\langle n|, \qquad p_n = \frac{\bar{n}^n}{(\bar{n}+1)^{n+1}}.$$

This is a density matrix, not a ket — thermal states cannot be written as wavefunctions.
Their purity is $\text{Tr}(\rho^2) = 1/(2\bar{n}+1) < 1$.

**Lab signature.**
Residual thermal photons — from hot microwave lines, imperfect attenuation, or insufficient thermalisation — appear as a broad, isotropic blob in the IQ plane.
Unlike a coherent state (single displaced point), a thermal field has no preferred phase.
The Wigner function of a thermal state is a Gaussian centred at the origin, always $W \geq 0$, and the CF decays as a Gaussian with no off-diagonal structure.""")))

# Cell 15 — Thermal sweep
CELLS.append(code_cell(L(r"""for n_mean in [0.5, 2.0, 5.0]:
    plot_state_quartet(thermal_state(n_mean, N=N), rf"Thermal $\bar{{n}} = {n_mean}$")""")))

# Cell 16 — Section 6: Cat states
CELLS.append(md_cell(L(r"""<a id="sec6"></a>
## 6  Cat States

A *cat state* is a superposition of two coherent states at opposite positions in phase space:

$$|\mathcal{C}^\pm_\alpha\rangle = \mathcal{N}\!\left(|\alpha\rangle \pm |-\alpha\rangle\right).$$

- **Even cat** ($+$, `phase=0`): only even Fock components, photon-number parity $\hat{\Pi} = e^{i\pi\hat{n}} = +1$.
- **Odd cat** ($-$, `phase=π`): only odd Fock components, parity $= -1$.

**Wigner function.** Two coherent-state blobs connected by oscillatory interference fringes.
The fringes are the quantum signature — a classical mixture of $|\alpha\rangle$ and $|-\alpha\rangle$ would show *no* fringes.
Negativity of $W$ confirms non-classicality.

**Characteristic function.** The CF has two separated Gaussian blobs along the imaginary axis (the Fourier dual of the real-axis fringes in Wigner space).

**Why cat states matter for error correction.**
Photon loss $\hat{a}$ maps $|\mathcal{C}^+_\alpha\rangle \to |\mathcal{C}^-_\alpha\rangle$ (parity flip).
By continuously monitoring parity via a coupled transmon — without measuring the field amplitude — you can detect single-photon loss without collapsing the logical state.
This is the cat-qubit error-correction protocol (Ofek *et al.*, Nature 2016).

**Preparation.** SNAP gates (Module 4) or driven Kerr resonator (Module 3).""")))

# Cell 17 — Even/odd cat
CELLS.append(code_cell(L(r"""for phase, label in [(0, "even"), (np.pi, "odd")]:
    plot_state_quartet(
        cat_state(alpha_cat, N=N, phase=phase),
        rf"{label.capitalize()} cat $|\alpha| = {alpha_cat}$",
    )""")))

# Cell 18 — Cat alpha sweep
CELLS.append(code_cell(L(r"""for alpha_c in [1.0, 2.0, 3.0]:
    plot_state_quartet(
        cat_state(alpha_c, N=N, phase=0),
        rf"Even cat $|\alpha| = {alpha_c}$",
    )""")))

# Cell 19 — Section 7: Binomial states
CELLS.append(md_cell(L(r"""<a id="sec7"></a>
## 7  Binomial States

A binomial state interpolates continuously between a Fock state and a coherent-like distribution:

$$|B(\theta, M)\rangle = \sum_{n=0}^{M} \sqrt{\binom{M}{n}} \cos^n\!\theta\;\sin^{M-n}\!\theta\;|n\rangle.$$

Limits: $\theta \to 0$ → Fock $|M\rangle$; $\theta = \pi/4$ → maximum spread (coherent-like); $\theta \to \pi/2$ → Fock $|0\rangle$.

Binomial codes are a family of bosonic quantum error correcting codes (Michael *et al.*, PRX 2016).
Choosing $M$ and the spacing between occupied Fock states allows protection against photon loss, gain, and dephasing events simultaneously.""")))

# Cell 20 — Binomial sweep
CELLS.append(code_cell(L(r"""for theta, label in [
    (np.pi / 8,     r"$\theta = \pi/8$"),
    (np.pi / 4,     r"$\theta = \pi/4$"),
    (3 * np.pi / 8, r"$\theta = 3\pi/8$"),
]:
    s = binomial_state(N=N, theta=theta, n_max=8)
    plot_state_quartet(s, rf"Binomial {label}, $n_{{\rm max}}=8$")""")))

# Cell 21 — Section 8: GKP states
CELLS.append(md_cell(L(r"""<a id="sec8"></a>
## 8  GKP States

Gottesman–Kitaev–Preskill (GKP) states are *grid states* in phase space (Gottesman, Kitaev & Preskill, PRA 2001).
The ideal logical-zero state is

$$|0_L\rangle \propto \sum_{k=-\infty}^{\infty} |2k\sqrt{\pi}\rangle_X,$$

a superposition of position eigenstates on a lattice with spacing $2\sqrt{\pi} \approx 3.54$.
Because physical states must be normalisable, the ideal (infinite-energy) GKP state is replaced by a *finite-energy* version: each delta function is replaced by a narrow squeezed Gaussian of width $\delta$, and the sum is truncated and weighted by an overall Gaussian envelope.

**Phase-space signatures.**
- **Wigner:** a regular comb of Gaussian blobs on a 2-D grid — the grid spacing is $2\sqrt{\pi}$ in both $X$ and $P$ (for the full square-lattice GKP code).
- **CF:** sharp peaks on the reciprocal lattice — this is the Fourier-dual grid.
- **Q function:** a smoothed version of the Wigner grid.

**Error correction.** Any displacement in phase space smaller than $\sqrt{\pi}$ can be corrected by measuring the residue of $\hat{X}$ or $\hat{P}$ modulo the grid.
This is the complementary error model to the cat qubit (which corrects photon-number parity jumps).

**Preparation.** Repeated ECD gate sequences (Campagne-Ibarcq *et al.*, Nature 2020; Module 4).""")))

# Cell 22 — GKP helper
CELLS.append(code_cell(L(r"""def gkp_state_approx(N, delta=0.4, n_grid=2):
    '''
    Finite-energy approximate GKP |0_L> state.

    Each grid point k is a squeezed coherent state displaced to X = 2k*sqrt(pi),
    squeezed so that its X-width is approximately delta (in phase-space units).
    The sum is weighted by a Gaussian envelope exp(-pi*delta^2*k^2).

    Parameters
    ----------
    N     : Hilbert space dimension
    delta : X-width of each Gaussian blob (smaller = closer to ideal)
    n_grid: number of grid points on each side of the origin
    '''
    spacing = 2.0 * np.sqrt(np.pi)
    r = np.log(1.0 / delta)              # squeeze r so DeltaX ~ delta/2 < vacuum
    sq_state = qt.squeeze(N, r) * qt.fock(N, 0)

    psi = qt.Qobj(np.zeros(N, dtype=complex))
    psi.dims = [[N], [1]]

    for k in range(-n_grid, n_grid + 1):
        alpha_k = float(k) * spacing     # real displacement along X
        weight = np.exp(-np.pi * delta**2 * k**2)
        psi = psi + weight * qt.displace(N, alpha_k) * sq_state

    return BosonicState(psi.unit(), N)""")))

# Cell 23 — GKP quartet
CELLS.append(code_cell(L(r"""s_gkp = gkp_state_approx(N=N, delta=0.4, n_grid=2)
plot_state_quartet(
    s_gkp,
    r"GKP $|0_L\rangle$ (finite energy, $\delta = 0.4$, $n_{\rm grid}=2$)",
    _x_max=7.0,
)""")))

# Cell 24 — Section 9: Zoo comparison
CELLS.append(md_cell(L(r"""<a id="sec9"></a>
## 9  State Zoo — Wigner Function Gallery

All six state families side by side for comparison.""")))

# Cell 25 — Zoo code
CELLS.append(code_cell(L(r"""state_zoo = [
    (fock_state(0, N=N),                             r"Vacuum $|0\rangle$"),
    (fock_state(3, N=N),                             r"Fock $|3\rangle$"),
    (coherent_state(alpha_coherent, N=N),            rf"Coherent $|\alpha|={alpha_coherent}$"),
    (cat_state(alpha_cat, N=N, phase=0),             rf"Even cat $|\alpha|={alpha_cat}$"),
    (binomial_state(N=N, theta=np.pi/4, n_max=8),   r"Binomial $\theta=\pi/4$"),
    (gkp_state_approx(N=N, delta=0.4, n_grid=2),    r"GKP $|0_L\rangle$"),
]

xvec_zoo = np.linspace(-7.0, 7.0, 120)
fig, axes = plt.subplots(2, 3, figsize=(13, 8))

for ax, (s, title) in zip(axes.flat, state_zoo):
    W = qt.wigner(s.density_matrix(), xvec_zoo, xvec_zoo)
    vmax = max(float(np.max(np.abs(W))), 1e-9)
    ax.contourf(xvec_zoo, xvec_zoo, W, levels=50, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$X$")
    ax.set_ylabel(r"$P$")

plt.suptitle("Wigner function zoo", fontsize=14, y=1.01)
plt.tight_layout()
plt.show()""")))

# =============================================================================
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Bosonic Gates",
            "language": "python",
            "name": "bosonic-gates",
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0",
        },
    },
    "cells": CELLS,
}

out_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..",
    "content", "module0_foundations",
    "nb0a_quantum_harmonic_oscillator.ipynb",
))

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Written {len(CELLS)} cells to {out_path}")
