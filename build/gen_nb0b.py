"""Generate nb0b_phase_space.ipynb."""
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
# BINDER_FAST: set N=10, xvec_pts=40, cf_pts=15 for fast cloud execution
N = 25
alpha = 2.0          # default phase-space amplitude
x_max = 5.0
xvec_pts = 80        # resolution for Wigner / Q grids
cf_pts = 25          # resolution for characteristic function grid"""), tags=["parameters"]))

# ── Cell 1  imports ──────────────────────────────────────────────────────────
CELLS.append(code_cell(L(r"""# hide
import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter
%matplotlib widget

from bosonic_gates import (
    BosonicState, coherent_state, fock_state, squeezed_state,
    thermal_state, cat_state, binomial_state,
)""")))

# ── Cell 2  helpers ──────────────────────────────────────────────────────────
CELLS.append(code_cell(L(r"""# hide
def _xvec(): return np.linspace(-x_max, x_max, xvec_pts)
def _bvec(): return np.linspace(-x_max, x_max, cf_pts)

def cf_direct(rho, bvec):
    '''Compute |chi(beta)| = |Tr[rho D(beta)]| on a square grid.'''
    nb = len(bvec)
    chi = np.zeros((nb, nb))
    N_ = rho.shape[0]
    for i, bx in enumerate(bvec):
        for j, by in enumerate(bvec):
            D = qt.displace(N_, bx + 1j * by)
            chi[i, j] = abs((D * rho).tr())
    return chi

def plot_state_quartet(s, title):
    rho = s.density_matrix()
    xvec = _xvec(); bvec = _bvec()
    W = qt.wigner(rho, xvec, xvec)
    Q = qt.qfunc(rho, xvec, xvec)
    chi = cf_direct(rho, bvec)
    dist = s.fock_distribution()
    n_max_plot = min(s.N - 1, max(int(np.argmax(dist)) * 2 + 6, 12))

    fig = plt.figure(figsize=(13, 3.4))
    gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.45)
    fig.suptitle(title, fontsize=12)
    ax0 = fig.add_subplot(gs[0])
    ax0.bar(range(s.N), dist, color="steelblue", alpha=0.85)
    ax0.set_xlabel("$n$"); ax0.set_ylabel("$P(n)$")
    ax0.set_title("Fock distribution"); ax0.set_xlim(-0.5, n_max_plot + 0.5)
    ax1 = fig.add_subplot(gs[1])
    vmax = max(float(np.max(np.abs(W))), 1e-9)
    ax1.contourf(xvec, xvec, W, levels=50, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax1.set_title(r"Wigner $W(\alpha)$"); ax1.set_aspect("equal")
    ax1.set_xlabel(r"$X$"); ax1.set_ylabel(r"$P$")
    ax2 = fig.add_subplot(gs[2])
    ax2.contourf(bvec, bvec, chi, levels=50, cmap="viridis")
    ax2.set_title(r"CF $|\chi(\beta)|$"); ax2.set_aspect("equal")
    ax2.set_xlabel(r"Re$(\beta)$"); ax2.set_ylabel(r"Im$(\beta)$")
    ax3 = fig.add_subplot(gs[3])
    ax3.contourf(xvec, xvec, Q, levels=50, cmap="inferno")
    ax3.set_title("Q function"); ax3.set_aspect("equal")
    ax3.set_xlabel(r"$X$"); ax3.set_ylabel(r"$P$")
    plt.tight_layout(); plt.show()
    print(f"  <n> = {s.photon_number():.3f}   purity = {s.purity():.4f}")

def mandel_Q(s):
    '''Mandel Q = (Var(n) - <n>) / <n>.  Q=0 Poissonian, Q<0 sub-, Q>0 super-.'''
    rho = s.density_matrix()
    n_op = qt.num(s.N)
    n_mean = float(qt.expect(n_op, rho))
    if n_mean < 1e-12:
        return 0.0
    n2_mean = float(qt.expect(n_op * n_op, rho))
    var_n = n2_mean - n_mean**2
    return (var_n - n_mean) / n_mean

def wigner_neg_volume(s, xvec=None):
    '''Integrated Wigner negativity = int max(0, -W) d^2 alpha.'''
    if xvec is None:
        xvec = _xvec()
    W = qt.wigner(s.density_matrix(), xvec, xvec)
    dx = xvec[1] - xvec[0]
    return float(np.sum(np.maximum(0.0, -W)) * dx**2)""")))

# ── Cell 3  title ────────────────────────────────────────────────────────────
CELLS.append(md_cell(L(r"""## Module 0b: Phase Space Representations

**Learning objectives:**
- State the precise mathematical definition of the Wigner function, Q function, characteristic function, and P function
- Connect the Wigner function marginals to quadrature distributions measured by homodyne detection
- State Hudson's theorem and use Wigner negativity to quantify non-classicality
- Describe the experimental protocol for Wigner tomography

---

Phase space is the language of continuous-variable quantum information.
Every quantum state has an equivalent description as a function over the $(\hat{X}, \hat{P})$ plane.
This notebook unpacks the four main representations, their mathematical relationships, and what they tell you at the bench.

**Sections:** [1 Four representations](#sec1) · [2 Marginals](#sec2) · [3 Fourier relations](#sec3) · [4 Non-classicality](#sec4) · [5 Tomography](#sec5)""")))

# ── Cell 4  Section 1 markdown ───────────────────────────────────────────────
CELLS.append(md_cell(L(r"""<a id="sec1"></a>
## 1  The Four Phase-Space Representations

All four functions represent the *same* quantum state — they are related by linear transforms.

| Function | Definition | Sign? | Lab access |
|---|---|---|---|
| **Wigner** $W(\alpha)$ | $(2/\pi)\,\text{Tr}[\rho\,\hat{D}(\alpha)(-1)^{\hat{n}}\hat{D}^\dagger(\alpha)]$ | Can be negative | Displace + parity |
| **Q function** $Q(\alpha)$ | $\langle\alpha|\rho|\alpha\rangle/\pi \geq 0$ | Always positive | Heterodyne detection |
| **Characteristic fn** $\chi(\beta)$ | $\text{Tr}[\rho\,\hat{D}(\beta)]$, $|\chi|\leq 1$ | Complex | Indirect via tomography |
| **P function** $P(\alpha)$ | $\rho = \int P(\alpha)|\alpha\rangle\langle\alpha|d^2\alpha$ | Can be singular | Indirect only |

**Key facts.**
- $W$ and $\chi$ are a Fourier pair: $W(\alpha) = \frac{1}{\pi^2}\int \chi(\beta)\,e^{\alpha\beta^*-\alpha^*\beta}\,d^2\beta$.
- $Q$ is $W$ *smoothed* by a coherent-state Gaussian: $Q(\alpha) = \frac{1}{\pi}\int W(\beta)\,e^{-|\alpha-\beta|^2}\,d^2\beta$.
- $P$ is the most singular; for non-classical states it is not a well-defined function.
- All three satisfy $\chi(0) = 1$ and $\int W\,d^2\alpha = \int Q\,d^2\alpha = 1$.""")))

# ── Cell 5  code: four panels for several states ─────────────────────────────
CELLS.append(code_cell(L(r"""states_demo = [
    (fock_state(2, N=N),               r"Fock $|2\rangle$"),
    (coherent_state(alpha, N=N),       rf"Coherent $\alpha={alpha}$"),
    (cat_state(alpha, N=N, phase=0),   rf"Even cat $|\alpha|={alpha}$"),
]
for s, title in states_demo:
    plot_state_quartet(s, title)""")))

# ── Cell 6  Section 2 markdown ───────────────────────────────────────────────
CELLS.append(md_cell(L(r"""<a id="sec2"></a>
## 2  Quadrature Marginals and Homodyne Detection

Integrating the Wigner function over one quadrature gives the probability distribution of the conjugate:

$$\int_{-\infty}^\infty W(X, P)\,dP = |\langle X|\psi\rangle|^2 \equiv \mathcal{P}(X)$$

$$\int_{-\infty}^\infty W(X, P)\,dX = |\langle P|\psi\rangle|^2 \equiv \mathcal{P}(P)$$

**Lab connection.** A homodyne detector measures the quadrature $\hat{X}_\phi = \hat{X}\cos\phi + \hat{P}\sin\phi$ directly.
Repeating at all angles $\phi$ from 0 to $\pi$ gives the *radon transform* of $W$, from which $W$ can be reconstructed (quantum state tomography).

**Key signatures:**
- Coherent state: Gaussian marginals centred at $\text{Re}(\alpha)$ and $\text{Im}(\alpha)$.
- Squeezed state: one quadrature narrower, the other wider than vacuum.
- Cat state: the $X$-marginal shows *two* peaks at $\pm|\alpha|$ but **no** interference fringes — fringes only appear in the 2-D Wigner function, not in homodyne traces.
- Thermal state: wider Gaussian than coherent for the same $\bar{n}$.""")))

# ── Cell 7  code: marginals ──────────────────────────────────────────────────
CELLS.append(code_cell(L(r"""xvec = _xvec()
dx = xvec[1] - xvec[0]

states_marg = [
    (coherent_state(alpha, N=N),       rf"Coherent $\alpha={alpha}$"),
    (squeezed_state(0.8, N=N),         r"Squeezed $r=0.8$"),
    (cat_state(alpha, N=N, phase=0),   rf"Even cat $|\alpha|={alpha}$"),
    (thermal_state(alpha**2, N=N),     rf"Thermal $\bar{{n}}={alpha**2}$"),
]

fig, axes = plt.subplots(len(states_marg), 3, figsize=(12, 2.8 * len(states_marg)),
                         gridspec_kw={"width_ratios": [2, 1, 1]})
fig.suptitle("Wigner function and its quadrature marginals", fontsize=13)

for row, (s, label) in enumerate(states_marg):
    rho = s.density_matrix()
    W = qt.wigner(rho, xvec, xvec)
    # Marginals: integrate over one quadrature axis
    # W returned with shape (len(xvec), len(xvec)):
    #   row index → X quadrature, col index → P quadrature
    marg_P = np.sum(W, axis=1) * dx    # integrate over P (cols) → P(X)
    marg_X = np.sum(W, axis=0) * dx    # integrate over X (rows) → P(P)

    vmax = max(float(np.max(np.abs(W))), 1e-9)
    axes[row, 0].contourf(xvec, xvec, W, levels=40, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[row, 0].set_aspect("equal")
    axes[row, 0].set_title(label, fontsize=10)
    axes[row, 0].set_xlabel(r"$X$"); axes[row, 0].set_ylabel(r"$P$")

    axes[row, 1].plot(xvec, marg_P, lw=2, color="steelblue")
    axes[row, 1].fill_between(xvec, marg_P, alpha=0.25, color="steelblue")
    axes[row, 1].set_xlabel(r"$X$"); axes[row, 1].set_ylabel(r"$\mathcal{P}(X)$")
    axes[row, 1].set_title("X marginal")

    axes[row, 2].plot(xvec, marg_X, lw=2, color="coral")
    axes[row, 2].fill_between(xvec, marg_X, alpha=0.25, color="coral")
    axes[row, 2].set_xlabel(r"$P$"); axes[row, 2].set_ylabel(r"$\mathcal{P}(P)$")
    axes[row, 2].set_title("P marginal")

plt.tight_layout()
plt.show()""")))

# ── Cell 8  Section 3 markdown ───────────────────────────────────────────────
CELLS.append(md_cell(L(r"""<a id="sec3"></a>
## 3  Fourier Relations Between the Representations

$W$ and $\chi$ form a 2-D Fourier pair:

$$\chi(\beta) = \pi \iint W(\alpha)\, e^{\alpha\beta^* - \alpha^*\beta}\, d^2\alpha$$

In Cartesian coordinates with $\alpha = x + ip$ and $\beta = u + iv$, the exponent becomes $2i(yu - xv)$, so $\chi(u,v) \propto \text{FT}[W](v, u)$ with swapped axes.

$Q$ is obtained from $W$ by convolution with the vacuum coherent-state kernel:

$$Q(\alpha) = \frac{1}{\pi}\iint W(\beta)\,e^{-|\alpha-\beta|^2}\,d^2\beta$$

This Gaussian smoothing of width $\sigma = 1/\sqrt{2}$ (in natural units) always erases negativity — explaining why $Q \geq 0$ while $W$ can dip below zero.""")))

# ── Cell 9  code: Q = W * Gaussian ──────────────────────────────────────────
CELLS.append(code_cell(L(r"""s = cat_state(alpha, N=N, phase=0)
rho = s.density_matrix()
xvec = _xvec()
dx = xvec[1] - xvec[0]

W = qt.wigner(rho, xvec, xvec)
Q_actual = qt.qfunc(rho, xvec, xvec)

# Gaussian convolution kernel width sigma = 1/sqrt(2) in phase-space units
sigma_pix = (1.0 / np.sqrt(2)) / dx
Q_from_W = gaussian_filter(np.maximum(0.0, W), sigma=sigma_pix)
# Renormalise to sum to 1
norm = Q_from_W.sum() * dx**2
if norm > 1e-12:
    Q_from_W /= norm

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
vmax_W = max(float(np.max(np.abs(W))), 1e-9)
axes[0].contourf(xvec, xvec, W, levels=50, cmap="RdBu_r", vmin=-vmax_W, vmax=vmax_W)
axes[0].set_title(r"Wigner $W(\alpha)$  (can be negative)"); axes[0].set_aspect("equal")
axes[0].set_xlabel(r"$X$"); axes[0].set_ylabel(r"$P$")

axes[1].contourf(xvec, xvec, Q_from_W, levels=50, cmap="inferno")
axes[1].set_title(r"$W$ convolved with vacuum Gaussian"); axes[1].set_aspect("equal")
axes[1].set_xlabel(r"$X$"); axes[1].set_ylabel(r"$P$")

axes[2].contourf(xvec, xvec, Q_actual, levels=50, cmap="inferno")
axes[2].set_title(r"Q function $Q(\alpha) = \langle\alpha|\rho|\alpha\rangle/\pi$")
axes[2].set_aspect("equal")
axes[2].set_xlabel(r"$X$"); axes[2].set_ylabel(r"$P$")

plt.suptitle(rf"Even cat $|\alpha|={alpha}$: W convolved → Q", fontsize=12)
plt.tight_layout(); plt.show()

# Numerical check: Q(0) should equal (1/pi) * int W(beta) exp(-|beta|^2) d^2beta
print(f"Q_actual max  = {Q_actual.max():.4f}")
print(f"Q_from_W max  = {Q_from_W.max():.4f}")
print("Smoothing erases all negativity — Q >= 0 always.")""")))

# ── Cell 10  code: CF chi(0) = 1 and grid structure ─────────────────────────
CELLS.append(code_cell(L(r"""bvec = _bvec()

state_cf_demo = [
    (fock_state(0, N=N),             r"Vacuum $|0\rangle$"),
    (coherent_state(alpha, N=N),     rf"Coherent $\alpha={alpha}$"),
    (cat_state(alpha, N=N, phase=0), rf"Even cat $|\alpha|={alpha}$"),
]

fig, axes = plt.subplots(1, len(state_cf_demo), figsize=(12, 4))
for ax, (s, title) in zip(axes, state_cf_demo):
    chi = cf_direct(s.density_matrix(), bvec)
    ax.contourf(bvec, bvec, chi, levels=50, cmap="viridis")
    ax.set_title(title, fontsize=10); ax.set_aspect("equal")
    ax.set_xlabel(r"Re$(\beta)$"); ax.set_ylabel(r"Im$(\beta)$")
    # Check chi(0) = 1
    ic = len(bvec) // 2
    print(f"{title}: chi(0) = {chi[ic, ic]:.4f}  (should be 1)")

plt.suptitle(r"Characteristic function $|\chi(\beta)| = |\text{Tr}[\rho\,\hat{D}(\beta)]|$", fontsize=12)
plt.tight_layout(); plt.show()""")))

# ── Cell 11  Section 4 markdown ──────────────────────────────────────────────
CELLS.append(md_cell(L(r"""<a id="sec4"></a>
## 4  Non-Classicality: Hudson's Theorem, Wigner Negativity, and Mandel Q

**Hudson's theorem (1974).** A *pure* state $|\psi\rangle$ has $W \geq 0$ everywhere if and only if it is a *Gaussian* state (coherent or squeezed vacuum). Every other pure state — Fock, cat, GKP, binomial — has a negative Wigner function somewhere.

**Wigner negativity volume.** A useful scalar non-classicality measure:

$$\mathcal{N}(W) = \int \max(0,\,-W(\alpha))\,d^2\alpha$$

$\mathcal{N}=0$ for classical (Gaussian) states; $\mathcal{N}>0$ for non-classical states.

**Mandel Q parameter.** Quantifies photon-number statistics:

$$Q_M = \frac{\text{Var}(\hat{n}) - \langle\hat{n}\rangle}{\langle\hat{n}\rangle}$$

| State | $Q_M$ | Regime |
|---|---|---|
| Fock $|n\rangle$ | $-1$ | maximally sub-Poissonian |
| Coherent $|\alpha\rangle$ | $0$ | Poissonian (shot-noise limited) |
| Thermal $\rho_{\rm th}$ | $\bar{n} > 0$ | super-Poissonian |
| Cat $|\mathcal{C}^+\rangle$ | varies | depends on $|\alpha|$ |

Sub-Poissonian statistics ($Q_M < 0$) are a signature of non-classicality detectable with a photon counter.""")))

# ── Cell 12  code: Wigner negativity ─────────────────────────────────────────
CELLS.append(code_cell(L(r"""from bosonic_gates import fock_superposition

xvec = _xvec()

states_nc = [
    (fock_state(0, N=N),             r"Vacuum"),
    (fock_state(1, N=N),             r"Fock $|1\rangle$"),
    (fock_state(3, N=N),             r"Fock $|3\rangle$"),
    (coherent_state(alpha, N=N),     r"Coherent"),
    (squeezed_state(0.8, N=N),       r"Squeezed"),
    (thermal_state(1.0, N=N),        r"Thermal"),
    (cat_state(alpha, N=N, phase=0), r"Even cat"),
    (cat_state(alpha, N=N, phase=np.pi), r"Odd cat"),
]

labels = [lbl for _, lbl in states_nc]
neg_vols = [wigner_neg_volume(s, xvec) for s, _ in states_nc]

fig, ax = plt.subplots(figsize=(9, 4))
colors = ["steelblue" if v < 1e-6 else "crimson" for v in neg_vols]
ax.bar(labels, neg_vols, color=colors, alpha=0.85, edgecolor="k", linewidth=0.5)
ax.axhline(0, color="k", lw=0.8)
ax.set_ylabel(r"Wigner negativity $\mathcal{N}(W)$")
ax.set_title("Non-classicality: Wigner negativity volume")
ax.set_xticklabels(labels, rotation=25, ha="right")
plt.tight_layout(); plt.show()

print("Negativity volume:")
for lbl, nv in zip(labels, neg_vols):
    tag = "✗ classical" if nv < 1e-6 else "✓ non-classical"
    print(f"  {lbl:20s}  N = {nv:.4f}  {tag}")""")))

# ── Cell 13  code: Mandel Q ───────────────────────────────────────────────────
CELLS.append(code_cell(L(r"""states_mq = [
    (fock_state(1, N=N),              r"Fock $|1\rangle$"),
    (fock_state(3, N=N),              r"Fock $|3\rangle$"),
    (coherent_state(1.0, N=N),        r"Coherent $|\alpha|=1$"),
    (coherent_state(alpha, N=N),      rf"Coherent $|\alpha|={alpha}$"),
    (squeezed_state(0.8, N=N),        r"Squeezed $r=0.8$"),
    (thermal_state(1.0, N=N),         r"Thermal $\bar{n}=1$"),
    (thermal_state(alpha**2, N=N),    rf"Thermal $\bar{{n}}={alpha**2}$"),
    (cat_state(alpha, N=N, phase=0),  rf"Even cat $|\alpha|={alpha}$"),
]

labels_mq = [lbl for _, lbl in states_mq]
Q_vals = [mandel_Q(s) for s, _ in states_mq]

fig, ax = plt.subplots(figsize=(9, 4))
colors_mq = ["crimson" if q < -0.01 else ("steelblue" if q < 0.01 else "goldenrod")
              for q in Q_vals]
ax.bar(labels_mq, Q_vals, color=colors_mq, alpha=0.85, edgecolor="k", linewidth=0.5)
ax.axhline(0, color="k", lw=1.2, label="Poissonian (coherent state)")
ax.set_ylabel(r"Mandel $Q_M$")
ax.set_title(r"Photon statistics: Mandel $Q_M = (\text{Var}(\hat{n}) - \langle\hat{n}\rangle) / \langle\hat{n}\rangle$")
ax.set_xticklabels(labels_mq, rotation=25, ha="right")
ax.legend()
plt.tight_layout(); plt.show()

print(f"{'State':30s}  Q_M")
for lbl, q in zip(labels_mq, Q_vals):
    regime = "sub-Poissonian" if q < -0.01 else ("Poissonian" if q < 0.01 else "super-Poissonian")
    print(f"  {lbl:30s}  {q:+.4f}  ({regime})")""")))

# ── Cell 14  Section 5 markdown ──────────────────────────────────────────────
CELLS.append(md_cell(L(r"""<a id="sec5"></a>
## 5  Wigner Tomography in the Lab

**Protocol (Lutterbach & Davidovich 1997; Bertet *et al.* 2002).**

$$W(\alpha) = \frac{2}{\pi} \langle (-1)^{\hat{n}} \rangle_{\hat{D}(-\alpha)\rho\hat{D}(\alpha)}$$

1. **Displace** the cavity by $-\alpha$: apply $\hat{D}(-\alpha)$ via a coherent pulse.
2. **Measure parity** $(-1)^{\hat{n}}$: couple to an ancilla qubit and read out.
3. Repeat for a grid of $\alpha$ values.
4. The expectation value of parity at each displacement point gives $W(\alpha)$.

**Key features.**
- No reconstruction algorithm needed — each point is directly $W(\alpha)$.
- Resolution is set by how many grid points you measure; shot noise averages down with repetitions.
- Displacement $\hat{D}(-\alpha)$ is a short microwave pulse; parity is measured via a $\pi/2$ — wait — $\pi/2$ Ramsey sequence on the transmon using the dispersive coupling.
- The full protocol is described in Ofek *et al.*, Nature 536, 441 (2016) and Vlastakis *et al.*, Science 342, 607 (2013).

Below we *simulate* this protocol: generate single-shot outcomes from the parity operator for a grid of displacements, then reconstruct $W$.""")))

# ── Cell 15  code: simulated Wigner tomography ───────────────────────────────
CELLS.append(code_cell(L(r"""# Simulate Wigner tomography for an even cat state
s_target = cat_state(alpha, N=N, phase=0)
rho_target = s_target.density_matrix()

# Tomography grid (coarser than display grid for speed)
tomo_pts = 20
xvec_tomo = np.linspace(-x_max, x_max, tomo_pts)
n_shots = 500   # Monte Carlo shot average per point (reduce for speed)

rng = np.random.default_rng(42)
W_tomo = np.zeros((tomo_pts, tomo_pts))
# Parity operator (-1)^n = diag(1, -1, 1, -1, ...)
parity_op = qt.Qobj(np.diag([(-1)**n for n in range(N)]))
parity_op.dims = [[N], [N]]

for i, ax_ in enumerate(xvec_tomo):
    for j, ap in enumerate(xvec_tomo):
        beta = ax_ + 1j * ap
        rho_disp = qt.displace(N, -beta) * rho_target * qt.displace(N, -beta).dag()
        p_parity = float(qt.expect(parity_op, rho_disp))   # exact <(-1)^n>
        # Add shot noise: p_parity is the true value; simulate binary outcomes
        sign = 1.0 if rng.random() < (1 + p_parity) / 2 else -1.0
        # Average over n_shots to reduce noise
        outcomes = rng.choice([-1.0, 1.0], size=n_shots,
                              p=[(1 - p_parity) / 2, (1 + p_parity) / 2])
        W_tomo[i, j] = (2 / np.pi) * outcomes.mean()

# Compare to exact Wigner
W_exact = qt.wigner(rho_target, xvec_tomo, xvec_tomo)

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
vmax = max(float(np.max(np.abs(W_exact))), 1e-9)
axes[0].contourf(xvec_tomo, xvec_tomo, W_exact, levels=30, cmap="RdBu_r",
                 vmin=-vmax, vmax=vmax)
axes[0].set_title("Exact Wigner function"); axes[0].set_aspect("equal")
axes[0].set_xlabel(r"$X$"); axes[0].set_ylabel(r"$P$")

axes[1].contourf(xvec_tomo, xvec_tomo, W_tomo, levels=30, cmap="RdBu_r",
                 vmin=-vmax, vmax=vmax)
axes[1].set_title(rf"Simulated tomography ({n_shots} shots/point)")
axes[1].set_aspect("equal")
axes[1].set_xlabel(r"$X$"); axes[1].set_ylabel(r"$P$")

plt.suptitle(rf"Wigner tomography: even cat $|\alpha|={alpha}$", fontsize=12)
plt.tight_layout(); plt.show()

# Negativity from tomography data
dx_t = xvec_tomo[1] - xvec_tomo[0]
neg_exact = float(np.sum(np.maximum(0, -W_exact)) * dx_t**2)
neg_tomo  = float(np.sum(np.maximum(0, -W_tomo))  * dx_t**2)
print(f"Wigner negativity — exact: {neg_exact:.3f},  tomography estimate: {neg_tomo:.3f}")
print(f"Noise degrades negativity estimate; increase n_shots to reduce error.")""")))

# ── assemble and write ────────────────────────────────────────────────────────
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
    "nb0b_phase_space.ipynb",
))
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)
print(f"Written {len(CELLS)} cells to {out_path}")
