"""
plots.py
========
Plotting utilities for the SNAIL oscillator.

All functions return (fig, ax) and use a minimal paper-compatible style.
Pass save_path to write a PNG.
"""

import numpy as np
import matplotlib.pyplot as plt


def _apply_style(ax=None):
    """Minimal paper-style tick formatting."""
    if ax is not None:
        ax.tick_params(direction="in", top=True, right=True, which="both")
        for spine in ax.spines.values():
            spine.set_linewidth(0.75)


# ---------------------------------------------------------------------------
# SNAIL potential
# ---------------------------------------------------------------------------

def plot_snail_potential(
    phi,
    U,
    phi_m=None,
    labels=None,
    colors=None,
    ylim=None,
    xlim=None,
    xlabel=None,
    ylabel=None,
    mark_barriers: bool = True,
    dpi: int = 150,
    save_path=None,
):
    """
    Plot the SNAIL potential U(phi) vs phase.

    Parameters
    ----------
    phi : array-like
        Phase values in radians.
    U : array-like or list of array-like
        Potential (normalised by E_J).  Pass a list for multiple flux points.
    phi_m : float or list of float, optional
        Potential minimum positions; marked with a dot if given.
    labels : list of str, optional
    colors : list of colors, optional
    ylim, xlim : tuple, optional
    xlabel : str, optional  defaults to r"Phase $\\varphi\\,/\\,\\pi$"
    ylabel : str, optional  defaults to r"$U_s\\,/\\,E_J$"
    mark_barriers : bool
        Draw dashed vertical lines at ±3π from minimum.
    dpi : int
    save_path : str or Path, optional

    Returns
    -------
    fig, ax
    """
    phi = np.asarray(phi, dtype=float)

    if isinstance(U, (list, tuple)) and isinstance(U[0], (list, np.ndarray)):
        Us = [np.asarray(u, dtype=float) for u in U]
    else:
        Us = [np.asarray(U, dtype=float)]

    fig, ax = plt.subplots(dpi=dpi)
    _apply_style(ax)

    for i, u in enumerate(Us):
        kw = {"linewidth": 1.25}
        if colors is not None:
            kw["color"] = colors[i]
        if labels is not None:
            kw["label"] = labels[i]
        ax.plot(phi / np.pi, u, **kw)

    if phi_m is not None:
        phi_ms = [float(phi_m)] if np.isscalar(phi_m) else [float(v) for v in phi_m]
        for i, pm in enumerate(phi_ms):
            u_at_min = Us[i % len(Us)]
            idx = int(np.argmin(np.abs(phi - pm)))
            kw_dot = {"marker": "o", "markersize": 4, "ls": "", "zorder": 5}
            if colors is not None:
                kw_dot["color"] = colors[i % len(colors)]
            ax.plot([pm / np.pi], [float(u_at_min[idx])], **kw_dot)

    if mark_barriers and phi_m is not None:
        pm0 = float(phi_m) if np.isscalar(phi_m) else float(phi_ms[0])
        for offset in [-3 * np.pi, 3 * np.pi]:
            ax.axvline(float((pm0 + offset) / np.pi), color="gray",
                       linestyle="--", linewidth=0.7, alpha=0.6,
                       label=r"$\pm 3\pi$ barrier" if offset > 0 else None)

    ax.set_xlabel(xlabel or r"Phase $\varphi\,/\,\pi$")
    ax.set_ylabel(ylabel or r"$U_s\,/\,E_J$")
    if xlim is not None:
        ax.set_xlim(float(xlim[0]) / np.pi, float(xlim[1]) / np.pi)
    if ylim is not None:
        ax.set_ylim(ylim)
    if labels is not None or mark_barriers:
        ax.legend(fontsize=8, frameon=False)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300)
    return fig, ax


# ---------------------------------------------------------------------------
# Nonlinearities vs flux
# ---------------------------------------------------------------------------

def plot_snail_nonlinearities_vs_flux(
    phi_e,
    g3,
    g4,
    g5=None,
    units: str = "MHz",
    ylim=None,
    colors=None,
    dpi: int = 150,
    save_path=None,
):
    """
    Plot SNAIL nonlinear coefficients g3, g4, (g5) vs external flux.

    Parameters
    ----------
    phi_e : array-like
        External flux in radians (0 to 2*pi).
    g3, g4 : array-like
        Cubic and quartic nonlinearities in GHz.
    g5 : array-like, optional
        Quintic nonlinearity in GHz.
    units : {'MHz', 'GHz'}
    ylim : tuple, optional
    colors : list of 2 or 3 colors, optional
    dpi : int
    save_path : str or Path, optional

    Returns
    -------
    fig, ax
    """
    scale = 1e3 if units == "MHz" else 1.0
    phi_e = np.asarray(phi_e, dtype=float) / (2 * np.pi)

    fig, ax = plt.subplots(dpi=dpi)
    _apply_style(ax)

    _colors = colors or ["C0", "C1", "C2"]
    curves = [(g3, r"$g_3$"), (g4, r"$g_4$")]
    if g5 is not None:
        curves.append((g5, r"$g_5$"))

    for (g, lbl), col in zip(curves, _colors):
        ax.plot(phi_e, np.asarray(g, dtype=float) * scale,
                linewidth=1.25, label=lbl, color=col)
    ax.axhline(0, color="k", linewidth=0.5, linestyle="--")

    ax.set_xlabel(r"External Flux, $\Phi_e/\Phi_0$")
    ax.set_ylabel(r"Nonlinearity$\,/2\pi$ " + f"({units})")
    ax.set_xlim(0, 0.5)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend(fontsize=8, frameon=False)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300)
    return fig, ax


# ---------------------------------------------------------------------------
# Circuit parameters vs flux
# ---------------------------------------------------------------------------

def plot_snail_circuit_params_vs_flux(
    phi_e,
    omega_c,
    phi_c,
    alpha_c,
    xi_crit,
    dpi: int = 150,
    save_path=None,
):
    """
    Four-panel figure: omega_c, phi_c, alpha_c, xi_crit vs external flux.

    Parameters
    ----------
    phi_e : array-like   External flux in radians.
    omega_c : array-like   Coupler frequency in GHz.
    phi_c : array-like     Zero-point phase fluctuation.
    alpha_c : array-like   Anharmonicity in GHz.
    xi_crit : array-like   Critical pump amplitude.

    Returns
    -------
    fig, axes  (2x2 grid)
    """
    x = np.asarray(phi_e, dtype=float) / (2 * np.pi)

    fig, axes = plt.subplots(2, 2, dpi=dpi, figsize=(6.8, 4.0))
    for ax in axes.flat:
        _apply_style(ax)

    panels = [
        (axes[0, 0], omega_c,                    r"$\omega_c/2\pi$ (GHz)",    "C0"),
        (axes[0, 1], phi_c,                      r"$\varphi_c$ (ZPF)",        "C1"),
        (axes[1, 0], np.array(alpha_c) * 1e3,   r"$\alpha_c/2\pi$ (MHz)",    "C2"),
        (axes[1, 1], xi_crit,                    r"$|\xi_\mathrm{crit}|$",    "C3"),
    ]
    for ax, y, ylabel, col in panels:
        ax.plot(x, y, linewidth=1.25, color=col)
        ax.set_xlabel(r"$\Phi_e/\Phi_0$")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, 0.5)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300)
    return fig, axes


# ---------------------------------------------------------------------------
# Beam-splitter rate vs pump amplitude
# ---------------------------------------------------------------------------

def plot_snail_gbs_vs_xi(
    xi,
    gbs_curves,
    labels=None,
    colors=None,
    units: str = "MHz",
    xlim=None,
    ylim=None,
    dpi: int = 150,
    save_path=None,
):
    """
    Plot beam-splitter rate g_bs vs dimensionless pump amplitude xi.

    Parameters
    ----------
    xi : array-like
    gbs_curves : array-like or list of array-like
        Beam-splitter rate(s) in GHz.
    labels : list of str, optional
    colors : list of colors, optional
    units : {'MHz', 'GHz'}
    xlim, ylim : tuple, optional

    Returns
    -------
    fig, ax
    """
    scale = 1e3 if units == "MHz" else 1.0
    xi = np.asarray(xi, dtype=float)

    if not isinstance(gbs_curves, (list, tuple)):
        gbs_curves = [gbs_curves]

    fig, ax = plt.subplots(dpi=dpi)
    _apply_style(ax)

    for i, gbs in enumerate(gbs_curves):
        kw = {"linewidth": 1.25}
        if colors is not None:
            kw["color"] = colors[i]
        if labels is not None:
            kw["label"] = labels[i]
        ax.plot(xi, np.asarray(gbs, dtype=float) * scale, **kw)

    ax.set_xlabel(r"Pump amplitude $|\xi|$")
    ax.set_ylabel(r"$g_\mathrm{bs}/2\pi$ " + f"({units})")
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if labels is not None:
        ax.legend(fontsize=8, frameon=False)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300)
    return fig, ax
