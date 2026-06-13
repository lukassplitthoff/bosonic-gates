"""
DrivenKerrConfig — single source of truth for all physical parameters.

All frequencies and rates are in angular-frequency units (ℏ = 1, rad/s).
Every parameter has an explicit note on the 2π convention so that mixing
ordinary frequency (GHz) and angular frequency (rad/s) is caught immediately.

Unit-convention note
--------------------
This module uses rad/s (angular frequencies), while ``hamiltonians`` and
``snail`` use plain GHz with ħ=1.  To connect them, multiply by 2π:

    from bosonic_gates.hamiltonians import resonator_hamiltonian
    from bosonic_gates.driven_kerr import DrivenKerrConfig

    w_GHz = 5.0                    # used in resonator_hamiltonian(w=5.0, ...)
    omega_rad_s = 2*np.pi * w_GHz  # used in DrivenKerrConfig(omega0=omega_rad_s)

Example
-------
>>> from bosonic_gates.driven_kerr import DrivenKerrConfig
>>> cfg = DrivenKerrConfig()
>>> cfg.omega12 / (2 * np.pi) / 1e9   # ≈ 4.8 GHz
"""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class DrivenKerrConfig:
    # ---- Hilbert space ----
    N: int = 8
    """Fock truncation. Converge by increasing."""

    # ---- System frequencies (rad/s; multiply by 2π from GHz) ----
    omega0: float = 2 * np.pi * 5.0e9
    """Bare 0→1 angular frequency. 2π·5.0 GHz."""

    K: float = 2 * np.pi * 0.2e9
    """Kerr nonlinearity > 0. Convention: ω₁₂ = ω₀ − K < ω₀. 2π·0.2 GHz."""

    omega_d: float = None  # type: ignore[assignment]
    """Drive angular frequency. Defaults to ω₀ (resonant drive) in __post_init__."""

    # ---- Drive ----
    epsilon: float = 0.0
    """Drive amplitude (rad/s). The control parameter to sweep."""

    # ---- Bath / dissipation ----
    kappa: float = 2 * np.pi * 1.0e6
    """Peak single-photon loss rate (on resonance with Lorentzian feature). 2π·1 MHz."""

    Gamma: float = 2 * np.pi * 5.0e6
    """FWHM of the Lorentzian bath feature. 2π·5 MHz.
    Set Gamma → very large (e.g. 1e15) for the white-bath limit."""

    nbar: float = 0.02
    """Thermal occupation of the mode. Dimensionless. At mK: nbar ≪ 1."""

    gamma_phi: float = 2 * np.pi * 0.05e6
    """Pure dephasing rate (flat/white, independent of ω). 2π·0.05 MHz."""

    omega_f: float = None  # type: ignore[assignment]
    """Center angular frequency of the Lorentzian bath feature.
    Defaults to ω₁₂ + ω_d so that a rising ε sweeps a dressed sideband onto ω_f."""

    # ---- Floquet settings ----
    k_max: int = 5
    """Sideband truncation for Floquet–Markov rate sum."""

    n_t: int = 512
    """Number of time points per drive period for Floquet mode propagation."""

    def __post_init__(self) -> None:
        if self.omega_d is None:
            self.omega_d = self.omega0
        if self.omega_f is None:
            self.omega_f = self.omega12 + self.omega_d

    # ---- Derived quantities ----

    @property
    def omega12(self) -> float:
        """1→2 angular frequency: ω₀ − K."""
        return self.omega0 - self.K

    @property
    def T_d(self) -> float:
        """Drive period: 2π / ω_d (seconds)."""
        return 2 * np.pi / self.omega_d

    def replace(self, **kwargs) -> "DrivenKerrConfig":
        """Return a copy with selected fields overridden.

        When ``omega_d`` is changed but ``omega_f`` is not provided, this method
        checks whether ``omega_f`` was using the default sideband formula
        ``omega_f = omega12 + omega_d``.  If so, it resets ``omega_f`` to
        ``None`` so ``__post_init__`` recomputes it from the new ``omega_d``.
        If ``omega_f`` was explicitly set to a different value, it is preserved.

        Example
        -------
        >>> cfg2 = cfg.replace(epsilon=2*np.pi*0.1e9, N=12)
        >>> # Change drive frequency — omega_f is recomputed automatically if
        >>> # it was using the default formula:
        >>> cfg3 = cfg.replace(omega_d=2*np.pi*4.8e9)
        """
        import dataclasses
        d = dataclasses.asdict(self)
        # If omega_d is being changed and omega_f is not explicitly provided,
        # check whether omega_f was auto-computed (matches the current default
        # formula omega12 + omega_d).  If so, reset to None so __post_init__
        # recomputes it from the new omega_d.  If omega_f was explicitly set
        # to a different value, leave it unchanged.
        if "omega_d" in kwargs and "omega_f" not in kwargs:
            auto_omega_f = self.omega12 + self.omega_d
            if abs(self.omega_f - auto_omega_f) < 1e-6 * abs(auto_omega_f):
                d["omega_f"] = None
        d.update(kwargs)
        return DrivenKerrConfig(**d)
