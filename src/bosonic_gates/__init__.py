"""
bosonic_gates — tutorial library for bosonic quantum computing with superconducting circuits.

Subpackages
-----------
states          : bosonic state preparation (coherent, Fock, squeezed, cat, ...)
hamiltonians    : qubit and resonator Hamiltonians (transmon, harmonic oscillator)
snail           : SNAIL potential and circuit parameters
driven_kerr     : driven anharmonic oscillator with Lindblad / Redfield / Floquet-Markov
gates           : SNAP gates, ECD gates, optimal-control wrappers
error_budget    : per-channel infidelity error budget framework

Note on units
-------------
``hamiltonians`` and ``snail`` use plain GHz with ħ=1 (e.g. ``w=5.0`` for 5 GHz).
``driven_kerr`` uses angular frequencies in rad/s (multiply GHz by 2π):
    omega0 = 2 * np.pi * 5e9   # 5 GHz in rad/s

Quick start
-----------
>>> from bosonic_gates import coherent_state
>>> from bosonic_gates.hamiltonians import resonator_hamiltonian
>>> from bosonic_gates.driven_kerr import DrivenKerrConfig, run_lindblad
>>> from bosonic_gates.gates import snap_operator
>>> from bosonic_gates.error_budget import compute_error_budget
"""

from .states import (
    BosonicState,
    coherent_state,
    fock_state,
    squeezed_state,
    thermal_state,
    cat_state,
    displaced_squeezed_state,
    binomial_state,
    fock_superposition,
)

__version__ = "0.1.0"
