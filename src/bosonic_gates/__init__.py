"""
bosonic_gates — tutorial library for bosonic quantum computing with superconducting circuits.

Subpackages
-----------
states          : bosonic state preparation (coherent, Fock, squeezed, cat, ...)
operations      : single- and two-mode operations (displacement, squeezing, beam splitter, ...)
measurements    : expectation values and measurement observables
visualization   : phase-space and Fock-distribution plots
hamiltonians    : qubit and resonator Hamiltonians (transmon, harmonic oscillator)
snail           : SNAIL potential and circuit parameters
driven_kerr     : driven anharmonic oscillator with Lindblad / Redfield / Floquet-Markov
gates           : SNAP gates, ECD gates, optimal-control wrappers
error_budget    : per-channel infidelity error budget framework
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
