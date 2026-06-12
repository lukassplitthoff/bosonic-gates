"""
State preparation for bosonic quantum states.

Provides functions to generate various quantum states of a bosonic mode:
coherent, Fock, squeezed, thermal, cat, displaced-squeezed, binomial, and
custom Fock superpositions.
"""

import math
import numpy as np
import qutip as qt
from typing import Union, Optional


class BosonicState:
    """
    Container for a bosonic quantum state.

    Attributes
    ----------
    state : qt.Qobj
        The QuTiP quantum state (ket or density matrix).
    N : int
        Hilbert space truncation dimension.
    """

    def __init__(self, state: qt.Qobj, N: int):
        self.state = state
        self.N = N

    def __repr__(self):
        return f"{self.__class__.__name__}(N={self.N})"

    def density_matrix(self) -> qt.Qobj:
        """Return the density matrix representation."""
        if self.state.type == 'ket':
            return qt.ket2dm(self.state)
        return self.state

    def purity(self) -> float:
        """Calculate the purity Tr(ρ²)."""
        rho = self.density_matrix()
        return np.real((rho * rho).tr())

    def photon_number(self) -> float:
        """Calculate the mean photon number ⟨a†a⟩."""
        a = qt.destroy(self.N)
        n_op = a.dag() * a
        rho = self.density_matrix()
        return np.real(qt.expect(n_op, rho))

    def fock_distribution(self) -> np.ndarray:
        """Return the Fock state distribution p_n = ⟨n|ρ|n⟩."""
        rho = self.density_matrix()
        return np.array([np.real(rho[n, n]) for n in range(self.N)])


def coherent_state(alpha: complex, N: int = 50) -> BosonicState:
    """
    Generate a coherent state |alpha⟩.

    Coherent states are eigenstates of the annihilation operator and are
    the closest quantum analogs to classical electromagnetic waves.

    Parameters
    ----------
    alpha : complex
        Complex amplitude of the coherent state.
    N : int
        Hilbert space dimension (default: 50).

    Returns
    -------
    BosonicState

    Example
    -------
    >>> state = coherent_state(2.0 + 1j, N=50)
    """
    state = qt.coherent(N, alpha)
    return BosonicState(state, N)


def fock_state(n: int, N: Optional[int] = None) -> BosonicState:
    """
    Generate a Fock (number) state |n⟩.

    Parameters
    ----------
    n : int
        Photon number.
    N : int, optional
        Hilbert space dimension (default: n + 20).

    Returns
    -------
    BosonicState

    Example
    -------
    >>> state = fock_state(5, N=50)
    """
    if N is None:
        N = n + 20
    if n >= N:
        raise ValueError(f"Photon number n={n} must be less than dimension N={N}")
    state = qt.fock(N, n)
    return BosonicState(state, N)


def squeezed_state(r: float, phi: float = 0, N: int = 50) -> BosonicState:
    """
    Generate a squeezed vacuum state S(r e^{iφ})|0⟩.

    Squeezed states have reduced quantum noise in one quadrature at the
    expense of increased noise in the conjugate quadrature.

    Parameters
    ----------
    r : float
        Squeezing magnitude.
    phi : float
        Squeezing angle (default: 0).
    N : int
        Hilbert space dimension (default: 50).

    Returns
    -------
    BosonicState

    Example
    -------
    >>> state = squeezed_state(0.5, phi=np.pi/4, N=50)
    """
    state = qt.squeeze(N, r * np.exp(1j * phi)) * qt.fock(N, 0)
    return BosonicState(state, N)


def thermal_state(n_mean: float, N: int = 50) -> BosonicState:
    """
    Generate a thermal (mixed) state with mean photon number n_mean.

    Parameters
    ----------
    n_mean : float
        Mean photon number (Bose-Einstein distribution).
    N : int
        Hilbert space dimension (default: 50).

    Returns
    -------
    BosonicState (density matrix)

    Example
    -------
    >>> state = thermal_state(2.5, N=50)
    """
    state = qt.thermal_dm(N, n_mean)
    return BosonicState(state, N)


def cat_state(alpha: complex, N: int = 50, phase: float = 0) -> BosonicState:
    """
    Generate a cat state (|α⟩ + e^{iφ}|−α⟩) / N.

    Parameters
    ----------
    alpha : complex
        Amplitude of the coherent states.
    N : int
        Hilbert space dimension (default: 50).
    phase : float
        Relative phase φ between |α⟩ and |−α⟩ (default: 0 → even cat).
        phase=π gives the odd cat state.

    Returns
    -------
    BosonicState

    Example
    -------
    >>> even_cat = cat_state(2.0, N=50, phase=0)
    >>> odd_cat = cat_state(2.0, N=50, phase=np.pi)
    """
    psi_plus = qt.coherent(N, alpha)
    psi_minus = qt.coherent(N, -alpha)
    state = (psi_plus + np.exp(1j * phase) * psi_minus).unit()
    return BosonicState(state, N)


def displaced_squeezed_state(alpha: complex, r: float, phi: float = 0, N: int = 50) -> BosonicState:
    """
    Generate a displaced squeezed state D(α) S(r e^{iφ})|0⟩.

    Parameters
    ----------
    alpha : complex
        Displacement amplitude.
    r : float
        Squeezing magnitude.
    phi : float
        Squeezing angle (default: 0).
    N : int
        Hilbert space dimension (default: 50).

    Returns
    -------
    BosonicState

    Example
    -------
    >>> state = displaced_squeezed_state(2.0, r=0.5, phi=0, N=50)
    """
    state = qt.displace(N, alpha) * qt.squeeze(N, r * np.exp(1j * phi)) * qt.fock(N, 0)
    return BosonicState(state, N)


def binomial_state(N: int, theta: float, n_max: Optional[int] = None) -> BosonicState:
    """
    Generate a binomial state.

    Binomial states are discrete superpositions of Fock states with binomial
    coefficients and can interpolate between Fock and coherent states.

    Parameters
    ----------
    N : int
        Hilbert space dimension.
    theta : float
        Binomial parameter in [0, π/2].
    n_max : int, optional
        Maximum Fock component (default: N-1).

    Returns
    -------
    BosonicState

    Example
    -------
    >>> state = binomial_state(50, theta=np.pi/4)
    """
    if n_max is None:
        n_max = N - 1

    state = sum(
        np.sqrt(math.comb(n_max, n)) * np.cos(theta)**n * np.sin(theta)**(n_max - n) * qt.fock(N, n)
        for n in range(n_max + 1)
    )
    state = state.unit()
    return BosonicState(state, N)


def fock_superposition(fock_numbers: list, coefficients: list, N: Optional[int] = None) -> BosonicState:
    """
    Generate a custom superposition of Fock states.

    Parameters
    ----------
    fock_numbers : list of int
        Fock state numbers to include.
    coefficients : list of complex
        Complex coefficients for each Fock state.
    N : int, optional
        Hilbert space dimension (default: max(fock_numbers) + 20).

    Returns
    -------
    BosonicState

    Example
    -------
    >>> state = fock_superposition([0, 1, 2], [1, 1j, -1])
    """
    if len(fock_numbers) != len(coefficients):
        raise ValueError("fock_numbers and coefficients must have the same length")

    if N is None:
        N = max(fock_numbers) + 20

    state = sum(c * qt.fock(N, n) for n, c in zip(fock_numbers, coefficients))
    state = state.unit()
    return BosonicState(state, N)
