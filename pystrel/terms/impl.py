"""
All `pystrel.terms.term.Term` implementations.
"""
import numpy as np
import scipy.sparse as nps  # type: ignore
from .. import combinadics
from .term import Term

# pylint: disable=C0103, R0903


class Term_Jz(Term):
    r"""
    Implementation of the spin-spin interaction Jᶻᵢⱼ σᶻᵢσᶻⱼ for
    spin 1/2 particles, given by

    $$
    \sum_{i,j} J_{ij}^z \, \sigma_i^z \sigma_j^z.
    $$
    """
    tag = "Jz"
    particle_type = "spins 1/2"
    ensemble = "canonical"
    repr = "∑ᵢⱼ Jᶻᵢⱼ σᶻᵢσᶻⱼ"
    mixing_rank = 0

    @staticmethod
    def apply(params, matrix, sector):
        for x in range(matrix.shape[0]):
            s = combinadics.tostate(x, sector[0], sector[1])
            for (i, j), Jz in params.items():
                matrix[x, x] += (+1 if s[i] == s[j] else -1) * Jz
        return matrix


class Term_hz(Term):
    r"""
    Implementation of the $z$-magnetic field hᶻᵢ σᶻᵢ operator for
    spin 1/2 particles, given by

    $$
    \sum_{i} h_{i}^z \, \sigma_i^z.
    $$
    """
    tag = "hz"
    particle_type = "spins 1/2"
    ensemble = "canonical"
    repr = "∑ᵢ hᶻᵢ σᶻᵢ"
    mixing_rank = 0

    @staticmethod
    def apply(params, matrix, sector):
        for x in range(matrix.shape[0]):
            s = combinadics.tostate(x, sector[0], sector[1])
            for i, hz in params.items():
                matrix[x, x] += (+1 if s[i] == "1" else -1) * hz
        return matrix


class Term_gamma(Term):
    r"""
    Implementation of the spin hop γᵢⱼ σ⁺ᵢσ⁻ⱼ operator for
    spin 1/2 particles, given by

    $$
    \sum_{i,j} \left(\gamma_{ij} \, \sigma_i^+\sigma_j^- + \text{h.c.}\right).
    $$
    """
    tag = "gamma"
    particle_type = "spins 1/2"
    ensemble = "canonical"
    repr = "∑ᵢⱼ (γᵢⱼ σ⁺ᵢσ⁻ⱼ + h.c.)"
    mixing_rank = 0

    @staticmethod
    def apply(params, matrix, sector):
        for x in range(matrix.shape[0]):
            s0 = combinadics.tostate(x, sector[0], sector[1])
            for (i, j), gamma in params.items():
                if s0[i] == "1" and s0[j] == "0":
                    s1 = s0[:i] + "0" + s0[i + 1 :]
                    s1 = s1[:j] + "1" + s1[j + 1 :]
                    y = combinadics.tonumber(s1)
                    if x < y:
                        matrix[x, y] += gamma
                    else:
                        matrix[y, x] += np.conj(gamma)
        return matrix


class Term_hx(Term):
    r"""
    Implementation of the $x$-magnetic field hˣᵢ σˣᵢ operator for
    spin 1/2 particles, given by

    $$
    \sum_{i} h_{i}^x \, \sigma_i^x.
    $$
    """
    tag = "hx"
    particle_type = "spins 1/2"
    ensemble = "grand canonical"
    repr = "∑ᵢ hˣᵢ σˣᵢ"
    mixing_rank = 1

    @staticmethod
    def apply(params, matrix, sector):
        for x in range(matrix.shape[0]):
            s0 = combinadics.tostate(x, sector[0], sector[1])
            for i, gamma in params.items():
                if s0[i] == "0":
                    s1 = s0[:i] + "1" + s0[i + 1 :]
                    y = combinadics.tonumber(s1)
                    matrix[x, y] += gamma
        return matrix


class Term_t(Term):
    r"""
    Implementation of the hopping operator tᵢⱼ a†ᵢaⱼ for
    spinless fermions, given by

    $$
    \sum_{i,j} \left(t_{ij} \, a_i^\dagger a_j + \text{h.c.}\right).
    $$
    """
    tag = "t"
    particle_type = "spinless fermions"
    ensemble = "canonical"
    repr = "∑ᵢⱼ (tᵢⱼ a†ᵢaⱼ + h.c.)"
    mixing_rank = 0

    @staticmethod
    def apply(params, matrix, sector):
        for x in range(matrix.shape[0]):
            s0 = combinadics.tostate(x, sector[0], sector[1])
            for (i, j), t in params.items():
                if s0[i] == "1" and s0[j] == "0":
                    s1 = s0[:i] + "0" + s0[i + 1 :]
                    s1 = s1[:j] + "1" + s1[j + 1 :]
                    y = combinadics.tonumber(s1)
                    sign = (-1.0) ** combinadics.count_particles_between(s0, i, j)
                    if x < y:
                        matrix[x, y] += sign * t
                    else:
                        matrix[y, x] += sign * np.conj(t)
        return matrix


class Term_V(Term):
    r"""
    Implementation of the 2-body operator Vᵢⱼ nᵢnⱼ for
    spinless fermions, given by

    $$
    \sum_{i,j} V_{ij} \, n_i n_j.
    $$
    """
    tag = "V"
    particle_type = "spinless fermions"
    ensemble = "canonical"
    repr = "∑ᵢⱼ Vᵢⱼ nᵢnⱼ"
    mixing_rank = 0

    @staticmethod
    def apply(params, matrix, sector):
        for x in range(matrix.shape[0]):
            s = combinadics.tostate(x, sector[0], sector[1])
            for (i, j), V in params.items():
                if s[i] == "1" and s[j] == "1":
                    matrix[x, x] += V
        return matrix


class Term_Delta(Term):
    r"""
    Implementation of the pair operator Δᵢⱼ a†ᵢa†ⱼ for
    spinless fermions, given by

    $$
    \sum_{i,j} \left(\Delta_{ij} \, a_i^\dagger a_j^\dagger + \text{h.c.}\right).
    $$
    """
    tag = "Delta"
    particle_type = "spinless fermions"
    ensemble = "parity grand canonical"
    repr = "∑ᵢⱼ (Δᵢⱼ a†ᵢa†ⱼ + h.c.)"
    mixing_rank = 2

    @staticmethod
    def apply(params, matrix, sector):
        for x in range(matrix.shape[0]):
            s0 = combinadics.tostate(x, sector[0], sector[1])
            for (i, j), Delta in params.items():
                if s0[i] == "0" and s0[j] == "0":
                    s1 = s0[:i] + "1" + s0[i + 1 :]
                    s1 = s1[:j] + "1" + s1[j + 1 :]
                    y = combinadics.tonumber(s1)
                    sign = (-1.0) ** combinadics.count_particles_between(s0, i, j)
                    sign = sign if i < j else -sign
                    matrix[x, y] += sign * Delta
        return matrix


class Term_epsilon(Term):
    r"""
    Implementation of chemical potential like operator εᵢ nᵢ for
    spinless fermions, given by

    $$
    \sum_{i} \epsilon_{i} \, n_i .
    $$
    """
    tag = "epsilon"
    particle_type = "spinless fermions"
    ensemble = "canonical"
    repr = "∑ᵢ εᵢ nᵢ"
    mixing_rank = 0

    @staticmethod
    def apply(params, matrix, sector):
        for x in range(matrix.shape[0]):
            s = combinadics.tostate(x, sector[0], sector[1])
            for i, mu in params.items():
                if s[i] == "1":
                    matrix[x, x] += mu
        return matrix


class Term_mu(Term):
    r"""
    Implementation of chemical potential operator μ N for
    spinless fermions, given by

    $$
    \mu N = \mu \sum_{i} \, n_i .
    $$
    """
    tag = "mu"
    particle_type = "spinless fermions"
    ensemble = "canonical"
    repr = "μ ∑ᵢ nᵢ"
    mixing_rank = 0

    @staticmethod
    def apply(params, matrix, sector):
        if isinstance(matrix, np.ndarray):
            np.fill_diagonal(matrix, matrix.diagonal() + params * sector[1])

        elif isinstance(matrix, nps.lil_array):
            matrix.setdiag(matrix.diagonal() + params * sector[1])

        return matrix


# pylint: enable=C0103, R0903
