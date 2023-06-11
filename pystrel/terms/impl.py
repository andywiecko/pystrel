"""
All `pystrel.terms.term.Term` implementations.
"""
import numpy as np
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
    tag = 'Jz'
    particle_type = 'spins 1/2'
    ensemble = 'canonical'
    repr = '∑ᵢⱼ Jᶻᵢⱼ σᶻᵢσᶻⱼ'
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
    tag = 'hz'
    particle_type = 'spins 1/2'
    ensemble = 'canonical'
    repr = '∑ᵢ hᶻᵢ σᶻᵢ'
    mixing_rank = 0

    @staticmethod
    def apply(params, matrix, sector):
        for x in range(matrix.shape[0]):
            s = combinadics.tostate(x, sector[0], sector[1])
            for i, hz in params.items():
                matrix[x, x] += (+1 if s[i] == '1' else -1) * hz
        return matrix


class Term_gamma(Term):
    r"""
    Implementation of the spin hop γᵢⱼ σ⁺ᵢσ⁻ⱼ operator for
    spin 1/2 particles, given by

    $$
    \sum_{i,j} \left(\gamma_{ij} \, \sigma_i^+\sigma_j^- + \text{h.c.}\right).
    $$
    """
    tag = 'gamma'
    particle_type = 'spins 1/2'
    ensemble = 'canonical'
    repr = '∑ᵢⱼ (γᵢⱼ σ⁺ᵢσ⁻ⱼ + h.c.)'
    mixing_rank = 0

    @staticmethod
    def apply(params, matrix, sector):
        for x in range(matrix.shape[0]):
            s0 = combinadics.tostate(x, sector[0], sector[1])
            for (i, j), gamma in params.items():
                if s0[i] == '1' and s0[j] == '0':
                    s1 = s0[:i] + '0' + s0[i+1:]
                    s1 = s1[:j] + '1' + s1[j+1:]
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
    tag = 'hx'
    particle_type = 'spins 1/2'
    ensemble = 'grand canonical'
    repr = '∑ᵢ hˣᵢ σˣᵢ'
    mixing_rank = 1

    @staticmethod
    def apply(params, matrix, sector):
        for x in range(matrix.shape[0]):
            s0 = combinadics.tostate(x, sector[0], sector[1])
            for i, gamma in params.items():
                if s0[i] == '0':
                    s1 = s0[:i] + '1' + s0[i+1:]
                    y = combinadics.tonumber(s1)
                    matrix[x, y] += gamma
        return matrix

# pylint: enable=C0103, R0903
