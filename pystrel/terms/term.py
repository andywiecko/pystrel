"""
Template class for term implementations.
All implementations can be found here: `pystrel.terms.impl`.
"""
import typing
import numpy as np
import scipy.sparse as nps # type: ignore

class Term: # pylint: disable=R0903
    """Abstract class for interaction implementation"""
    tag: str
    particle_type: typing.Literal[
        'spinless fermions',
        'spins 1/2',
        'spinfull fermions'
    ]
    ensemble: typing.Literal[
        'canonical',
        'parity grand canonical',
        'grand canonical'
    ]
    repr: str
    mixing_rank: typing.Literal[0, 1, 2]

    @staticmethod  # pylint: disable=W0613
    def apply(params: dict, matrix: np.ndarray | nps.dok_matrix, sector: tuple[int, int]):
        """
        Applies terms on given `matrix` using `params`.
        Terms are applied only on triangle upper part of the `matrix`.
        When operator mixing rank is non-zero, then `matrix` should have shape
        `size(sector(L, N)) × size(sector(L, N + rank))`.

        Parameters
        ----------
        params : dict
            Dictionary of given parameters, e.g. `{(0, 1): 1.0, (1, 2): 2.0}`.
        matrix : npt.ndarray | nps.dok_matrix
            View on matrix on which terms should be applied.
        sector : tuple[int, int]
            Corresponding particle sector for given `matrix`.
        """
        raise NotImplementedError()
