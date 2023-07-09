"""
All `pystrel.operators.operator.Operator` implementations.
"""
import typing
import numpy.typing as npt
import numpy as np
import scipy.sparse as nps  # type: ignore

from ..sectors import Sectors
from .operator import Operator
from .hermitian_operator import HermitianOperator

# pylint: disable=C0103, R0903


class Operator_H(Operator):
    r"""
    Hamiltonian operator

    $$
        H = \sum_{ij} h_{ij}
    $$
    """
    tag = "H"
    name = "Hamiltonian"
    repr = "H"

    @staticmethod
    def build(
        sectors: Sectors,
        terms: dict,
        sparsity: typing.Literal["sparse", "dense"] = "dense",
        dtype: npt.DTypeLike = None,
        **kwargs,
    ) -> np.ndarray | nps.csr_array:
        return HermitianOperator.build(sectors, terms, sparsity, dtype)


class Operator_N(Operator):
    r"""
    Total particle number operator, given by

    $$
        N = \sum_i n_i
    $$
    """
    tag = "N"
    name = "total particle number operator"
    repr = "∑ᵢ nᵢ"

    @staticmethod
    def build(
        sectors: Sectors,
        terms: dict,
        sparsity: typing.Literal["sparse", "dense"] = "dense",
        dtype: npt.DTypeLike = None,
        **kwargs,
    ) -> np.ndarray | nps.csr_array:
        return HermitianOperator.build(sectors, {"mu": 1}, sparsity, dtype)


class Operator_n(Operator):
    """
    Particle number operator given by

    $$
        n_i.
    $$

    Parameters
    ----------
    **kwargs : dict, optional
        `i`: select site.

    """

    tag = "n"
    name = "particle number operator"
    repr = "nᵢ"

    @staticmethod
    def build(
        sectors: Sectors,
        terms: dict,
        sparsity: typing.Literal["sparse", "dense"] = "dense",
        dtype: npt.DTypeLike = None,
        **kwargs,
    ) -> np.ndarray | nps.csr_array:
        return HermitianOperator.build(
            sectors, {"epsilon": {kwargs["i"]: 1.0}}, sparsity, dtype
        )


# pylint: enable=C0103, R0903
