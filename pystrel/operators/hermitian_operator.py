"""
Module with generic utility for hermitian operator construction.
"""
import typing
import numpy as np
import numpy.typing as npt
import scipy.sparse as nps  # type: ignore

from ..sectors import Sectors
from ..terms import utils as terms_utils
from ..terms.typing import Terms
from ..sparse import Sparse


class HermitianOperator:  # pylint: disable=R0903
    """Hermitian operator implementation"""

    @staticmethod
    def __build_local_sectors(
        sectors: Sectors, terms: Terms, matrix: np.ndarray | Sparse
    ):
        for start, end, sector in sectors:
            terms_utils.apply(
                terms=terms,
                matrix=matrix[start:end, start:end],
                sector=sector,
                rank=0,
            )

    @staticmethod
    def __build_mixing_sectors(
        sectors: Sectors, terms: Terms, matrix: np.ndarray | Sparse
    ):
        for (start0, end0, sector0), (
            start1,
            end1,
            sector1,
        ) in sectors.mixing_iter():
            terms_utils.apply(
                terms=terms,
                matrix=matrix[start0:end0, start1:end1],
                sector=sector0,
                rank=sector1[1] - sector0[1],
            )

    @staticmethod
    def build(
        sectors: Sectors,
        terms: Terms,
        sparsity: typing.Literal["sparse", "dense"] = "dense",
        dtype: npt.DTypeLike = None,
    ) -> np.ndarray | nps.csr_array:
        """
        Construct hermitian operator.

        Parameters
        ----------
        sectors : Sectors
            Sectors in which operator should be constructed.
        terms : Terms
            Dictionary with terms.
            See `pystrel.terms` for more details.
        sparsity : typing.Literal["sparse", "dense"]
            Matrix sparsity type which is used, by default "dense".
            If "sparse" is selected, returns matrix in CSR format.
        dtype : npt.DTypeLike
            Any object that can be interpreted as a numpy data type.

        Returns
        -------
        np.ndarray | nps.csr_array
            Constructed hermitian operator.

        Raises
        ------
        ValueError
            When provided `sparsity` is not supported.
        """
        shape = (sectors.size, sectors.size)
        match sparsity:
            case "dense":
                matrix = np.zeros(shape, dtype=dtype)
                HermitianOperator.__build_local_sectors(sectors, terms, matrix)
                HermitianOperator.__build_mixing_sectors(sectors, terms, matrix)
                matrix = matrix + np.conjugate(np.triu(matrix, 1)).T
                return matrix

            case "sparse":
                sparse = Sparse(shape)
                HermitianOperator.__build_local_sectors(sectors, terms, sparse)
                HermitianOperator.__build_mixing_sectors(sectors, terms, sparse)
                csr = sparse.to_csr(dtype=dtype)
                return csr + nps.triu(csr, 1).H

            case _:
                raise ValueError()
