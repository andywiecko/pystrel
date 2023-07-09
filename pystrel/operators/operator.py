"""
Template class for operator implementations.
All implementations can be found here: `pystrel.operators.impl`.
"""
import typing
import numpy.typing as npt
import numpy as np
import scipy.sparse as nps  # type: ignore
from ..sectors import Sectors


class Operator:  # pylint: disable=R0903
    """Abstract class for operator implementations"""

    tag: str
    name: str
    repr: str

    @staticmethod  # pylint: disable=W0613
    def build(
        sectors: Sectors,
        terms: dict,
        sparsity: typing.Literal["sparse", "dense"] = "dense",
        dtype: npt.DTypeLike = None,
        **kwargs,
    ) -> np.ndarray | nps.csr_array:
        """
        Construct matrix for given operator.

        Parameters
        ----------
        sectors : Sectors
            Sectors in which operator should be constructed.
        terms : dict
            Terms of the model in which operator should be constructed.
        sparsity : typing.Literal["sparse", "dense"], optional
            Matrix sparsity type which is used, by default "dense".
            If "sparse" is selected, returns matrix in CSR format.
        dtype : npt.DTypeLike, optional
            Any object that can be interpreted as a numpy data type.
        **kwargs : dict, optional
            Additional operators arguments.

        Returns
        -------
        np.ndarray | nps.csr_array
            Constructed operator.
        """
        raise NotImplementedError()
