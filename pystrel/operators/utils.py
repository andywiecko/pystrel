"""
Utilities related to operators.
"""
import typing
import inspect
import sys
import numpy as np
import numpy.typing as npt
import scipy.sparse as nps  # type: ignore
from . import impl
from .operator import Operator
from ..terms.typing import Terms
from ..sectors import Sectors

__tag_to_operator: dict[str, typing.Type[Operator]] = {}


def build(
    tag: str,
    sectors: Sectors,
    terms: Terms,
    sparsity: typing.Literal["dense", "sparse"],
    dtype: npt.DTypeLike,
    **kwargs,
) -> np.ndarray | nps.csr_array:
    """
    Construct operator with the given `tag`.

    Parameters
    ----------
    tag : str
        Corresponding `tag` for operator to create.
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
    **kwargs : dict, optional
        Additional operators arguments.

    Returns
    -------
    np.ndarray | nps.csr_array
        Constructed operator.
    """
    return __tag_to_operator[tag].build(sectors, terms, sparsity, dtype, **kwargs)


class _DuplicateTagError(Exception):  # pylint: disable=C0115
    pass


def register_term_type(operator_type: type):
    """
    Registers given `operator_type` in global mappings
    used in `pystrel.operators.utils`.

    Parameters
    ----------
    operator_type : type
        Operator type to register. It should inherit from `pystrel.operators.operator.Operator`.

    Raises
    ------
    _DuplicateTagError
        when tag for given `operator_type` is already used.
    """
    assert issubclass(operator_type, Operator)

    tag = operator_type.tag

    if tag in __tag_to_operator:
        raise _DuplicateTagError(
            f"Duplicate tag is found: `{tag}`! "
            f"Operators `{operator_type.__name__}` and `{__tag_to_operator[tag].__name__}`"
            f" have the same tag `{tag}`."
        )

    __tag_to_operator[tag] = operator_type


for name, obj in inspect.getmembers(sys.modules[impl.__name__]):
    if inspect.isclass(obj) and issubclass(obj, Operator) and obj != Operator:
        register_term_type(obj)
