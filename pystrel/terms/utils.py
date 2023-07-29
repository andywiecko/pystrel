"""
Utilities related to terms.
"""
import typing
import sys
import inspect
import numpy as np
import scipy.sparse as nps  # type: ignore
from .term import Term
from . import impl
from .typing import Terms

__tag_to_term: dict[str, typing.Type[Term]] = {}

__particle_type_tags: dict[str, set[str]] = {
    "spinless fermions": {*()},
    "spins 1/2": {*()},
    "spinfull fermions": {*()},
    "tj quasiparticles": {*()},
}

__ensemble_tags: dict[str, set[str]] = {
    "grand canonical": {*()},
    "parity grand canonical": {*()},
    "canonical": {*()},
}

__mixing_sectors_ranks_tags: dict[int, set[str]] = {0: {*()}, 1: {*()}, 2: {*()}}


class _DuplicateTagError(Exception):  # pylint: disable=C0115
    pass


def term__str__(term_type: str) -> str:
    """
    Returns string representation of the given `term_type`.

    Parameters
    ----------
    term_type : str
        Tag used for given term type.

    Returns
    -------
    str
        `term_type` string representation.
    """
    return __tag_to_term[term_type].repr


def identify_particle_type(
    terms: dict[str, dict]
) -> typing.Literal["spinless fermions", "spins 1/2", "spinfull fermions", "undefined"]:
    """
    Identifies the particle type for given `terms`.

    Parameters
    ----------
    terms : dict[str, dict]
        Dictionary of terms tags with corresponding parameters.

    Returns
    -------
    'spinless fermions' | 'spins 1/2' | 'spinfull fermions' | 'undefined'
        Particle string.
    """
    if all(t in __particle_type_tags["spinless fermions"] for t in terms):
        return "spinless fermions"
    if all(t in __particle_type_tags["spinfull fermions"] for t in terms):
        return "spinfull fermions"
    if all(t in __particle_type_tags["spins 1/2"] for t in terms):
        return "spins 1/2"
    return "undefined"


def identify_ensemble(
    terms: Terms,
) -> typing.Literal[
    "grand canonical", "parity grand canonical", "canonical", "undefined"
]:
    """
    Identify minimal ensemble required for representing the `terms`.

    Parameters
    ----------
    terms : Terms
        Dictionary with terms.
        See `pystrel.terms` for more details.

    Returns
    -------
    'grand canonical' | 'parity grand canonical' | 'canonical' | 'undefined'
        Ensemble string.
    """
    if any(t in __ensemble_tags["grand canonical"] for t in terms):
        return "grand canonical"
    if any(t in __ensemble_tags["parity grand canonical"] for t in terms):
        return "parity grand canonical"
    if any(t in __ensemble_tags["canonical"] for t in terms):
        return "canonical"
    return "undefined"


def collect_mixing_sector_ranks(terms: Terms) -> set[int]:
    """
    Collects non-zero mixing sector ranks for given `terms`.

    Parameters
    ----------
    terms : Terms
        Dictionary with terms.
        See `pystrel.terms` for more details.

    Returns
    -------
    set[int]
        Set of non-zer ranks.
    """
    ret = set()

    if any(i in __mixing_sectors_ranks_tags[1] for i in terms):
        ret.add(1)
    if any(i in __mixing_sectors_ranks_tags[2] for i in terms):
        ret.add(2)

    return ret


def apply(
    terms: Terms,
    matrix: np.ndarray | nps.dok_array,
    sector: tuple[int, int],
    rank: int,
) -> np.ndarray | nps.dok_array:
    """
    Applies all `terms` of given `rank` on `matrix` within given `sector`.

    Parameters
    ----------
    terms : Terms
        Dictionary with terms.
        See `pystrel.terms` for more details.
    matrix : _type_
        View on matrix on which terms should be applied.
    sector : tuple[int, int]
        Corresponding particle sector for given `matrix`.
    rank : int
        Mixing rank.

    Returns
    -------
    np.ndarray | nps.dok_array
        Matrix with applied terms.
    """
    for t, params in terms.items():
        term = __tag_to_term[t]
        if rank == term.mixing_rank:
            matrix = term.apply(params, matrix, sector)
    return matrix


def info():
    """
    Displays information related to available terms with corresponding tags.
    """
    ret = ""
    for ptype, tags in __particle_type_tags.items():
        ret += 15 * "-" + "\n"
        ret += ptype + "\n"
        ret += 15 * "-" + "\n"
        for t in tags:
            ret += "- " + t + ": " + term__str__(t) + "\n"
        if len(tags) == 0:
            ret += "not implemented yet\n"

        ret += "\n"
    print(ret)


def register_term_type(term_type: type):
    """
    Registers given `term_type` in global mappings
    used in `utils.py`.

    Parameters
    ----------
    term_type : type
        Term type to register. It should inherit from `Term`.

    Raises
    ------
    _DuplicateTagError
        when tag for given `term_type` is already used.
    ValueError
        when unsupported mixing rank is defined in `term_type`.
    """
    assert issubclass(term_type, Term)

    tag = term_type.tag

    if tag in __tag_to_term:
        raise _DuplicateTagError(
            f"Duplicate tag is found {tag}! "
            f"Terms `{term_type.__name__}` and `{__tag_to_term[tag].__name__}`"
            f" have the same tag `{tag}`."
        )

    if term_type.mixing_rank not in [0, 1, 2]:
        raise ValueError(f"Rank {term_type.mixing_rank} is not supported!")

    __tag_to_term[tag] = term_type
    __particle_type_tags[term_type.particle_type].add(tag)
    __ensemble_tags[term_type.ensemble].add(tag)
    __mixing_sectors_ranks_tags[term_type.mixing_rank].add(tag)


for name, obj in inspect.getmembers(sys.modules[impl.__name__]):
    if inspect.isclass(obj) and issubclass(obj, Term) and obj != Term:
        register_term_type(obj)
