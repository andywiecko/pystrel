"""
Type hints class for `pystrel.model.Model`'s parameter.
"""
import typing


class Parameters(typing.TypedDict):
    """
    `TypedDict` for `pystrel.model.Model`'s parameters representation.
    """

    terms: dict
    """
    To see supported terms see `pystrel.terms`
    """
    sites: typing.NotRequired[int]
    """
    (Optional) Number of sites in the given model.
    """
    particles: typing.NotRequired[int]
    """
    (Optional) Number of particles in the given model. 
    It can be used only with *canonical ensemble* terms
    """
    parity: typing.NotRequired[typing.Literal[0, 1]]
    """
    (Optional) Parity sector of the given model.
    It can be used only with *parity grand canonical ensemble*.
    It accepts `0`/`1`, for even/odd sectors, respectively.
    """
    sectors: typing.NotRequired[list[tuple[int, int]]]
    """
    (Optional) List of sectors in the form of tuples `(L, N)`, where `L` and `N`
    correspond to sites and particles, respectively.

    Note
    ----
    This is experimental feature, use with caution.
    Sectors should be sorted with respect to particles.
    """
