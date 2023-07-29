"""
Type hints class for `pystrel.model.Model`'s parameters terms.
"""
import typing


class Terms(typing.TypedDict):
    """
    `TypedDict` for `pystrel.model.Model`'s parameters terms representation.
    """

    t: typing.NotRequired[dict[tuple[int, int], typing.Any]]
    """
    Hopping operator for spinless fermions: tᵢⱼ a†ᵢaⱼ + h.c.
    See `pystrel.terms.impl.Term_t` for more details.
    """
    V: typing.NotRequired[dict[tuple[int, int], typing.Any]]
    """
    2-body operator for spinless fermions: Vᵢⱼ nᵢnⱼ.
    See `pystrel.terms.impl.Term_V` for more details.
    """
    Delta: typing.NotRequired[dict[tuple[int, int], typing.Any]]
    """
    Pair operator for spinless fermions: Δᵢⱼ a†ᵢa†ⱼ + h.c.
    See `pystrel.terms.impl.Term_Delta` for more details.
    """
    mu: typing.NotRequired[typing.Any]
    """
    Chemical potential operator for spinless fermions: μ N.
    See `pystrel.terms.impl.Term_mu` for more details.
    """
    epsilon: typing.NotRequired[dict[int, typing.Any]]
    """
    Chemical potential *like* operator for spinless fermions: εᵢ nᵢ.
    See `pystrel.terms.impl.Term_epsilon` for more details.
    """

    Jz: typing.NotRequired[dict[tuple[int, int], typing.Any]]
    """
    Spin-spin interaction for spin 1/2 particles: Jᶻᵢⱼ σᶻᵢσᶻⱼ.
    See `pystrel.terms.impl.Term_Jz` for more details.
    """
    hz: typing.NotRequired[dict[int, typing.Any]]
    """
    $z$-magnetic field operator for spin 1/2 particles:  hᶻᵢ σᶻᵢ.
    See `pystrel.terms.impl.Term_hz` for more details.
    """
    gamma: typing.NotRequired[dict[tuple[int, int], typing.Any]]
    """
    Spin hop operator for spin 1/2 particles: γᵢⱼ σ⁺ᵢσ⁻ⱼ + h.c.
    See `pystrel.terms.impl.Term_gamma` for more details.
    """
    hx: typing.NotRequired[dict[int, typing.Any]]
    """
    $x$-magnetic field operator for spin 1/2 particles: hˣᵢ σˣᵢ.
    See `pystrel.terms.impl.Term_hx` for more details.
    """
