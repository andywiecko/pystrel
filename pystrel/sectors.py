"""
Module contains utility functions related to `Model`'s particle sectors.
"""
import typing


def generate_sectors(
    ensemble: typing.Literal[
        "grand canonical", "parity grand canonical", "canonical", "undefined"
    ],
    params: dict,
) -> list[tuple[int, int]]:
    """
    Generates sectors for the corresponding `ensemble` with given `params`.

    Parameters
    ----------
    ensemble : typing.Literal["grand canonical", "parity grand canonical", "canonical", "undefined"]
        Ensemble string.
    params : dict
        Supports same parameters as `Model`.

    Returns
    -------
    list[tuple[int,int]]
        Sectors list with made of tuples `(sites, particles)`.
    """

    def max_site(terms):
        return max(
            (
                id + 1
                for key in terms
                if isinstance(terms[key], dict)
                for t in terms[key]
                for id in (t if isinstance(t, tuple) else (t,))
            ),
            default=None,
        )

    terms = params.get("terms", None)
    L = params.get("sites", max_site(terms) if terms is not None else None)
    if L is None:
        return []

    match ensemble:
        case "grand canonical":
            return [(L, i) for i in range(L + 1)]

        case "parity grand canonical":
            parity = params.get("parity", 0)
            parity = parity if parity in [0, 1] else 0
            size = (L + 1) // 2 if L % 2 == 1 else L // 2 + (parity + 1) % 2
            return [(L, 2 * i + parity) for i in range(size)]

        case "canonical":
            N = params.get("particles", L // 2)
            return [(L, N)]

        case "undefined":
            return []


def generate_mixing_sectors(
    ranks: set[int], sectors: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """
    Generates mixing sectors assuming `ranks` and `sectors`.

    Parameters
    ----------
    ranks : set[int]
        Set with mixing ranks used for generation.
    sectors : list[tuple[int, int]]
        List of sectors with `(sites, particles)`.
        Sectors **must** be sorted by `sites` and `particles`.

    Returns
    -------
    list[tuple[int, int]]
        List of mixing sectors with `(sector id0, sector id1)`.
    """
    mixing_sectors: list[tuple[int, int]] = []
    count = len(sectors)
    for id0 in range(count):
        s0 = sectors[id0]
        for id1 in range(id0 + 1, count):
            s1 = sectors[id1]
            if s1[1] - s0[1] in ranks:
                mixing_sectors.append((id0, id1))

    return mixing_sectors
