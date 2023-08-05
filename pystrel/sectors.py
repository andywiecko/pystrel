"""
Module contains utilities related to `Model`'s particle sectors.
"""
import typing
import scipy.special as sps  # type: ignore
from . import terms
from .parameters import Parameters
from . import combinadics


class Sectors:
    """
    Helper class used for representing the sectors of the Hilbert space.

    Example
    -------
    `Sectors` allows for iteration and easy access to related matrix elements.

    >>> sectors = Sectors({"terms": { ... }})
    >>> for start, end, sector in sectors:
    >>>     matrix[start:end, start:end] = ...

    It can be used for mixing sectors iterations as well

    >>> sectors = Sectors({"terms":{ ... }})
    >>> for (x0, y0, s0), (x1, y1, s1) in sectors.mixing_iter():
    >>>      matrix[x0:y0, x1:y1] = ...
    """

    def __init__(self, params: Parameters):
        """
        `Sectors` constructor.

        Parameters
        ----------
        params : Parameters
            Dictionary of parameters.
            See `pystrel.parameters.Parameters` for more details.
        """
        self.sectors: list[tuple[int, int]] = params.get(
            "sectors",
            generate_sectors(
                terms.utils.identify_ensemble(params.get("terms", {})), params
            ),
        )
        """
        List of sectors in the form of tuples `(L, N)`, where `L` and `N`
        correspond to sites and particles, respectively.
        """

        self.mixing_sectors: list[tuple[int, int]] = generate_mixing_sectors(
            terms.utils.collect_mixing_sector_ranks(params.get("terms", {})),
            self.sectors,
        )
        """List of tuples representing mixing sector IDs."""

        self.size: int = 0
        """Size of the Hilbert space corresponding to `Sectors.sectors`."""
        self.starts: list[int] = []
        """List of start indices corresponding to the `Sectors.sectors`."""
        self.ends: list[int] = []
        """List of end indices corresponding to the `Sectors.sectors`."""

        offset = 0
        for L, N in self.sectors:
            size = int(sps.binom(L, N))
            self.size += size
            self.starts.append(offset)
            self.ends.append(offset + size)
            offset += size

    def __iter__(self):
        return zip(self.starts, self.ends, self.sectors)

    def __str__(self):
        return str(self.sectors)

    def get_base_state(self, index: int) -> str:
        """
        Returns base state for given `index`.

        Parameters
        ----------
        index : int
            Index of base state, in range [0, `self.size`).

        Returns
        -------
        str
            State for corresponding `index`, e.g. `'001010'`.

        Raises
        ------
        IndexError
            When `index` is out of range sectors size.
        """
        if index >= self.size or index < 0:
            raise IndexError(f"Index {index} is out or range sectors size {self.size}!")

        ret = ""
        for i, (start, end) in enumerate(zip(self.starts, self.ends)):
            if start <= index < end:
                s = self.sectors[i]
                ret = combinadics.tostate(number=index - start, n=s[0], k=s[1])
                break
        return ret

    def get_base_state_id(self, state: str) -> int:
        """
        Returns index for the given `state`.

        Parameters
        ----------
        state : str
            Base state e.g. `'00001100'`.

        Returns
        -------
        int
            Index which corresponds to the given base `state`.

        Raises
        ------
        ValueError
            If there is no compatible sector with the given `state`.
        """
        L = len(state)
        N = state.count("1")
        sector = (L, N)
        if (sector) not in self.sectors:
            raise ValueError(f"There is no sector with {L} sites and {N} particles!")

        i = self.sectors.index(sector)
        return self.starts[i] + combinadics.tonumber(state)

    def mixing_iter(self):
        """
        Iterator for mixing sectors, enabling convenient access to matrix elements.

        Example
        -------
        >>> sectors = Sectors({"terms": { ... }})
        >>> for (s0, e0, sector0), (s1, e1, sector1) in sectors.mixing_iter():
        >>>     ... = matrix[s0:e0, s1:e1]
        >>>     ...

        Returns
        -------
        _MixingSectorIterator
            Iterator for mixing sectors.
        """
        return _MixingSectorIterator(self)


# pylint: disable=R0903
class _MixingSectorIterator:
    def __init__(self, sectors: Sectors):
        self.sectors = sectors

    def __iter__(self):
        s = self.sectors
        return iter(
            (
                (s.starts[s0], s.ends[s0], s.sectors[s0]),
                (s.starts[s1], s.ends[s1], s.sectors[s1]),
            )
            for s0, s1 in s.mixing_sectors
        )


# pylint: enable=R0903


def generate_sectors(
    ensemble: typing.Literal[
        "grand canonical", "parity grand canonical", "canonical", "undefined"
    ],
    params: Parameters,
) -> list[tuple[int, int]]:
    """
    Generates sectors for the corresponding `ensemble` with given `params`.

    Parameters
    ----------
    ensemble : typing.Literal["grand canonical", "parity grand canonical", "canonical", "undefined"]
        Ensemble string.
    params : Parameters
        Dictionary of parameters.
        See `pystrel.parameters.Parameters` for more details.

    Returns
    -------
    list[tuple[int,int]]
        Sectors list with made of tuples `(sites, particles)`.
    """

    def max_site(_terms):
        return max(
            (
                id + 1
                for key in _terms
                if isinstance(_terms[key], dict)
                for t in _terms[key]
                for id in (t if isinstance(t, tuple) else (t,))
            ),
            default=None,
        )

    _terms = params.get("terms", None)
    L = params.get("sites", max_site(_terms) if _terms is not None else None)
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
