"""
Module contains utility methods for conversion between numbers and combinations, i.e. 
it implements [combinadics](https://en.wikipedia.org/wiki/Combinatorial_number_system).
"""
import scipy.special as sps  # type: ignore


def tonumber(state: str) -> int:
    """
    Converts `state` into `number`

    Parameters
    ----------
    state : str
        State, which represent the combination, e.g. `0010110`, `1110101`, ...
        where `1` correspond to selected element in combination,
        and `0` otherwise.

    Returns
    -------
    int
        Number which corresponds to combination with given `state`.
    """
    ret = 0
    k = state.count("1")
    L = len(state)
    for i, s in enumerate(state):
        if s == "1":
            ret += int(sps.binom(L - 1 - i, k))
            k -= 1

    return ret


def tostate(number: int, n: int, k: int) -> str:
    """
    Converts `number` to corresponding `k`-combination from `n`-element set.

    Parameters
    ----------
    number : int
        Number to convert.
    n : int
        All elements count.
    k : int
        Elements taken count.

    Returns
    -------
    str
        State, which represent the combination, e.g. `0010110`, `1110101`, ...
        where `1` correspond to selected element in combination,
        and `0` otherwise.
    """
    ret = "0" * n
    Nt = number

    for i in range(k):
        n = k - i - 1
        while sps.binom(n, k - i) <= Nt:
            n += 1

        if sps.binom(n, k - i) > Nt:
            n -= 1

        ret = ret[:n] + "1" + ret[n + 1 :]
        Nt -= int(sps.binom(n, k - i))
    return ret[::-1]


def count_particles_between(state: str, i: int, j: int) -> int:
    """Counts particles, i.e. '1' between `i` and `j` site for given `state`.

    Parameters
    ----------
    state : str
        State, which represent the combination, e.g. `0010110`, `1110101`, ...
        where `1` correspond to selected element in combination,
        and `0` otherwise.
    i : int
        Site start.
    j : int
        Site end.

    Returns
    -------
    int
        Particles count.
    """
    return (state[i + 1 : j] if i < j else state[j + 1 : i]).count("1")
