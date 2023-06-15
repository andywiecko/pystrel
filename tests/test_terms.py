"""
**Hint:** during test for terms, consider assertion with eigenvalues.
This approach make these unit-test combinadics agnostic.
We could introduce abstraction for the combinadics, however, this is not worth the effort.
"""
import pytest
import numpy as np
import numpy.testing as npt
import scipy.special as sps # type: ignore
import pystrel.terms as ps


# pylint: disable=R0903,C0115,R0801


class FakeTermRank0(ps.Term):
    tag = "fake0"
    repr = "Fake repr 0"
    mixing_rank = 0
    particle_type = "spinless fermions"
    ensemble = "canonical"

    @staticmethod
    def apply(params: dict, matrix: np.ndarray, sector: tuple[int, int]):
        matrix[:] += 1.0
        return matrix


class FakeTermRank1(ps.Term):
    tag = "fake1"
    repr = "Fake repr 1"
    mixing_rank = 1
    particle_type = "spinfull fermions"
    ensemble = "grand canonical"

    @staticmethod
    def apply(params: dict, matrix: np.ndarray, sector: tuple[int, int]):
        matrix[:] += 2.0
        return matrix


class FakeTermRank2(ps.Term):
    tag = "fake2"
    repr = "Fake repr 2"
    mixing_rank = 2
    particle_type = "spins 1/2"
    ensemble = "parity grand canonical"

    @staticmethod
    def apply(params: dict, matrix: np.ndarray, sector: tuple[int, int]):
        matrix[:] += 4.0
        return matrix

# pylint: enable=R0903,C0115,R0801


ps.utils.register_term_type(FakeTermRank0)
ps.utils.register_term_type(FakeTermRank1)
ps.utils.register_term_type(FakeTermRank2)


def test_term__str():
    assert ps.utils.term__str__("fake1") == "Fake repr 1"


@pytest.mark.parametrize('terms, expected', [
    ({"fake0": {}}, "spinless fermions"),
    ({"fake1": {}}, "spinfull fermions"),
    ({"fake2": {}}, "spins 1/2"),
    ({"fake0": {}, "fake1": {}}, "undefined"),
])
def test_identify_particle_type(terms: dict[str, dict], expected: str):
    assert ps.utils.identify_particle_type(terms) == expected


@pytest.mark.parametrize('terms, expected', [
    ({"fake0": {}}, "canonical"),
    ({"fake1": {}}, "grand canonical"),
    ({"fake2": {}}, "parity grand canonical"),
    ({"fake0": {}, "fake1": {}}, "grand canonical"),
    ({"fake0": {}, "fake1": {}, "fake2": {}}, "grand canonical"),
    ({"fake0": {}, "fake2": {}}, "parity grand canonical"),
])
def test_identify_ensemble(terms: dict[str, dict], expected: str):
    assert ps.utils.identify_ensemble(terms) == expected


@pytest.mark.parametrize('terms, expected', [
    ({"fake0": {}}, {*()}),
    ({"fake1": {}}, {1}),
    ({"fake2": {}}, {2}),
    ({"fake0": {}, "fake1": {}}, {1}),
    ({"fake0": {}, "fake1": {}, "fake2": {}}, {1, 2}),
    ({"fake0": {}, "fake2": {}}, {2}),
])
def test_collect_mixing_sector_ranks(terms: dict[str, dict], expected: set[int]):
    assert ps.utils.collect_mixing_sector_ranks(terms) == expected


@pytest.mark.parametrize('terms, rank, expected', [
    ({"fake0": {}}, 0, 1.0 * np.ones((10, 10))),
    ({"fake1": {}}, 1, 2.0 * np.ones((10, 10))),
    ({"fake2": {}}, 2, 4.0 * np.ones((10, 10))),
    ({"fake0": {}, "fake1": {}}, 2, np.zeros((10, 10))),
    ({"fake0": {}, "fake1": {}, "fake2": {}}, 0,  1.0 * np.ones((10, 10))),
    ({"fake0": {}, "fake2": {}}, 1, np.zeros((10, 10))),
])
def test_apply(terms: dict[str, dict], rank, expected):
    assert (ps.utils.apply(terms, np.zeros(
        (10, 10)), (0, 0), rank) == expected).all()


def test_term_Jz():  # pylint: disable=C0103
    L = 4
    sector = (L, L//2)
    size = 6
    params = {(i, (i+1) % L): 1.0 for i in range(L-1)}
    matrix = np.zeros((size, size))

    matrix = ps.Term_Jz.apply(params, matrix, sector)
    eig = np.linalg.eigvalsh(matrix, 'U')

    assert (eig == np.array([-3.0, -3, -1, -1, 1, 1])).all()


def test_term_hz():
    L = 4
    sector = (L, L//2)
    size = 6
    params = {0: 1.0, 1: 2.0, 2: 4.0, 3: 8.0}
    matrix = np.zeros((size, size))

    matrix = ps.Term_hz.apply(params, matrix, sector)
    eig = np.linalg.eigvalsh(matrix, 'U')

    assert (eig == np.array([-9.0, -5, -3, +3, +5, +9])).all()


def test_term_gamma():
    L = 4
    sector = (L, L//2)
    size = 6
    params = {(i, (i+1) % L): 1.0 for i in range(L)}
    matrix = np.zeros((size, size))

    matrix = ps.Term_gamma.apply(params, matrix, sector)
    eig = np.linalg.eigvalsh(matrix, 'U')

    npt.assert_allclose(eig, np.array(
        [-np.sqrt(8), 0, 0, 0, 0, +np.sqrt(8)]), atol=1e-7)


def test_term_hx():
    L = 4
    sector = (L, L//2)
    params = {i: 1.0 for i in range(L)}
    matrix = np.zeros((6, 4))

    matrix = ps.Term_hx.apply(params, matrix, sector)

    m = np.zeros((10, 10))
    m[:6, 6:] = matrix
    eig = np.linalg.eigvalsh(m, 'U')

    npt.assert_allclose(eig, np.array([
        -np.sqrt(6),
        -np.sqrt(2),
        -np.sqrt(2),
        -np.sqrt(2),
        0,
        0,
        +np.sqrt(2),
        np.sqrt(2),
        np.sqrt(2),
        np.sqrt(6)
    ]), atol=1e-7)


@pytest.mark.parametrize("L, N", [
    (4, 0), (4, 1), (4, 2), (4, 3), (4, 4),
    (5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5),
    (6, 0), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6),
])
def test_term_t(L, N):
    sector = (L, N)
    size = int(sps.binom(L, N))
    params = {(i, (i+1) % L): 1.0 for i in range(L)}
    matrix = np.zeros((size, size))

    matrix = ps.Term_t.apply(params, matrix, sector)
    eig = np.linalg.eigvalsh(matrix, 'U')

    # It can be shown that model ∑ᵢⱼ a†ᵢaⱼ + h.c.
    # analytical solution for dispersion relation is given by:
    #
    # e(k) = 2 cos(k),
    #
    # where k = 2pi/L*l, l = 0, 1, ..., L-1
    ek = [2 * np.cos(2 * np.pi / L * i) for i in range(L)]
    E = [sum(ek[id] for id, i in
             enumerate(ps.combinadics.tostate(j, L, N)) if i == '1') for j in range(size)]
    E.sort()
    npt.assert_allclose(eig, E, atol=1e-7)


def test_term_V(): # pylint: disable=C0103
    L = 4
    N = 2
    size = int(sps.binom(L, N))
    params = {(0, 1): 1.0, (1, 2): 2.0, (2, 3): 4.0, (3, 0): -1}
    matrix = np.zeros((size, size))

    matrix = ps.Term_V.apply(params, matrix, (L, N))
    eig = np.linalg.eigvalsh(matrix, 'U')

    npt.assert_array_equal(eig, [-1.0, 0, 0, 1, 2, 4])


@pytest.mark.parametrize("L, P", [
    (3, 0), (3, 1),
    (4, 0), (4, 1),
    (5, 0), (5, 1),
    (6, 0), (6, 1),
    (7, 0), (7, 1),
])
def test_term_Delta(L: int, P: int): # pylint: disable=C0103,R0914
    size = 2**(L-1)
    params = {(i, (i+1) % L): 1.0 for i in range(L)}
    matrix = np.zeros((size, size))
    particles = [i for i in range(L+1) if i % 2 == P]

    offset = 0
    for N in particles[:-1]:
        sizeA = int(sps.binom(L, N))
        sizeB = int(sps.binom(L, N+2))
        a0 = offset
        a1 = a0 + sizeA
        b0 = offset + sizeA
        b1 = b0 + sizeB
        matrix[a0:a1, b0:b1] = ps.Term_Delta.apply(
            params, matrix[a0:a1, b0:b1], (L, N))
        offset += sizeA
    eig = np.linalg.eigvalsh(matrix, 'U')

    # It can be shown that model ∑ᵢⱼ a†ᵢa†ⱼ + h.c.
    # analytical solution for dispersion relation is given by:
    #
    # e(k) = ±2 |sin(k)|
    #
    # where k = 2pi/L*l, l = 0, 1, ..., L-1.
    ek = [2 * np.abs(np.sin(2 * np.pi / L * i)) for i in range((L+1) // 2)]
    ek += [-e for e in (ek if L % 2 == 0 else ek[1:])]
    E = [sum(ek[id] for id, i in
             enumerate(ps.combinadics.tostate(j, L, N)) if i == '1') for N in
         particles for j in range(int(sps.binom(L, N)))]
    E.sort()
    npt.assert_allclose(eig, E, atol=1e-7)


def test_term_mu():
    L = 4
    N = 2
    size = int(sps.binom(L, N))
    params = {0: 1.0, 1: 2, 2: 4, 3: 8}
    matrix = np.zeros((size, size))

    matrix = ps.Term_mu.apply(params, matrix, (L, N))
    eig = np.linalg.eigvalsh(matrix, 'U')

    npt.assert_array_equal(eig, [3.0, 5, 6, 9, 10, 12])
