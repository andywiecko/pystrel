"""
**Hint:** during test for terms, consider assertion with eigenvalues.
This approach make these unit-test combinadics agnostic.
We could introduce abstraction for the combinadics, however, this is not worth the effort.
"""
import pytest
import numpy as np
import numpy.testing as npt
import pystrel.terms as ps

# pylint: disable=R0903,C0115


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
