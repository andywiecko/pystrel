import pytest
import numpy as np
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
