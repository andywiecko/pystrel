import pytest
import numpy as np
import pystrel.sectors as ps

# pylint: disable=R0903,C0115,R0801


class FakeSectorsA(ps.terms.Term):
    tag = "fake sectors A"
    repr = "Fake repr 0"
    mixing_rank = 0
    particle_type = "spinless fermions"
    ensemble = "canonical"

    @staticmethod
    def apply(params: dict, matrix: np.ndarray, sector: tuple[int, int]):
        matrix[:] += 1.0
        return np.triu(matrix)


class FakeSectorsB(ps.terms.Term):
    tag = "fake sectors B"
    repr = "Fake repr 1"
    mixing_rank = 1
    particle_type = "spinfull fermions"
    ensemble = "grand canonical"

    @staticmethod
    def apply(params: dict, matrix: np.ndarray, sector: tuple[int, int]):
        matrix[:] += 2.0
        return matrix


class FakeSectorsC(ps.terms.Term):
    tag = "fake sectors C"
    repr = "Fake repr 2"
    mixing_rank = 2
    particle_type = "spins 1/2"
    ensemble = "parity grand canonical"

    @staticmethod
    def apply(params: dict, matrix: np.ndarray, sector: tuple[int, int]):
        matrix[:] += 4.0
        return matrix


# pylint: enable=R0903,C0115,R0801


ps.terms.utils.register_term_type(FakeSectorsA)
ps.terms.utils.register_term_type(FakeSectorsB)
ps.terms.utils.register_term_type(FakeSectorsC)


@pytest.mark.parametrize(
    "params, expected",
    [
        (
            {"sites": 4, "particles": 2, "terms": {"fake sectors A": {(0, 1): 1.0}}},
            [(4, 2)],
        ),
        (
            {"sites": 4, "terms": {"fake sectors B": {(0, 1): 1.0}}},
            [(4, 0), (4, 1), (4, 2), (4, 3), (4, 4)],
        ),
        (
            {"sites": 4, "parity": 0, "terms": {"fake sectors C": {(0, 1): 1.0}}},
            [(4, 0), (4, 2), (4, 4)],
        ),
    ],
)
def test_sectors_sectors_collection(params, expected):
    sectors = ps.Sectors(params)
    assert sectors.sectors == expected


@pytest.mark.parametrize(
    "params, expected",
    [
        ({"sites": 4, "particles": 2, "terms": {"fake sectors A": {(0, 1): 1.0}}}, []),
        (
            {"sites": 4, "terms": {"fake sectors B": {(0, 1): 1.0}}},
            [(0, 1), (1, 2), (2, 3), (3, 4)],
        ),
        (
            {"sites": 4, "parity": 0, "terms": {"fake sectors C": {(0, 1): 1.0}}},
            [(0, 1), (1, 2)],
        ),
    ],
)
def test_sectors_mixing_sectors_collection(params, expected):
    sectors = ps.Sectors(params)
    assert sectors.mixing_sectors == expected


def test_sectors_str():
    sectors = ps.Sectors({"sectors": [(4, 1), (4, 3)]})
    assert str(sectors) == "[(4, 1), (4, 3)]"


def test_sectors_iter():
    sectors = ps.Sectors({"sectors": [(3, 0), (3, 1), (3, 2), (3, 3)]})
    assert list(sectors) == [
        (0, 1, (3, 0)),
        (1, 4, (3, 1)),
        (4, 7, (3, 2)),
        (7, 8, (3, 3)),
    ]


def test_sectors_mixing_iter():
    sectors = ps.Sectors(
        {"terms": {"fake sectors B": {}}, "sectors": [(2, 0), (2, 1), (2, 2)]}
    )
    assert list(sectors.mixing_iter()) == [
        ((0, 1, (2, 0)), (1, 3, (2, 1))),
        ((1, 3, (2, 1)), (3, 4, (2, 2))),
    ]


@pytest.mark.parametrize(
    "ensemble, params, expected",
    [
        ("undefined", {}, []),
        ("undefined", {"sites": 1}, []),
        ("canonical", {}, []),
        ("canonical", {"sites": 6}, [(6, 3)]),
        ("canonical", {"sites": 7}, [(7, 3)]),
        ("canonical", {"sites": 7, "particles": 5}, [(7, 5)]),
        ("canonical", {"terms": {"t": {0: 1.0, 6: 1.0}}}, [(7, 3)]),
        ("parity grand canonical", {}, []),
        ("parity grand canonical", {"sites": 5}, [(5, 0), (5, 2), (5, 4)]),
        ("parity grand canonical", {"sites": 6}, [(6, 0), (6, 2), (6, 4), (6, 6)]),
        ("parity grand canonical", {"sites": 6, "parity": 1}, [(6, 1), (6, 3), (6, 5)]),
        (
            "parity grand canonical",
            {"sites": 7, "parity": 0},
            [(7, 0), (7, 2), (7, 4), (7, 6)],
        ),
        (
            "parity grand canonical",
            {"terms": {"t": {0: 1.0, 5: 1.0}}},
            [(6, 0), (6, 2), (6, 4), (6, 6)],
        ),
        (
            "parity grand canonical",
            {"terms": {"t": {0: 1.0, 5: 1.0}}, "parity": 1},
            [(6, 1), (6, 3), (6, 5)],
        ),
        ("grand canonical", {}, []),
        ("grand canonical", {"sites": 4}, [(4, 0), (4, 1), (4, 2), (4, 3), (4, 4)]),
        (
            "grand canonical",
            {"terms": {"t": {(0, 4): 1.0}}},
            [(5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5)],
        ),
    ],
)
def test_generate_sectors(ensemble, params, expected):
    sectors = ps.generate_sectors(ensemble, params)
    assert sectors == expected


@pytest.mark.parametrize(
    "ranks, sectors, expected",
    [
        ([], [], []),
        ([1], [], []),
        ([1], [(10, 1), (10, 2), (10, 3)], [(0, 1), (1, 2)]),
        ([2], [], []),
        ([2], [(10, 1), (10, 2)], []),
        ([2], [(10, 1), (10, 2), (10, 3)], [(0, 2)]),
        ([1, 2], [], []),
        ([1, 2], [(10, 1), (10, 2)], [(0, 1)]),
        ([1, 2], [(10, 1), (10, 3)], [(0, 1)]),
        ([1, 2], [(10, 1), (10, 2), (10, 3)], [(0, 1), (0, 2), (1, 2)]),
    ],
)
def test_generate_mixing_sectors(ranks, sectors, expected):
    mixing_sectors = ps.generate_mixing_sectors(ranks, sectors)
    assert mixing_sectors == expected


def test_get_base_state():
    s = ps.Sectors({"sectors": [(3, 0), (3, 1), (3, 2)]})
    states = [s.get_base_state(i) for i in range(7)]
    assert states == ["000", "001", "010", "100", "011", "101", "110"]


@pytest.mark.parametrize("i", [-1, 8, 9])
def test_get_base_state_out_of_range(i):
    with pytest.raises(IndexError):
        _ = ps.Sectors({"sectors": [(3, 0), (3, 1), (3, 2)]}).get_base_state(i)


def test_get_base_state_id():
    s = ps.Sectors({"sectors": [(3, 0), (3, 1), (3, 2)]})
    ids = [s.get_base_state_id(i) for i in ["001", "010", "100"]]
    assert ids == [1, 2, 3]


@pytest.mark.parametrize("state", ["0000", "111"])
def test_get_base_state_id_value_error(state):
    with pytest.raises(ValueError):
        _ = ps.Sectors({"sectors": [(3, 0), (3, 1), (3, 2)]}).get_base_state_id(state)
