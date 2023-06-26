import pytest
import pystrel.sectors as ps


@pytest.mark.parametrize(
    "ensemble, params, expected",
    [
        ("undefined", {}, []),
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
def test_(ranks, sectors, expected):
    mixing_sectors = ps.generate_mixing_sectors(ranks, sectors)
    assert mixing_sectors == expected
