import pytest
import pystrel as ps


@pytest.mark.parametrize(
    "state, expected",
    [
        ("000", 0),
        ("001", 0),
        ("010", 1),
        ("100", 2),
        ("011", 0),
        ("101", 1),
        ("110", 2),
        ("111", 0),
        ("0000", 0),
        ("0001", 0),
        ("0010", 1),
        ("0100", 2),
        ("1000", 3),
        ("0011", 0),
        ("0101", 1),
        ("0110", 2),
        ("1001", 3),
        ("1010", 4),
        ("1100", 5),
        ("0111", 0),
        ("1011", 1),
        ("1101", 2),
        ("1110", 3),
        ("1111", 0),
    ],
)
def test_tonumber(state: str, expected: int):
    assert ps.combinadics.tonumber(state) == expected


@pytest.mark.parametrize(
    "number, L, N, expected",
    [
        (0, 3, 0, "000"),
        (0, 3, 1, "001"),
        (1, 3, 1, "010"),
        (2, 3, 1, "100"),
        (0, 3, 2, "011"),
        (1, 3, 2, "101"),
        (2, 3, 2, "110"),
        (0, 3, 3, "111"),
        (0, 4, 0, "0000"),
        (0, 4, 1, "0001"),
        (1, 4, 1, "0010"),
        (2, 4, 1, "0100"),
        (3, 4, 1, "1000"),
        (0, 4, 2, "0011"),
        (1, 4, 2, "0101"),
        (2, 4, 2, "0110"),
        (3, 4, 2, "1001"),
        (4, 4, 2, "1010"),
        (5, 4, 2, "1100"),
        (0, 4, 3, "0111"),
        (1, 4, 3, "1011"),
        (2, 4, 3, "1101"),
        (3, 4, 3, "1110"),
        (0, 4, 4, "1111"),
    ],
)
def test_tostate(number: int, L: int, N: int, expected: str):
    assert ps.combinadics.tostate(number, L, N) == expected


@pytest.mark.parametrize(
    "state, i, j, expected",
    [
        ("0000", 0, 3, 0),
        ("0100", 0, 3, 1),
        ("0010", 0, 3, 1),
        ("0110", 0, 3, 2),
        ("1110", 0, 3, 2),
        ("1111", 0, 3, 2),
        ("0000", 3, 0, 0),
        ("0100", 3, 0, 1),
        ("0010", 3, 0, 1),
        ("0110", 3, 0, 2),
        ("1110", 3, 0, 2),
        ("1111", 3, 0, 2),
        ("1111", 0, 0, 0),
    ],
)
def test_count_particles_between(state: str, i: int, j: int, expected: int):
    assert ps.combinadics.count_particles_between(state, i, j) == expected
