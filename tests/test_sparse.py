import pytest
import scipy.sparse as sp  # type: ignore
import numpy as np
import numpy.testing as npt
import pystrel.sparse as ps


def test_add_value():
    s = ps.Sparse(shape=(10, 10))
    s.add((2, 4), 5.9)

    assert s.vals == [5.9]
    assert s.rows == [2]
    assert s.cols == [4]


def test_to_csr():
    s = ps.Sparse(shape=(10, 10))
    s.add((2, 4), 5.9)
    s.add((1, 7), 1.2)
    csr = s.to_csr(dtype=np.float64)

    expected = np.zeros((10, 10))
    expected[2, 4] = 5.9
    expected[1, 7] = 1.2
    npt.assert_equal(csr.toarray(), expected)


def test_view_add_value():
    s = ps.Sparse(shape=(10, 10))
    s[2, 4] += 5.9
    s[1, 7] += 1.2

    expected = np.zeros((10, 10))
    expected[2, 4] = 5.9
    expected[1, 7] = 1.2
    npt.assert_equal(s.to_csr().toarray(), expected)


def test_view_add_value_duplicates():
    s = ps.Sparse(shape=(10, 10))
    s[2, 4] += 5.0
    s[2, 4] += 1.0

    expected = np.zeros((10, 10))
    expected[2, 4] = 6.0
    npt.assert_equal(s.to_csr().toarray(), expected)


def test_span():
    s = ps.Sparse(shape=(10, 10))
    view = s[5:10, 5:10]
    view[1, 2] += 1.0
    view[1, 3] += 2.0

    expected = sp.lil_array((10, 10))
    expected_view = expected[5:10, 5:10]
    expected_view[1, 2] += 1.0
    expected_view[1, 3] += 2.0
    expected[5:10, 5:10] = expected_view
    expected = expected.toarray()

    npt.assert_equal(s.to_csr().toarray(), expected)


@pytest.mark.parametrize(
    "index",
    [
        (0, 11),
        (11, 0),
        (11, -1),
        (-11, 1),
        (slice(0, 11), slice(0, 10)),
        (slice(0, 1), slice(0, 11)),
    ],
)
def test_raise_index_error(index):
    with pytest.raises(IndexError):
        s = ps.Sparse((10, 10))
        _ = s[index]


@pytest.mark.parametrize("index", [4.3, slice(0, 1), 23])
def test_raise_value_error(index):
    with pytest.raises(ValueError):
        s = ps.Sparse((10, 10))
        _ = s[index]
