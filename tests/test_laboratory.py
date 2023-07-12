import pytest
import numpy as np
import scipy.sparse as nps  # type: ignore
import pystrel as ps


@pytest.mark.parametrize(
    "operator, state, expected",
    [
        (np.array([[0, 1], [1, 0]]), np.array([1, 0]), 0.0),
        (np.array([[0, 1], [1, 0]]), np.array([1, 1]), 2.0),
        (nps.csr_array(np.array([[0, 1], [1, 0]])), np.array([1, 0]), 0.0),
        (nps.csr_array(np.array([[0, 1], [1, 0]])), np.array([1, 1]), 2.0),
        (np.array([[0, 1.0j], [-1.0j, 0]]), np.array([1 + 1.0j, 1 - 1.0j]), 4.0 + 0.0j),
        (
            nps.csr_array(np.array([[0, 1.0j], [-1.0j, 0]])),
            np.array([1 + 1.0j, 1 - 1.0j]),
            4.0 + 0.0j,
        ),
    ],
)
def test_measure(operator, state, expected):
    assert ps.measure(operator, state) == expected
