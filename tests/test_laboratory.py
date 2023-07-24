import pytest
import numpy as np
import numpy.testing as npt
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


@pytest.mark.parametrize(
    "phi, psi, expected",
    [
        (np.array([0.0, 1]), np.array([1.0, 0]), 0.0),
        (np.array([1.0, 0]), np.array([1.0, 0]), 1.0),
        (np.array([1.0, 1.0]) / np.sqrt(2), np.array([1.0, -1.0]) / np.sqrt(2), 0.0),
        (np.array([1.0, 1.0j]) / np.sqrt(2), np.array([1.0, 1.0j]) / np.sqrt(2), 1.0),
        (np.array([1.0, 1.0j]) / np.sqrt(2), np.array([1.0, -1.0j]) / np.sqrt(2), 0.0),
        (
            np.array([0.5 + 0.5j, 0.75 + 0.25j]) / np.sqrt(2),
            np.array([0.3 - 0.7j, 0.9 - 0.1j]) / np.sqrt(2),
            0.210625,
        ),
    ],
)
def test_project(phi, psi, expected):
    npt.assert_almost_equal(ps.project(phi, psi), expected)
