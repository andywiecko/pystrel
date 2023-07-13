import pytest
import numpy as np
import numpy.testing as npt
import scipy.sparse as nps  # type: ignore
import pystrel as ps


@pytest.mark.parametrize(
    "operator, expected_e, expected_v",
    [
        (
            np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]),
            np.array([1.0, 2.0]),
            np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        ),
        (
            np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 2.0]]),
            np.array([-1.0, 1.0]),
            1.0 / np.sqrt(2) * np.array([[1.0, -1.0, 0.0], [1.0, 1.0, 0.0]]),
        ),
        (
            nps.csr_array(
                np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
            ),
            np.array([1.0, 2.0]),
            np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        ),
        (
            nps.csr_array(
                np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 2.0]])
            ),
            np.array([-1.0, 1.0]),
            1.0 / np.sqrt(2) * np.array([[1.0, -1.0, 0.0], [-1.0, -1.0, 0.0]]),
        ),
    ],
)
def test_get_partial_spectrum(operator, expected_e, expected_v):
    e, v = ps.spectrum.get_partial_spectrum(operator, k=2)
    npt.assert_array_almost_equal(e, expected_e)
    npt.assert_array_almost_equal(v, expected_v)


@pytest.mark.parametrize(
    "operator, expected_e",
    [
        (
            np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]),
            np.array([1.0, 2.0]),
        ),
        (
            np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 2.0]]),
            np.array([-1.0, 1.0]),
        ),
        (
            nps.csr_array(
                np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
            ),
            np.array([1.0, 2.0]),
        ),
        (
            nps.csr_array(
                np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 2.0]])
            ),
            np.array([-1.0, 1.0]),
        ),
    ],
)
def test_get_partial_spectrum_without_vectors(operator, expected_e):
    e = ps.spectrum.get_partial_spectrum(operator, k=2, return_eigenvectors=False)
    npt.assert_array_almost_equal(e, expected_e)


@pytest.mark.parametrize(
    "operator, expected_e, expected_v",
    [
        (
            np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]),
            np.array([1.0, 2.0, 3.0]),
            np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        ),
        (
            np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 2.0]]),
            np.array([-1.0, 1.0, 2.0]),
            1.0
            / np.sqrt(2)
            * np.array([[-1.0, +1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, np.sqrt(2)]]),
        ),
    ],
)
def test_get_full_spectrum(operator, expected_e, expected_v):
    e, v = ps.spectrum.get_full_spectrum(operator)
    npt.assert_array_almost_equal(e, expected_e)
    npt.assert_array_almost_equal(v, expected_v)


@pytest.mark.parametrize(
    "operator, expected_e",
    [
        (
            np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]),
            np.array([1.0, 2.0, 3.0]),
        ),
        (
            np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 2.0]]),
            np.array([-1.0, 1.0, 2.0]),
        ),
    ],
)
def test_get_full_spectrum_without_vectors(operator, expected_e):
    e = ps.spectrum.get_full_spectrum(operator, compute_eigenvectors=False)
    npt.assert_array_almost_equal(e, expected_e)
