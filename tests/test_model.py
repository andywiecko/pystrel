import pytest
import numpy as np
import numpy.testing as npt
import pystrel as ps

# pylint: disable=R0903,C0115


class FakeA(ps.terms.Term):
    tag = "fake A"
    repr = "Fake repr 0"
    mixing_rank = 0
    particle_type = "spinless fermions"
    ensemble = "canonical"

    @staticmethod
    def apply(params: dict, matrix: np.ndarray, sector: tuple[int, int]):
        matrix[:] += 1.0
        return np.triu(matrix)


class FakeB(ps.terms.Term):
    tag = "fake B"
    repr = "Fake repr 1"
    mixing_rank = 1
    particle_type = "spinfull fermions"
    ensemble = "grand canonical"

    @staticmethod
    def apply(params: dict, matrix: np.ndarray, sector: tuple[int, int]):
        matrix[:] += 2.0
        return matrix


class FakeC(ps.terms.Term):
    tag = "fake C"
    repr = "Fake repr 2"
    mixing_rank = 2
    particle_type = "spins 1/2"
    ensemble = "parity grand canonical"

    @staticmethod
    def apply(params: dict, matrix: np.ndarray, sector: tuple[int, int]):
        matrix[:] += 4.0
        return matrix


# pylint: enable=R0903,C0115


ps.terms.utils.register_term_type(FakeA)
ps.terms.utils.register_term_type(FakeB)
ps.terms.utils.register_term_type(FakeC)


@pytest.mark.parametrize(
    "params, expected",
    [
        ({"sites": 4, "particles": 2, "terms": {"fake A": {(0, 1): 1.0}}}, [(4, 2)]),
        (
            {"sites": 4, "terms": {"fake B": {(0, 1): 1.0}}},
            [(4, 0), (4, 1), (4, 2), (4, 3), (4, 4)],
        ),
        (
            {"sites": 4, "parity": 0, "terms": {"fake C": {(0, 1): 1.0}}},
            [(4, 0), (4, 2), (4, 4)],
        ),
    ],
)
def test_model_init_sectors(params, expected):
    model = ps.Model(params)
    assert model.sectors == expected


@pytest.mark.parametrize(
    "params, expected",
    [
        ({"sites": 4, "particles": 2, "terms": {"fake A": {(0, 1): 1.0}}}, []),
        (
            {"sites": 4, "terms": {"fake B": {(0, 1): 1.0}}},
            [(0, 1), (1, 2), (2, 3), (3, 4)],
        ),
        (
            {"sites": 4, "parity": 0, "terms": {"fake C": {(0, 1): 1.0}}},
            [(0, 1), (1, 2)],
        ),
    ],
)
def test_model_init_mixing_sectors(params, expected):
    model = ps.Model(params)
    assert model.mixing_sectors == expected


def test_model_build_hamiltonian_case1():
    params = {
        "sites": 2,
        "terms": {
            "fake A": {(0, 1): 1.0},
            "fake B": {(0, 1): 1.0},
            "fake C": {(0, 1): 1.0},
        },
    }
    model = ps.Model(params)
    h = model.build_hamiltonian()

    npt.assert_array_equal(
        h,
        np.array(
            [
                [1.0, 2, 2, 4],
                [2.0, 1, 1, 2],
                [2.0, 1, 1, 2],
                [4.0, 2, 2, 1],
            ]
        ),
    )


def test_model_build_hamiltonian_case2():
    params = {
        "sites": 3,
        "terms": {
            "fake A": {(0, 1): 1.0},
            "fake B": {(0, 1): 1.0},
            "fake C": {(0, 1): 1.0},
        },
    }
    model = ps.Model(params)
    h = model.build_hamiltonian()

    npt.assert_array_equal(
        h,
        np.array(
            [
                [1.0, 2.0, 2.0, 2.0, 4.0, 4.0, 4.0, 0.0],
                [2.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 4.0],
                [2.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 4.0],
                [2.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 4.0],
                [4.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 2.0],
                [4.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 2.0],
                [4.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 2.0],
                [0.0, 4.0, 4.0, 4.0, 2.0, 2.0, 2.0, 1.0],
            ]
        ),
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
@pytest.mark.parametrize("sparsity", ["dense", "sparse"])
def test_model_dtype(dtype, sparsity):
    m = ps.Model({})
    h = m.build_hamiltonian(sparsity=sparsity, dtype=dtype)
    assert h.dtype == dtype
