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


def test_model_update_terms():
    params = {"terms": {"t": {(0, 1): 1.0}, "Delta": {(0, 1): 2.0}}}
    model = ps.Model(params)
    model.update_terms({"t": {(0, 1): 2.0, (1, 2): 2.0}})
    assert model.terms == {"t": {(0, 1): 2.0, (1, 2): 2.0}, "Delta": {(0, 1): 2.0}}


def test_model_update_terms_exception():
    with pytest.raises(ValueError):
        params = {"terms": {"t": {(0, 1): 1.0}}}
        model = ps.Model(params)
        model.update_terms({"t": {(0, 1): 2.0, (1, 2): 2.0}, "Delta": {(0, 1): 2.0}})


@pytest.mark.parametrize(
    "state, expected",
    [
        ("000", np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])),
        ("001", np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])),
        ("010", np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])),
        ("100", np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])),
        ("111", np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])),
    ],
)
def test_model_build_base_state(state: str, expected: np.ndarray):
    model = ps.Model({"sites": 3, "terms": {"hx": {}}})
    s = model.build_base_state(state)
    npt.assert_array_equal(s, expected)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_model_build_base_state_dtype(dtype):
    model = ps.Model({"sites": 3, "terms": {"hx": {}}})
    s = model.build_base_state("000", dtype=dtype)
    assert s.dtype == dtype


def test_model_build_base_state_gpu_error():
    module = ps.model.cp
    with pytest.raises(ImportError):
        ps.model.cp = None
        model = ps.Model({"sites": 3, "terms": {"hx": {}}})
        _ = model.build_base_state("000", device="gpu")
        ps.model.cp = module


def test_model_build_base_state_not_supported_device():
    with pytest.raises(ValueError):
        model = ps.Model({"sites": 3, "terms": {"hx": {}}})
        _ = model.build_base_state("000", device="qwerty")
