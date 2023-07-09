import pytest
import numpy as np
import numpy.testing as npt
from pystrel import operators
from pystrel.sectors import Sectors

# pylint: disable=C0103,W0223,R0903,C0115


def test_hermitian_operator_dense():
    sectors = Sectors({"sectors": [(2, 0), (2, 1), (2, 2)]})
    terms = {
        "t": {(0, 1): 1.0 + 1.0j},
        "Delta": {(0, 1): 0.5j},
    }

    h = operators.HermitianOperator.build(
        sectors, terms, sparsity="dense", dtype=np.complex128
    )

    npt.assert_array_equal(h, np.conj(h.T))


def test_hermitian_operator_sparse():
    sectors = Sectors({"sectors": [(2, 0), (2, 1), (2, 2)]})
    terms = {
        "t": {(0, 1): 1.0 + 1.0j},
        "Delta": {(0, 1): 0.5j},
    }

    h = operators.HermitianOperator.build(
        sectors, terms, sparsity="sparse", dtype=np.complex128
    )

    npt.assert_array_equal(h.toarray(), h.conj().T.toarray())


def test_operator_not_implemented():
    class Fake(operators.Operator):
        ...

    sectors = Sectors({"sectors": [(2, 0), (2, 1), (2, 2)]})
    terms = {
        "t": {(0, 1): 1.0 + 1.0j},
        "Delta": {(0, 1): 0.5j},
    }

    with pytest.raises(NotImplementedError):
        Fake.build(sectors, terms)


def test_hermitian_operator_throw():
    sectors = Sectors({"sectors": [(2, 0), (2, 1), (2, 2)]})
    terms = {
        "t": {(0, 1): 1.0 + 1.0j},
        "Delta": {(0, 1): 0.5j},
    }

    with pytest.raises(ValueError):
        _ = operators.HermitianOperator.build(
            sectors, terms, sparsity="qwerty", dtype=np.complex128
        )


def test_utils_duplicated_tag():
    class FakeA(operators.Operator):
        tag = "fake"

    class FakeB(operators.Operator):
        tag = "fake"

    with pytest.raises(operators.utils._DuplicateTagError):  # pylint: disable=W0212
        operators.utils.register_term_type(FakeA)
        operators.utils.register_term_type(FakeB)


def test_operator_H():
    sectors = Sectors({"sectors": [(2, 0), (2, 1), (2, 2)]})
    terms = {"t": {(0, 1): 1.0}}

    h = operators.Operator_H.build(sectors, terms)

    eig = np.linalg.eigvalsh(h, "U")
    npt.assert_array_equal(eig, [-1, 0, 0, 1])


def test_operator_N():
    sectors = Sectors({"sectors": [(3, 0), (3, 1), (3, 3)]})
    terms = {"t": {(0, 1): 1.0}}

    N = operators.Operator_N.build(sectors, terms)

    eig = np.linalg.eigvalsh(N, "U")
    npt.assert_array_equal(eig, [0, 1, 1, 1, 3])


def test_operator_n():
    sectors = Sectors({"sectors": [(3, 0), (3, 1), (3, 3)]})
    terms = {"t": {(0, 1): 1.0}}

    n = operators.Operator_n.build(sectors, terms, i=1)

    eig = np.linalg.eigvalsh(n, "U")
    npt.assert_array_equal(eig, [0, 0, 0, 1, 1])


def test_operator_Nn():
    sectors = Sectors({"sectors": [(3, 0), (3, 1), (3, 3)]})
    terms = {"t": {(0, 1): 1.0}}

    N = operators.Operator_N.build(sectors, terms)
    Ni = np.zeros((5, 5))
    for i in range(3):
        Ni += operators.Operator_n.build(sectors, terms, i=i)

    npt.assert_array_equal(N, Ni)


# pylint: enable=C0103,W0223,R0903,C0115
