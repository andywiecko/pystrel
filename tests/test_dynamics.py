import pytest
import numpy as np
import numpy.testing as npt
import pystrel as ps


@pytest.mark.parametrize("a", [0.1, 1.0, 2.0])
@pytest.mark.parametrize("b", [0.5, 1.0, 2.5])
@pytest.mark.parametrize("method", ["rk4"])
def test_time_independent_hamiltonian_rabi_oscilations(a: float, b: float, method):
    r"""
    Rabi oscillations example
    ------

    The system is described with the Hamiltonian:

    $$
    H = a \sigma_z + b \sigma_x
    $$

    The analytic solution for the Schrödinger equation is given by:

    $$
    |\langle 1|\psi(t)\rangle|^2 = \frac {b^2}{a^2 + b^2} \sin^2(\sqrt{a^2 + b^2} t)
    $$
    """
    psi = np.array([1.0, 0], dtype=np.complex128)
    sz = np.array([[1.0, 0], [0, -1]])
    sx = np.array([[0.0, 1], [1, 0]])
    h = a * sz + b * sx

    t = 0.0
    dt = 0.01
    T = []
    p = []

    for _ in range(100):
        psi = ps.propagate(psi, h, t, dt, method)
        t += dt

        T.append(t)
        p.append(np.abs(psi[1]) ** 2)

    expected = (
        b**2 / (a**2 + b**2) * np.sin(np.sqrt(a**2 + b**2) * np.array(T)) ** 2
    )
    result = np.array(p)

    error = np.sum((expected - result) ** 2)
    npt.assert_almost_equal(error, 0)


@pytest.mark.parametrize("a", [2.0, 1.1, 0.33])
@pytest.mark.parametrize("b", [1.0, 0.1, 3.5])
@pytest.mark.parametrize("method", ["rk4"])
def test_time_dependent_hamiltonian_landau_zener_transition(a: float, b: float, method):
    r"""
    Landau-Zener transition example
    -------

    The system is described with the Hamiltonian:

    $$
    H(t) = \frac12 (a \sigma_z t + b \sigma_x)
    $$

    The analytic solution for the probability of state $|0\rangle$ is given by:

    $$
    P = 1 - \exp[-\pi b^2 / (2a)]
    $$
    """
    sz = np.array([[1.0, 0], [0, -1]])
    sx = np.array([[0.0, 1], [1, 0]])
    h = lambda t: 0.5 * (a * sz * t + b * sx)  # pylint: disable=C3001

    T = 10.0
    t = -T
    dt = 0.05

    _, v = ps.spectrum.get_full_spectrum(h(-T))
    psi = v[0].astype(np.complex128)

    while t < T:
        psi = ps.propagate(psi, h, t, dt, method)
        t += dt

    _, v = ps.spectrum.get_full_spectrum(h(+T))
    psi0 = v[0].astype(np.complex128)
    p = ps.project(psi0, psi)
    p_expected = 1 - np.exp(-np.pi / 2.0 * b**2 / a)

    assert np.abs(p - p_expected) < 0.05


def test_propagate_method_not_implemented():
    with pytest.raises(NotImplementedError):
        ps.dynamics.propagate(
            state=None, hamiltonian=None, t0=0.0, dt=0.0, method="qwerty"  # type: ignore
        )
