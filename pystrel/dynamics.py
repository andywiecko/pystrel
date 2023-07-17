r"""
The `dynamics` module contains utilities for solving time-dependent Schrödinger equation.
"""
import typing
import numpy as np
import scipy.sparse as nps  # type: ignore


def propagate(
    state: np.ndarray | typing.Any,
    hamiltonian: typing.Callable[[float], np.ndarray | nps.csr_array | typing.Any]
    | np.ndarray
    | nps.csr_array
    | typing.Any,
    t0: float,
    dt: float,
    method: typing.Literal["rk4"] = "rk4",
) -> np.ndarray | typing.Any:
    r"""
    Propagate the `state` in time assuming `hamiltonian`.
    In other words it solves step of the time-dependent Schrödinger equation,
    given by

    $$
        i\partial_t|\psi\rangle = H |\psi\rangle
    $$

    Parameters
    ----------
    state : np.ndarray | typing.Any
        State $|\psi\rangle$ to propagate.
    hamiltonian :  ((float) -> (ndarray | csr_array | Any)) | ndarray | csr_array | Any
        Hamiltonian used for time propagation, `hamiltonian` can be:
        - "callable -> array", then $H = H(t)$,
        - const, i.e. "array".

    t0 : float
        Initial time $t$.
    dt : float
        Time step $\Delta t$.
    method : typing.Literal["rk4"], optional
        Numerical scheme used for propagation, by default `"rk4"`:
        - if `"rk4"` is selected then fourth order Runge-Kutta algorithm is used.
          Propagation of the state is obtained in the following way:
        $$
            |\psi(t+\Delta t)\rangle =
                |\psi(t)\rangle + \frac16(|k_1\rangle+2|k_2\rangle+2|k_3\rangle+|k_4\rangle),
        $$
          where corresponding states $|k_1\rangle,\,|k_2\rangle,\,|k_3\rangle,\,|k_4\rangle$
          are given by:

            - $|k_1\rangle =-i\Delta t \hat H(t) |\psi(t)\rangle $
            - $|k_2\rangle =-i\Delta t \hat H(t+\Delta t/2) (|\psi(t)\rangle +\frac12|k_1\rangle)$
            - $|k_3\rangle =-i\Delta t \hat H(t+\Delta t/2) (|\psi(t)\rangle +\frac12|k_2\rangle)$
            - $|k_4\rangle =-i\Delta t \hat H(t+\Delta t) (|\psi(t)\rangle +|k_3\rangle)$

           See [wikipage][rk4] for more details.

        [rk4]: https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
    Returns
    -------
    np.ndarray | typing.Any
        Propagated state $|\psi(t + \Delta t)\rangle$.

    Raises
    ------
    NotImplementedError
        When provided `method` is not implemented.
    """
    if callable(hamiltonian):
        match method:
            case "rk4":
                return __rk4_solver(state, hamiltonian, t0, dt)

    match method:
        case "rk4":
            return __rk4_solver_const(state, hamiltonian, dt)

    raise NotImplementedError()


def __rk4_solver(
    state: np.ndarray,
    hamiltonian: typing.Callable[[float], np.ndarray | nps.csr_array | typing.Any],
    t0: float,
    dt: float,
) -> np.ndarray | typing.Any:
    psi = state
    h = hamiltonian(t0)
    k1 = -1.0j * dt * h @ psi

    h = hamiltonian(t0 + 0.5 * dt)
    k2 = -1.0j * dt * h @ (psi + 0.5 * k1)
    k3 = -1.0j * dt * h @ (psi + 0.5 * k2)

    h = hamiltonian(t0 + dt)
    k4 = -1.0j * dt * h @ (psi + k3)

    psi += 1.0 / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return psi


def __rk4_solver_const(
    state: np.ndarray,
    hamiltonian: np.ndarray | nps.csr_array | typing.Any,
    dt: float,
) -> np.ndarray | typing.Any:
    psi = state
    h = hamiltonian
    psi += (
        -1.0j * dt * h @ psi
        + (-1.0j * dt) ** 2 / 2.0 * h @ (h @ psi)
        + (-1.0j * dt) ** 3 / 6.0 * h @ (h @ (h @ psi))
        + (-1.0j * dt) ** 4 / 24.0 * h @ (h @ (h @ (h @ psi)))
    )
    return psi
