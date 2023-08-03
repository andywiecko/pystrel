r"""
The `dynamics` module contains utilities for solving time-dependent Schrödinger equation.
"""
import typing
import numpy as np
import scipy.sparse as nps  # type: ignore
import scipy.special as sps  # type: ignore

# pylint: disable=R0801
try:
    import cupy as cp  # type: ignore
    import cupy.sparse as cps  # type: ignore
except ImportError:
    cp = None
    cps = None
# pylint: enable=R0801


def propagate(
    state: np.ndarray | typing.Any,
    hamiltonian: typing.Callable[[float], np.ndarray | nps.csr_array | typing.Any]
    | np.ndarray
    | nps.csr_array
    | typing.Any,
    t0: float,
    dt: float,
    method: typing.Literal["rk4", "cheb"] = "rk4",
    **kwargs,
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
    method : typing.Literal["rk4", "cheb"], optional
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

          **kwargs:** None

           ---
           
        - if `"cheb"` is selected then [Chebyshev polynomial expansion algorithm][cheb] is used.
          Propagation of the state is obtained in the following way:
          $$
              |\psi(t + \Delta t)\rangle = \sum_{k=0}^m \alpha_k(\rho\Delta t) |\nu_k\rangle,
          $$
          where $m$ is polynomial order and:
        
          - $|\nu_0\rangle = |\psi(t)\rangle$, $|\nu_1\rangle =  H' |\nu_0\rangle$, 
            and $|\nu_{k+1}\rangle = 2 H' |\nu_k\rangle - |\nu_{k-1}\rangle$,
          - $H' = H / \rho$, the $\rho$ scaling factor ensures that the spectrum of $H$
            is in the range $[-1, 1]$, which is required by Chebyshev polynomials. 
            We use the matrix spectral theorem and the Frobenius norm in calculations: 
            $\rho = \| H \|_F = \sqrt{\sum_{ij}|H_{ij}|^2}$. **Note:** the *full* linear scaling 
            can also be used in the numerical scheme, i.e. $H = a H' + b$.
          - $
              \alpha_k(t) = \begin{cases}
                1, \, \text{for}\, k = 0\\\\
                2, \, \text{else}
               \end{cases}\,\,\,\cdot(-i)^k J_k(t)
            $, and $J_k(t)$ is $k^{\text{th}}$ order Bessel function of the first kind.

          **kwargs:**
        
          - `m`: polynomial order $m$, default `m = 5`.

    Returns
    -------
    np.ndarray | typing.Any
        Propagated state $|\psi(t + \Delta t)\rangle$.

    Raises
    ------
    NotImplementedError
        When provided `method` is not implemented.

    References
    ----------
    1. Runge, Carl. "Über die numerische Auflösung von Differentialgleichungen." 
       [Math. Ann. 46.2 (1895): 167-178][runge].
    2. Kutta, Wilhelm. "Beitrag zur näherungsweisen Integration totaler Differentialgleichungen."
       [Zeitschrift für Mathematik und Physik, 46: 435–453][kutta]
    3. Fehske, Holger, et al. "Numerical approaches to time evolution of complex quantum systems."
       [Phys. Lett. A 373.25 (2009): 2182-2188][cheb].
    
    [rk4]: https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
    [cheb]: https://doi.org/10.1016/j.physleta.2009.04.022
    [runge]: https://doi.org/10.1007/BF01446807
    [kutta]: https://archive.org/details/zeitschriftfrma12runggoog/page/434/mode/2up
    """
    if callable(hamiltonian):
        match method:
            case "rk4":
                return __rk4_solver(state, hamiltonian, t0, dt)
            case "cheb":
                return __cheb_solver_const(
                    state, hamiltonian(t0 + 0.5 * dt), dt, kwargs=kwargs
                )

    match method:
        case "rk4":
            return __rk4_solver_const(state, hamiltonian, dt)
        case "cheb":
            return __cheb_solver_const(state, hamiltonian, dt, kwargs=kwargs)
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


def __cheb_solver_const(
    state: np.ndarray,
    hamiltonian: np.ndarray | nps.csr_array | typing.Any,
    dt: float,
    **kwargs,
) -> np.ndarray | typing.Any:
    def c_k(k, t):
        return (-1.0j) ** k * sps.jv(k, t)

    def norm(h):
        xp = np if cp is None else cp.get_array_module(hamiltonian)
        xps = nps if xp is np else cps
        return (xps if xps.issparse(h) else xp).linalg.norm(h)

    m: int = kwargs.get("m", 5)
    rho = 1.01 * norm(hamiltonian)
    alpha = [(2 if i > 0 else 1.0) * c_k(i, rho * dt) for i in range(m)]
    h = hamiltonian / rho

    k0 = state
    k1 = h @ k0
    psi = alpha[0] * k0 + alpha[1] * k1
    for i in range(2, m):
        ki = 2 * h @ k1 - k0
        psi += alpha[i] * ki
        k0, k1 = k1, ki

    return psi
