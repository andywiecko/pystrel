"""
Utility functions related for obtaining matrix spectra.
"""
import typing
import numpy as np
import scipy.sparse as nps  # type: ignore
import scipy.sparse.linalg as nps_linalg  # type: ignore

try:
    import cupy as cp  # type: ignore
    import cupy.sparse.linalg as cps_linalg  # type: ignore
except ImportError:
    cp = None
    cps_linalg = None


def get_partial_spectrum(
    operator: np.ndarray | nps.csr_array | typing.Any,
    k: int = 6,
    *,
    which: typing.Literal["SA", "SM", "LA", "LM"] = "SA",
    ncv: int | None = None,
    maxiter: int | None = None,
    tol: float = 0,
    return_eigenvectors: bool = True,
) -> np.ndarray | tuple[np.ndarray, np.ndarray] | typing.Any:
    """
    Finds `k` solutions to eigenproblem for the given real symmetric/complex Hermitian `operator`.

    Parameters
    ----------
    operator : np.ndarray | nps.csr_array | typing.Any
        Real symmetric/complex Hermitian operator.
    k : int, optional
        The number of eigenvalues and eigenvectors to compute. Must be `1 <= k < n.`, by default 6
    which : typing.Literal["SA", "SM", "LA", "LM"], optional
        - `"LM"`: finds `k` largest (in magnitude) eigenvalues.
        - `"LA"`: finds `k` largest (algebraic) eigenvalues.
        - `"SA"`: finds `k` smallest (algebraic) eigenvalues.

        **Note:** `scipy.sparse.linalg.eigsh` have additional option `"SM"` for smallest magnitude,
        however, it's not implemented for `cupy.sparse` yet.

        By default `"SA"`.
    ncv : int | None, optional
        The number of Lanczos vectors generated. Must be `k + 1 < ncv < n`.
        If `None`, default value is used, by default `None`.
    maxiter : int | None, optional
        Maximum number of Lanczos update iterations.
        If `None`, default value is used, by default `None`.
    tol : float, optional
        Tolerance for residuals `||Ax - wx||`. If `0`, machine precision is used, by default 0.
    return_eigenvectors : bool, optional
        If `True`, returns eigenvectors in addition to eigenvalues, by default `True`.
        Eigenvectors are stored in rows.

    Returns
    -------
    np.ndarray | tuple[np.ndarray, np.ndarray] | typing.Any
        Eigenvalues (and eigenvectors) sorted in ascending order.

    See also
    --------
    See [`scipy.sparse.linalg.eigsh`][scipy_eigs] and [`cupy.sparse.linalg.eigsh`][cupy_eigs]
    for more details.

    [scipy_eigs]:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html#scipy.sparse.linalg.eigsh
    [cupy_eigs]:
        https://docs.cupy.dev/en/stable/reference/generated/cupyx.scipy.sparse.linalg.eigsh.html
    """
    xp = np if cp is None else cp.get_array_module(operator)
    xp_linalg = nps_linalg if xp is np else cps_linalg

    ev = xp_linalg.eigsh(
        operator,
        k,
        which=which,
        ncv=ncv,
        maxiter=maxiter,
        tol=tol,
        return_eigenvectors=return_eigenvectors,
    )

    return (ev[0], ev[1].T) if return_eigenvectors else ev[ev.argsort()]


def get_full_spectrum(
    operator: np.ndarray | typing.Any, compute_eigenvectors: bool = True
) -> np.ndarray | tuple[np.ndarray, np.ndarray] | typing.Any:
    """
    Finds solutions to eigenproblem for the given real symmetric/complex Hermitian `operator`.

    Parameters
    ----------
    operator : np.ndarray | typing.Any
        Real symmetric/complex Hermitian operator.
    compute_eigenvectors : bool, optional
        If `True`, computes eigenvectors in addition to eigenvalues, by default `True`.
        Eigenvectors are stored in rows.

    Returns
    -------
    np.ndarray | tuple[np.ndarray, np.ndarray] | typing.Any
        Eigenvalues (and eigenvectors).

    See also
    --------

    See [`numpy.linalg.eigh`][numpy_eigh], [`numpy.linalg.eigvalsh`][numpy_eigvalsh],
    [`cupy.linalg.eigh`][cupy_eigh], and [`cupy.linalg.eigvalsh`][cupy_eigvalsh] for more details.

    [numpy_eigh]:
        https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigh.html
    [numpy_eigvalsh]:
        https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigvalsh.html
    [cupy_eigh]:
        https://docs.cupy.dev/en/stable/reference/generated/cupy.linalg.eigh.html
    [cupy_eigvalsh]:
        https://docs.cupy.dev/en/stable/reference/generated/cupy.linalg.eigvalsh.html
    """
    xp = np if cp is None else cp.get_array_module(operator)

    match compute_eigenvectors:
        case True:
            e, v = xp.linalg.eigh(operator)
            return e, v.T

        case False:
            return xp.linalg.eigvalsh(operator)
