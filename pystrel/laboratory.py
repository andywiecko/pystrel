"""
The laboratory module contains utilities related to the measurement of quantum states.
"""
import typing
import numpy as np
import scipy.sparse as nps  # type: ignore


def measure(
    operator: np.ndarray | nps.csr_array | typing.Any, state: np.ndarray
) -> np.float64 | typing.Any:
    r"""
    Measure `operator` $A$ in a `state` $|\psi\rangle$

    $$
        \langle \psi | A | \psi \rangle.
    $$

    Parameters
    ----------
    operator : np.ndarray | nps.csr_array | typing.Any
        Operator instance (sparse/dense/gpu/cpu)
    state : np.ndarray
        State instance (cpu/gpu)

    Returns
    -------
    np.float64
        Operator measurement.
    """
    return np.conj(state) @ operator @ state
