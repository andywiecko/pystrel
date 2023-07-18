"""

Supported keys for `Model` `params`:

| key         | values                  | Description                                      |
| ----------- | ----------------------- | ------------------------------------------------ |
| `terms`     | `dict[str, dict]`       | To see supported terms see `terms.py`            |
| `sites`     | `int`                   | number of sites                                  |
| `particles` | `int`                   | number of particles                              |
| `parity`    | `Literal[0, 1]`         | parity sector in parity grand canonical ensemble |
| `sectors`   | `list[tuple[int, int]]` | custom particle sectors                          |

"""
import typing
import numpy as np
import numpy.typing as npt
import scipy.sparse as nps  # type: ignore

try:
    import cupy as cp  # type: ignore
    import cupy.sparse as cps  # type: ignore
except ImportError:
    cp = None
    cps = None

from . import sectors
from .terms import utils as terms_utils
from .operators import utils as operators_utils


class Model:
    """
    Basic class used for representing the system.
    It is used for constructing Hamiltonian as well as operators
    in the given Hilbert space.
    """

    def __init__(self, params: dict):
        """
        Constructs `Model` using the given `params`.

        Parameters
        ----------
        params : dict
            Dictionary of parameters. Can include keys like:
            - `terms`
            - `particles`
            - `sites`
            - `sectors`

            See module documentation for more details.
        """
        self.params: dict[str, typing.Any] = params
        self.terms: dict[str, dict] = params.get("terms", {})
        self.sectors = sectors.Sectors(params)

    def update_terms(self, terms: dict):
        """
        Update the `terms` for the model.

        Parameters
        ----------
        terms : dict
            Dictionary with terms to update.

        Examples
        --------
        This method allows for updating the values of existing terms or adding new terms.
        Terms updates is based on dictionary `|=` operator behavior

        >>> import pystrel as ps
        >>> params = { "terms": {"t": {(0, 1): 1.0 }, "Delta": {(1, 2): 2.0} } }
        >>> model = ps.Model(params)
        >>> model.update_terms({"t": {(0, 1): 2.0, (1, 2): 2.0}})
        >>> # The model now has t(0, 1) = t(1, 2) = 2.0 and Delta(1, 2) = 2.0

        Raises
        ------
        ValueError
            If the provided `terms` contain keys that were not present during initialization.
            This restriction is in place for safety reasons.
            Consider creating a new `Model` instance
            or adding new terms with 'zero value' during initialization.
        """
        if self.terms.keys() != (self.terms | terms).keys():
            raise ValueError(
                "The provided `terms` contains keys that were not present during initialization! "
                "This restriction is in place for safety reasons. "
                "Consider creating a new `Model` instance "
                "or adding new terms with 'zero value' during initialization."
            )

        self.terms |= terms

    def build_hamiltonian(
        self,
        device: typing.Literal["cpu", "gpu"] = "cpu",
        sparsity: typing.Literal["sparse", "dense"] = "dense",
        dtype: npt.DTypeLike = None,
    ) -> np.ndarray | nps.csr_array | typing.Any:
        """
        Constructs Hamiltonian on the selected `device` and matrix `sparsity` type.

        Parameters
        ----------
        device : typing.Literal["cpu", "gpu"], optional
            Device on which hamiltonian should be constructed, by default "cpu".
        sparsity : typing.Literal["sparse", "dense"], optional
            Matrix sparsity type which is used, by default "dense".
            If "sparse" is selected, returns matrix in CSR format.
        dtype : npt.DTypeLike, optional
            Any object that can be interpreted as a numpy data type.

        Returns
        -------
        np.ndarray | nps.csr_array | Any
            Hamiltonian matrix representation.
        """
        return self.build_operator("H", device, sparsity, dtype)

    def build_operator(
        self,
        tag: str,
        device: typing.Literal["cpu", "gpu"] = "cpu",
        sparsity: typing.Literal["sparse", "dense"] = "dense",
        dtype: npt.DTypeLike = None,
        **kwargs,
    ) -> np.ndarray | nps.csr_array | typing.Any:
        """
        Constructs operator of given `tag` on the selected `device` and matrix `sparsity` type.

        Parameters
        ----------
        tag : str
            Tag of selected operator. See `pystrel.operators` to see available operators.
        device : typing.Literal["cpu", "gpu"], optional
            Device on which hamiltonian should be constructed, by default "cpu".
        sparsity : typing.Literal["sparse", "dense"], optional
            Matrix sparsity type which is used, by default "dense".
            If "sparse" is selected, returns matrix in CSR format.
        dtype : npt.DTypeLike, optional
            Any object that can be interpreted as a numpy data type.
        **kwargs : dict, optional
            Additional operators arguments, see `pystrel.operators` for more details.

        Returns
        -------
        np.ndarray | nps.csr_array | typing.Any
            Operator matrix representation

        Raises
        ------
        ImportError
            If `device="gpu"` is selected and `cupy` is not found.
        ValueError
            If not recognized option for `device` or `sparsity` is selected.
        """
        if device == "gpu" and cp is None:
            raise ImportError()
        if device not in ["gpu", "cpu"]:
            raise ValueError()

        matrix = operators_utils.build(
            tag,
            sectors=self.sectors,
            terms=self.terms,
            sparsity=sparsity,
            dtype=dtype,
            **kwargs,
        )

        match (device, sparsity):
            case ("gpu", "dense"):
                return cp.array(matrix)

            case ("gpu", "sparse"):
                return cps.csr_matrix(matrix)

            case _:
                return matrix

    def __str__(self):
        return (
            "# " + 16 * "#" + " Info " + 16 * "#" + "\n"
            f"# Particle type: {terms_utils.identify_particle_type(self.terms)}\n"
            "# Model: Ĥ = "
            + " +\n#            ".join(terms_utils.term__str__(t) for t in self.terms)
            + "\n"
            f"# Space size: {self.sectors.size} × {self.sectors.size}\n"
            f"# Ensemble: {terms_utils.identify_ensemble(self.terms)}\n"
            f"# Sectors: {self.sectors}\n"
            "# Terms:\n"
            + "\n".join("# - " + i + ": " + str(j) for i, j in self.terms.items())
        )
