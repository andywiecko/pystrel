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
import scipy.special as sps  # type: ignore
try:
    import cupy as cp  # type: ignore
    import cupy.sparse as cps  # type: ignore
except ImportError:
    cp = None
    cps = None

from . import terms
from . import sectors


class Model:
    """
    Basic class used for representing the system.
    It is used for constructing hamiltonian as well as operators
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
        self.terms: dict[str, dict] = params.get('terms', {})

        self.__sector_offset: list[int] = []
        self.__sector_size: list[int] = []
        self.__size = 0
        self.sectors: list[tuple[int, int]] = []
        ensemble = terms.utils.identify_ensemble(self.terms)
        self.sectors = params.get(
            'sectors', sectors.generate_sectors(ensemble, self.params))

        ranks = terms.utils.collect_mixing_sector_ranks(self.terms)
        self.mixing_sectors: list[tuple[int, int]] = sectors\
            .generate_mixing_sectors(ranks, self.sectors)

        self.__init_sectors()

    def __init_sectors(self):
        offset = 0
        for L, N in self.sectors:
            size = int(sps.binom(L, N))
            self.__sector_size.append(size)
            self.__sector_offset.append(offset)
            offset += size
            self.__size += size

    def build_hamiltonian(
        self,
        device: typing.Literal["cpu", "gpu"] = "cpu",
        sparsity: typing.Literal["sparse", "dense"] = "dense",
        dtype: npt.DTypeLike = None
    ) -> np.matrix | nps.csr_matrix | typing.Any:
        """
        Constructs hamiltonian on the selected `device` and matrix `sparsity` type.

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
        np.matrix | nps.csr_matrix | Any
            Hamiltonian matrix representation.

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

        shape = (self.__size, self.__size)
        match sparsity:
            case "dense":
                matrix = np.matrix(np.zeros(shape), dtype=dtype)
                self.__build_local_sectors(matrix)
                self.__build_mixing_sectors(matrix)
                matrix = matrix + np.matrix(np.triu(matrix, 1)).H
                if device == "cpu":
                    return matrix
                if device == "gpu":
                    return cp.array(matrix)
                raise ValueError()

            case "sparse":
                matrix = nps.dok_matrix(shape, dtype=dtype)
                self.__build_local_sectors(matrix)
                self.__build_mixing_sectors(matrix)
                matrix = nps.csr_matrix(matrix + nps.triu(matrix, 1).H)
                if device == "cpu":
                    return matrix
                if device == "gpu":
                    return cps.csr_matrix(matrix)
                raise ValueError()

            case _:
                raise ValueError()

    def __build_local_sectors(self, matrix: np.ndarray | nps.dok_matrix):
        for i, sector in enumerate(self.sectors):
            start = self.__sector_offset[i]
            end = start + self.__sector_size[i]
            matrix[start:end, start:end] = terms.utils.apply(
                terms=self.terms,
                matrix=matrix[start:end, start:end],
                sector=sector,
                rank=0
            )

    def __build_mixing_sectors(self, matrix: np.ndarray | nps.dok_matrix):
        for id0, id1 in self.mixing_sectors:
            start0 = self.__sector_offset[id0]
            start1 = self.__sector_offset[id1]
            end0 = start0 + self.__sector_size[id0]
            end1 = start1 + self.__sector_size[id1]
            matrix[start0:end0, start1:end1] = terms.utils.apply(
                terms=self.terms,
                matrix=matrix[start0:end0, start1:end1],
                sector=self.sectors[id0],
                rank=self.sectors[id1][1]-self.sectors[id0][1]
            )

    def __str__(self):
        return \
            "# " + 16 * '#' + ' Info ' + 16 * '#' + "\n"\
            f"# Particle type: {terms.utils.identify_particle_type(self.terms)}\n"\
            "# Model: Ĥ = " + ' +\n#            '\
            .join(terms.utils.term__str__(t) for t in self.terms) + '\n'\
            f"# Space size: {self.__size} × {self.__size}\n"\
            f"# Ensemble: {terms.utils.identify_ensemble(self.terms)}\n"\
            f"# Sectors: {self.sectors}\n"\
            "# Terms:\n"\
            + "\n".join('# - ' + i + ': ' + str(j)
                        for i, j in self.terms.items())
