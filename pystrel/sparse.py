"""
This module contains a simple helper class `Sparse` 
designed to facilitate the incremental construction of sparse matrices.

Example:

```python
m = Sparse(shape=(10, 10))

m[0, 1] += 4.0
m[2, 1] += 3.0
m[1, 1] += 1.0
```

This class supports the concept of spans, enabling 
the addition of elements within a specific submatrix view:

```python
m = Sparse(shape=(10, 10))
view = m[5:10, 5:10]
view[0, 0] += 1.0 # will insert 1.0 for m(5, 5)
```

Once elements have been inserted, you can construct a `scipy.sparse.csr_matrix` using:

```python
m = Sparse(shape=(10, 10))
...
m.to_csr()
```

The helper implementation includes index range safety checks, even for spans.
However, it's important to note that only full slicing is currently implemented, 
meaning that both the `start` and `end` indices must be non-`None`.
"""
import typing
import numpy as np
import scipy.sparse as sp  # type: ignore


class Sparse:
    """
    Sparse matrix helper for efficient incremental construction.
    """

    class __AddElementView:  # pylint: disable=C0103
        def __init__(self, sparse: "Sparse", index):
            self.sparse = sparse
            self.index = index

        def __iadd__(self, other):
            return other

        def __setitem__(self, index, value):
            self.sparse.add(index, value)

    def __init__(self, shape):
        self.vals = []
        self.rows = []
        self.cols = []
        self.shape = shape
        self.offset = (0, 0)
        self.__add_element_view = Sparse.__AddElementView(self, (-1, -1))

    def __getitem__(self, index):
        if isinstance(index, tuple) and len(index) == 2:
            x, y = index

            if isinstance(x, int) and isinstance(y, int):
                if index[0] not in range(self.shape[0]) or index[1] not in range(
                    self.shape[1]
                ):
                    raise IndexError(f"Index ({index}) is out of range {self.shape}!")
                self.__add_element_view.index = (
                    index[0] + self.offset[0],
                    index[1] + self.offset[1],
                )
                return self.__add_element_view

            if isinstance(x, slice) and isinstance(y, slice):
                x, y = index
                x0, x1 = x.start, x.stop
                y0, y1 = y.start, y.stop

                if (
                    x0 not in range(self.shape[0])
                    or x1 not in range(self.shape[0] + 1)
                    or y0 not in range(self.shape[1])
                    or y1 not in range(self.shape[1] + 1)
                ):
                    raise IndexError(f"Index {index} is out of range {self.shape}!")

                shape = (x1 - x0, y1 - y0)
                span = Sparse(shape)
                span.offset = (x0, y0)
                span.vals = self.vals
                span.rows = self.rows
                span.cols = self.cols
                return span

        raise ValueError("Invalid index format")

    def __setitem__(self, index, value):
        if (
            isinstance(index, tuple)
            and len(index) == 2
            and isinstance(index[0], int)
            and isinstance(index[1], int)
        ):
            self.add(index, value)
            return

        raise ValueError("Invalid index format")

    def add(
        self, index: tuple[int, int], value: np.float64 | np.complex128 | typing.Any
    ):
        """
        Add `value` with `index`.

        **Note:** duplicates will be added when constructing `csr` matrix.

        Parameters
        ----------
        index : (int, int)
            2d-index.
        value : np.float64 | np.complex128 | typing.Any
            Value to insert.
        """
        self.vals.append(value)
        self.rows.append(index[0] + self.offset[0])
        self.cols.append(index[1] + self.offset[1])

    def to_csr(self, dtype=None) -> sp.csr_array:
        """Construct `scipy.sparse.csr_array`.

        Parameters
        ----------
        dtype : npt.DTypeLike, optional
            Any object that can be interpreted as a numpy data type.

        Returns
        -------
        sp.csr_array
            Sparse array.
        """
        coo = sp.coo_matrix(
            (self.vals, (self.rows, self.cols)), shape=self.shape, dtype=dtype
        )
        coo.sum_duplicates()
        return coo.tocsr()
