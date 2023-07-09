r"""
## Supported operators

| tag          | operator       |        name                        | `kwargs`        |
|--------------|----------------|------------------------------------|-----------------|
| `H`          | $H$            | Hamiltonian                        |   None          |
| `N`          | $N=\sum_i n_i$ | total particle number operator     |   None          |
| `n`          | $n_i$          | particle number operator           | `i`: site index |

"""
from .impl import *
from . import utils
