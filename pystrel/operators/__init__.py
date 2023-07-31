r"""
## Supported operators

| tag          | operator       |        name                        | `kwargs`        |
|--------------|----------------|------------------------------------|-----------------|
| [`"H"`][H]   | $H$            | Hamiltonian                        |   None          |
| [`"N"`][N]   | $N=\sum_i n_i$ | total particle number operator     |   None          |
| [`"n"`][n]   | $n_i$          | particle number operator           | `i`: site index |

[H]: operators/impl.html#Operator_H
[N]: operators/impl.html#Operator_N
[n]: operators/impl.html#Operator_n
"""
from .impl import *
from . import utils
