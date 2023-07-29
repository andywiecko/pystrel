# pylint: disable=C0301
r"""
## Supported terms

**Spinless fermions**

| tag                    | Operator                                                                         |
|------------------------|----------------------------------------------------------------------------------|
| [`"t"`][t]             | $\sum_{i,j} \left(t_{ij} \, a_i^\dagger a_j + \text{h.c.}\right)$                |
| [`"V"`][V]             | $\sum_{i,j} V_{ij} \, n_i n_j$                                                   |
| [`"Delta"`][Delta]     | $\sum_{i,j} \left(\Delta_{ij} \, a_i^\dagger a_j^\dagger + \text{h.c.}\right)$   |
| [`"mu"`][mu]           | $\mu N = \mu \sum_{i} \, n_i$                                                    |
| [`"epsilon"`][epsilon] | $\sum_i \epsilon_i \, n_i$                                                       |

**Spins 1/2**

| tag                | Operator                                                                         |
|--------------------|----------------------------------------------------------------------------------|
| [`"Jz"`][Jz]       | $\sum_{i,j} J_{ij}^z \, \sigma_i^z \sigma_j^z$                                   |
| [`"hz"`][hz]       | $\sum_{i} h_{i}^z \, \sigma_i^z$                                                 |
| [`"gamma"`][gamma] | $\sum_{i,j} \left(\gamma_{ij} \, \sigma_i^+\sigma_j^- + \text{h.c.}\right)$      |
| [`"hx"`][hx]       | $\sum_{i} h_{i}^x \, \sigma_i^x$                                                 |

[t]:terms/impl.html#Term_t
[V]:terms/impl.html#Term_V
[Delta]:terms/impl.html#Term_Delta
[mu]:terms/impl.html#Term_mu
[epsilon]:terms/impl.html#Term_epsilon
[Jz]:terms/impl.html#Term_Jz
[hz]:terms/impl.html#Term_hz
[gamma]:terms/impl.html#Term_gamma
[hx]:terms/impl.html#Term_hx
"""
from .impl import *
from . import utils
