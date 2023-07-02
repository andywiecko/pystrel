r"""
## Supported terms

**Spinless fermions**

| tag          | Operator                                                                         |
|--------------|----------------------------------------------------------------------------------|
| `t`          | $\sum_{i,j} \left(t_{ij} \, a_i^\dagger a_j + \text{h.c.}\right)$                |
| `V`          | $\sum_{i,j} V_{ij} \, n_i n_j$                                                   |
| `Delta`      | $\sum_{i,j} \left(\Delta_{ij} \, a_i^\dagger a_j^\dagger + \text{h.c.}\right)$   |
| `mu`         | $\mu N = \mu \sum_{i} \, n_i$                                                    |
| `epsilon`    | $\sum_i \epsilon_i \, n_i$                                                       |

**Spins 1/2**

| tag          | Operator                                                                         |
|--------------|----------------------------------------------------------------------------------|
| `Jz`         | $\sum_{i,j} J_{ij}^z \, \sigma_i^z \sigma_j^z$                                   |
| `hz`         | $\sum_{i} h_{i}^z \, \sigma_i^z$                                                 |
| `gamma`      | $\sum_{i,j} \left(\gamma_{ij} \, \sigma_i^+\sigma_j^- + \text{h.c.}\right)$      |
| `hx`         | $\sum_{i} h_{i}^x \, \sigma_i^x$                                                 |

"""
from .impl import *
from . import utils
