r"""
## Supported terms

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
