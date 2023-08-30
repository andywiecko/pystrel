# pystrel

Library for *exact* calculations of strongly correlated systems.

> **Warning**  
> This package is currently in a preview state.
> The API is subject to change without advance notice.
> It is not recommended for production usage.

[![test](https://github.com/andywiecko/pystrel/actions/workflows/test.yml/badge.svg)](https://github.com/andywiecko/pystrel/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/andywiecko/pystrel/branch/main/graph/badge.svg?token=Q9QS5ISW1E)](https://codecov.io/gh/andywiecko/pystrel)
[![gh-pages](https://img.shields.io/github/deployments/andywiecko/pystrel/github-pages?label=gh-pages)][docs]

The main goal of this package is to provide an easy-to-use interface for performing precise calculations on strongly correlated systems. The package offers the following features:

- CPU/GPU agnostic code,
- Customizable system topologies, particle types, ensembles, and more,
- Support for different data types, including real/complex numbers and dense/sparse matrices.

If you encounter any issues or have suggestions for improvement, please feel free to open an issue or submit a pull request.

We hope this library enhances your research on strongly correlated systems. Happy computing!

## Installation

<details open>
<summary> Install with wheel file </summary>

1. Download wheel file from the [releases page](https://github.com/andywiecko/pystrel/releases).
2. Install wheel with `pip`
```
pip install pystrel-0.1.0-py3-none-any.whl
```
</details>

<details>
<summary> Install with <code>git</code></summary>

1. Clone repository 
```
git clone https://github.com/andywiecko/pystrel
```

2. Install `flit` (if necessary)
```
pip install flit
```

3. Install package (including dev tools)
```
flit install
```
</details>

<details>
<summary> Install with <code>pip</code> (TBA with v1.0) </summary>
</details>

## Supported models

### Spinless fermions

| tag          | Operator                                                                         |
|--------------|----------------------------------------------------------------------------------|
| `t`          | $\sum_{i,j} \left(t_{ij} \, a_i^\dagger a_j + \text{h.c.}\right)$                |
| `V`          | $\sum_{i,j} V_{ij} \, n_i n_j$                                                   |
| `Delta`      | $\sum_{i,j} \left(\Delta_{ij} \, a_i^\dagger a_j^\dagger + \text{h.c.}\right)$   |
| `mu`         | $\mu N = \mu \sum_{i} \, n_i$                                                    |
| `epsilon`    | $\sum_i \epsilon_i \, n_i$                                                       |

### Spins 1/2

| tag          | Operator                                                                         |
|--------------|----------------------------------------------------------------------------------|
| `Jz`         | $\sum_{i,j} J_{ij}^z \, \sigma_i^z \sigma_j^z$                                   |
| `hz`         | $\sum_{i} h_{i}^z \, \sigma_i^z$                                                 |
| `gamma`      | $\sum_{i,j} \left(\gamma_{ij} \, \sigma_i^+\sigma_j^- + \text{h.c.}\right)$      |
| `hx`         | $\sum_{i} h_{i}^x \, \sigma_i^x$                                                 |

## Accelerate computation with GPU

This package is designed to easily switch between CPU and GPU target devices. 
To enable GPU computation, you need to install [`CUDA`][CUDA]/[`ROCm`][ROCM] (depending on the GPU manufacturer) and then [`cupy`][cupy].


## Examples

Below, you can find an example usage of the package for solving the eigenproblem of the given Hamiltonian (with periodic boundary condition):

$$
H = -J\sum_{i=1}^L \sigma^z_i\sigma^z_{i+1} + \sum_{i=1}^L (-1)^i \sigma^z_i.
$$

```python
import pystrel as ps

L = 10
J = 1.0
params = {
    "sites": L,
    "terms": {
        'Jz': {(i, (i+1) % L): -J for i in range(L)},
        'hz': {i: (-1.0)**i for i in range(L)},
    },
}
model = ps.Model(params)
h = model.build_hamiltonian(device='cpu', sparsity='dense')
e, v = ps.spectrum.get_full_spectrum(h)
```

Visit the [documentation page][docs] for more details.
To learn about advanced usage, please refer to the following tutorials:

- [**Example 01** – Basics][example01]
- [**Example 02** – Dynamics][example02]
- [**Example 03** – $t$-$V$ model][example03]
- [**Example 04** – Quantum annealing][example04]
- [**Example 05** – Quantum braiding][example05]

Tutorial notebooks can be found and downloaded from the project repository: [`examples/`][examples].

## Benchmark

For performance benchmarking using CPU/GPU, please refer to the results available [**here**][benchmark].
Keep in mind that performance may vary based on your machine's specifications and the specific problem being solved. 

It's worth noting that GPUs tend to perform better with dense matrices, whereas CPUs show better performance with sparse matrices. 
However, in dynamic contexts, there might not be a noticeable speedup from parallelization.

> **Note**  
> Please be aware that performance characteristics may change in future versions.

## Roadmap for v1.0.0

List of the tasks to consider before the `v1.0.0` release:

- [ ] perf: state class
- [ ] perf: gpu kernels
- [ ] perf: numba acceleration
- [ ] feat: topology utils
- [ ] feat: additional terms: 3dim (tJ like), 4 dim (Hubbard like)
- [ ] feat: lioms
- [ ] feat: additional operators
- [ ] feat: parameters json/yaml load


[CUDA]:https://developer.nvidia.com/cuda-downloads
[ROCm]:https://github.com/RadeonOpenCompute/ROCm
[cupy]:https://cupy.dev/
[docs]:https://andywiecko.github.io/pystrel
[examples]: https://github.com/andywiecko/pystrel/tree/main/examples
[example01]: https://andywiecko.github.io/pystrel/01-basics.html
[example02]: https://andywiecko.github.io/pystrel/02-dynamics.html
[example03]: https://andywiecko.github.io/pystrel/03-tv-model.html
[example04]: https://andywiecko.github.io/pystrel/04-quantum-annealing.html
[example05]: https://andywiecko.github.io/pystrel/05-quantum-braiding.html
[benchmark]: https://andywiecko.github.io/pystrel/benchmark.html