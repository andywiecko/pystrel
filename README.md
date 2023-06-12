# pystrel

## Example usage

```python
import pystrel as ps

L = 10
model = ps.Model({
    "sites": L,
    "terms": {
        'Jz': {(i, (i+1) % L): 1.0 for i in range(L)},
        'hz': {i: (-1.0)**i for i in range(L)},
    },
})
```

## TODO

- [ ] topology utils
- [ ] state
- [ ] additional terms: 3dim (tJ like), 4 dim (Hubbard like)
- [ ] quantum dynamics: RK4, chebyshev
- [ ] energy calculation: ground state, full spectrum
- [ ] LIOMs
- [ ] operators
- [ ] parameters json/yaml load
- [ ] example notebooks
- [ ] benchmarks
- [ ] ci/cd: test, pylint, mypy, black, pdoc