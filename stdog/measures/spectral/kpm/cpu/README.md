
# Introduction



# Example

### Create a graph

```
import igraph as ig

N = int(10e4)
p = 4/N
g = ig.Graph.Erdos_Rtheenyi(N, p)
```

### Convert the adjacency matrix into a sparse scipy matrix

```
from stdog.utils import ig2sparse
H = ig2sparse(g)
```

### Choose kpm parameters

```
n_moments = 100
n_vecs = 20
n_points = 10
```

### Estimating the spectral density (rho) associated with a set of energy points (xk)

#### A direct version
```
from stdog.measures.kpm.cpu import kpm
rho, xk = kpm(
    H,
    n_moments = n_moments,
    n_vecs = n_vecs,
    n_points = n_points  
    )
```

#### A more transparent version

```
from stdog.measures.kpm.cpu import kpm_dos, get_kpm_moments, rescale

n_vertices = H.shape[0]

H_rescaled, scale_fact_a, scale_fact_b = rescale(H)

mus = [
    get_kpm_moments(H_rescaled, n_moments)
    for i in range(n_vecs)
]

rho, xk = kpm_dos(
    mus,
    n_moments,
    n_vecs,
    n_points,
    n_vertices,
    scale_fact_a,
    scale_fact_b
)
```

#### Parallelizing using ipycluster


# References

[1] - [The Kernel Polynomial Method applied to tight binding systems with time-dependence](https://repository.tudelft.nl/islandora/object/uuid%3Ac7c70ef2-6f19-4b70-b7ef-deaa6f1b1d45)
