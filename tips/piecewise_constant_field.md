---
jupytext:
  formats: md:myst,ipynb
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.0
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Marking facets with geometrical functions

We provide here a utility function to create a piecewise constant field for each region of a domain

```{code-cell} ipython3
from mpi4py import MPI
import numpy as np
from dolfinx import mesh


def create_piecewise_constant_field(domain, cell_markers, property_dict, name=None):
    """Create a piecewise constant field with different values per subdomain.

    Parameters
    ----------
    domain : Mesh
        `dolfinx` mesh object
    cell_markers : MeshTag
        cell marker MeshTag
    property_dict : dict
        A dictionary mapping region tags to physical values {tag: value}

    Returns
    -------
    A DG-0 function
    """
    V0 = fem.functionspace(domain, ("DG", 0))
    k = fem.Function(V0, name=name)
    for tag, value in property_dict.items():
        cells = cell_markers.find(tag)
        k.x.array[cells] = np.full_like(cells, value, dtype=np.float64)
    return k
```

For instance, creating a square mesh with a central region:

```{code-cell} ipython3
N = 10
domain = mesh.create_unit_square(MPI.COMM_WORLD, N, N)


def inclusions(x):
    return np.logical_and(np.isclose(x[0], 0.5, 0.25), np.isclose(x[1], 0.5, 0.25))


k = create_piecewise_constant_field(domain, cell_markers, {1: 1.0, 2: 10.0})
xdofs = k.function_space().tabulate_dofs_coordinates()
plt.scatter(*xdofs, k.vector.array)
plt.show()
```
