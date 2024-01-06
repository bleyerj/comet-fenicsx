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

# Create a piecewise constant field for spatially varying properties

We provide here a utility function to create a piecewise constant field for each region of a domain

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from dolfinx import mesh, fem


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

For instance, we first create a square mesh with a central region:

```{code-cell} ipython3
N = 8
domain = mesh.create_unit_square(
    MPI.COMM_WORLD, N, N, cell_type=mesh.CellType.quadrilateral
)


def inclusions(x):
    return np.logical_and(np.abs(x[0] - 0.5) <= 0.25, np.abs(x[1] - 0.5) <= 0.25)
```

We then use the function `mark_entities` introduced in [](/tips/mark_facets) to mark the corresponding cells and create a piecewise constant field.

```{code-cell} ipython3
:tags: [hide-input]

def mark_entities(domain, dim, entities_dict):
    """Mark entities of specified dimension according to geometrical location

    Parameters
    ----------
    domain : Mesh
        `dolfinx` mesh object
    dim : int
        Dimension of the entities to mark
    entities_dict : dict
        A dictionary mapping integer tags with geometrical location function {tag: locator(x)}

    Returns
    -------
    entities_tag array
    """
    marked_values = []
    marked_entities = []
    # number of non-ghosted entities
    num_entities_local = domain.topology.index_map(dim).size_local
    # Concatenate and sort the arrays based on indices
    for tag, location in entities_dict.items():
        entities = mesh.locate_entities(domain, dim, location)
        entities = entities[entities < num_entities_local]  # remove ghost entities
        marked_entities.append(entities)
        marked_values.append(np.full_like(entities, tag))
    marked_entities = np.hstack(marked_entities)
    marked_values = np.hstack(marked_values)
    sorted_entities = np.argsort(marked_entities)
    entities_tags = mesh.meshtags(
        domain, dim, marked_entities[sorted_entities], marked_values[sorted_entities]
    )
    return entities_tags
```

```{code-cell} ipython3
tdim = domain.topology.dim
cell_markers = mark_entities(domain, tdim, {1: inclusions})
k = create_piecewise_constant_field(domain, cell_markers, {0: 1.0, 1: 2.0})
xdofs = k.function_space.tabulate_dof_coordinates()

plt.scatter(xdofs[:, 0], xdofs[:, 1], c=k.vector.array, cmap="bwr")
plt.colorbar()
plt.gca().set_aspect("equal")
plt.show()
```
