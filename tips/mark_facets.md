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

We provide here a utility function to tag some facets of a mesh by providing geometrical locator functions

```{code-cell} ipython3
from mpi4py import MPI
import numpy as np
from dolfinx import mesh


def mark_facets(domain, surfaces_dict):
    """Mark facets of the domain according to geometrical location

    Parameters
    ----------
    domain : Mesh
        `dolfinx` mesh object
    surfaces_dict : dict
        A dictionary mapping integer tags with geometrical location function {tag: locator(x)}

    Returns
    -------
    facet_tag array
    """
    fdim = domain.topology.dim - 1
    marked_values = []
    marked_facets = []
    # Concatenate and sort the arrays based on facet indices
    for tag, location in surfaces_dict.items():
        facets = mesh.locate_entities_boundary(domain, fdim, location)
        marked_facets.append(facets)
        marked_values.append(np.full_like(facets, tag))
    marked_facets = np.hstack(marked_facets)
    marked_values = np.hstack(marked_values)
    sorted_facets = np.argsort(marked_facets)
    facet_tag = mesh.meshtags(
        domain, fdim, marked_facets[sorted_facets], marked_values[sorted_facets]
    )
    return facet_tag
```

For instance, tagging the bottom, right, top and left boundary of a square mesh will look like this:

```{code-cell} ipython3
N = 4
domain = mesh.create_unit_square(MPI.COMM_WORLD, N, N)


def left(x):
    return np.isclose(x[0], 0.0)


def bottom(x):
    return np.isclose(x[1], 0.0)


def right(x):
    return np.isclose(x[0], 1.0)


def top(x):
    return np.isclose(x[1], 1.0)


facets = mark_facets(domain, {1: bottom, 2: right, 3: top, 4: left})
print(facets.values)
```
