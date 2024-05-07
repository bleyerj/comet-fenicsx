# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Marking facets with geometrical functions
#
# ```{admonition} Objectives
# :class: objectives
#
# We provide here a utility function to tag some facets of a mesh by providing geometrical marker functions
# ```
#
# ```{admonition} Download sources
# :class: download
#
# * {Download}`Python script<./mark_facets.py>`
# * {Download}`Jupyter notebook<./mark_facets.ipynb>`
# ```

# +
from mpi4py import MPI
import numpy as np
from dolfinx import mesh


def mark_facets(domain, surfaces_dict):
    """Mark facets of the domain according to a geometrical marker

    Parameters
    ----------
    domain : Mesh
        `dolfinx` mesh object
    surfaces_dict : dict
        A dictionary mapping integer tags with a geometrical marker function {tag: marker(x)}

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


# -

# For instance, tagging the bottom, right, top and left boundary of a square mesh will look like this:

# +
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


# -

# Note that we can also adapt the function to mark entities of specified dimension i.e. subdomains if `dim=tdim`, facets if `dim=tdim-1`, etc. where `tdim` is the domain topological dimension.

# +
def mark_entities(domain, dim, entities_dict):
    """Mark entities of specified dimension according to a geometrical marker function

    Parameters
    ----------
    domain : Mesh
        `dolfinx` mesh object
    dim : int
        Dimension of the entities to mark
    entities_dict : dict
        A dictionary mapping integer tags with a geometrical marker function {tag: marker(x)}

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


def half_left(x):
    return x[0] <= 0.5


def half_right(x):
    return x[0] >= 0.5


tdim = domain.topology.dim
cell_markers = mark_entities(domain, tdim, {1: half_left, 2: half_right})
print(cell_markers.values)
# -

# ```{warning}
#
# When calling `mesh.locate_entities` for a cell or a facet, the geometrical marker function gets evaluated for all vertices of the cell/facet. The marker must therefore evaluate to `True` for all vertices to properly identify the entity.
# ```
