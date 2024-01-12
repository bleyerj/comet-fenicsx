# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Lumping a mass matrix
#
# ```{admonition} Objectives
# :class: objectives
#
# We discuss here how to lump a mass matrix.
# ```
#
# ```{admonition} Download sources
# :class: download
#
# * {Download}`Python script<./mass_lumping.py>`
# * {Download}`Jupyter notebook<./mass_lumping.ipynb>`
# ```
#
#
# Explicit dynamics simulations require the usage of lumped mass matrices i.e. diagonal mass matrices for which inversion can be done explicitly.
#
# We show how to do this for Lagrange elements using the Gauss-Lobatto-Legendre quadrature rule, see also the [](/tips/quadrature_schemes/quadrature_schemes.md) tour. For the high order case in wave propagation, this setting defines the so-called *spectral element method*.

# +
import numpy as np
from mpi4py import MPI
import ufl
import dolfinx.fem.petsc
from dolfinx import fem, mesh

domain = mesh.create_unit_square(
    MPI.COMM_WORLD, 1, 1, cell_type=mesh.CellType.quadrilateral
)

for order in range(1, 3):
    V = fem.FunctionSpace(domain, ("P", order))
    v = ufl.TestFunction(V)
    u = ufl.TrialFunction(V)

    dx = ufl.Measure("dx", domain=domain)
    dx_lumped = dx(metadata={"quadrature_rule": "GLL", "quadrature_degree": order})
    mass_form = v * u * dx
    lumped_mass_form = v * u * dx_lumped

    M_consistent = fem.assemble_matrix(fem.form(mass_form))
    print(
        "Consistent mass matrix:\n", np.array_str(M_consistent.to_dense(), precision=3)
    )

    M_lumped = fem.assemble_matrix(fem.form(lumped_mass_form))
    print("Lumped mass matrix:\n", np.array_str(M_lumped.to_dense(), precision=3))
# -

# For explicit dynamics simulation, the mass matrix can then be manipulated using the diagonal vector, to compute for instance $M^{-1}w$ where $w$ is some Function.

mass_diagonal = u * dx_lumped  # defines a linear form corresponding to the diagonal
M_vect = fem.petsc.assemble_vector(fem.form(mass_diagonal))
w = fem.Function(V)
iMw = fem.Function(V)
iMw.vector.pointwiseDivide(M_vect, w.vector)
