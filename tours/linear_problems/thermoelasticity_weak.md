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

# Linear thermoelasticity (weak coupling)

## Introduction

In this tour, we will solve a linear thermoelastic problem. In permanent regime, the temperature field is uncoupled from the mechanical fields whereas the latter depend on the temperature due to presence of thermal strains in the thermoelastic constitutive relation. This situation can be described as *weak* thermomechanical coupling. 
*Full* thermomechanical coupling for a transient evolution problem is treated [in this tour](thermoelasticity_transient.ipynb).

In the present setting, the temperature field can be either given as a `Constant` or an `Expression` throughout the domain or obtained as the solution of a steady-state heat (Poisson) equation. This last case will be implemented in this tour.

## Problem position

We consider the case of a rectangular 2D domain of dimensions $L\times H$ fully clamped on both lateral sides and subjected to a self-weight loading. The bottom side is subjected to a uniform temperature increase of $\Delta T = +50^{\circ}C$ while the top and lateral boundaries remain at the initial temperature $T_0$. The geometry and boundary regions are first defined.

```{code-cell} ipython3
:raw_mimetype: text/restructuredtext

import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
import ufl
from dolfinx import fem, mesh, io, plot
import dolfinx.fem.petsc
import pyvista


L, H = 5, 0.3
Nx, Ny = 20, 5
domain = mesh.create_rectangle(MPI.COMM_WORLD, [(0., 0.), (L, H)], [Nx, Ny], cell_type=mesh.CellType.quadrilateral)
gdim = domain.geometry.dim

def lateral_sides(x):
    return np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[0], L))
def bottom(x):
    return np.isclose(x[1], 0.0)
def top(x):
    return np.isclose(x[1], H)
```

Because of the weak coupling discussed before, the thermal and mechanical problem can be solved separately. As a result, we don't need to resort to Mixed FunctionSpaces but can just define separately both problems.

## Resolution of the thermal problem

The temperature is solution to the following equation $\text{div}(k\nabla T) = 0$ where $k$ is the thermal conductivity (here we have no heat source). Since $k$ is assumed to be homogeneous, it will not influence the solution. We therefore obtain a standard Poisson equation without forcing terms. Its formulation and resolution in FEniCS is quite standard with the temperature variation $\Delta T$ as the main unknown.

> **Note**: We needed to define the linear form `LT` in order to use the standard `solve(aT == LT, T, bcT)` syntax for linear problems. We could also have equivalently defined the residual `res = dot(grad(T), grad(T_))*dx` and used the nonlinear-like syntax `solve(res == 0, T, bcT)`.

```{code-cell} ipython3
VT = fem.FunctionSpace(domain, ("Q", 1))
T_, dT = ufl.TestFunction(VT), ufl.TrialFunction(VT)
Delta_T = fem.Function(VT, name="Temperature_variation")
aT = ufl.dot(ufl.grad(dT), ufl.grad(T_)) * ufl.dx
LT = fem.Constant(domain, 0.0) * T_ * ufl.dx

bot_dofs_T = fem.locate_dofs_geometrical(VT, bottom)
top_dofs_T = fem.locate_dofs_geometrical(VT, top)
sides_dofs_T = fem.locate_dofs_geometrical(VT, lateral_sides)
bcT = [
    fem.dirichletbc(50.0, bot_dofs_T, VT),
    fem.dirichletbc(0.0, top_dofs_T, VT),
    fem.dirichletbc(0.0, sides_dofs_T, VT),
]


# solve problem
problem = fem.petsc.LinearProblem(aT, LT, u=Delta_T, bcs=bcT)
problem.solve();
```

We use Pyvista to plot the corresponding temperature field:

```{code-cell} ipython3
pyvista.set_jupyter_backend("static")

topology, cell_types, geometry = plot.vtk_mesh(domain, gdim)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

T_topology, T_cell_types, T_geometry = plot.vtk_mesh(VT)
T_grid = pyvista.UnstructuredGrid(T_topology, T_cell_types, T_geometry)
T_grid.point_data["Temperature"] = Delta_T.x.array
T_grid.set_active_scalars("Temperature")
contours = T_grid.contour()

plotter = pyvista.Plotter()
plotter.window_size = (800, 300)
plotter.add_mesh(grid, show_edges=True)
plotter.add_mesh(contours, line_width=3)
plotter.view_xy()
plotter.zoom_camera(2.5)
plotter.show()
```

## Mecanical problem

+++

The linearized thermoelastic constitutive equation is given by:

$$
\boldsymbol{\sigma} = \mathbb{C}:(\boldsymbol{\varepsilon}-\alpha(T-T_0)\boldsymbol{1}) = \lambda\text{tr}(\boldsymbol{\varepsilon})\boldsymbol{1}+2\mu\boldsymbol{\varepsilon} -\alpha(3\lambda+2\mu)(T-T_0)\boldsymbol{1} 
$$

where $\lambda,\mu$ are the Lamé parameters and $\alpha$ is the thermal expansion coefficient. As regards the current problem, the last term corresponding to the thermal strains is completely known. The following formulation can thus be generalized to any kind of known initial stress or eigenstrain state such as pre-stress or phase changes.

```{raw-cell}
:raw_mimetype: text/restructuredtext

The weak formulation for the mechanical problem builds upon the presentation of the :ref:`LinearElasticity2D` tour. 
The main difference lies in the presence of the temperature term in the constitutive relation which introduces a linear term with respect to the TestFunction ``u_`` when writing the work of internal forces ``inner(sigma(du, Delta_T), eps(u_))*dx``. As a result, the bilinear form is extracted using the left-hand side ``lhs`` function whereas the thermal strain term, acting as a loading term, is extracted using the right-hand side ``rhs`` function and added to the work of external forces when defining the linear form ``LM``.
```

```{code-cell} ipython3
E = fem.Constant(domain, 50e3)
nu = fem.Constant(domain, 0.2)
mu = E / 2 / (1 + nu)
lmbda = E * nu / (1 + nu) / (1 - 2 * nu)
alpha = fem.Constant(domain, 1e-5)

f = fem.Constant(domain, (0.0, 0.0))


def eps(v):
    return ufl.sym(ufl.grad(v))


def sigma(v, Delta_T):
    return (
        lmbda * ufl.tr(eps(v)) - alpha * (3 * lmbda + 2 * mu) * Delta_T
    ) * ufl.Identity(gdim) + 2.0 * mu * eps(v)


Vu = fem.FunctionSpace(domain, ("Q", 2, (gdim,)))
du = ufl.TrialFunction(Vu)
u_ = ufl.TestFunction(Vu)
Wint = ufl.inner(sigma(du, Delta_T), eps(u_)) * ufl.dx
aM = ufl.lhs(Wint)
LM = ufl.rhs(Wint) + ufl.inner(f, u_) * ufl.dx

lateral_dofs_u = fem.locate_dofs_geometrical(Vu, lateral_sides)
bcu = [fem.dirichletbc(np.zeros((gdim,)), lateral_dofs_u, Vu)]
```

`````{tab-set}
```{tab-item} With ufl.lhs/rhs
```{code-cell} ipython3
Wint = ufl.inner(sigma(du, Delta_T), eps(u_)) * ufl.dx
aM = ufl.lhs(Wint)
LM = ufl.rhs(Wint) + ufl.inner(f, u_) * ufl.dx
```
```

```{tab-item} Tab 2 title
My second tab with `some code`!
```
````

First, the self-weight loading is deactivated, only thermal stresses are computed.

```{code-cell} ipython3
u = fem.Function(Vu, name="Displacement")
problem = fem.petsc.LinearProblem(aM, LM, u=u, bcs=bcu)
problem.solve();

```

We now take into account the self-weight.

```{code-cell} ipython3
rho_g = 2400*9.81e-6
f.value[1] = -rho_g
problem.solve();
```
