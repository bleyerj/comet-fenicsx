---
jupytext:
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

# Computing consistent reaction forces

```{admonition} Objectives
:class: objectives
These snippets show how to compute reaction forces using two methods:
* post-processing the stress and perform integration (poor quality)
* use the weak form and compute the residual action in a well chosen test function (good quality)
$\newcommand{\dS}{\,\text{dS}}
\newcommand{\dx}{\,\text{dx}}
\newcommand{\bsig}{\boldsymbol{\sigma}}
\newcommand{\bu}{\boldsymbol{u}}
\newcommand{\bv}{\boldsymbol{v}}$
```

One often needs to compute a resulting reaction force on some part of the boundary as post-processing of a mechanical resolution.

Quite often, such a reaction will be computed using the stress field associated with the computed displacement. We will see that this may lead to slight inaccuracies whereas another more consistent approach using the virtual work principle is more accurate.


## A cantilever beam problem

We use here a simple 2D small strain elasticity script of a rectangular domain of dimensions $L\times H$ representing a cantilever beam clamped on the left hand-side and under uniform body forces $\boldsymbol{f}=(f_x, f_y)$. $\mathbb{P}^2$ Lagrange elements are used for the displacement discretization.

For the sake of illustration, we are interested in computing the horizontal and vertical reaction forces $R_x$ and $R_y$ on the left boundary as well as the resulting moment $M_z$ around the out-of-plane direction. In the present simple case, they can all be computed explicitly using global balance equations as:

$$
\begin{align*}
R_x &= \int_{x=0} \boldsymbol{T}\cdot \boldsymbol{e}_x \dS = \int_{x=0} (-\sigma_{xx}) dS = -f_x \cdot L \cdot H \\
R_y &= \int_{x=0} \boldsymbol{T}\cdot \boldsymbol{e}_y \dS = \int_{x=0} (-\sigma_{xy}) dS = -f_y \cdot L \cdot H\\
M_z &= \int_{x=0} (\vec{\boldsymbol{OM}}\times \boldsymbol{T})\cdot \boldsymbol{e}_z \dS = \int_{x=0} (y \sigma_{xx}) dS = f_y \cdot \dfrac{L^2}{2} \cdot H
\end{align*}
$$

We first define a standard small strain elasticity problem.

```{code-cell} ipython3
:tags: [hide-input]

import numpy as np
from mpi4py import MPI
import ufl
from dolfinx import fem, mesh
import dolfinx.fem.petsc

L = 5.0
H = 1.0
Nx = 20
Ny = 5
domain = mesh.create_rectangle(
    MPI.COMM_WORLD,
    [(0.0, -H / 2), (L, H / 2)],
    (Nx, Ny),
    diagonal=mesh.DiagonalType.crossed,
)


E = fem.Constant(domain, 1e5)
nu = fem.Constant(domain, 0.3)
mu = E / 2 / (1 + nu)
lmbda = E * nu / (1 + nu) / (1 - 2 * nu)


def eps(v):
    return ufl.sym(ufl.grad(v))


def sigma(v):
    return lmbda * ufl.tr(eps(v)) * ufl.Identity(2) + 2.0 * mu * eps(v)


fx = 0.1
fy = -1.0
f = fem.Constant(domain, (fx, fy))

V = fem.functionspace(domain, ("P", 2, (2,)))
du = ufl.TrialFunction(V)
u_ = ufl.TestFunction(V)
a = ufl.inner(sigma(du), eps(u_)) * ufl.dx
l = ufl.inner(f, u_) * ufl.dx


def left(x):
    return np.isclose(x[0], 0.0)


fdim = domain.topology.dim - 1
marked_values = []
marked_facets = []
# Concatenate and sort the arrays based on facet indices
facets = mesh.locate_entities_boundary(domain, fdim, left)
marked_facets.append(facets)
marked_values.append(np.full_like(facets, 1))
marked_facets = np.hstack(marked_facets)
marked_values = np.hstack(marked_values)
sorted_facets = np.argsort(marked_facets)
facet_tag = mesh.meshtags(
    domain, fdim, marked_facets[sorted_facets], marked_values[sorted_facets]
)


ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tag)
left_dofs = fem.locate_dofs_geometrical(V, left)
u_bc = fem.Function(V)
bcs = [fem.dirichletbc(u_bc, left_dofs)]

u = fem.Function(V, name="Displacement")

problem = dolfinx.fem.petsc.LinearProblem(a, l, bcs, u=u)
problem.solve()
```

```{seealso}
See the [](/tours/linear_problems/isotropic_orthotropic_elasticity/isotropic_orthotropic_elasticity.md) or [](/intro/linear_elasticity/linear_elasticity.md) demos for more details on setting up a linear elasticity problem.
```

## First method: using the post-processed stress

The first, and most widely used, method for computing the above reactions relies on the stress field computed from the obtained displacement `sigma(u)` and perform `assemble_scalar` over the left boundary (measure `ds(1)`). Unfortunately, this procedure does not ensure an exact computation as seen below. Indeed, the stress field, implicitly known only at the quadrature points only is extended to the structure boundary and does not satisfy global equilibrium anymore.

```{code-cell} ipython3
x = ufl.SpatialCoordinate(domain)

Rx = fem.assemble_scalar(fem.form(-sigma(u)[0, 0] * ds(1)))
print(f"Horizontal reaction Rx = {Rx:.6f}")
print(f"             (analytic = {-L * H * fx})")
print("-" * 50)

Ry = fem.assemble_scalar(fem.form(-sigma(u)[0, 1] * ds(1)))
print(f"Vertical reaction Ry = {Ry:.6f}")
print(f"           (analytic = {-L * H * fy})")
print("-" * 50)

Mz = fem.assemble_scalar(fem.form(-x[1] * sigma(u)[0, 0] * ds(1)))
print(f"Bending moment Mz = {Mz:.6f}")
print(f"        (analytic = {H * L**2 / 2 * fy})".format())
print("-" * 50)
print("\n")
```

## Second method: using the work of internal forces

The second approach relies on the virtual work principle (or weak formulation) which writes in the present case:

$$
\int_\Omega \bsig(\boldsymbol{u}):\nabla^s \boldsymbol{v} \dx =\int_\Omega \boldsymbol{f}\cdot\boldsymbol{v} \dx + \int_{\partial \Omega_N} \boldsymbol{T}\cdot\boldsymbol{v}\dS + \int_{\partial \Omega_D} \boldsymbol{T}\cdot\boldsymbol{v}\dS \quad \forall \boldsymbol{v}\in V
$$

in which $\boldsymbol{v}$ does not necessarily satisfy the Dirichlet boundary conditions on $\partial \Omega_D$.

The solution $\bu$ is precisely obtained by enforcing the Dirichlet boundary conditions on $\boldsymbol{v}$ such that:

$$
\int_\Omega \bsig(\boldsymbol{u}):\nabla^s \boldsymbol{v} \dx =\int_\Omega \boldsymbol{f}\cdot\boldsymbol{v} \dx + \int_{\partial \Omega_N} \boldsymbol{T}\cdot\boldsymbol{v}\dS \quad \forall \boldsymbol{v}\in V \text{ and } \boldsymbol{v}=0 \text{ on }\partial \Omega_D
$$

Defining the **residual**:

$$
Res(\bv) = \int_\Omega \bsig(\boldsymbol{u}):\nabla^s \boldsymbol{v} \dx - \int_\Omega \boldsymbol{f}\cdot\boldsymbol{v} \dx - \int_{\partial \Omega_N} \boldsymbol{T}\cdot\boldsymbol{v}\dS = a(\boldsymbol{u}, \boldsymbol{v}) -\ell(\boldsymbol{v})
$$

we have that $Res(\bv)= 0$ if $\boldsymbol{v}=0$ on $\partial \Omega_D$.

Now, if $\boldsymbol{v}\neq0$ on $\partial \Omega_D$, say, for instance, $\boldsymbol{v}=(1,0)$ on $\partial \Omega_D$, we have that:

$$
Res(\bv) = \int_{\partial \Omega_D} \boldsymbol{T}\cdot\boldsymbol{v}\dS = \int_{\partial \Omega_D} \boldsymbol{T}_x\dS = \int_{\partial \Omega_D} -\sigma_{xx}\dS = R_x
$$

Similarly, we obtain the vertical reaction $R_y$ by considering $\boldsymbol{v}=(0,1)$ and the bending moment $M_z$ by considering $\boldsymbol{v}=(y,0)$.

As regards implementation, the residual is defined using the action of the bilinear form on the displacement solution: `residual = action(a, u) - l`. We then define boundary conditions on the left boundary and apply them to an empty Function `v_reac` to define the required test field $\bv$. We observe that the computed reactions are now exact.

```{code-cell} ipython3
residual = ufl.action(a, u) - l

v_reac = fem.Function(V)
virtual_work_form = fem.form(ufl.action(residual, v_reac))


def one(x):
    values = np.zeros((1, x.shape[1]))
    values[0] = 1.0
    return values


def y(x):
    values = np.zeros((1, x.shape[1]))
    values[0] = x[1]
    return values


u_bc.sub(0).interpolate(one)
fem.set_bc(v_reac.vector, bcs)
print(f"Horizontal reaction Rx = {fem.assemble_scalar(virtual_work_form):.6f}")
print(f"             (analytic = {-L * H * fx})")
print("-" * 50)

u_bc.vector.set(0.0)
v_reac.vector.set(0.0)
u_bc.sub(1).interpolate(one)
fem.set_bc(v_reac.vector, bcs)
print(f"Vertical reaction Ry = {fem.assemble_scalar(virtual_work_form):.6f}")
print(f"           (analytic = {-L * H * fy})")
print("-" * 50)

u_bc.vector.set(0.0)
v_reac.vector.set(0.0)
u_bc.sub(0).interpolate(y)
fem.set_bc(v_reac.vector, bcs)
print(f"Bending moment Mz = {fem.assemble_scalar(virtual_work_form):.6f}")
print(f"        (analytic = {H * L**2 / 2 * fy})")
print("-" * 50)
```
