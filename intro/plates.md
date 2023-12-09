---
jupyter:
  jupytext:
    formats: md,ipynb
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.0
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Reissner-Mindlin plates
$$\newcommand{\bM}{\boldsymbol{M}}
\newcommand{\bQ}{\boldsymbol{Q}}
\newcommand{\bgamma}{\boldsymbol{\gamma}}
\newcommand{\btheta}{\boldsymbol{\theta}}
\newcommand{\bchi}{\boldsymbol{\chi}}
\renewcommand{\div}{\operatorname{div}}$$

This demo illustrates how to implement a Reissner-Mindlin thick plate model. The main specificity of such models is that one needs to solve for two different fields: a vertical deflection field $w$ and a rotation vector field $\btheta$. We recall below the main relations defining this model in the linear elastic case.

## Governing equations
### Generalized strains

* *Bending curvature* strain $\bchi = \dfrac{1}{2}(\nabla \btheta + \nabla^\text{T} \btheta) = \nabla^\text{s}\btheta$
* *Shear* strain $\bgamma = \nabla w - \btheta$

### Generalized stresses 
* *Bending moment* $\bM$
* *Shear force* $\bQ$

### Equilibrium conditions
For a distributed transverse surface loading $f$,
* Vertical equilibrium: $\div \bQ + f = 0$
* Moment equilibrium: $\div \bM + \bQ = 0$

In weak form:
$$\int_{\Omega} (\bM:\nabla^\text{s}\widehat{\btheta} + \bQ\cdot(\nabla \widehat{w} - \widehat{\btheta}))\text{d}\Omega = \int_{\Omega} f w \text{d}\Omega \quad \forall \widehat{w},\widehat{\btheta}$$

### Isotropic linear elastic constitutive relation
* Bending/curvature relation:
\begin{align*}
\begin{Bmatrix}M_{xx}\\ M_{yy} \\M_{xy} \end{Bmatrix} &= \textsf{D} \begin{bmatrix}1 & \nu & 0 \\ \nu & 1 & 0 \\ 0 & 0 & (1-\nu)/2 \end{bmatrix}\begin{Bmatrix}\chi_{xx} \\ \chi_{yy} \\ 2\chi_{xy}  \end{Bmatrix}\\
\text{ where } \textsf{D} &= \dfrac{E h^3}{12(1-\nu^2)}
\end{align*}

* Shear strain/stress relation:
\begin{align*}
\bQ &= \textsf{F}\bgamma\\
\text{ where } \textsf{F} &= \dfrac{5}{6}\dfrac{E h}{2(1+\nu)}
\end{align*}


## Implementation

We first load relevant modules and functions and generate a unit square mesh of triangles. 

```python
import numpy as np
import ufl

from mpi4py import MPI
from dolfinx import fem, io
from dolfinx.mesh import create_unit_square, CellType, locate_entities_boundary
from ufl import (
    as_matrix,
    as_vector,
    inner,
    dot,
    grad,
    split,
    FiniteElement,
    VectorElement,
    MixedElement,
    TestFunction,
    TrialFunction,
)


N = 10
mesh = create_unit_square(MPI.COMM_WORLD, N, N, CellType.triangle)
```

Next we define material properties and functions which will be used for defining the variational formulation.

```python
# material parameters
thick = 0.05
E = 210.0e3
nu = 0.3

# bending stiffness
D = fem.Constant(mesh, E * thick**3 / (1 - nu**2) / 12.0)
# shear stiffness
F = fem.Constant(mesh, E / 2 / (1 + nu) * thick * 5.0 / 6.0)

# uniform transversal load
f = fem.Constant(mesh, -100.0)

# Useful function for defining strains and stresses
def curvature(u):
    (w, theta) = split(u)
    return as_vector([theta[0].dx(0), theta[1].dx(1), theta[0].dx(1) + theta[1].dx(0)])

def shear_strain(u):
    (w, theta) = split(u)
    return theta - grad(w)

def bending_moment(u):
    DD = as_matrix([[D, nu * D, 0], [nu * D, D, 0], [0, 0, D * (1 - nu) / 2.0]])
    return dot(DD, curvature(u))

def shear_force(u):
    return F * shear_strain(u)

```

Now we define the corresponding function space. Our dofs are $w$ and $\btheta$, so the full function space $V$ will be a **mixed** function space consisting of a scalar subspace related to $w$ and a vectorial subspace related to $\btheta$. We first use a continuous $P^2$ interpolation for $w$ and a continuous $P^1$ interpolation for $\btheta$. We then define the corresponding linear and bilinear forms.

```python
# Definition of function space for U:displacement, T:rotation
Ue = FiniteElement("CG", mesh.ufl_cell(), 2)
Te = VectorElement("CG", mesh.ufl_cell(), 1)
V = fem.FunctionSpace(mesh, MixedElement([Ue, Te]))

# Functions
u = fem.Function(V, name="Unknown")
u_ = TestFunction(V)
(w_, theta_) = split(u_)
du = TrialFunction(V)

# Linear and bilinear forms
dx = ufl.Measure("dx", domain=mesh)
L = f * w_ * dx
a = (dot(bending_moment(u_), curvature(du)) + dot(shear_force(u_), shear_strain(du)))*dx
```

Boundary conditions are now defined. We consider a fully clamped boundary. Note that since we are using a mixed function space, we cannnot use the `locate_dofs_geometrical` function. Instead, we locate the facets on the boundary using `dolfinx.mesh.locate_entities_boundary`. Then we locate the dofs on such facets using `locate_dofs_topological`.

```python
# Boundary of the plate
def border(x):
    return np.logical_or(
        np.logical_or(np.isclose(x[0], 0), np.isclose(x[0], 1)),
        np.logical_or(np.isclose(x[1], 0), np.isclose(x[1], 1)),
    )

facet_dim = 1
clamped_facets = locate_entities_boundary(mesh, facet_dim, border)
clamped_dofs = fem.locate_dofs_topological(V, facet_dim, clamped_facets)

u0 = fem.Function(V)
bcs = [fem.dirichletbc(u0, clamped_dofs)]
```

We now solve the problem and output the result. To get the deflection $w$, we use `u.sub(0).collapse()` to extract a new function living in the corresponding subspace. Note that $u.sub(0)$ provides only an indexed view of the $w$ component of `u`.

```python
problem = fem.petsc.LinearProblem(
    a, L, u=u, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
)
problem.solve()

with io.XDMFFile(mesh.comm, "plates.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    w = u.sub(0).collapse()
    w.name = "Deflection"
    xdmf.write_function(w)

w_LK = 1.265319087e-3 * -f.value / D.value
print(f"Reissner-Mindlin deflection: {max(abs(w.vector.array)):.5f}")
print(f"Love-Kirchhoff deflection: {w_LK:.5f}")
```

## Modal analysis

Now we define the form corresponding to the definition of the mass matrix and we assemble the stiffness and mass forms into corresponding PETSc matrix objects. We use a value of 1 on the diagonal for K and 0 for M for the rows corresponding to the boundary conditions. Doing so, eigenvalues associated to boundary conditions are equal to infinity and will not pollute the low-frequency spectrum.

```python
rho = fem.Constant(mesh, 1.0)
m_form = rho * dot(du, u_) * dx

K = fem.petsc.assemble_matrix(fem.form(a), bcs, diagonal=1)
K.assemble()
M = fem.petsc.assemble_matrix(fem.form(m_form), bcs, diagonal=0)
M.assemble()
```

We now use `slepc4py` to define a eigenvalue solver (EPS -- *Eigenvalue Problem Solver* -- in SLEPc vocable) and solve the corresponding generalized eigenvalue problem. Functions defined in the `eigenvalue_problem.py` module enable to define the corresponding objects, set up the parameters, monitor the resolution and extract the corresponding eigenpairs. Here the problem is of type `GHEP` (Generalized Hermitian eigenproblem) and we use a shift-invert transform to compute the smallest eigenvalues.

```python
from slepc4py import SLEPc
from eigenvalue_solver import solve_GEP_shiftinvert, EPS_get_spectrum

N_eig = 6  # number of requested eigenvalues
eigensolver = solve_GEP_shiftinvert(
    K,
    M,
    problem_type=SLEPc.EPS.ProblemType.GHEP,
    solver=SLEPc.EPS.Type.KRYLOVSCHUR,
    nev=N_eig,
)
# Extract results
(eigval, eigvec_r, eigvec_i) = EPS_get_spectrum(eigensolver, V)
# Output eigenmodes
with io.XDMFFile(mesh.comm, "plates_eigenvalues.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    for i in range(N_eig):
        w = eigvec_r[i].sub(0).collapse()
        w.name = "Deflection"
        xdmf.write_function(w, i)
```

```python

```
