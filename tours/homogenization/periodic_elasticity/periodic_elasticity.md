---
jupytext:
  formats: md:myst,ipynb
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---


# Periodic homogenization of linear elasticity {far}`star`

```{admonition} Objectives
:class: objectives

This tour will show how to perform periodic homogenization of linear elastic heterogeneous materials. In particular, we show how to define periodic boundary conditions and compute the effective stiffness tensor.$\newcommand{\bu}{\boldsymbol{u}}
\newcommand{\bv}{\boldsymbol{v}}
\newcommand{\bt}{\boldsymbol{t}}
\newcommand{\be}{\boldsymbol{e}}
\newcommand{\beps}{\boldsymbol{\varepsilon}}
\newcommand{\bsig}{\boldsymbol{\sigma}}
\newcommand{\T}{^{\text{T}}}$
```

```{attention}
This tour requires the `dolfinx_mpc` add-on package to enforce periodic boundary conditions. More details and installation instructions are to be found here https://github.com/jorgensd/dolfinx_mpc.
```

The considered 2D plane strain problem deals with a skewed unit cell of dimensions $1\times \sqrt{3}/2$ consisting of circular inclusions (numbered $1$) of radius $R$ with elastic properties $(E_r, \nu_r)$ and embedded in a matrix material (numbered $0$) of properties $(E_m, \nu_m)$ following an hexagonal pattern. A classical result of homogenization theory ensures that the resulting overall behavior will be isotropic, a property that will be numerically verified later.

## Periodic homogenization framework

The goal of homogenization theory consists in computing the apparent elastic moduli of the homogenized medium associated with a given microstructure. In a linear elastic setting, this amounts to solving the following auxiliary problem defined on the unit cell $\mathcal{A}$:

$$\begin{equation}\begin{cases}\operatorname{div} \boldsymbol{\sigma} = \boldsymbol{0} & \text{in } \mathcal{A} \\ 
\boldsymbol{\sigma} = \mathbb{C}(\boldsymbol{y}):\boldsymbol{\varepsilon} & \text{for }\boldsymbol{y}\in\mathcal{A} \\
\boldsymbol{\varepsilon} = \boldsymbol{E} + \nabla^s \boldsymbol{v} & \text{in } \mathcal{A} \\
\boldsymbol{v} & \text{is } \mathcal{A}\text{-periodic} \\
\boldsymbol{T}=\boldsymbol{\sigma}\cdot\boldsymbol{n} & \text{is } \mathcal{A}\text{-antiperiodic}
\end{cases} \label{auxiliary-problem}
\end{equation}$$

where $\boldsymbol{E}$ is the **given** macroscopic strain, $\boldsymbol{v}$ a periodic fluctuation and $\mathbb{C}(\boldsymbol{y})$ is the heterogeneous elasticity tensor depending on the microscopic space variable $\boldsymbol{y}\in\mathcal{A}$. By construction, the local microscopic strain is equal on average to the macroscopic strain: $\langle \boldsymbol{\varepsilon} \rangle = \boldsymbol{E}$. Upon defining the macroscopic stress $\boldsymbol{\Sigma}$ as the microscopic stress average: $\langle \boldsymbol{\sigma} \rangle = \boldsymbol{\Sigma}$, there will be a linear relationship between the auxiliary problem loading parameters $\boldsymbol{E}$ and the resulting average stress:

$$\boldsymbol{\Sigma}  = \mathbb{C}^{hom}:\boldsymbol{E}$$

where $\mathbb{C}^{hom}$ represents the apparent elastic moduli of the homogenized medium. Hence, its components can be computed by solving elementary load cases corresponding to the different components of $\boldsymbol{E}$ and performing a unit cell average of the resulting microscopic stress components.

### Total displacement as the main unknown

The previous problem can also be reformulated by using the total displacement $\boldsymbol{u} = \boldsymbol{E}\cdot\boldsymbol{y} + \boldsymbol{v}$ as the main unknown with now $\boldsymbol{\varepsilon} = \nabla^s \boldsymbol{u}$. The periodicity condition is therefore equivalent to the following constraint: 

$$\boldsymbol{u}(\boldsymbol{y}^+)-\boldsymbol{u}(\boldsymbol{y}^-) = \boldsymbol{E}\cdot(\boldsymbol{y}^+-\boldsymbol{y}^-)$$

where $\boldsymbol{y}^{\pm}$ are opposite points on the unit cell boundary related by the periodicity condition. This formulation is widely used in solid mechanics FE software as it does not require specific change of the problem formulation but just adding tying constraints between some degrees of freedom.

This formulation is however not the easiest to deal with in `dolfinx_mpc`.

### Periodic fluctuation as the main unknown

Instead, we will keep the initial formulation and consider the periodic fluctuation $\boldsymbol{v}$ as the main unknown. The periodicity constraint on $\boldsymbol{v}$ will be imposed using the `create_periodic_constraint` functions of `dolfinx_mpc`. To do so, one must define the periodic map linking the different unit cell boundaries. Here the unit cell is 2D and its boundary is represented by a parallelogram of vertices ``corners`` and the corresponding base vectors `a1` and `a2` are computed. The right part is then mapped onto the left part, the top part onto the bottom part and the top-right corner onto the bottom-left one.

```{code-cell} ipython3
import numpy as np
from mpi4py import MPI
import gmsh
import ufl
from dolfinx import fem, io
import dolfinx.fem.petsc
from dolfinx.io.gmshio import model_to_mesh
import dolfinx_mpc.utils
from dolfinx_mpc import LinearProblem

Lx = 1.0
Ly = np.sqrt(3) / 2.0 * Lx
c = 0.5 * Lx
R = 0.2 * Lx
h = 0.01 * Lx

corners = np.array([[0.0, 0.0], [Lx, 0.0], [Lx + c, Ly], [c, Ly]])

a1 = corners[1,:]-corners[0,:] # first vector generating periodicity
a2 = corners[3,:]-corners[0,:] # second vector generating periodicity


def periodic_relation_left_right(x):
    out_x = np.zeros(x.shape)
    out_x[0] = x[0] - a1[0]
    out_x[1] = x[1] - a1[1]
    out_x[2] = x[2]
    return out_x


def periodic_relation_bottom_top(x):
    out_x = np.zeros(x.shape)
    out_x[0] = x[0] - a2[0]
    out_x[1] = x[1] - a2[1]
    out_x[2] = x[2]
    return out_x
```
The geometry is then generated using `gmsh` Python API and the Open Cascade kernel. We tag the matrix with tag `1` and the inclusions with tag `2`. The bottom, right, top and left boundaries are respectively tagged `1, 2, 3, 4`.

```{code-cell} ipython3
gdim = 2  # domain geometry dimension
fdim = 1  # facets dimension
gmsh.initialize()

occ = gmsh.model.occ
mesh_comm = MPI.COMM_WORLD
model_rank = 0
if model_rank == 0:
    points = [occ.add_point(*corner, 0) for corner in corners]
    lines = [occ.add_line(points[i], points[(i + 1) % 4]) for i in range(4)]
    loop = occ.add_curve_loop(lines)
    unit_cell = occ.add_plane_surface([loop])
    inclusions = [occ.add_disk(*corner, 0, R, R) for corner in corners]
    vol_dimTag = (gdim, unit_cell)
    out = occ.intersect(
        [vol_dimTag], [(gdim, incl) for incl in inclusions], removeObject=False
    )
    incl_dimTags = out[0]
    occ.synchronize()
    occ.cut([vol_dimTag], incl_dimTags, removeTool=False)
    occ.synchronize()

    # tag physical domains and facets
    gmsh.model.addPhysicalGroup(gdim, [vol_dimTag[1]], 1, name="Matrix")
    gmsh.model.addPhysicalGroup(
        gdim,
        [tag for _, tag in incl_dimTags],
        2,
        name="Inclusions",
    )
    gmsh.model.addPhysicalGroup(fdim, [7, 20, 10], 1, name="bottom")
    gmsh.model.addPhysicalGroup(fdim, [9, 19, 16], 2, name="right")
    gmsh.model.addPhysicalGroup(fdim, [15, 18, 12], 3, name="top")
    gmsh.model.addPhysicalGroup(fdim, [11, 17, 5], 4, name="left")
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)

    gmsh.model.mesh.generate(gdim)
```

```{code-cell} ipython3
domain, cells, facets = model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim=gdim)
gmsh.finalize()
```

```{code-cell} ipython3
:tags: [hide-input]

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

```{code-cell} ipython3
E = create_piecewise_constant_field(domain, cells, {1: 50e3, 2: 210e3})
nu = create_piecewise_constant_field(domain, cells, {1: 0.2, 2: 0.3})
```

```{code-cell} ipython3
lmbda = E * nu / (1 + nu) / (1 - 2 * nu)
mu = E / 2 / (1 + nu)
```

```{code-cell} ipython3
Eps = fem.Constant(domain, np.zeros((2, 2)))
Eps_ = fem.Constant(domain, np.zeros((2, 2)))
x = ufl.SpatialCoordinate(domain)
```

```{code-cell} ipython3
vol = fem.assemble_scalar(fem.form(1 * ufl.dx(domain=domain)))
print("Volume:", vol)
```

```{code-cell} ipython3
def epsilon(v):
    return ufl.sym(ufl.grad(v))
```

```{code-cell} ipython3
def sigma(v):
    eps = Eps + epsilon(v)
    return lmbda * ufl.tr(eps) * ufl.Identity(gdim) + 2 * mu * eps
```

```{code-cell} ipython3
# Define function space
V = fem.functionspace(domain, ("P", 2, (gdim,)))
```

```{code-cell} ipython3
# Define variational problem
du = ufl.TrialFunction(V)
u_ = ufl.TestFunction(V)
a_form, L_form = ufl.system(ufl.inner(sigma(du), epsilon(u_)) * ufl.dx)
```

```{code-cell} ipython3
elementary_load = [
    np.array([[1.0, 0.0], [0.0, 0.0]]),
    np.array([[0.0, 0.0], [0.0, 1.0]]),
    np.array([[0.0, 0.5], [0.5, 0.0]]),
]
dim_load = len(elementary_load)
```

```{code-cell} ipython3

point_dof = fem.locate_dofs_geometrical(
    V, lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0)
)
bcs = [fem.dirichletbc(np.zeros((gdim,)), point_dof, V)]
```


```{code-cell} ipython3
mpc = dolfinx_mpc.MultiPointConstraint(V)
mpc.create_periodic_constraint_topological(
    V, facets, 2, periodic_relation_left_right, bcs
)
mpc.create_periodic_constraint_topological(
    V, facets, 3, periodic_relation_bottom_top, bcs
)
mpc.finalize()
```

```{code-cell} ipython3
u = fem.Function(mpc.function_space, name="Displacement")
v = fem.Function(mpc.function_space, name="Periodic_fluctuation")
problem = LinearProblem(
    a_form,
    L_form,
    mpc,
    bcs=bcs,
    u=v,
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
)
```

```{code-cell} ipython3
vtk = io.VTKFile(domain.comm, "linear_elasticity.pvd", "w")
C = np.zeros((dim_load, dim_load))
for nload in range(dim_load):
    Eps.value = elementary_load[nload]
    u.interpolate(
        fem.Expression(
            ufl.dot(Eps, x), mpc.function_space.element.interpolation_points()
        )
    )

    problem.solve()
    u.x.array[:] += v.x.array[:]

    vtk.write_function(u, nload)
    vtk.write_function(E, nload)

    for nload_ in range(dim_load):
        Eps_.value = elementary_load[nload_]

        C[nload, nload_] = (
            fem.assemble_scalar(fem.form(ufl.inner(sigma(v), Eps_) * ufl.dx)) / vol
        )
print(C)
np.savetxt("elasticity_tensor.csv", C)
```

```{code-cell} ipython3
vtk.close()
```
