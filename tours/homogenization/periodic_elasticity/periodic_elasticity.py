# ---
# jupyter:
#   jupytext:
#     default_lexer: ipython3
#     formats: md:myst,ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: fenicsx-v0.11
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Periodic homogenization of linear elasticity {far}`star`
#
# ```{admonition} Objectives
# :class: objectives
#
# This tour shows how to perform periodic homogenization of linear elastic heterogeneous materials. In particular, we show how to define periodic boundary conditions without redundant constraints at the cell corners, and how to compute the effective stiffness tensor.
# $\newcommand{\bu}{\boldsymbol{u}}
# \newcommand{\bv}{\boldsymbol{v}}
# \newcommand{\bt}{\boldsymbol{t}}
# \newcommand{\be}{\boldsymbol{e}}
# \newcommand{\ba}{\boldsymbol{a}}
# \newcommand{\by}{\boldsymbol{y}}
# \newcommand{\bE}{\boldsymbol{E}}
# \newcommand{\beps}{\boldsymbol{\varepsilon}}
# \newcommand{\bsig}{\boldsymbol{\sigma}}
# \newcommand{\T}{^{\text{T}}}$
# ```
#
# ```{attention}
# This tour requires DOLFINx v0.11 and the `dolfinx_mpc` add-on package (v0.11) to enforce periodic boundary conditions. Installation instructions are given in the [dolfinx_mpc repository](https://github.com/jorgensd/dolfinx_mpc).
# ```
#
# ```{admonition} Download sources
# :class: download
#
# * {Download}`Python script<./periodic_elasticity.py>`
# * {Download}`Jupyter notebook<./periodic_elasticity.ipynb>`
# ```
#
# ```{seealso}
# The finite-strain, three-dimensional extension of this tour is [](/tours/homogenization/periodic_hyperelasticity/periodic_hyperelasticity.md).
# ```
#
# We consider a 2D plane strain problem on a skewed unit cell of dimensions $1\times \sqrt{3}/2$. Circular inclusions (tag `2`) of radius $R$ and elastic properties $(E_r, \nu_r)$ are embedded in a matrix material (tag `1`) of properties $(E_m, \nu_m)$, following a hexagonal pattern. The hexagonal symmetry implies that the effective behavior is isotropic in the plane, a property that we verify numerically below.
#
# ```{image} unit_cell.png
# :align: center
# :width: 600px
# ```
#
# ## Periodic homogenization framework
#
# The goal of homogenization theory is to compute the effective elastic moduli of the homogenized medium associated with a given microstructure. In a linear elastic setting, this amounts to solving the following auxiliary problem on the unit cell $\mathcal{A}$:
#
# $$\begin{cases}\operatorname{div} \boldsymbol{\sigma} = \boldsymbol{0} & \text{in } \mathcal{A} \\
# \boldsymbol{\sigma} = \mathbb{C}(\boldsymbol{y}):\boldsymbol{\varepsilon} & \text{for }\boldsymbol{y}\in\mathcal{A} \\
# \boldsymbol{\varepsilon} = \boldsymbol{E} + \nabla^\text{s} \boldsymbol{v} & \text{in } \mathcal{A} \\
# \boldsymbol{v} & \text{is } \mathcal{A}\text{-periodic} \\
# \boldsymbol{T}=\boldsymbol{\sigma}\cdot\boldsymbol{n} & \text{is } \mathcal{A}\text{-antiperiodic}
# \end{cases}$$
#
# where $\boldsymbol{E}$ is the **given** macroscopic strain, $\boldsymbol{v}$ a periodic fluctuation and $\mathbb{C}(\boldsymbol{y})$ the heterogeneous elasticity tensor depending on the microscopic position $\boldsymbol{y}\in\mathcal{A}$. By construction, the local strain is equal on average to the macroscopic strain: $\langle \boldsymbol{\varepsilon} \rangle = \boldsymbol{E}$. Defining the macroscopic stress $\boldsymbol{\Sigma}$ as the average of the local stress, $\boldsymbol{\Sigma} = \langle \boldsymbol{\sigma} \rangle$, the auxiliary problem yields a linear relation between the loading $\boldsymbol{E}$ and the resulting average stress:
#
# $$\boldsymbol{\Sigma}  = \mathbb{C}^\text{hom}:\boldsymbol{E}$$
#
# where $\mathbb{C}^\text{hom}$ is the effective stiffness tensor of the homogenized medium. Its components are computed by solving elementary load cases corresponding to the components of $\boldsymbol{E}$ and averaging the resulting local stresses.
#
# ### Total displacement as the main unknown
#
# The problem can also be written in terms of the total displacement $\boldsymbol{u} = \boldsymbol{E}\cdot\boldsymbol{y} + \boldsymbol{v}$, with $\boldsymbol{\varepsilon} = \nabla^\text{s} \boldsymbol{u}$. The periodicity condition then becomes:
#
# $$\boldsymbol{u}(\boldsymbol{y}^+)-\boldsymbol{u}(\boldsymbol{y}^-) = \boldsymbol{E}\cdot(\boldsymbol{y}^+-\boldsymbol{y}^-)$$
#
# where $\boldsymbol{y}^{\pm}$ are opposite points on the unit cell boundary. This formulation is widely used in finite-element software since it only requires tying constraints between degrees of freedom. It is however not the easiest one to handle with `dolfinx_mpc`.
#
# ### Periodic fluctuation as the main unknown
#
# Instead, we keep the initial formulation and use the periodic fluctuation $\boldsymbol{v}$ as the main unknown. Periodicity is imposed with the `create_periodic_constraint_geometrical` method of `dolfinx_mpc`, which requires a map from the *slave* degrees of freedom to their *master* counterparts on the opposite side.
#
# This elimination is consistent only if each slave is constrained once and maps onto masters that are neither slaves nor subject to a Dirichlet condition. A corner of the cell belongs to two edges: mapping the right edge onto the left one and the top edge onto the bottom one would constrain the top-right corner twice and chain the corners back to the bottom-left one. We therefore proceed hierarchically:
#
# * the interiors of the right and top edges are mapped onto the interiors of the left and bottom edges;
# * the four corners, which carry the same fluctuation by periodicity, are fixed to $\bv = 0$ by a Dirichlet condition. This also removes the rigid body translations, which periodicity does not fix.
#
# ```{important}
# Mapping the corners through two successive edge constraints creates chained slaves. Effective properties still look reasonable, but local fields near the corners are wrong and the discrete effective stiffness loses its symmetry. On the mesh of this tour, doing so changes the effective moduli by less than 0.01%, but introduces spurious shear–normal couplings in $\mathbb{C}^\text{hom}$ and local stress errors near the corners reaching 3.6 times the maximum stress. The same hierarchy (faces, then ridges, then vertices) is used in 3D in [](/tours/homogenization/periodic_hyperelasticity/periodic_hyperelasticity.md).
# ```
#
# ## Implementation
#
# ### Unit cell geometry and periodic mesh
#
# The unit cell is a parallelogram generated by the vectors $\ba_1$ and $\ba_2$. It is convenient to work with the lattice coordinates $(\xi_1, \xi_2)$ of a point $\by = \xi_1\ba_1 + \xi_2\ba_2$: the cell is $[0, 1]^2$ in these coordinates, the right edge is $\xi_1 = 1$ and the top edge is $\xi_2 = 1$.

# %%
import numpy as np
from mpi4py import MPI
import gmsh
import ufl
from dolfinx import fem, io, plot
import dolfinx.fem.petsc
import dolfinx_mpc

Lx = 1.0
Ly = np.sqrt(3) / 2.0 * Lx
c = 0.5 * Lx
R = 0.2 * Lx
h = 0.02 * Lx

corners = np.array([[0.0, 0.0], [Lx, 0.0], [Lx + c, Ly], [c, Ly]])
a1 = corners[1, :] - corners[0, :]  # first periodicity vector
a2 = corners[3, :] - corners[0, :]  # second periodicity vector
lattice = np.column_stack([a1, a2])


def lattice_coordinates(x):
    """Lattice coordinates (xi_1, xi_2) of points x of shape (2 or 3, N)."""
    return np.linalg.solve(lattice, x[:2])


gdim = 2
comm = MPI.COMM_WORLD
model_rank = 0
MATRIX_TAG, INCL_TAG = 1, 2

# %% [markdown]
# The geometry is generated with the `gmsh` Python API and the Open Cascade kernel. A disk centered at each corner is fragmented with the cell, and the disk parts outside the cell are removed. The matrix is tagged `1` and the inclusions `2`. The `setPeriodic` command then makes the meshes of opposite edges identical, so that each slave node has a matching master node.

# %% tags=["hide-input", "hide-output"]
gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)  # to disable meshing info
if comm.rank == model_rank:
    occ = gmsh.model.occ
    points = [occ.addPoint(*corner, 0) for corner in corners]
    lines = [occ.addLine(points[i], points[(i + 1) % 4]) for i in range(4)]
    unit_cell = occ.addPlaneSurface([occ.addCurveLoop(lines)])
    disks = [occ.addDisk(*corner, 0, R, R) for corner in corners]
    occ.fragment([(gdim, unit_cell)], [(gdim, disk) for disk in disks])
    occ.synchronize()

    # remove the disk parts lying outside the unit cell
    eps = 1e-8
    inside = []
    for _, tag in gmsh.model.getEntities(gdim):
        xi = lattice_coordinates(np.array(occ.getCenterOfMass(gdim, tag))[:, None])
        if np.all((xi > -eps) & (xi < 1 + eps)):
            inside.append(tag)
        else:
            occ.remove([(gdim, tag)], recursive=True)
    occ.synchronize()

    areas = [occ.getMass(gdim, tag) for tag in inside]
    matrix = inside[int(np.argmax(areas))]
    inclusions = [tag for tag in inside if tag != matrix]
    gmsh.model.addPhysicalGroup(gdim, [matrix], MATRIX_TAG, name="Matrix")
    gmsh.model.addPhysicalGroup(gdim, inclusions, INCL_TAG, name="Inclusions")

    # periodic edge meshes: edge xi_i = 1 is the image of edge xi_i = 0 by a translation a_i
    curves = [tag for _, tag in gmsh.model.getEntities(gdim - 1)]
    xi_curves = {
        tag: lattice_coordinates(np.array(occ.getCenterOfMass(gdim - 1, tag))[:, None])[:, 0]
        for tag in curves
    }
    for i, ai in enumerate([a1, a2]):
        j = 1 - i  # sort edge pieces along the other lattice direction
        masters = sorted(
            [t for t in curves if np.isclose(xi_curves[t][i], 0.0)],
            key=lambda t: xi_curves[t][j],
        )
        slaves = sorted(
            [t for t in curves if np.isclose(xi_curves[t][i], 1.0)],
            key=lambda t: xi_curves[t][j],
        )
        translation = np.eye(4)
        translation[:2, 3] = ai
        gmsh.model.mesh.setPeriodic(gdim - 1, slaves, masters, translation.flatten().tolist())

    gmsh.option.setNumber("Mesh.MeshSizeMin", h)
    gmsh.option.setNumber("Mesh.MeshSizeMax", h)
    gmsh.model.mesh.generate(gdim)

mesh_data = io.gmsh.model_to_mesh(gmsh.model, comm, model_rank, gdim=gdim)
gmsh.finalize()

domain = mesh_data.mesh
cells = mesh_data.cell_tags

# %% [markdown]
# We compute the unit cell area from the mesh for verification:

# %%
vol = comm.allreduce(fem.assemble_scalar(fem.form(1 * ufl.dx(domain=domain))), op=MPI.SUM)
print(f"Area: {vol:.6f} (expected {Lx * Ly:.6f})")


# %% [markdown]
# ### Variational formulation with heterogeneous material properties
#
# In the [legacy demo](https://comet-fenics.readthedocs.io/en/latest/demo/periodic_homog_elas/periodic_homog_elas.html), the bilinear form used two terms with the `dx(1)` and `dx(2)` cell measures, one per phase. This approach does not scale well to many phases or to spatially varying properties. Instead, we use a single constitutive law for all phases, with Young's modulus and Poisson's ratio represented as piecewise constant (`DG0`) functions, see [](/tips/piecewise_constant_field/piecewise_constant_field.md).

# %%
def create_piecewise_constant_field(domain, cell_markers, property_dict, name=None):
    """Create a DG0 function taking a constant value on each tagged subdomain.

    Parameters
    ----------
    domain : dolfinx.mesh.Mesh
        Mesh object
    cell_markers : dolfinx.mesh.MeshTags
        Cell markers
    property_dict : dict
        Dictionary mapping region tags to values {tag: value}
    name : str, optional
        Name of the function

    Returns
    -------
    A DG0 function
    """
    V0 = fem.functionspace(domain, ("DG", 0))
    k = fem.Function(V0, name=name)
    for tag, value in property_dict.items():
        k.x.array[cell_markers.find(tag)] = value
    k.x.scatter_forward()
    return k


E = create_piecewise_constant_field(domain, cells, {1: 50e3, 2: 210e3}, name="YoungModulus")
nu = create_piecewise_constant_field(domain, cells, {1: 0.2, 2: 0.3}, name="PoissonRatio")

lmbda = E * nu / (1 + nu) / (1 - 2 * nu)
mu = E / 2 / (1 + nu)

# %% [markdown]
# Our main unknown is the periodic fluctuation $\bv$. The loading is imposed through the macroscopic strain $\bE$, defined as a `fem.Constant`, and the total displacement $\bu = \bE\cdot\by + \bv$ is rebuilt afterwards using `ufl.SpatialCoordinate`. The form `ufl.inner(sigma(du), epsilon(u_)) * ufl.dx` contains both bilinear and linear terms, the latter due to the macroscopic strain acting as a loading. We use `ufl.system` to split it into its bilinear and linear parts.

# %%
Eps = fem.Constant(domain, np.zeros((2, 2)))
Eps_ = fem.Constant(domain, np.zeros((2, 2)))
y = ufl.SpatialCoordinate(domain)


def epsilon(v):
    return ufl.sym(ufl.grad(v))


def sigma(v):
    eps = Eps + epsilon(v)
    return lmbda * ufl.tr(eps) * ufl.Identity(gdim) + 2 * mu * eps


V = fem.functionspace(domain, ("P", 2, (gdim,)))
du = ufl.TrialFunction(V)
u_ = ufl.TestFunction(V)
a_form, L_form = ufl.system(ufl.inner(sigma(du), epsilon(u_)) * ufl.dx)

# %% [markdown]
# ### Periodic boundary conditions with `dolfinx_mpc`
#
# We first fix the four corners with a Dirichlet condition. Points are classified from their lattice coordinates.

# %%
tol = 1e-8


def on_cell_boundary(x):
    """Array of shape (2, N): True where lattice coordinate i equals 0 or 1."""
    xi = lattice_coordinates(x)
    return np.isclose(xi, 0.0, atol=tol) | np.isclose(xi, 1.0, atol=tol)


def corners_marker(x):
    return np.all(on_cell_boundary(x), axis=0)


corner_dofs = fem.locate_dofs_geometrical(V, corners_marker)
bcs = [fem.dirichletbc(np.zeros((gdim,)), corner_dofs, V)]


# %% [markdown]
# We then map the interior of the right edge ($\xi_1 = 1$) onto the left edge with the translation $-\ba_1$, and the interior of the top edge ($\xi_2 = 1$) onto the bottom edge with the translation $-\ba_2$. Corners are excluded from both slave sets, and passing `bcs` ensures that Dirichlet dofs are never used as slaves.

# %%
def edge_slaves(i):
    """Interior of the edge xi_i = 1."""

    def indicator(x):
        xi = lattice_coordinates(x)
        return np.isclose(xi[i], 1.0, atol=tol) & ~corners_marker(x)

    return indicator


def translation(ai):
    def relation(x):
        out_x = x.copy()
        out_x[0] -= ai[0]
        out_x[1] -= ai[1]
        return out_x

    return relation


mpc = dolfinx_mpc.MultiPointConstraint(V)
mpc.create_periodic_constraint_geometrical(V, edge_slaves(0), translation(a1), bcs)
mpc.create_periodic_constraint_geometrical(V, edge_slaves(1), translation(a2), bcs)
mpc.finalize()

# %% [markdown]
# ### Solving the elementary load cases
#
# We now define the problem instance. We must use the `LinearProblem` provided by `dolfinx_mpc` rather than that of `dolfinx.fem.petsc`, since it takes the `MultiPointConstraint` object as an argument. The solution $\bv$, and the total displacement $\bu$, must be defined on the constrained function space `mpc.function_space` rather than on `V`.

# %%
u = fem.Function(mpc.function_space, name="Displacement")
v = fem.Function(mpc.function_space, name="Periodic_fluctuation")
problem = dolfinx_mpc.LinearProblem(
    a_form,
    L_form,
    mpc,
    bcs=bcs,
    u=v,
    petsc_options_prefix="periodic_elasticity_",
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
)

# %% [markdown]
# We consider three elementary load cases: macroscopic uniaxial strain in the $x$ and $y$ directions and macroscopic shear in the $xy$ plane. For each load case, we update the macroscopic strain $\bE$, solve the problem and rebuild the total displacement from the periodic fluctuation. We then compute the components of the macroscopic stress $\boldsymbol{\Sigma}$. Since the load cases are unit strains, they directly give the components of the effective stiffness $\mathbb{C}^\text{hom}$ in Voigt notation.

# %% tags=["hide-input"]
import pyvista

pyvista.set_jupyter_backend("static")


def plot_warped(u, scale=1.0, plot_mesh=False, title=None):
    Vu = u.function_space
    u_topology, u_cell_types, u_geometry = plot.vtk_mesh(Vu)
    u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
    u_3D = np.zeros((u_geometry.shape[0], 3))
    u_3D[:, :2] = u.x.array.reshape(-1, 2)
    u_grid.point_data[u.name] = u_3D
    u_grid.set_active_vectors(u.name)
    warped = u_grid.warp_by_vector(u.name, factor=scale)

    plotter = pyvista.Plotter()
    plotter.window_size = (800, 300)
    plotter.add_mesh(warped)
    if plot_mesh:
        edges = warped.extract_all_edges()
        plotter.add_mesh(edges, color="k", line_width=1, opacity=0.5)
    plotter.view_xy()
    if title:
        plotter.add_text(title, font_size=14)
    plotter.show()


# %%
elementary_load = [
    np.array([[1.0, 0.0], [0.0, 0.0]]),
    np.array([[0.0, 0.0], [0.0, 1.0]]),
    np.array([[0.0, 0.5], [0.5, 0.0]]),
]
load_labels = ["Exx", "Eyy", "Exy"]
dim_load = len(elementary_load)

Eps_y = fem.Expression(ufl.dot(Eps, y), mpc.function_space.element.interpolation_points)
C_hom = np.zeros((dim_load, dim_load))
for nload in range(dim_load):
    Eps.value = elementary_load[nload]
    u.interpolate(Eps_y)

    problem.solve()
    u.x.array[:] += v.x.array[:]

    plot_warped(u, scale=0.5, plot_mesh=True, title=load_labels[nload])

    for nload_ in range(dim_load):
        Eps_.value = elementary_load[nload_]
        C_hom[nload, nload_] = (
            comm.allreduce(
                fem.assemble_scalar(fem.form(ufl.inner(sigma(v), Eps_) * ufl.dx)), op=MPI.SUM
            )
            / vol
        )

# %% [markdown]
# ### Effective stiffness

# %%
with np.printoptions(suppress=True, precision=0):
    print(C_hom)

# %% [markdown]
# The effective stiffness is symmetric and isotropic up to the finite element discretization error. With $\lambda^\text{hom} = \mathbb{C}^\text{hom}_{xxyy}$ and $\mu^\text{hom} = \mathbb{C}^\text{hom}_{xyxy}$, isotropy requires $\mathbb{C}^\text{hom}_{xxxx} = \mathbb{C}^\text{hom}_{yyyy} = \lambda^\text{hom}+2\mu^\text{hom}$.

# %%
lmbda_hom = C_hom[0, 1]
mu_hom = C_hom[2, 2]
print(f"Symmetry check: max |C - C^T| / max |C| = {np.max(np.abs(C_hom - C_hom.T)) / np.max(np.abs(C_hom)):.1e}")
print(f"Isotropy check: C_xxxx = {C_hom[0, 0]:.0f}, C_yyyy = {C_hom[1, 1]:.0f}, lambda + 2 mu = {lmbda_hom + 2 * mu_hom:.0f}")

# %% [markdown]
# ```{note}
# The discrete effective stiffness is symmetric to solver precision. For load cases $i$ and $j$ with fluctuations $\bv_i$ and $\bv_j$, the averaged stress gives $\mathbb{C}^\text{hom}_{ij} = \langle\bsig_i\rangle:\bE_j = \langle\bsig_i:(\bE_j+\nabla^\text{s}\bv_j)\rangle$, because $\bv_j$ is an admissible test function of load case $i$. The right-hand side is symmetric in $i$ and $j$. This requires $\bv_j$ to belong exactly to the constrained test space, which is precisely what an inconsistent treatment of the corners breaks.
# ```
#
# Since the problem is solved in plane strain, $\lambda^\text{hom}$ and $\mu^\text{hom}$ are the in-plane Lamé coefficients of the transversely isotropic homogenized medium. The equivalent Young's modulus and Poisson's ratio for in-plane loading follow from the usual isotropic formulas:
#
# $$E^\text{hom} = \mu^\text{hom}\dfrac{3\lambda^\text{hom}+2\mu^\text{hom}}{\lambda^\text{hom}+\mu^\text{hom}}, \qquad \nu^\text{hom} = \dfrac{\lambda^\text{hom}}{2(\lambda^\text{hom}+\mu^\text{hom})}$$

# %%
E_hom = mu_hom * (3 * lmbda_hom + 2 * mu_hom) / (lmbda_hom + mu_hom)
nu_hom = lmbda_hom / (lmbda_hom + mu_hom) / 2
print(f"Effective Young's modulus: {E_hom:.0f}")
print(f"Effective Poisson's ratio: {nu_hom:.3f}")

# %% [markdown]
# ### Local stress field
#
# We finally plot the local stress $\sigma_{xx}$ of the last load case (macroscopic shear) on $2\times 2$ repeated unit cells. The field is continuous across the cell edges and, in particular, around the corners.

# %%
W0 = fem.functionspace(domain, ("DG", 0))
sig_xx = fem.Function(W0, name="sigma_xx")
sig_xx.interpolate(fem.Expression(sigma(v)[0, 0], W0.element.interpolation_points))

# %% tags=["hide-input"]
num_cells = domain.topology.index_map(gdim).size_local
grid = pyvista.UnstructuredGrid(*plot.vtk_mesh(domain, gdim, np.arange(num_cells, dtype=np.int32)))
grid.cell_data["sigma_xx"] = sig_xx.x.array[:num_cells]
plotter = pyvista.Plotter()
plotter.window_size = (800, 600)
for i in range(2):
    for j in range(2):
        shift = i * a1 + j * a2
        tile = grid.translate((shift[0], shift[1], 0.0), inplace=False)
        plotter.add_mesh(tile, scalars="sigma_xx", cmap="coolwarm", scalar_bar_args={"title": "Local stress sigma_xx"})
        plotter.add_mesh(
            tile.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False),
            color="k",
            line_width=1,
        )
plotter.view_xy()
plotter.show()
