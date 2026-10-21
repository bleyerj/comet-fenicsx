# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     default_lexer: ipython3
#     formats: md:myst,ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
# ---

# %% [markdown]
# # Cohesive zone modeling restricted to an interface {far}`star`{far}`star`{far}`star`
#
# ```{admonition} Objectives
# :class: objectives
#
# This tutorial demonstrates how to formulate a cohesive zone model (CZM) involving two subdomains connected by a cohesive interface. Contrary to the previous tour on [](/tours/interfaces/intrinsic_czm/intrinsic_czm.md) which uses a Discontinuous Galerkin formulation everywhere (*case (a)*), discontinuities modeled as CZM are here considered at the interface only. This prevents the use of a single continuous function space across the whole mesh and requires instead to work with **disconnected submeshes** and mixed Continuous Galerkin interpolation (*case (b)*).$
# \newcommand{\bu}{\boldsymbol{u}}
# \newcommand{\bv}{\boldsymbol{v}}
# \newcommand{\bn}{\boldsymbol{n}}
# \newcommand{\jump}[1]{[\![#1]\!]}$
# ```
#
# ```{image} CZM_mesh.svg
# :width: 800px
# :align: center
# ```
#
# ```{admonition} Download sources
# :class: download
#
# * {Download}`Python script<./czm_interface_only.py>`
# * {Download}`Jupyter notebook<./czm_interface_only.ipynb>`
# * {Download}`Complete sources files<./czm_interface_only.zip>`
# ```
#
# ```{seealso}
# This tour borrows from different resources discussing mixed domain formulations and DG couplings such as:
# - [Joe P. Dean mixed domain demos](https://github.com/jpdean/mixed_domain_demos)
# - [This gist by Jørgen Dokken](https://gist.github.com/jorgensd/8a5c32f491195e838f5863ca88b27bce)
# - [This post](https://fenicsproject.discourse.group/t/discontinuity-at-interface-using-mixed-domains/15040)
# ```
#
# ## Introduction
#
# We consider the same geometry and the same cohesive zone model formulation as in [](/tours/interfaces/intrinsic_czm/intrinsic_czm.md). The only difference here is that we do not want to introduce a DG formulation everywhere which might be too costly for large meshes due to the increased number of degrees of freedom. We are only concerned here with modeling the debonding of the matrix/inclusion interface.
#
# To do so, we will build a formulation involving the matrix and the inclusion submeshes with a standard Continuous Galerkin formulation in both of them. The two domains will then be tied by the formulation of a CZM on the interface. The interface itself will also be defined as a submesh (of codimension 1) to define the damage internal state variable of the CZM law, see again the previous tour.
#
# ```{attention}
# - The approach proposed here is well suited for interfaces separating a moderate number of subdomains as we explicitly need to define each subdomain and each interface between pairs of subdomains separately. For more complex problems, it might be more beneficial to resort to a fully DG formulation.
#
# - Currently, parallel computations are not fully supported.
# ```
#
# ## Implementation
#
# All the mixed-domain features needed for this tour, such as entity maps, transfer of mesh tags to submeshes, interpolation between a submesh and its parent mesh or the resolution of block linear systems, are directly available in `dolfinx`. We only define two small helper functions along the way.

# %%
from IPython.display import clear_output
import numpy as np
import matplotlib.pyplot as plt
import gmsh
import pyvista
from mpi4py import MPI
import ufl
import basix
import dolfinx
from dolfinx import fem, io
from dolfinx.fem.petsc import LinearProblem


# %% [markdown]
# ### Mesh and subdomains
#
# The same mesh is defined using `gmsh`.
#
# ```{note}
# For `gmsh`, there is no discontinuity between the matrix and the inclusion. Elements from each side of the interface share common nodes from the mesh point of view. It is when defining two function spaces on both submeshes and when formulating the CZM law that we introduce the possibility of a jump at the interface.
# ```

# %%
def create_matrix_inclusion_mesh(L, W, R, hsize):
    comm = MPI.COMM_WORLD

    gmsh.initialize()
    gdim = 2
    model_rank = 0
    if comm.rank == model_rank:
        gmsh.option.setNumber("General.Terminal", 0)  # to disable meshing info
        gmsh.model.add("Model")

        gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, L, W, tag=1)
        gmsh.model.occ.addDisk(0.4, 0.0, 0.0, R, R, tag=2)
        gmsh.model.occ.addDisk(0.6, W, 0.0, R, R, tag=3)
        gmsh.model.occ.fragment([(gdim, 1)], [(gdim, 2), (gdim, 3)], removeObject=True)

        gmsh.model.occ.synchronize()

        gmsh.model.occ.remove([(gdim, 5), (gdim, 4)], recursive=True)

        gmsh.model.occ.synchronize()

        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", hsize)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", hsize)

        gmsh.model.addPhysicalGroup(gdim, [1], 1, name="Matrix")
        gmsh.model.addPhysicalGroup(gdim, [2, 3], 2, name="Inclusions")

        gmsh.model.addPhysicalGroup(gdim - 1, [3], 1, name="left")
        gmsh.model.addPhysicalGroup(gdim - 1, [7], 2, name="right")
        gmsh.model.addPhysicalGroup(gdim - 1, [1, 5], 3, name="interface")
        gmsh.model.addPhysicalGroup(gdim - 1, [2, 9, 8, 4, 10, 6], 4, name="sides")
        gmsh.model.mesh.generate(gdim)

    partitioner = dolfinx.cpp.mesh.create_cell_partitioner(
        dolfinx.mesh.GhostMode.shared_facet, 2
    )
    mesh_data = dolfinx.io.gmsh.model_to_mesh(
        gmsh.model, MPI.COMM_WORLD, model_rank, gdim=gdim, partitioner=partitioner
    )
    gmsh.finalize()
    return (mesh_data.mesh, mesh_data.cell_tags, mesh_data.facet_tags)


# %% [markdown]
# We first create the mesh and define the different tags for identifying physical domains and interfaces.

# %%
length = 1.0
width = 0.5
radius = 0.25
hsize = 0.02
domain, cells, facets = create_matrix_inclusion_mesh(length, width, radius, hsize)
MATRIX_TAG = 1  # tag of matrix phase
INCL_TAG = 2  # tag of inclusion phase
INT_TAG = 3  # tag of interface
LEFT_TAG = 1  # tag of left boundary
RIGHT_TAG = 2  # tag of right boundary
interface_facets = facets.find(INT_TAG)

tdim = domain.topology.dim
fdim = tdim - 1

# %% [markdown]
# We define three submeshes: two submeshes (of codim. 0) corresponding to the matrix and inclusion 2D domains and one submesh (of codim. 1) corresponding to the facet restriction on the interface. Besides the submesh itself, `dolfinx.mesh.create_submesh` returns `EntityMap` objects relating the cells (resp. the vertices) of the submesh to the corresponding entities of the parent mesh `domain`. For the interface submesh, the cells of `interface_mesh` are thus related to facets of `domain`.

# %%
subdomain2, subdomain2_cell_map, subdomain2_vertex_map, _ = dolfinx.mesh.create_submesh(
    domain, tdim, cells.find(INCL_TAG)
)
subdomain1, subdomain1_cell_map, subdomain1_vertex_map, _ = dolfinx.mesh.create_submesh(
    domain, tdim, cells.find(MATRIX_TAG)
)
interface_mesh, interface_cell_map, _, _ = dolfinx.mesh.create_submesh(
    domain, fdim, interface_facets
)

# %% [markdown]
# Each submesh is plotted individually:

# %%
plotter = pyvista.Plotter(off_screen=True)
grid1 = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(subdomain1))
plotter.add_mesh(grid1, show_edges=True, color="gold")
grid2 = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(subdomain2))
plotter.add_mesh(grid2, show_edges=True, color="darkblue")
gridi = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(interface_mesh))
plotter.add_mesh(gridi, show_edges=True, color="darkred", line_width=5)
plotter.show_axes()
plotter.view_xy()
plotter.show()

# %% [markdown]
# Now that we have defined submeshes, we need to transfer the facet mesh tags defined on the parent domain to their submesh counterparts. This is done with `dolfinx.mesh.transfer_meshtags_to_submesh`, which uses the vertex and cell entity maps of each submesh.

# %%
subdomain1_facet_tags = dolfinx.mesh.transfer_meshtags_to_submesh(
    facets, subdomain1, subdomain1_vertex_map, subdomain1_cell_map
)
subdomain2_facet_tags = dolfinx.mesh.transfer_meshtags_to_submesh(
    facets, subdomain2, subdomain2_vertex_map, subdomain2_cell_map
)
subdomain1.topology.create_connectivity(fdim, tdim)
subdomain2.topology.create_connectivity(fdim, tdim)

# %% [markdown]
# ### Entity maps and integration measures
#
# Similarly to the previous CZM tour, *entity maps* are required to assemble forms involving functions defined on a different mesh than the integration domain. Here, all forms will be integrated over the parent `domain` while the unknown displacements and the interfacial damage live on the three submeshes. The `EntityMap` objects returned by `create_submesh` are passed directly to the forms through the `entity_maps` argument.

# %%
entity_maps = [interface_cell_map, subdomain1_cell_map, subdomain2_cell_map]


# %% [markdown]
# Integrating terms on the interface requires a specific treatment. Indeed, the interface is a set of interior facets of `domain`, each of them being shared by one cell of subdomain 1 and one cell of subdomain 2. For an interior facet integral, these two cells correspond to the `"+"` and `"-"` sides of the facet, but their ordering is arbitrary. We must therefore define the integration entities manually, i.e. a list of `(cell, local facet index)` pairs for both sides of each interface facet. These are first computed with `fem.compute_integration_domains` and then reordered so that cells of subdomain 1 always correspond to the `"+"` side and cells of subdomain 2 to the `"-"` side. Functions defined on subdomain 1 (resp. 2) must therefore only be restricted to the `"+"` (resp. `"-"`) side in the interface forms.
#
# ````{note}
# Having a consistent facet orientation does not really impact the results here as we do not distinguish between positive and negative opening of the cohesive zone in the formulation of the `effective_opening`. In case one wants to distinguish between positive and negative opening, an inconsistent facet orientation would produce incorrect results.
#
# ```{image} facet_orientation.png
# :width: 700px
# :align: center
# ```
# ````

# %%
def interface_int_entities(domain, interface_facets, plus_cells):
    """Integration entities of an interface between two subdomains.

    Returns the list of (cell, local facet index) pairs on both sides of
    each interface facet, ordered such that cells in `plus_cells`
    correspond to the "+" side.
    """
    tdim = domain.topology.dim
    domain.topology.create_connectivity(tdim - 1, tdim)
    domain.topology.create_connectivity(tdim, tdim - 1)
    entities = fem.compute_integration_domains(
        fem.IntegralType.interior_facet, domain.topology, interface_facets
    ).reshape(-1, 4)
    swap = ~np.isin(entities[:, 0], plus_cells)
    entities[swap] = entities[swap][:, [2, 3, 0, 1]]
    return entities.flatten()


interface_entities = interface_int_entities(
    domain, interface_facets, cells.find(MATRIX_TAG)
)

# %% [markdown]
# We are now in position to define the various integration measures. The key point here is that the `dInt` interface measure is defined using the prescribed integration entities computed above. This is done by passing them to `subdomain_data` as follows.

# %%
dx = ufl.Measure("dx", domain=domain, subdomain_data=cells)
ds = ufl.Measure("ds", domain=domain, subdomain_data=facets)
dx_int = ufl.Measure("dx", domain=interface_mesh)
dInt = ufl.Measure(
    "dS",
    domain=domain,
    subdomain_data=[(INT_TAG, interface_entities)],
    subdomain_id=INT_TAG,
)

# %% [markdown]
# We can check that the length of the interface is properly computed.

# %%
Gamma = fem.assemble_scalar(fem.form(1 * dInt))
print(f"Check that Gamma is such that {Gamma} ~ {np.pi * 2 * radius}")


# %% [markdown]
# ### Weak form definition
#
# We generate piecewise constant fields of elastic properties, as in the previous tour, see also [](/tips/piecewise_constant_field/piecewise_constant_field.md).

# %%
def create_piecewise_constant_field(
    domain, cell_markers, property_dict, name=None, default_value=0
):
    """Create a DG-0 field taking constant values {tag: value} on each subdomain."""
    V0 = fem.functionspace(domain, ("DG", 0))
    k = fem.Function(V0, name=name)
    k.x.array[:] = default_value
    for tag, value in property_dict.items():
        cells = cell_markers.find(tag)
        k.x.array[cells] = np.full_like(cells, value, dtype=np.float64)
    return k


E = create_piecewise_constant_field(domain, cells, {MATRIX_TAG: 3.09e3, INCL_TAG: 10e3})
nu = create_piecewise_constant_field(domain, cells, {MATRIX_TAG: 0.25, INCL_TAG: 0.4})
mu = E / (2.0 * (1.0 + nu))
lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))


def epsilon(v):
    return ufl.sym(ufl.grad(v))


def sigma(v):
    return lmbda * ufl.tr(epsilon(v)) * ufl.Identity(tdim) + 2.0 * mu * epsilon(v)


# %% [markdown]
# Interfacial mechanical properties and expressions for the CZM law are defined similarly except that we consider constant properties since we define the CZM at the interface only.

# %%
Gc = fem.Constant(domain, 0.5)
sig_max = fem.Constant(domain, 50.0)
delta_0 = Gc / sig_max / ufl.exp(1)
beta = fem.Constant(domain, 2.0)


def normal_opening(v, n):
    return ufl.dot(v, n)


def tangential_opening(v, n):
    return v - normal_opening(v, n) * n


def effective_opening(v, n):
    return ufl.sqrt(normal_opening(v, n) ** 2 + beta**2 * tangential_opening(v, n) ** 2)


def T(opening, d):
    K_int = ufl.avg(Gc / delta_0**2) * (1 - d)
    return K_int * opening


# %% [markdown]
# In the previous tour, it was possible to use `ufl.jump` to define $\jump{\bu}$. Here, we need to define it manually from two displacement fields `u1` and `u2` which live on two different subdomains. Here, we define $\jump{\bu} = \bu^{(2)} - \bu^{(1)}$ where $(1)$ denotes subdomain 1 (the matrix) and $(2)$ denotes subdomain 2 (the inclusions). Note that we need to restrict quantities since we work with a facet measure. Since each subdomain only exists on one side of the interface, the restriction is imposed by the ordering of the integration entities: we must use the `"+"` side for subdomain 1 and the `"-"` side for subdomain 2.

# %%
def jump(u1, u2):
    return u2("-") - u1("+")


# %% [markdown]
# We now define the relevant function spaces. As hinted before, the unknown $\bu$ will consist of two displacements $(\bu^{(1)},\bu^{(2)})$ respectively belonging to a continuous Lagrange space defined on subdomains 1 and 2. We use a `MixedFunctionSpace` for this, meaning that we will end up with a block system.

# %%
V1 = fem.functionspace(subdomain1, ("Lagrange", 1, (tdim,)))
V2 = fem.functionspace(subdomain2, ("Lagrange", 1, (tdim,)))
W = ufl.MixedFunctionSpace(V1, V2)
u1 = fem.Function(V1, name="Displacement_1")
u2 = fem.Function(V2, name="Displacement_2")
v1, v2 = ufl.TestFunctions(W)
du1, du2 = ufl.TrialFunctions(W)

# %% [markdown]
# For easier post-processing, the computed displacement will be stored as a `DG` function, with jumps being non zero only at the interface. This space will not be used for defining the weak forms.

# %%
V = fem.functionspace(domain, ("DG", 1, (tdim,)))  # for post-processing only
u = fem.Function(V, name="Displacement")

# %% [markdown]
# Similarly to the previous tour, a `DG-1` function space on the interface `V_int` will be used to represent the damage fields. Again, we could have used a Quadrature space but this choice proves easier for visualization purposes.

# %%
V_int = fem.functionspace(interface_mesh, ("DG", 1))
d = fem.Function(V_int, name="Interfacial_damage")
d_prev = fem.Function(V_int, name="Previous_interfacial_damage")
d_old = fem.Function(V_int, name="Interfacial_damage_old")

# %% [markdown]
# We can now define the expression for the interfacial damage based on the effective opening $\delta$. The latter involves $\jump{\bu}$ and the interface normal. Since $\jump{\bu} = \bu^{(2)} - \bu^{(1)}$, we need the normal $\bn^{(1)\to(2)}$ from subdomain 1 to subdomain 2. This is `n("+")` since subdomain 1 is on `"+"` side of the interface and the facet normal points outwards the cell.

# %%
n1_to_2 = ufl.FacetNormal(domain)("+")
delta = effective_opening(jump(u1, u2), n1_to_2)
d_expr = ufl.max_value(ufl.avg(d_prev), 1 - ufl.exp(-delta / ufl.avg(delta_0)))

# %% [markdown]
# Bulk and interface contributions to the bilinear form are defined by separating the contributions of both subdomains.

# %%
a_bulk = ufl.inner(sigma(du1), epsilon(v1)) * dx(1) + ufl.inner(
    sigma(du2), epsilon(v2)
) * dx(2)
a_interface = ufl.dot(T(jump(du1, du2), ufl.avg(d)), jump(v1, v2)) * dInt
a = a_bulk + a_interface
f = fem.Constant(domain, np.zeros((tdim,)))
L = ufl.dot(f, v1) * dx(1) + ufl.dot(f, v2) * dx(2)

# %% [markdown]
# Finally, we use `ufl.extract_blocks` to obtain the different blocks $\begin{bmatrix} \text{a}_{11} & \text{a}_{12}\\ \text{a}_{21} & \text{a}_{22}\end{bmatrix}$ of the bilinear form associated with $\bu^{(1)}$ and $\bu^{(2)}$. These blocks will be compiled when defining the linear problem below.

# %%
a_blocks = ufl.extract_blocks(a)
L_blocks = ufl.extract_blocks(L)

# %% [markdown]
# As in the previous tour, we need to evaluate expressions that live on the facet mesh. We follow the same approach as discussed in detail in {ref}`facet:expressions:interpolation`. Since `d_expr` involves functions defined on the interface mesh and on both subdomains, all entity maps are passed to the corresponding form.

# %%
q_p = V_int.element.interpolation_points
weights = np.full(q_p.shape[0], 1.0, dtype=dolfinx.default_scalar_type)
q_el = basix.ufl.quadrature_element(
    interface_mesh.basix_cell(), scheme="custom", points=q_p, weights=weights
)
Q = fem.functionspace(interface_mesh, q_el)
q_ = ufl.TestFunction(Q)
dS_custom = ufl.Measure(
    "dS",
    domain=domain,
    metadata={
        "quadrature_rule": "custom",
        "quadrature_points": q_p,
        "quadrature_weights": weights,
    },
    subdomain_data=[(INT_TAG, interface_entities)],
    subdomain_id=INT_TAG,
)

facet_interp = fem.form(
    1 / ufl.FacetArea(domain)("+") * d_expr * ufl.avg(q_) * dS_custom,
    entity_maps=entity_maps,
)

# %% [markdown]
# ### Boundary conditions
#
# Dirichlet boundary conditions are now defined. Note that they involve only subdomain 1.

# %%
Uimp = fem.Constant(domain, (1.0, 0.0))
left_dofs = fem.locate_dofs_topological(V1, fdim, subdomain1_facet_tags.find(1))
right_dofs = fem.locate_dofs_topological(V1, fdim, subdomain1_facet_tags.find(2))

bcs = [
    fem.dirichletbc(np.zeros((tdim,)), left_dofs, V1),
    fem.dirichletbc(Uimp, right_dofs, V1),
]

# %% [markdown]
# The imposed displacement is initialized with unitary values so as to define the virtual displacement fields `v_reac1` and `v_reac2` to be used for measuring the reaction force on the boundary in a consistent manner based on the equilibrium residual $a(\bu,\bv_\text{reac})-L(\bv_\text{reac})$. Since boundary conditions only involve subdomain 1, `v_reac2` is simply zero.

# %%
v_reac1 = fem.Function(V1)
fem.set_bc(v_reac1.x.array, bcs)
v_reac2 = fem.Function(V2)
virtual_work_form = fem.form(
    ufl.replace(a - L, {du1: u1, du2: u2, v1: v_reac1, v2: v_reac2}),
    entity_maps=entity_maps,
)

# %% [markdown]
# The linear problem associated with the computation of the displacement at fixed damage is now defined outside the load-stepping loop. Since we use a mixed function space, we obtain a block linear system. `LinearProblem` directly supports such block problems: it expects the blocks of the bilinear and linear forms, the list of functions in which to store the solution and the entity maps. By default, the different blocks are assembled into a single PETSc matrix which is then factorized with MUMPS.

# %%
problem = LinearProblem(
    a_blocks,
    L_blocks,
    u=[u1, u2],
    bcs=bcs,
    entity_maps=entity_maps,
    petsc_options={
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    },
    petsc_options_prefix="czm_interface_",
)


# %% [markdown]
# ### Load-stepping
#
# The fixed-point resolution scheme is implemented as before. The main difference here lies in the post-processing steps. Data in `u1` and `u2` fields are transferred to a parent function `u` defined as a `DG` function on the parent mesh. We thus have a single displacement field in ParaView which presents jumps at the interface only. This interpolation from a submesh to its parent mesh is performed with `u.interpolate(u1, cells0=..., cells1=...)` where `cells0` are the submesh cells and `cells1` the corresponding cells of the parent mesh, as given by the cell entity map.

# %%
def submesh_to_parent_cells(submesh, cell_map):
    """Submesh cells and their corresponding cells in the parent mesh."""
    num_cells = submesh.topology.index_map(submesh.topology.dim).size_local
    sub_cells = np.arange(num_cells, dtype=np.int32)
    return sub_cells, cell_map.sub_topology_to_topology(sub_cells, inverse=False)


cells1_sub, cells1_parent = submesh_to_parent_cells(subdomain1, subdomain1_cell_map)
cells2_sub, cells2_parent = submesh_to_parent_cells(subdomain2, subdomain2_cell_map)

Nincr = 40
loading = np.linspace(0, 0.04, Nincr + 1)

Niter_max = 200
tol = 1e-4
Force = [0.0]
iterations = [0]

out_file = io.VTKFile(MPI.COMM_WORLD, "results/czm.pvd", "w")
for i, t in enumerate(loading[1:]):
    print(f"Load step {i + 1}/{len(loading) - 1}")
    Uimp.value[0] = t
    nRes = 1.0
    j = 0
    while j < Niter_max:
        # displacement problem resolution
        problem.solve()
        # interpolation of damage on facets
        d.x.array[:] = fem.assemble_vector(facet_interp).array
        # normalized residual for convergence check
        nRes = (
            np.sqrt((fem.assemble_scalar(fem.form((d - d_old) ** 2 * dx_int)))) / Gamma
        )
        d_old.x.array[:] = d.x.array[:]
        j += 1
        print(f"    Iteration {j:2d} | Residual: {nRes}")
        if nRes < tol:
            break
    else:
        raise ValueError(
            f"Fixed-point solver did not converge within {Niter_max} iterations"
        )
    iterations.append(j)
    d_prev.x.array[:] = d.x.array[:]

    Force.append(fem.assemble_scalar(virtual_work_form))

    # We interpolate both u1 and u2 into u for easier visualization with ParaView
    u.interpolate(u1, cells0=cells1_sub, cells1=cells1_parent)
    u.interpolate(u2, cells0=cells2_sub, cells1=cells2_parent)
    out_file.write_function(u, i)
    out_file.write_function(d, i)
    clear_output(wait=True)

out_file.close()

# %% [markdown]
# ## Results
#
# We finally plot the resulting load-displacement curve.

# %%
plt.figure()
plt.plot(loading, Force)
plt.xlabel("Imposed displacement")
plt.ylabel("Reaction force")
plt.show()
