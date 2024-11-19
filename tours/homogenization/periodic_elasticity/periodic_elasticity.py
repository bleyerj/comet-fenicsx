import numpy as np
from mpi4py import MPI
import gmsh
import ufl
from dolfinx import fem, io
import dolfinx.fem.petsc

from dolfinx.io.gmshio import model_to_mesh

import dolfinx_mpc.utils
from dolfinx_mpc import LinearProblem

L = 1.0
R = 0.2
h = 0.02

centers = np.array(
    [
        [0.0, 0.0, 0.0],
        [L, 0.0, 0.0],
        [0.0, L, 0.0],
        [L, L, 0.0],
        [L / 2, L / 2, 0.0],
    ]
)


gdim = 2  # domain geometry dimension
fdim = 1  # facets dimension

gmsh.initialize()

occ = gmsh.model.occ
mesh_comm = MPI.COMM_WORLD
model_rank = 0
if model_rank == 0:
    rectangle = gmsh.model.occ.addRectangle(0, 0, 0, L, L)
    inclusions = []
    for c in centers:
        inclusion = gmsh.model.occ.addDisk(*c, R, R)
        inclusions.append(inclusion)
    vol_dimTag = (gdim, rectangle)
    out = gmsh.model.occ.intersect(
        [vol_dimTag], [(gdim, incl) for incl in inclusions], removeObject=False
    )
    incl_dimTags = out[0]
    gmsh.model.occ.synchronize()
    gmsh.model.occ.cut([vol_dimTag], incl_dimTags, removeTool=False)
    gmsh.model.occ.synchronize()

    gmsh.model.addPhysicalGroup(gdim, [vol_dimTag[1]], 1, name="Matrix")
    gmsh.model.addPhysicalGroup(
        gdim,
        [tag for dim, tag in incl_dimTags],
        2,
        name=f"Inclusions",
    )
    eps = 1e-3
    left = gmsh.model.getEntitiesInBoundingBox(
        -eps, -eps, -eps, eps, L + eps, eps, dim=fdim
    )
    bottom = gmsh.model.getEntitiesInBoundingBox(
        -eps, -eps, -eps, L + eps, eps, eps, dim=fdim
    )
    right = gmsh.model.getEntitiesInBoundingBox(
        L - eps, -eps, -eps, L + eps, L + eps, eps, dim=fdim
    )
    top = gmsh.model.getEntitiesInBoundingBox(
        -eps, L - eps, -eps, L + eps, L + eps, eps, dim=fdim
    )

    # tag physical domains and facets
    gmsh.model.addPhysicalGroup(fdim, [b[1] for b in bottom], 1)
    gmsh.model.addPhysicalGroup(fdim, [b[1] for b in right], 2)
    gmsh.model.addPhysicalGroup(fdim, [b[1] for b in top], 3)
    gmsh.model.addPhysicalGroup(fdim, [b[1] for b in left], 4)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)

    gmsh.write("unit_cell.geo_unrolled")

    gmsh.model.mesh.generate(gdim)

domain, cells, facets = model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim=gdim)
gmsh.finalize()


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


E = create_piecewise_constant_field(domain, cells, {1: 1.0, 2: 25.0})
nu = create_piecewise_constant_field(domain, cells, {1: 0.3, 2: 0.2})

lmbda = E * nu / (1 + nu) / (1 - 2 * nu)
mu = E / 2 / (1 + nu)

Eps = fem.Constant(domain, np.zeros((2, 2)))
Eps_ = fem.Constant(domain, np.zeros((2, 2)))
x = ufl.SpatialCoordinate(domain)

vol = fem.assemble_scalar(fem.form(1 * ufl.dx(domain=domain)))
print("Volume:", vol)


def epsilon(v):
    return ufl.sym(ufl.grad(v))


def sigma(v):
    eps = Eps + epsilon(v)
    return lmbda * ufl.tr(eps) * ufl.Identity(gdim) + 2 * mu * eps


# Define function space
V = fem.functionspace(domain, ("P", 2, (gdim,)))

# Define variational problem
du = ufl.TrialFunction(V)
u_ = ufl.TestFunction(V)
a_form, L_form = ufl.system(ufl.inner(sigma(du), epsilon(u_)) * ufl.dx)


elementary_load = [
    np.array([[1.0, 0.0], [0.0, 0.0]]),
    np.array([[0.0, 0.0], [0.0, 1.0]]),
    np.array([[0.0, 0.5], [0.5, 0.0]]),
]
dim_load = len(elementary_load)

bcs = []


def periodic_relation_left_right(x):
    out_x = np.zeros(x.shape)
    out_x[0] = L - x[0]
    out_x[1] = x[1]
    out_x[2] = x[2]
    return out_x


def periodic_relation_bottom_top(x):
    out_x = np.zeros(x.shape)
    out_x[0] = x[0]
    out_x[1] = L - x[1]
    out_x[2] = x[2]
    return out_x


mpc = dolfinx_mpc.MultiPointConstraint(V)
mpc.create_periodic_constraint_topological(
    V, facets, 2, periodic_relation_left_right, bcs
)
mpc.create_periodic_constraint_topological(
    V, facets, 3, periodic_relation_bottom_top, bcs
)
mpc.finalize()

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

vtk.close()
