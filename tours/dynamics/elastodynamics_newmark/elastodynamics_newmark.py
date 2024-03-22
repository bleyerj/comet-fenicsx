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

# # Transient elastodynamics with Newmark time-integration {far}`star`{far}`star`
#
# ```{admonition} Objectives
# :class: objectives
#
# This demo shows how to perform time integration of transient elastodynamics using the Newmark scheme.$\newcommand{\bsig}{\boldsymbol{\sigma}}
# \newcommand{\beps}{\boldsymbol{\varepsilon}}
# \newcommand{\bu}{\boldsymbol{u}}
# \newcommand{\bv}{\boldsymbol{v}}
# \newcommand{\bT}{\boldsymbol{T}}
# \newcommand{\dOm}{\,\text{d}\Omega}
# \newcommand{\dS}{\,\text{d}S}
# \newcommand{\Neumann}{{\partial \Omega_\text{N}}}
# \newcommand{\Dirichlet}{{\partial \Omega_\text{D}}}$
# ```
#
# ```{image} beam.gif
# :width: 600px
# :align: center
# ```
#
# ```{admonition} Download sources
# :class: download
#
# * {Download}`Python script<./elastodynamics_newmark.py>`
# * {Download}`Jupyter notebook<./elastodynamics_newmark.ipynb>`
# ```

# +
import numpy as np
import matplotlib.pyplot as plt

from mpi4py import MPI
import ufl
from dolfinx import fem, io
import dolfinx.fem.petsc
from dolfinx.mesh import create_box, CellType

# +
L = 8.0
H = 0.2
B = 0.1
domain = create_box(
    MPI.COMM_WORLD,
    [[0.0, -B / 2, -H / 2], [L, B / 2, H / 2]],
    [8, 2, 2],
    CellType.hexahedron,
)

dim = domain.topology.dim
dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 4})

degree = 2
shape = (dim,)
V = fem.functionspace(domain, ("Q", degree, shape))

u = fem.Function(V, name="Displacement")

# +
E = fem.Constant(domain, 210e3)
nu = fem.Constant(domain, 0.3)
rho = fem.Constant(domain, 7.8e-3)
f = fem.Constant(domain, (0.0,) * dim)

lmbda = E * nu / (1 + nu) / (1 - 2 * nu)
mu = E / 2 / (1 + nu)


def epsilon(v):
    return ufl.sym(ufl.grad(v))


def sigma(v):
    return lmbda * ufl.tr(epsilon(v)) * ufl.Identity(dim) + 2 * mu * epsilon(v)


# +
def left(x):
    return np.isclose(x[0], 0.0)


def point(x):
    return np.isclose(x[0], L) & np.isclose(x[1], 0) & np.isclose(x[2], 0)


clamped_dofs = fem.locate_dofs_geometrical(V, left)
point_dof = fem.locate_dofs_geometrical(V, point)[0]
point_dofs = np.arange(point_dof * dim, (point_dof + 1) * dim)


bcs = [fem.dirichletbc(np.zeros((dim,)), clamped_dofs, V)]

# +
u_old = fem.Function(V)
v_old = fem.Function(V)
a_old = fem.Function(V)
a_new = fem.Function(V)

gamma_ = 0.5
beta_ = 0.25
beta = fem.Constant(domain, beta_)
dt = fem.Constant(domain, 0.0)

a = 1 / beta / dt**2 * (u - u_old - dt * v_old) + a_old * (1 - 1 / 2 / beta)
a_expr = fem.Expression(a, V.element.interpolation_points())


u_ = ufl.TestFunction(V)
du = ufl.TrialFunction(V)
E_kin = 0.5 * ufl.dot(rho * v_old, v_old) * ufl.dx
E_pot = 0.5 * ufl.inner(sigma(u_old), epsilon(u_old)) * ufl.dx
E_tot = fem.form(E_pot + E_kin)
Residual = (
    ufl.inner(sigma(u), epsilon(u_)) * ufl.dx
    + ufl.dot(rho * a, u_) * ufl.dx
    - ufl.dot(f, u_) * ufl.dx
)

Residual_du = ufl.replace(Residual, {u: du})
a_form = ufl.lhs(Residual_du)
L_form = ufl.rhs(Residual_du)

problem = fem.petsc.LinearProblem(
    a_form, L_form, u=u, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
)

# +
vtk = io.VTKFile(domain.comm, "results/elastodynamics.pvd", "w")

t = 0.0

Nsteps = 400
times = np.linspace(0, 2, Nsteps + 1)
save_freq = Nsteps // 100
total_energy = np.zeros_like(times)
tip_displacement = np.zeros((Nsteps + 1, 2))
for i, dti in enumerate(np.diff(times)):
    if i % save_freq == 0:
        vtk.write_function(u, t)

    dt.value = dti
    t += dti

    if t <= 0.2:
        f.value = np.array([0.0, 1.0, 1.5]) * t / 0.2
    else:
        f.value *= 0.0

    problem.solve()

    u.x.scatter_forward()  # updates ghost values for parallel computations

    # compute new acceleration a_n+1
    a_new.interpolate(a_expr)

    # update u_n with u_n+1
    u.vector.copy(u_old.vector)

    # update v_n with v_n+1
    v_old.x.array[:] += dti * ((1 - gamma_) * a_old.x.array + gamma_ * a_new.x.array)
    v_old.x.scatter_forward()

    # update a_n with a_n+1
    a_new.vector.copy(a_old.vector)

    total_energy[i] = fem.assemble_scalar(E_tot)

    tip_displacement[i, :] = u.x.array[point_dofs][1:]

vtk.close()

# +
cmap = plt.get_cmap("plasma")
colors = cmap(times / max(times))
plt.scatter(tip_displacement[:, 0], tip_displacement[:, 1], color=colors)

I_y = B * H**3 / 12
omega_y = 1.875**2 * np.sqrt(float(E) * I_y / (float(rho) * B * H * L**4))
omega_x = omega_y * B / H
plt.plot(
    max(tip_displacement[:, 0]) * np.sin(omega_x * times),
    max(tip_displacement[:, 1]) * np.sin(omega_y * times),
    "--k",
    alpha=0.7,
)
plt.gca().set_aspect("equal")
