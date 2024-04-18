#
# ..    # gedit: set fileencoding=utf8 :
#
# .. raw:: html
#
#  <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><p align="center"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png"/></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a></p>
#
# .. _ReissnerMindlinDG:
#
# ==============================================================
# Reissner-Mindlin plate with a Discontinuous-Galerkin approach
# ==============================================================
#
# -------------
# Introduction
# -------------
#
# This program solves the Reissner-Mindlin plate equations on the unit
# square with uniform transverse loading and clamped boundary conditions.
# The corresponding file can be obtained from :download:`reissner_mindlin_dg.py`.
#
# It uses a Discontinuous Galerkin interpolation for the rotation field to
# remove shear-locking issues in the thin plate limit. Details of the formulation
# can be found in [HAN2011]_.
#
# The solution for :math:`\theta_x` on the middle line of equation :math:`y=0.5`
# will look as follows for 10 elements and a stabilization parameter :math:`s=1`:
#
# .. image:: dg_rotation_N10_s1.png
#    :scale: 15%
#
#
#
# ---------------
# Implementation
# ---------------
#
#
# Material properties and loading are the same as in :ref:`ReissnerMindlinQuads`::

import numpy as np
from mpi4py import MPI
import ufl
from dolfinx import fem, mesh
import dolfinx.fem.petsc

N = 11
domain = mesh.create_unit_square(MPI.COMM_WORLD, N, N, cell_type=mesh.CellType.triangle)
# -
# Material parameters for isotropic linear elastic behavior are first defined::

E = fem.Constant(domain, 210e3)
nu = fem.Constant(domain, 0.3)

# Plate bending stiffness $\textsf{D}=\dfrac{Eh^3}{12(1-\nu^2)}$ and shear stiffness $\textsf{F} = \kappa G h$ with a shear correction factor $\kappa = 5/6$ for a homogeneous plate of thickness $h$:

thick = fem.Constant(domain, 1e-3)
D = E * thick**3 / (1 - nu**2) / 12.0
F = E / 2 / (1 + nu) * thick * 5.0 / 6.0

# The uniform loading $f$ is scaled by the Love-Kirchhoff solution so that the deflection converges to a
# constant value of 1 in the thin plate. This thin plate limit will be used to check the sensitivity of the element to shear locking. Formulations which exhibit shear locking are expected to provide overly stiff results (low values of the deflection) for a given mesh in the limit of small thickness $h \to 0$.

f = -D / 1.265319087e-3  # with this we have w_Love-Kirchhoff = 1.0

# Continuous interpolation using of degree 2 is chosen for the deflection :math:`w`
# whereas the rotation field :math:`\underline{\theta}` is discretized using discontinuous linear polynomials::

We = ufl.FiniteElement("P", domain.ufl_cell(), 2)
Te = ufl.VectorElement("DG", domain.ufl_cell(), 1)
V = fem.FunctionSpace(domain, ufl.MixedElement([We, Te]))


# Boundary of the plate
def border(x):
    return np.logical_or(
        np.logical_or(np.isclose(x[0], 0), np.isclose(x[0], 1)),
        np.logical_or(np.isclose(x[1], 0), np.isclose(x[1], 1)),
    )


# Clamped boundary conditions on the lateral boundary are defined. Note that BCs on the rotation field will be imposed using the weak form
facet_dim = 1
clamped_facets = mesh.locate_entities_boundary(domain, facet_dim, border)
clamped_dofs = fem.locate_dofs_topological(V, facet_dim, clamped_facets)

u0 = fem.Function(V)
bcs = [fem.dirichletbc(u0, clamped_dofs)]

# Standard part of the variational form is the same (without full integration)::


# +
def strain2voigt(eps):
    return ufl.as_vector([eps[0, 0], eps[1, 1], 2 * eps[0, 1]])


def voigt2stress(S):
    return ufl.as_tensor([[S[0], S[2]], [S[2], S[1]]])


def curv(u):
    (w, theta) = ufl.split(u)
    return ufl.sym(ufl.grad(theta))


def shear_strain(u):
    (w, theta) = ufl.split(u)
    return theta - ufl.grad(w)


def bending_moment(u):
    DD = ufl.as_tensor([[D, nu * D, 0], [nu * D, D, 0], [0, 0, D * (1 - nu) / 2.0]])
    return voigt2stress(ufl.dot(DD, strain2voigt(curv(u))))


def shear_force(u):
    return F * shear_strain(u)


u = fem.Function(V)
u_ = ufl.TestFunction(V)
du = ufl.TrialFunction(V)

dx = ufl.Measure("dx")

L = f * u_[0] * dx
a = (
    ufl.inner(bending_moment(u_), curv(du)) * dx
    + ufl.dot(shear_force(u_), shear_strain(du)) * dx
)


# We then add the contribution of jumps in rotation across all internal facets plus
# a stabilization term involving a user-defined parameter :math:`s`::

n = ufl.FacetNormal(domain)
c_vol = ufl.CellVolume(domain)
f_area = ufl.FacetArea(domain)
h_avg = ufl.avg(c_vol) / f_area
h = c_vol / f_area
stabilization = fem.Constant(domain, 10.0)

(dw, dtheta) = ufl.split(du)
(w_, theta_) = ufl.split(u_)

a -= (
    ufl.dot(ufl.avg(ufl.dot(bending_moment(u_), n)), ufl.jump(dtheta)) * ufl.dS
    + ufl.dot(ufl.avg(ufl.dot(bending_moment(du), n)), ufl.jump(theta_)) * ufl.dS
    - stabilization * D / h_avg * ufl.dot(ufl.jump(theta_), ufl.jump(dtheta)) * ufl.dS
)

# Because of the clamped boundary conditions, we also need to add the corresponding
# contributions of the external facets (the imposed rotation is zero on the boundary
# so that no term arise in the linear functional)::

a -= (
    ufl.dot(ufl.dot(bending_moment(u_), n), dtheta) * ufl.ds
    + ufl.dot(ufl.dot(bending_moment(du), n), theta_) * ufl.ds
    - stabilization * D / h * ufl.dot(theta_, dtheta) * ufl.ds
)


# We then solve for the solution and print the deflection normalized with respect to the Love-Kirchhoff thin plate analytical solution:

# +
problem = fem.petsc.LinearProblem(
    a, L, u=u, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
)
problem.solve()

w = u.sub(0).collapse()
w.name = "Deflection"
theta = u.sub(1).collapse()
theta.name = "Rotation"

print(f"Reissner-Mindlin FE deflection: {max(abs(w.vector.array)):.5f}")

tol = 0.001  # Avoid hitting the outside of the domain
Npoints = 500
x = np.linspace(tol, 1 - tol, Npoints)
points = np.zeros((3, Npoints))
points[0] = x
points[1] = 0.5
u_values = []
t_values = []
from dolfinx import geometry

bb_tree = geometry.bb_tree(domain, domain.topology.dim)
cells = []
points_on_proc = []
# Find cells whose bounding-box collide with the the points
cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
# Choose one of the cells that contains the point
colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
for i, point in enumerate(points.T):
    if len(colliding_cells.links(i)) > 0:
        points_on_proc.append(point)
        cells.append(colliding_cells.links(i)[0])
points_on_proc = np.array(points_on_proc, dtype=np.float64)
t_values = theta.eval(points_on_proc, cells)

import matplotlib.pyplot as plt

plt.figure()
plt.plot(
    points_on_proc[:, 0],
    t_values[:, 0],
    ".k",
    linewidth=2,
)
plt.grid(True)
plt.xlabel("x")
plt.legend()
plt.show()
# For :math:`h=0.001` and 50 elements per side, one finds :math:`w_{FE} = 1.38322\text{e-5}`  against :math:`w_{\text{Kirchhoff}} = 1.38173\text{e-5}` for the thin plate solution.
#
# -----------
# References
# -----------
#
# .. [HAN2011] Peter Hansbo, David Heintz, Mats G. Larson, A finite element method with discontinuous rotations for the Mindlin-Reissner plate model, *Computer Methods in Applied Mechanics and Engineering*, 200, 5-8, 2011, pp. 638-648, https://doi.org/10.1016/j.cma.2010.09.009.
