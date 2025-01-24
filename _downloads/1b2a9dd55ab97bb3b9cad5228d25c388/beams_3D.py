# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Elastic 3D beam structures {far}`star`
#
# ```{admonition} Objectives
# :class: objectives
#
# This tour explores the formulation of 3D beam-like elastic structures. One particularity of this tour is that the mesh topology is 1D (beams) but is embedded in a 3D ambient space. We will also show how to define a local frame containing the beam axis direction and two perpendicular directions for the definition of the cross-section geometrical properties.
# ```
#
# ```{seealso}
# This tour bears similarities with the [](/tours/beams/linear_truss/linear_truss.md) and [](/tours/shells/linear_shell/linear_shell.md) tours.
# ```
#
# ```{admonition} Download sources
# :class: download
#
# * {Download}`Python script<./beams_3D.py>`
# * {Download}`Jupyter notebook<./beams_3D.ipynb>`
# ```
#
# ## Beam kinematics
#
# The variational formulation for 3D beams requires to distinguish between motion along the beam direction and along both perpendicular directions. These two directions often correspond to principal directions of inertia and distinguish between strong and weak bending directions. The user must therefore specify this local orientation frame throughout the whole structure. In this example, we will first compute the vector $\boldsymbol{t}$ tangent to the beam direction, and two perpendicular directions $\boldsymbol{a}_1$ and $\boldsymbol{a}_2$. $\boldsymbol{a}_1$ will always be perpendicular to $t$ and the vertical direction, while $\boldsymbol{a}_2$ will be such that $(\boldsymbol{t}, \boldsymbol{a}_1, \boldsymbol{a}_2)$ is a direct orthonormal frame.
#
# Classical Euler-Bernoulli beam theory would typically require $C^1$-continuous interpolation of the displacement field, using Hermite polynomials for instance. Such discretization scheme is unfortunately not available natively in `dolfinx`. Instead, we will consider a Timoshenko beam model with St-Venant uniform torsion theory for the torsional part. The beam kinematics will then be described by a 3D displacement field $\boldsymbol{u}$ and an independent 3D rotation field $\boldsymbol{\theta}$, that is 6 degrees of freedom at each point. $(u_t,u_1,u_2)$ will respectively denote the displacement components in the section local frame $(\boldsymbol{t}, \boldsymbol{a}_1, \boldsymbol{a}_2)$. Similarly, $\theta_1$ and $\theta_2$ will denote the section rotations about the two section axes whereas $\theta_t$ is the twist angle.
#
# The beam strains are given by (straight beam with constant local frame):
#
# * the **normal strain**: $\delta = \dfrac{d u_t}{ds}$
# * the **bending curvatures**: $\chi_1 = \dfrac{d\theta_1}{ds}$ and $\chi_2 = \dfrac{d\theta_2}{ds}$
# * the **shear strains**: $\gamma_1 = \dfrac{du_1}{ds}-\theta_2$ and
#     $\gamma_2 = \dfrac{du_2}{ds}+\theta_1$. 
# * the **torsional strain**: $\omega = \dfrac{d\theta_t}{ds}$. 
#
# where $\dfrac{dv}{ds} = \nabla v \cdot t$ is the tangential gradient. Associated with these generalized strains are the corresponding generalized stresses:
#
# * the **normal force** $N$
# * the **bending moments** $M_1$ and $M_2$
# * the **shear forces** $Q_1$ and $Q_2$
# * the **torsional moment** $M_T$
#
# ## Constitutive equations
#
# The beam constitutive equations (assuming no normal force/bending moment coupling and that $a_1$ and $a_2$ are the principal axes of inertia) read as:
#
# $$\begin{Bmatrix}
# N \\ Q_1 \\ Q_2 \\ M_T \\ M_1 \\ M_2 
# \end{Bmatrix} = \begin{bmatrix}
# ES & 0 & 0 & 0 & 0 & 0\\
# 0 & GS_1 & 0 & 0 & 0 & 0\\
# 0 & 0 & GS_2 & 0 & 0 & 0\\
# 0 & 0 & 0 & GJ & 0 & 0\\
# 0 & 0 & 0 & 0 & EI_1 & 0\\
# 0 & 0 & 0 & 0 & 0 & EI_2
# \end{bmatrix}\begin{Bmatrix}
# \delta \\ \gamma_1 \\ \gamma_2 \\ \omega \\ \chi_1 \\ \chi_2 
# \end{Bmatrix}$$
#
# where $S$ is the cross-section area, $E$ the material Young modulus and $G$ the shear modulus, $I_1=\int_S x_2^2 dS$ (resp. $I_2=\int_S x_1^2 dS$) are the bending second moment of inertia about axis $a_1$ (resp. $a_2$), $J$ is the torsion inertia, $S_1$ (resp. $S_2$) is the shear area in direction $a_1$ (resp. $a_2$).
#
# ## Variational formulation
#
# The 3D beam variational formulation finally reads as: Find $(\boldsymbol{u},\boldsymbol{\theta})\in V$ such that:
#
# $$\int_S (N\widehat{\delta}+Q_1\widehat{\gamma}_1+Q_2\widehat{\gamma}_2+M_T\widehat{\omega}+M_1\widehat{\chi}_1+M_2\widehat{\chi}_2)dS = \int \boldsymbol{f}\cdot\widehat{\boldsymbol{u}}dS \quad \forall (\widehat{\boldsymbol{u}},\widehat{\boldsymbol{\theta}})\in V$$
#
# where we considered only a distributed loading $\boldsymbol{f}$ and where $\widehat{\delta},\ldots,\widehat{\chi}_2$ are the generalized strains associated with test functions $(\widehat{\boldsymbol{u}},\widehat{\boldsymbol{\theta}})$.
#
# ## Implementation
#
# We start by loading the relevant modules.

import numpy as np
import gmsh
import ufl
import basix
from mpi4py import MPI
from dolfinx import mesh, fem, io
import dolfinx.fem.petsc

# We generate the mesh (`domain`) using the `Gmsh` Python API.

# + tags=["hide-input", "hide-output"]
# Initialize gmsh
gmsh.initialize()

# Parameters
N = 20
h = 10
a = 20
M = 20
d = 1

# Creating points and lines
p0 = gmsh.model.occ.addPoint(a, 0, 0, d)
lines = []

for i in range(1, N + 1):
    p1 = gmsh.model.occ.addPoint(
        a * np.cos(0.99*i / N * np.pi / 2.0),
        0,
        h * np.sin(i / N * np.pi / 2.0),
        d,
    )
    ll = gmsh.model.occ.addLine(p0, p1)
    lines.append(ll)
    p0 = p1
in_lines = [(1, line) for line in lines]
# Extruding lines
for j in range(1, M + 1):
    out = gmsh.model.occ.revolve(
        in_lines,
        0,
        0,
        0,
        0,
        0,
        1,
        angle=2 * np.pi / M,
    )
    in_lines = out[::4]

# Coherence (remove duplicate entities and ensure topological consistency)
# gmsh.model.occ.dilate(entities, 0.0, 0.0, 0.0, 1, 0.1, 1)
# gmsh.model.occ.synchronize()
# Save the mesh
gmsh.model.occ.remove_all_duplicates()
gmsh.model.occ.synchronize()
lines = gmsh.model.get_entities(1)

gmsh.model.add_physical_group(1, [l[1] for l in lines], 1)
gmsh.model.mesh.generate(dim=1)

domain, _, _ = io.gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0)
domain.geometry.x[:, 1] *= 2/3.

# Finalize gmsh
gmsh.finalize()
# -

# We can check that the mesh is embedded in a 3D geometrical space while being of 1D topology.

# +
gdim = domain.geometry.dim
tdim = domain.topology.dim

print(f"Geometrical dimension = {gdim}")
print(f"Topological dimension = {tdim}")
# -

# We use the `ufl.Jacobian` function to compute the transformation Jacobian between a reference element (interval here) and the current element. In our case, the Jacobian is of shape (3,1). Transforming it into a vector of unit length will give us the local tangent vector $\boldsymbol{t}$.

dx_dX = ufl.Jacobian(domain)[:, 0]
t = dx_dX / ufl.sqrt(ufl.inner(dx_dX, dx_dX))

# We now compute the section local axis. As mentioned earlier, $\boldsymbol{a}_1$ will be perpendicular to $\boldsymbol{t}$ and the vertical direction $\boldsymbol{e}_z=(0,0,1)$. After normalization, $\boldsymbol{a}_2$ is built by taking the cross product between $\boldsymbol{t}$ and $\boldsymbol{a}_1$, $\boldsymbol{a}_2$ will therefore belong to the plane made by $\boldsymbol{t}$ and the vertical direction.

ez = ufl.as_vector([0, 0, 1])
a1 = ufl.cross(t, ez)
a1 /= ufl.sqrt(ufl.dot(a1, a1))
a2 = ufl.cross(t, a1)
a2 /= ufl.sqrt(ufl.dot(a2, a2))

# We now define the material and geometrical constants which will be used in the constitutive relation. We consider the case of a rectangular cross-section of width $b$ and height $h$ in directions $\boldsymbol{a}_1$ and $\boldsymbol{a}_2$. The bending inertia will therefore be $I_1 = bh^3/12$ and $I_2=hb^3/12$. The torsional inertia is $J=\beta hb^3$ with $\beta\approx 0.26$ for $h=3b$. Finally, the shear areas are approximated by $S_1=S_2=\kappa S$ with $\kappa=5/6$.

# +
thick = fem.Constant(domain, 0.3)
width = thick/3
E = fem.Constant(domain, 70e3)
nu = fem.Constant(domain, 0.3)
G = E/2/(1+nu)
rho = fem.Constant(domain, 2.7e-3)
g = fem.Constant(domain, 9.81)

S = thick*width
ES = E*S
EI1 = E*width*thick**3/12
EI2 = E*width**3*thick/12
GJ = G*0.26*thick*width**3
kappa = fem.Constant(domain, 5./6.)
GS1 = kappa*G*S
GS2 = kappa*G*S
# -

# We now consider a mixed $\mathbb{P}_1/\mathbb{P}_1$-Lagrange interpolation for the displacement and rotation fields. The variational form is built using a function `generalized_strains` giving the vector of six generalized strains as well as a function `generalized_stresses` which computes the dot product of the strains with the above-mentioned constitutive matrix (diagonal here). Note that since the 1D beams are embedded in an ambient 3D space, the gradient operator has shape (3,), we therefore define a tangential gradient operator `tgrad` by taking the dot product with the local tangent vector $t$.
#
# Finally, similarly to Reissner-Mindlin plates, shear-locking issues might arise in the thin beam limit. To avoid this, reduced integration is performed on the shear part $Q_1\widehat{\gamma}_1+Q_2\widehat{\gamma}_2$ of the variational form using a one-point rule.

# +
Ue = basix.ufl.element("P", domain.basix_cell(), 1, shape=(gdim,))
W = fem.functionspace(domain, basix.ufl.mixed_element([Ue, Ue]))


u_, theta_ = ufl.TestFunctions(W)
du, dtheta = ufl.TrialFunctions(W)

def tgrad(u):
    return ufl.dot(ufl.grad(u), t)
def generalized_strains(u, theta):
    return ufl.as_vector([ufl.dot(tgrad(u), t),
                      ufl.dot(tgrad(u), a1)-ufl.dot(theta, a2),
                      ufl.dot(tgrad(u), a2)+ufl.dot(theta, a1),
                      ufl.dot(tgrad(theta), t),
                      ufl.dot(tgrad(theta), a1),
                      ufl.dot(tgrad(theta), a2)])
def generalized_stresses(u, theta):
    return ufl.dot(ufl.diag(ufl.as_vector([ES, GS1, GS2, GJ, EI1, EI2])), generalized_strains(u, theta))

Sig = generalized_stresses(du, dtheta)
Eps_ =  generalized_strains(u_, theta_)

dx_shear = ufl.dx(scheme="default",metadata={"quadrature_scheme":"default", "quadrature_degree": 0})
k_form = sum([Sig[i]*Eps_[i]*ufl.dx for i in [0, 3, 4, 5]]) + (Sig[1]*Eps_[1]+Sig[2]*Eps_[2])*dx_shear
l_form = -rho*S*g*u_[2]*ufl.dx


# -

# Clamped boundary conditions are considered at the bottom $z=0$ level and the linear problem is finally solved.

# +
def bottom(x):
    return np.isclose(x[2], 0.)

Vu, _ = W.sub(0).collapse()
Vt, _ = W.sub(1).collapse()
u_dofs = fem.locate_dofs_geometrical((W.sub(0), Vu), bottom)
theta_dofs = fem.locate_dofs_geometrical((W.sub(1), Vt), bottom)
u0 = fem.Function(Vu)
theta0 = fem.Function(Vt)
bcs = [fem.dirichletbc(u0, u_dofs, W.sub(0)), fem.dirichletbc(theta0, theta_dofs, W.sub(1))]

w = fem.Function(W, name="Generalized_displacement")

problem = fem.petsc.LinearProblem(
    k_form, l_form, u=w, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
)
problem.solve();
# -

# We finally plot the deflected shape and the bending moment distribution using `pyvista`.

# + tags=["hide-input"]
import pyvista
from dolfinx import plot

u = w.sub(0).collapse()

plotter = pyvista.Plotter()
topology, cell_types, geometry = plot.vtk_mesh(Vu)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
grid.point_data["Deflection"] = u.x.array.reshape(-1, 3)
warped = grid.warp_by_vector("Deflection", factor=500.0)
plotter.add_mesh(grid, show_edges=True, color="k", line_width=1, opacity=0.5)
plotter.add_mesh(warped, show_edges=True, line_width=5)
plotter.show()


M = fem.Function(Vu, name="Bending moments (M1,M2)")
Sig = generalized_stresses(w.sub(0), w.sub(1))
M_exp = fem.Expression(ufl.as_vector([Sig[4], Sig[5], 0]), Vu.element.interpolation_points())
M.interpolate(M_exp)

grid.point_data["Bending_moments"] = M.x.array.reshape(-1, 3)
plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=True, line_width=5, scalars="Bending_moments")
plotter.show()
# -

# ## Attention
#
# ```{attention}
# - The $\mathbb{P}_1-\mathbb{P}_1$ discretization choice made here might be too poor for achieving a decent error with a moderate mesh size. Increasing the order of interpolation, e.g. for the displacement $\boldsymbol{u}$ is obviously possible but the integration degree for the reduced integration measure `dx_shear` must be adapted to this case.
#
# - To add concentrated forces, one can use the `dS` measure with appropriate tags or follow the strategy hinted in [](/tours/beams/linear_truss/linear_truss.md).
# ```
