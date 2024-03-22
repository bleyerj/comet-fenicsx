---
jupytext:
  formats: md:myst,ipynb
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.0
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Transient elastodynamics with Newmark time-integration {far}`star`{far}`star`

```{admonition} Objectives
:class: objectives

This demo shows how to perform time integration of transient elastodynamics using the Newmark scheme.$\newcommand{\bsig}{\boldsymbol{\sigma}}
\newcommand{\beps}{\boldsymbol{\varepsilon}}
\newcommand{\bu}{\boldsymbol{u}}
\newcommand{\bv}{\boldsymbol{v}}
\newcommand{\ba}{\boldsymbol{a}}
\newcommand{\bn}{\boldsymbol{n}}
\newcommand{\bT}{\boldsymbol{T}}
\newcommand{\bf}{\boldsymbol{f}}
\newcommand{\dOm}{\,\text{d}\Omega}
\newcommand{\dS}{\,\text{d}S}
\newcommand{\Neumann}{{\partial \Omega_\text{N}}}
\newcommand{\Dirichlet}{{\partial \Omega_\text{D}}}$
```

```{image} beam.gif
:width: 600px
:align: center
```

```{admonition} Download sources
:class: download

* {Download}`Python script<./elastodynamics_newmark.py>`
* {Download}`Jupyter notebook<./elastodynamics_newmark.ipynb>`
```


## Introduction and elastodynamics equation

The elastodynamics equation combine the balance of linear momentum:

$$
\nabla \cdot \bsig + \rho \bf = \rho \ddot{\bu}
$$

where $\bu$ is the displacement vector field, $\ba=\ddot{\bu}=\dfrac{\partial^2 \bu}{\partial t^2}$ is the acceleration,
$\rho$ the material density, $\bf$ a given body force and $\bsig$ the stress tensor which is related
to the displacement through a constitutive equation. We consider in the following the case of isotropic linear elasticity.

The weak form is readily obtained by integrating by part the balance equation using a test function $\bv\in V$
with $V$ being a suitable function space that satisfies the displacement boundary conditions:

$$
\int_{\Omega} \rho \ddot{\bu}\cdot \bv \dOm + \int_{\Omega} \bsig(\bu):\beps(v) \dOm = \int_{\Omega} \rho \bf \cdot \bv \dOm + \int_{\partial\Omega} (\bsig\bn) \cdot \bv \dS \quad \text{for all } \bv\in V
$$

The previous equation can be written as follows:

$$
\text{Find }\bu\in V\text{ such that } m(\ddot{\bu},\bv) + k(\bu,\bv) = L(\bv) \quad \text{for all } \bv\in V
$$

where $m$ is the symmetric bilinear form associated with the mass matrix and $k$ the one associated with the stiffness matrix.

After introducing the finite element space interpolation, one obtains the corresponding discretized evolution equation:

$$
\text{Find }\{u\}\in\mathbb{R}^n\text{ such that } \{v\}^T[M]\{\ddot{u}\} + \{v\}^T[K]\{u\} = \{v\}^T\{F\} \quad \text{for all } \{v\}\in\mathbb{R}^n
$$

which is a generalized $n$-dof harmonic oscillator equation.

Quite often in structural dynamics, structures do not oscillate perfectly but lose energy through various dissipative mechanisms (friction with air or supports, internal dissipation through plasticity, damage, etc.). Dissipative terms can be introduced at the level of the constitutive equation if these mechanisms are well
known but quite often it is not the case. Dissipation can then be modeled by adding an *ad hoc* damping term depending on the structure velocity $\dot{u}$ to the previous evolution equation:

$$
\text{Find }\bu\in V\text{ such that } m(\ddot{\bu},\bv) + c(\dot{\bu},\bv) + k(\bu,\bv) = L(\bv) \quad \text{for all } \bv\in V
$$

The damping form will be considered here as bilinear and symmetric, being therefore associated with a damping matrix $[C]$.

### Rayleigh damping

When little is known about the origin of damping in the structure, a popular choice for the damping matrix, known as *Rayleigh damping*, consists in using
a linear combination of the mass and stiffness matrix $[C] = \eta_M[M]+\eta_K[K]$ with two positive parameters $\eta_M,\eta_K$ which
can be fitted against experimental measures for instance (usually by measuring the damping ratio of two natural modes of vibration).

## Time discretization using the Newmark scheme

We now introduce a time discretization of the interval study $[0;T]$ in $N+1$ time increments $t_0=0,t_1,\ldots,t_N,t_{N+1}=T$
with $\Delta t=T/N$ denoting the time step (supposed constant). The resolution will make use of the Newmark-$\beta$ method in structural dynamics. As an implicit method, it is unconditionally
stable for a proper choice of coefficients so that quite large time steps can be used. 

The method consists in solving the dynamic evolution equation at intermediate time between $t_n$ and $t_{n+1}$ as follows:

$$
[M]\{\ddot{u}_{n+1}\} + [C]\{\dot{u}_{n+1}\}+[K]\{u_{n+1}\} = \{F(t_{n+1})\}
$$

In addition, the following approximation for the displacement and velocity
at $t_{n+1}$ are used:

$$
\begin{align*} \{u_{n+1}\} &= \{u_{n}\}+\Delta t \{\dot{u}_{n}\} + \dfrac{\Delta t^2}{2}\left((1-2\beta)\{\ddot{u}_{n}\}+2\beta\{\ddot{u}_{n+1}\}\right) \\ \{\dot{u}_{n+1}\} &= \{\dot{u}_{n}\} + \Delta t\left((1-\gamma)\{\ddot{u}_{n}\}+\gamma\{\ddot{u}_{n+1}\}\right) \end{align*}
$$

where thee parameters $\gamma\in[0;1]$ and $\beta\in[0;1/2]$ determine the stability and accuracy of the approximation. 

We can use the previous expressions to express the acceleration $\ddot{u}_{n+1}$ in terms of unknown displacement at $t_{n+1}$ with:

$$
\{\ddot{u}_{n+1}\} = \dfrac{1}{\beta\Delta t^2}\left(\{u_{n+1}\} - \{u_{n}\}-\Delta t \{\dot{u}_{n}\} \right) - \dfrac{1-2\beta}{2\beta}\{\ddot{u}_{n}\}
$$



### Popular choice of parameters

The most popular choice for the parameters is: $\gamma=\dfrac{1}{2}$,
$\beta=\dfrac{1}{4}$ which ensures unconditional stability, energy conservation and second-order accuracy.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML, clear_output

from mpi4py import MPI
import ufl
from dolfinx import fem, io
import dolfinx.fem.petsc
from dolfinx.mesh import create_box, CellType
```

```{code-cell} ipython3
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
```

```{code-cell} ipython3
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
```

```{code-cell}
def left(x):
    return np.isclose(x[0], 0.0)


def point(x):
    return np.isclose(x[0], L) & np.isclose(x[1], 0) & np.isclose(x[2], 0)


clamped_dofs = fem.locate_dofs_geometrical(V, left)
point_dof = fem.locate_dofs_geometrical(V, point)[0]
point_dofs = np.arange(point_dof * dim, (point_dof + 1) * dim)


bcs = [fem.dirichletbc(np.zeros((dim,)), clamped_dofs, V)]
```

```{code-cell} ipython3
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
E_kin = fem.form(0.5 * ufl.dot(rho * v_old, v_old) * ufl.dx)
E_el = fem.form(
    0.5 * ufl.inner(sigma(u_old), epsilon(u_old)) * ufl.dx
)
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
```

```{code-cell} ipython3
vtk = io.VTKFile(domain.comm, "results/elastodynamics.pvd", "w")

t = 0.0

Nsteps = 400
times = np.linspace(0, 2, Nsteps + 1)
save_freq = Nsteps // 100
energies = np.zeros((Nsteps+1, 2))
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

    energies[i+1, :] = (fem.assemble_scalar(E_el), fem.assemble_scalar(E_kin))

    tip_displacement[i+1, :] = u.x.array[point_dofs][1:]

    clear_output(wait=True)
    print(f"Time increment {i+1}/{Nsteps}")

vtk.close()
```

```{code-cell} ipython3
cmap = plt.get_cmap("plasma")
colors = cmap(times / max(times))

I_y = B * H**3 / 12
omega_y = 1.875**2 * np.sqrt(float(E) * I_y / (float(rho) * B * H * L**4))
omega_x = omega_y * B / H
fig = plt.figure()
ax = fig.gca()
ax.set_aspect("equal")
lines = ax.plot(
    max(tip_displacement[:, 0]) * np.sin(omega_x * times),
    max(tip_displacement[:, 1]) * np.sin(omega_y * times),
    "--k",
    alpha=0.7,
)
ax.set_ylim(-2, 2)
ax.set_xlim(-3, 3)
markers = []
def draw_frame(n):
    markers.append(ax.plot(tip_displacement[n, 0], tip_displacement[n, 1], "o", color=colors[n])[0])
    return markers
anim = animation.FuncAnimation(fig, draw_frame, frames=Nsteps, interval=20, blit=True)
plt.close()
HTML(anim.to_html5_video())
```

```{code-cell} ipython3
plt.plot(times, energies[:, 0], '-b', label="Elastic")
plt.plot(times, energies[:, 1], "-g", label="Kinetic")
plt.plot(times, np.sum(energies, axis=1), "-r", label="Total")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Energies")
```