from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
import dolfinx.fem.petsc
import ufl


class TAOProblem:
    """Nonlinear problem class compatible with PETSc TAO solver."""

    def __init__(
        self,
        f: ufl.Form,
        F: ufl.Form,
        J: ufl.Form,
        u: dolfinx.fem.Function,
        bcs: list[dolfinx.fem.DirichletBC],
    ):
        """This class sets up structures for solving a bound-constrained
        optimization problem with PETSc TAO solvers.

        Parameters
        ==========
        f: Objective.
        F: Gradient of the objective (residual).
        J: Hessian of the objective (Jacobian).
        u: Solution.
        bcs: Dirichlet boundary conditions.
        """
        self.obj = dolfinx.fem.form(f)
        self.L = dolfinx.fem.form(F)
        self.a = dolfinx.fem.form(J)
        self.bcs = bcs
        self.u = u
        self.comm = u.function_space.mesh.comm

        # Create matrix and vector to be used for assembly
        # of the non-linear problem
        self.A = dolfinx.fem.petsc.create_matrix(self.a)
        self.b = dolfinx.fem.petsc.create_vector(self.L.function_spaces)

    def update_solution(self, x: PETSc.Vec):
        """Copy the current TAO iterate x into the solution function u.

        TAO locks x for read-only access in the callbacks, and x may be the
        solution vector u.x.petsc_vec itself. We therefore copy the owned
        values with numpy and then update the ghost values.
        """
        size_local = self.u.x.index_map.size_local * self.u.x.block_size
        self.u.x.array[:size_local] = x.array_r
        self.u.x.scatter_forward()

    def f(self, tao: PETSc.TAO, x: PETSc.Vec):
        """Assemble the objective f.

        Parameters
        ==========
        tao: the tao object
        x: Vector containing the latest solution.
        """
        self.update_solution(x)
        return self.comm.allreduce(dolfinx.fem.assemble_scalar(self.obj), op=MPI.SUM)

    def F(self, tao: PETSc.TAO, x: PETSc.Vec, F: PETSc.Vec):
        """Assemble the gradient F of the objective.

        Parameters
        ==========
        tao: the tao object
        x: Vector containing the latest solution.
        F: Vector to assemble the gradient into.
        """
        self.update_solution(x)
        x0 = self.u.x.petsc_vec

        with F.localForm() as f_local:
            f_local.set(0.0)
        dolfinx.fem.petsc.assemble_vector(F, self.L)
        dolfinx.fem.petsc.apply_lifting(
            F, [self.a], bcs=[self.bcs], x0=[x0], alpha=-1.0
        )
        F.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        dolfinx.fem.petsc.set_bc(F, self.bcs, x0, alpha=-1.0)

    def J(self, tao: PETSc.TAO, x: PETSc.Vec, A: PETSc.Mat, P: PETSc.Mat):
        """Assemble the Hessian matrix.

        Parameters
        ==========
        tao: the tao object
        x: Vector containing the latest solution.
        A: Matrix to assemble the Hessian into.
        P: Matrix to assemble the preconditioner into (unused).
        """
        self.update_solution(x)
        A.zeroEntries()
        dolfinx.fem.petsc.assemble_matrix(A, self.a, bcs=self.bcs)
        A.assemble()
