from typing import List

from petsc4py import PETSc

import dolfinx
import ufl


class TAOProblem:
    """Nonlinear problem class compatible with PETSC.TAO solver."""

    def __init__(
        self,
        f: ufl.form.Form,
        F: ufl.form.Form,
        J: ufl.form.Form,
        u: dolfinx.fem.Function,
        bcs: List[dolfinx.fem.dirichletbc],
    ):
        """This class set up structures for solving a non-linear problem using Newton's method.

        Parameters
        ==========
        f: Objective.
        F: Residual.
        J: Jacobian.
        u: Solution.
        bcs: Dirichlet boundary conditions.
        """
        self.obj = dolfinx.fem.form(f)
        self.L = dolfinx.fem.form(F)
        self.a = dolfinx.fem.form(J)
        self.bcs = bcs
        self.u = u

        # Create matrix and vector to be used for assembly
        # of the non-linear problem
        self.A = dolfinx.fem.petsc.create_matrix(self.a)
        self.b = dolfinx.fem.petsc.create_vector(self.L)

    def f(self, tao, x: PETSc.Vec):
        """Assemble the objective f.

        Parameters
        ==========
        tao: the tao object
        x: Vector containing the latest solution.
        """

        """Assemble residual vector."""
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        x.copy(self.u.vector)
        self.u.vector.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )
        return dolfinx.fem.assemble_scalar(self.obj)

    def F(self, tao: PETSc.TAO, x: PETSc.Vec, F):
        """Assemble the residual F into the vector b.

        Parameters
        ==========
        tao: the tao object
        x: Vector containing the latest solution.
        b: Vector to assemble the residual into.
        """
        # We need to assign the vector to the function
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        x.copy(self.u.vector)
        self.u.vector.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )

        with F.localForm() as f_local:
            f_local.set(0.0)
        dolfinx.fem.petsc.assemble_vector(F, self.L)
        dolfinx.fem.petsc.apply_lifting(F, [self.a], bcs=[self.bcs], x0=[x], scale=-1.0)
        F.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        dolfinx.fem.petsc.set_bc(F, self.bcs, x, -1.0)

    def J(self, tao: PETSc.TAO, x: PETSc.Vec, A: PETSc.Mat, P: PETSc.Mat):
        """Assemble the Jacobian matrix.

        Parameters
        ==========
        x: Vector containing the latest solution.
        A: Matrix to assemble the Jacobian into.
        """
        A.zeroEntries()
        dolfinx.fem.petsc.assemble_matrix(A, self.a, self.bcs)
        A.assemble()
