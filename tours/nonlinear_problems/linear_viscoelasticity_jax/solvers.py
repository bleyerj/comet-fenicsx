from petsc4py import PETSc
from dolfinx import fem
import dolfinx.fem.petsc


class CustomLinearProblem(fem.petsc.LinearProblem):
    """A `LinearProblem` with separate assembly of the rhs, of the lhs and solve steps."""

    def assemble_rhs(self, u=None):
        """Assemble right-hand side and lift Dirichlet bcs.

        Parameters
        ----------
        u : dolfinx.fem.Function, optional
            For non-zero Dirichlet bcs u_D, use this function to assemble rhs with the value u_D - u_{bc}
            where u_{bc} is the value of the given u at the corresponding dofs. Typically used for custom
            Newton methods with non-zero Dirichlet bcs.
        """

        # Assemble rhs
        with self.b.localForm() as b_loc:
            b_loc.set(0)
        fem.petsc.assemble_vector(self.b, self.L)

        # Apply boundary conditions to the rhs
        x0 = None if u is None else [u.x.petsc_vec]
        fem.petsc.apply_lifting(self.b, [self.a], bcs=[self.bcs], x0=x0, alpha=1.0)
        self.b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        x0 = None if u is None else u.x.petsc_vec
        fem.petsc.set_bc(self.b, self.bcs, x0, alpha=1.0)

    def assemble_lhs(self):
        """Assemble left-hand side matrix."""
        self.A.zeroEntries()
        fem.petsc.assemble_matrix(self.A, self.a, bcs=self.bcs)
        self.A.assemble()

    def solve_system(self):
        """Solve linear system directly into the solution vector and update ghost values."""
        self.solver.solve(self.b, self.u.x.petsc_vec)
        self.u.x.scatter_forward()


class CustomNewtonSolver:
    """A Newton solver calling a user-defined callback before each residual evaluation."""

    def __init__(self, tangent_problem):
        self.tangent_problem = tangent_problem
        self.du = self.tangent_problem.u

    def callback(self, u, *args):
        pass

    def solve(self, u, *args, tol=1e-8, atol=1e-10, Nitermax=200):
        """Solve the nonlinear problem starting from the current value of u.

        Convergence is reached when the residual norm is smaller than
        `tol` times the initial residual norm or smaller than `atol`.
        """
        self.callback(u, *args)

        # compute the initial residual (pass u to handle non homogeneous bcs)
        self.tangent_problem.assemble_rhs(u=u)
        nRes0 = self.tangent_problem.b.norm()
        nRes = nRes0
        threshold = max(tol * nRes0, atol)

        niter = 0
        while nRes > threshold and niter < Nitermax:
            # solve for the displacement correction
            self.tangent_problem.assemble_lhs()
            self.tangent_problem.solve_system()

            # update the displacement with the current correction
            u.x.petsc_vec.axpy(1, self.du.x.petsc_vec)  # u = u + 1*du
            u.x.scatter_forward()

            self.callback(u, *args)
            # compute the new residual (pass u to handle non homogeneous bcs)
            self.tangent_problem.assemble_rhs(u=u)
            nRes = self.tangent_problem.b.norm()

            niter += 1

        converged = nRes <= threshold
        return niter, converged
