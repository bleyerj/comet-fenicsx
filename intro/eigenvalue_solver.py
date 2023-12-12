from typing import List, Tuple

import dolfinx.fem as fem
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from slepc4py import SLEPc
from dolfinx.fem import FunctionSpace


def print0(string: str):
    """Print on rank 0 only"""
    if MPI.COMM_WORLD.rank == 0:
        print(string)


def monitor_EPS_short(
    EPS: SLEPc.EPS, it: int, nconv: int, eig: list, err: list, it_skip: int
):
    """
    Concise monitor for EPS.solve().
    Parameters
    ----------
    eps
        Eigenvalue Problem Solver class.
    it
       Current iteration number.
    nconv
       Number of converged eigenvalue.
    eig
       Eigenvalues
    err
       Computed errors.
    it_skip
        Iteration skip.
    """
    if it == 1:
        print0("******************************")
        print0("***  SLEPc Iterations...   ***")
        print0("******************************")
        print0("Iter. | Conv. | Max. error")
        print0(f"{it:5d} | {nconv:5d} | {max(err):1.1e}")
    elif not it % it_skip:
        print0(f"{it:5d} | {nconv:5d} | {max(err):1.1e}")


def EPS_print_results(EPS: SLEPc.EPS):
    """Print summary of solution results."""
    print0("\n******************************")
    print0("*** SLEPc Solution Results ***")
    print0("******************************")
    its = EPS.getIterationNumber()
    print0(f"Iteration number: {its}")
    nconv = EPS.getConverged()
    print0(f"Converged eigenpairs: {nconv}")

    if nconv > 0:
        # Create the results vectors
        vr, vi = EPS.getOperators()[0].createVecs()
        print0("\nConverged eigval.  Error ")
        print0("----------------- -------")
        for i in range(nconv):
            k = EPS.getEigenpair(i, vr, vi)
            error = EPS.computeError(i)
            if k.imag != 0.0:
                print0(f" {k.real:2.2e} + {k.imag:2.2e}j {error:1.1e}")
            else:
                pad = " " * 11
                print0(f" {k.real:2.2e} {pad} {error:1.1e}")


def EPS_get_spectrum(
    EPS: SLEPc.EPS, V: FunctionSpace
) -> Tuple[List[complex], List[PETSc.Vec], List[PETSc.Vec]]:
    """Retrieve eigenvalues and eigenfunctions from SLEPc EPS object.
    Parameters
    ----------
    EPS
       The SLEPc solver
    V
       The function space
    Returns
    -------
        Tuple consisting of: List of complex converted eigenvalues,
         lists of converted eigenvectors (real part) and (imaginary part)
    """
    # Get results in lists
    eigval = list()
    eigvec_r = list()
    eigvec_i = list()
    for i in range(EPS.getConverged()):
        vr = fem.Function(V)
        vi = fem.Function(V)

        eigval.append(EPS.getEigenpair(i, vr.vector, vi.vector))
        eigvec_r.append(vr)
        eigvec_i.append(vi)  # Sort by increasing magnitude
    idx = np.argsort(np.abs(np.array(eigval)), axis=0)
    eigval = [eigval[i] for i in idx]
    eigvec_r = [eigvec_r[i] for i in idx]
    eigvec_i = [eigvec_i[i] for i in idx]
    return (eigval, eigvec_r, eigvec_i)


def solve_GEP_shiftinvert(
    A: PETSc.Mat,
    B: PETSc.Mat,
    problem_type: SLEPc.EPS.ProblemType = SLEPc.EPS.ProblemType.GNHEP,
    solver: SLEPc.EPS.Type = SLEPc.EPS.Type.KRYLOVSCHUR,
    nev: int = 10,
    tol: float = 1e-7,
    max_it: int = 10,
    target: float = 0.0,
    shift: float = 0.0,
) -> SLEPc.EPS:
    """
     Solve generalized eigenvalue problem A*x=lambda*B*x using shift-and-invert
     as spectral transform method.
     Parameters
     ----------
     A
        The matrix A
     B
        The matrix B
     problem_type
        The problem type, for options see: https://bit.ly/3gM5pth
    solver:
        Solver type, for options see: https://bit.ly/35LDcMG
     nev
         Number of requested eigenvalues.
     tol
        Tolerance for slepc solver
     max_it
        Maximum number of iterations.
     target
        Target eigenvalue. Also used for sorting.
     shift
        Shift 'sigma' used in shift-and-invert.
     Returns
     -------
     EPS
        The SLEPc solver
    """

    # Build an Eigenvalue Problem Solver object
    EPS = SLEPc.EPS()
    EPS.create(comm=MPI.COMM_WORLD)
    EPS.setOperators(A, B)
    EPS.setProblemType(problem_type)
    # set the number of eigenvalues requested
    EPS.setDimensions(nev=nev)
    # Set solver
    EPS.setType(solver)
    # set eigenvalues of interest
    EPS.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
    EPS.setTarget(target)  # sorting
    # set tolerance and max iterations
    EPS.setTolerances(tol=tol, max_it=max_it)
    # Set up shift-and-invert
    # Only work if 'whichEigenpairs' is 'TARGET_XX'
    ST = EPS.getST()
    ST.setType(SLEPc.ST.Type.SINVERT)
    ST.setShift(shift)
    EPS.setST(ST)
    # set monitor
    it_skip = 1
    EPS.setMonitor(
        lambda eps, it, nconv, eig, err: monitor_EPS_short(
            eps, it, nconv, eig, err, it_skip
        )
    )
    # parse command line options
    EPS.setFromOptions()
    # Display all options (including those of ST object)
    # EPS.view()
    EPS.solve()
    EPS_print_results(EPS)
    return EPS
