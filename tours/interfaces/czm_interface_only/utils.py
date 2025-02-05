import numpy as np

import numpy.typing as npt
import dolfinx
from dolfinx import fem
import ufl
import basix
import typing
from petsc4py import PETSc

from dolfinx.fem.forms import form as _create_form
from dolfinx.fem.function import Function as _Function
from dolfinx.fem.bcs import DirichletBC


class BlockedLinearProblem:
    """Class for solving a blocked linear variational problem."""

    def __init__(
        self,
        a_blocked_compiled,
        L_blocked_compiled,
        w: list[dolfinx.fem.Function],
        bcs: list[DirichletBC] = [],
        petsc_options: typing.Optional[dict] = None,
    ):
        """Initialize solver for a blocked linear variational problem.

        Args:
            a: A bilinear UFL form, the left hand side of the
                variational problem.
            L: A linear UFL form, the right hand side of the variational
                problem.
            w: A list of Functions to store the result.
            bcs: A list of Dirichlet boundary conditions.

        """

        self._a = a_blocked_compiled
        self._L = L_blocked_compiled
        self._A = fem.petsc.create_matrix_block(self._a)
        self._b = fem.petsc.create_vector_block(self._L)
        self.w = w

        # Set PETSc options
        opts = PETSc.Options()
        if petsc_options is not None:
            for k, v in petsc_options.items():
                opts[k] = v

        # Define KSP solver
        self._solver = PETSc.KSP().create(self._b.getComm().tompi4py())
        self._solver.setOperators(self._A)
        self._solver.setFromOptions()

        # # Set matrix and vector PETSc options
        self._A.setFromOptions()
        self._b.setFromOptions()

        self._x = self._A.createVecRight()
        self.bcs = bcs

    def __del__(self):
        self._solver.destroy()
        self._A.destroy()
        self._b.destroy()
        self._x.destroy()

    def assemble(self):
        # Assemble lhs
        self._A.zeroEntries()
        fem.petsc.assemble_matrix_block(self._A, self._a, bcs=self.bcs)
        self._A.assemble()

        with self._b.localForm() as b_loc:
            b_loc.set(0)
        fem.petsc.assemble_vector_block(self._b, self._L, self._a, bcs=self.bcs)

    def solve(self) -> _Function:
        """Solve the problem."""
        self.assemble()

        # Solve linear system and update ghost values in the solution
        self._solver.solve(self._b, self._x)
        # self._x.ghostUpdate(
        #     addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        # )
        assert self._solver.getConvergedReason() > 0, "Solve failed"

        blocked_maps = [
            (
                si.function_space.dofmap.index_map,
                si.function_space.dofmap.index_map_bs,
            )
            for si in self.w
        ]
        local_values = dolfinx.cpp.la.petsc.get_local_vectors(self._x, blocked_maps)
        for lx, u in zip(local_values, self.w):
            u.x.array[:] = lx
            # u.x.scatter_forward()

        return self.w

    @property
    def L(self) -> fem.Form:
        """The compiled linear form"""
        return self._L

    @property
    def a(self) -> fem.Form:
        """The compiled bilinear form"""
        return self._a

    @property
    def A(self) -> PETSc.Mat:
        """Matrix operator"""
        return self._A

    @property
    def b(self) -> PETSc.Vec:
        """Right-hand side vector"""
        return self._b

    @property
    def solver(self) -> PETSc.KSP:
        """Linear solver object"""
        return self._solver


def transfer_meshtags_to_submesh(
    mesh: dolfinx.mesh.Mesh,
    entity_tag: dolfinx.mesh.MeshTags,
    submesh: dolfinx.mesh.Mesh,
    sub_vertex_to_parent: npt.NDArray[np.int32],
    sub_cell_to_parent: npt.NDArray[np.int32],
) -> tuple[dolfinx.mesh.MeshTags, npt.NDArray[np.int32]]:
    """
    Transfer a meshtag from a parent mesh to a sub-mesh.

    Args:
        mesh: Mesh containing the meshtags
        entity_tag: The meshtags object to transfer
        submesh: The submesh to transfer the `entity_tag` to
        sub_to_vertex_map: Map from each vertex in `submesh` to the corresponding
            vertex in the `mesh`
        sub_cell_to_parent: Map from each cell in the `submesh` to the corresponding
            entity in the `mesh`
    Returns:
        The entity tag defined on the submesh, and a map from the entities in the
        `submesh` to the entities in the `mesh`.

    """

    tdim = mesh.topology.dim
    cell_imap = mesh.topology.index_map(entity_tag.dim)
    num_cells = cell_imap.size_local + cell_imap.num_ghosts
    mesh_to_submesh = np.full(num_cells, -1)
    mesh_to_submesh[sub_cell_to_parent] = np.arange(
        len(sub_cell_to_parent), dtype=np.int32
    )
    sub_vertex_to_parent = np.asarray(sub_vertex_to_parent)

    submesh.topology.create_connectivity(entity_tag.dim, 0)

    num_child_entities = (
        submesh.topology.index_map(entity_tag.dim).size_local
        + submesh.topology.index_map(entity_tag.dim).num_ghosts
    )
    submesh.topology.create_connectivity(submesh.topology.dim, entity_tag.dim)

    c_c_to_e = submesh.topology.connectivity(submesh.topology.dim, entity_tag.dim)
    c_e_to_v = submesh.topology.connectivity(entity_tag.dim, 0)

    child_markers = np.full(num_child_entities, 0, dtype=np.int32)

    mesh.topology.create_connectivity(entity_tag.dim, 0)
    mesh.topology.create_connectivity(entity_tag.dim, mesh.topology.dim)
    p_f_to_v = mesh.topology.connectivity(entity_tag.dim, 0)
    p_f_to_c = mesh.topology.connectivity(entity_tag.dim, mesh.topology.dim)
    sub_to_parent_entity_map = np.full(num_child_entities, -1, dtype=np.int32)
    for facet, value in zip(entity_tag.indices, entity_tag.values):
        facet_found = False
        for cell in p_f_to_c.links(facet):
            if facet_found:
                break
            if (child_cell := mesh_to_submesh[cell]) != -1:
                for child_facet in c_c_to_e.links(child_cell):
                    child_vertices = c_e_to_v.links(child_facet)
                    child_vertices_as_parent = sub_vertex_to_parent[child_vertices]
                    is_facet = np.isin(
                        child_vertices_as_parent, p_f_to_v.links(facet)
                    ).all()
                    if is_facet:
                        child_markers[child_facet] = value
                        facet_found = True
                        sub_to_parent_entity_map[child_facet] = facet

    tags = dolfinx.mesh.meshtags(
        submesh,
        entity_tag.dim,
        np.arange(num_child_entities, dtype=np.int32),
        child_markers,
    )
    tags.name = entity_tag.name
    return tags, sub_to_parent_entity_map


def create_piecewise_constant_field(
    domain, cell_markers, property_dict, name=None, default_value=0
):
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
    V0 = dolfinx.fem.functionspace(domain, ("DG", 0))
    k = dolfinx.fem.Function(V0, name=name)
    k.x.array[:] = default_value
    for tag, value in property_dict.items():
        cells = cell_markers.find(tag)
        k.x.array[cells] = np.full_like(cells, value, dtype=np.float64)
    return k


def interpolate_submesh_to_parent(u_parent, u_sub, sub_to_parent_cells):
    """
    Interpolate results from functions defined in submeshes to a function defined on the parent mesh.

    Parameters
    ----------
    u_parent : dolfinx.fem.Function
        Parent function to interpolate to.
    u_sub : list[dolfinx.fem.Function]
        _description_
    sub_to_parent_cells : list
        Submesh cells to parent cells mapping
    """
    V_parent = u_parent.function_space
    V_sub = u_sub.function_space
    for i, cell in enumerate(sub_to_parent_cells):
        bs = V_parent.dofmap.bs
        bs_sub = V_sub.dofmap.bs
        assert bs == bs_sub
        parent_dofs = V_parent.dofmap.cell_dofs(cell)
        sub_dofs = V_sub.dofmap.cell_dofs(i)
        for p_dof, s_dof in zip(parent_dofs, sub_dofs):
            for j in range(bs):
                u_parent.x.array[p_dof * bs + j] = u_sub.x.array[s_dof * bs + j]


def interface_int_entities(
    msh,
    interface_facets,
    domain_to_domain_0,
    domain_to_domain_1,
):
    """
    This function computes the integration entities (as a list of pairs of
    (cell, local facet index) pairs) required to assemble mixed domain forms
    over the interface. It assumes there is a domain with two sub-domains,
    domain_0 and domain_1, that have a common interface. Cells in domain_0
    correspond to the "+" restriction and cells in domain_1 correspond to
    the "-" restriction.

    Parameters:
        interface_facets: A list of facets on the interface
        domain_0_cells: A list of cells in domain_0
        domain_1_cells: A list of cells in domain_1
        c_to_f: The cell to facet connectivity for the domain mesh
        f_to_c: the facet to cell connectivity for the domain mesh
        facet_imap: The facet index_map for the domain mesh
        domain_to_domain_0: A map from cells in domain to cells in domain_0
        domain_to_domain_1: A map from cells in domain to cells in domain_1

    Returns:
        A tuple containing:
            1) The integration entities
            2) A modified map (see HACK below)
            3) A modified map (see HACK below)
    """
    # Create measure for integration. Assign the first (cell, local facet)
    # pair to the cell in domain_0, corresponding to the "+" restriction.
    # Assign the second pair to the domain_1 cell, corresponding to the "-"
    # restriction.
    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_connectivity(tdim, fdim)
    msh.topology.create_connectivity(fdim, tdim)
    facet_imap = msh.topology.index_map(fdim)
    c_to_f = msh.topology.connectivity(tdim, fdim)
    f_to_c = msh.topology.connectivity(fdim, tdim)
    # FIXME This can be done more efficiently
    interface_entities = []
    domain_to_domain_0_new = np.array(domain_to_domain_0)
    domain_to_domain_1_new = np.array(domain_to_domain_1)
    for facet in interface_facets:
        # Check if this facet is owned
        if facet < facet_imap.size_local:
            cells = f_to_c.links(facet)
            assert len(cells) == 2
            if domain_to_domain_0[cells[0]] > 0:
                cell_plus = cells[0]
                cell_minus = cells[1]
            else:
                cell_plus = cells[1]
                cell_minus = cells[0]
            assert (
                domain_to_domain_0[cell_plus] >= 0
                and domain_to_domain_0[cell_minus] < 0
            )
            assert (
                domain_to_domain_1[cell_minus] >= 0
                and domain_to_domain_1[cell_plus] < 0
            )

            local_facet_plus = np.where(c_to_f.links(cell_plus) == facet)[0][0]
            local_facet_minus = np.where(c_to_f.links(cell_minus) == facet)[0][0]

            interface_entities.extend(
                [cell_plus, local_facet_plus, cell_minus, local_facet_minus]
            )

            # FIXME HACK cell_minus does not exist in the left submesh, so it
            # will be mapped to index -1. This is problematic for the
            # assembler, which assumes it is possible to get the full macro
            # dofmap for the trial and test functions, despite the restriction
            # meaning we don't need the non-existant dofs. To fix this, we just
            # map cell_minus to the cell corresponding to cell plus. This will
            # just add zeros to the assembled system, since there are no
            # u("-") terms. Could map this to any cell in the submesh, but
            # I think using the cell on the other side of the facet means a
            # facet space coefficient could be used
            domain_to_domain_0_new[cell_minus] = domain_to_domain_0[cell_plus]
            # Same hack for the right submesh
            domain_to_domain_1_new[cell_plus] = domain_to_domain_1[cell_minus]

    return interface_entities, domain_to_domain_0_new, domain_to_domain_1_new
