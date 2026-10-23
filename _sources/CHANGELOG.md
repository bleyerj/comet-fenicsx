## v0.3 — FEniCSx v0.11

All tours now run with `dolfinx` v0.11.

### New demos
- **Finite-strain periodic homogenization** of a soft open-cell foam (3D, hyperelastic)
- **Bloch waves** in a periodic medium: dispersion diagram and band gaps of a cell with four holes
- **Obstacle problem** added to the bound-constrained tour, solved with PETSc TAO and with SNES variational inequality solvers
- **Form-finding** using the force density method of a funicular gridshell
- **Topology optimization** tour porting the legacy demo, including now filtering and thresholding

### Migration notes towards dolfinx v0.11

- `u.vector` → `u.x.petsc_vec` / `u.x.array`
- `LinearProblem` now requires `petsc_options_prefix` and solves into its own vector, so custom Newton solvers solve directly into `du.x.petsc_vec`
- Block linear problems use `LinearProblem` directly (custom `BlockedLinearProblem` removed)
- `dolfinx.io.gmshio` → `dolfinx.io.gmsh`; `model_to_mesh` returns a `MeshData` object
- Submeshes: `create_submesh` returns `EntityMap` objects passed as `entity_maps=[...]`; `transfer_meshtags_to_submesh` and submesh-to-parent interpolation (`u.interpolate(u_sub, cells0=..., cells1=...)`) are now part of dolfinx, so the CZM interface tour no longer needs `utils.py`
- Quadrature: `element.interpolation_points` is a property, the measure metadata key is `"quadrature_rule"`, and quadrature functions are updated with `Function.interpolate(fem.Expression(...))`
- Interior facet integrals: geometric quantities must be restricted, e.g. `ufl.FacetArea(domain)("+")`

### Fixes
- Periodic homogenization: periodicity constraints applied hierarchically (faces, then edges, then fixed corners) to avoid double constraints, which corrects the local stress fields
- Viscoelasticity (JAX): shear term of the plane stress stiffness in Mandel notation
- Bound-constrained tour: the objective is summed over processes, so TAO solves are correct in parallel
- Missing $2$ factor in fully coupled thermoelasticity demo

### Citing
- The landing page now cites the all-versions Zenodo DOI and the FEniCSx references
- `CITATION.cff`