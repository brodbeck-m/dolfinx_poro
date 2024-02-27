"""
Testcase: Footing problem

Solves a 2D footing problem on a rectangular domain. Symmetry is applied 
on surface 1, while traction is applied on surface 5.

The domain:

                5             4 
     -     |--------|-----------------|
     |     |                          |          h -> hight
     |     |---w1---'                 |          w -> width
   h |   1 |                          | 3        w1 = w/3
     |     |                          |
     |     |                          |
     -     |--------------------------|            
                        2
     
           '--------------------------'
                         w


The boundary conditions:
    1: u = 0, nhFwtFS x n = 0
    2: u = v = 0, nhFwtFS x n = 0
    3: p = 0
    4: p = 0
    5: t x n = q, nhFwtFS x n = 0

"""

# --- Imports ---
from mpi4py import MPI
import numpy as np
import typing

import dolfinx.fem as dfem
import dolfinx.mesh as dmesh
import ufl

from dolfinx.mesh import CellType, DiagonalType, create_rectangle

from dolfinx_poro import (
    EquationType,
    PrimaryVariables,
    FeSpaces,
    VolumeTerms,
    set_default_spaces,
    set_default_orders,
    set_discretisation,
    set_weakform,
    scale_primal_variables,
)
from dolfinx_poro.material import AbstractMaterialPoro
from dolfinx_poro.fem import Domain, DirichletBC, set_finite_element_problem


# --- Mesh generation  ---
def create_geometry_footing(
    domain_geometry: typing.List[float],
    n_elmt: typing.List[int],
    diagonal: DiagonalType = DiagonalType.left,
):
    # Parameters
    tol = 1.0e-14
    width = domain_geometry[0]
    hight = domain_geometry[1]
    w1 = domain_geometry[0] / 3

    # Create mesh
    mesh = create_rectangle(
        MPI.COMM_WORLD,
        [np.array([0, 0]), np.array([width, hight])],
        [n_elmt[0], n_elmt[1]],
        cell_type=CellType.triangle,
        diagonal=diagonal,
    )

    boundaries = [
        (1, lambda x: np.isclose(x[0], 0)),
        (2, lambda x: np.isclose(x[1], 0)),
        (3, lambda x: np.isclose(x[0], width)),
        (4, lambda x: np.logical_and(np.isclose(x[1], hight), x[0] > w1 - tol)),
        (5, lambda x: np.logical_and(np.isclose(x[1], hight), x[0] < w1 + tol)),
    ]

    facet_indices, facet_markers = [], []
    for marker, locator in boundaries:
        facets = dmesh.locate_entities(mesh, 1, locator)
        facet_indices.append(facets)
        facet_markers.append(np.full(len(facets), marker))

    facet_indices = np.array(np.hstack(facet_indices), dtype=np.int32)
    facet_markers = np.array(np.hstack(facet_markers), dtype=np.int32)
    sorted_facets = np.argsort(facet_indices)
    facet_tag = dmesh.meshtags(
        mesh, 1, facet_indices[sorted_facets], facet_markers[sorted_facets]
    )
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tag)

    return Domain(mesh, facet_tag, ds)


# --- Setter function ---
def setup_calculation(
    equation_type: EquationType,
    primary_variables: PrimaryVariables,
    material: typing.Type[AbstractMaterialPoro],
    q_top: float,
    domain_geometry: typing.Optional[typing.List[float]] = None,
    sdisc_fetype: typing.Optional[FeSpaces] = None,
    sdisc_discont_volfrac: typing.Optional[bool] = False,
    sdisc_eorder: typing.Optional[typing.List[int]] = None,
    sdisc_nelmt: typing.Optional[typing.List[int]] = None,
    initcond_nhSt0S: typing.Optional[float] = 0.5,
    scale_output: typing.Optional[bool] = False,
):
    # Check input
    if domain_geometry is None:
        domain_geometry = [0.5, 1.0]

    if sdisc_fetype is None:
        sdisc_fetype = set_default_spaces(primary_variables)

    if sdisc_eorder is None:
        sdisc_eorder = set_default_orders(primary_variables, sdisc_fetype)

    if sdisc_nelmt is None:
        sdisc_nelmt = [36, 72]

    # Set volume fraction (if it is no field-quantity)
    if primary_variables == PrimaryVariables.up:
        material.set_volumetric_term(VolumeTerms.nhSt0S, initcond_nhSt0S)

    # The mesh
    domain = create_geometry_footing(domain_geometry, sdisc_nelmt)

    # The solution spaces
    solution_space = set_discretisation(
        equation_type,
        primary_variables,
        domain.mesh,
        sdisc_fetype,
        sdisc_eorder,
        nhS_is_dg=sdisc_discont_volfrac,
    )

    # The finite-element problem
    fem_problem = set_finite_element_problem(
        domain, solution_space, material, equation_type, primary_variables, sdisc_fetype
    )

    # The weak form
    set_weakform(equation_type, primary_variables, fem_problem)

    if scale_output:
        scale_primal_variables(primary_variables, fem_problem)

    # Set boundary conditions
    if (
        primary_variables == PrimaryVariables.up
        or primary_variables == PrimaryVariables.upn
    ):
        # Initialise essential boundary conditions
        fem_problem.initialise_essential_bc(
            DirichletFootingUPN(
                fem_problem.domain,
                fem_problem.solution_space.function_space,
                fem_problem.solution_space.sub_function_spaces,
            )
        )

        # Set traction BC on top surface
        fem_problem.initialise_natural_bc(
            -q_top * ufl.FacetNormal(fem_problem.domain.mesh), 5, 0
        )
    else:
        raise ValueError("Unknown formulation type.")

    # Set initial conditions
    if primary_variables == PrimaryVariables.up:
        fem_problem.initialise_solution([0], [0])
    elif primary_variables == PrimaryVariables.upn:
        fem_problem.initialise_solution([0, initcond_nhSt0S], [0, 2])

    return fem_problem


# --- The Dirichlet boundary conditions ---
class DirichletFootingUPN(DirichletBC):
    def set_dirichletbc(
        self,
        domain: Domain,
        V: dfem.FunctionSpace,
        V_sub: typing.List[dfem.FunctionSpace],
    ):
        # Extract facet functions
        fct_fkts = domain.facet_functions

        # Initialise boundary values
        self.initialise_dirichlet_values(V_sub[0], const_value=0.0, id_subspace=0)
        self.initialise_dirichlet_values(V_sub[1], const_value=0.0, id_subspace=1)

        # --- Displacement BCs
        # Boundary 1: No horizontal displacement
        facets = fct_fkts.indices[fct_fkts.values == 1]
        dofs = dfem.locate_dofs_topological(
            (V.sub(0).sub(0), V_sub[0].sub(0)), 1, facets
        )
        self.list.append(dfem.dirichletbc(self.uD[0], dofs, V.sub(0)))

        # Boundary 2: No displacement
        facets = fct_fkts.indices[fct_fkts.values == 2]
        dofs = dfem.locate_dofs_topological((V.sub(0), V_sub[0]), 1, facets)
        self.list.append(dfem.dirichletbc(self.uD[0], dofs, V.sub(0)))

        # --- Pressure BCs
        # Boundary 3: Outflow
        facets = fct_fkts.indices[fct_fkts.values == 3]
        dofs = dfem.locate_dofs_topological((V.sub(1), V_sub[1]), 1, facets)
        self.list.append(dfem.dirichletbc(self.uD[1], dofs, V.sub(1)))

        # Boundary 4: Outflow
        facets = fct_fkts.indices[fct_fkts.values == 4]
        dofs = dfem.locate_dofs_topological((V.sub(1), V_sub[1]), 1, facets)
        self.list.append(dfem.dirichletbc(self.uD[1], dofs, V.sub(1)))
