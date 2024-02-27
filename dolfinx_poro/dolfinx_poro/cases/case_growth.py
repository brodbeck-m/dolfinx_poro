"""
Testcase: Footing problem

Solves a 2D problem on a rectangular domain. Symmetry is applied 
on surface 1 and 2, outflow is allowed on the surfaces 3 and 4.

The domain:

                        4 
     -     |--------------------------|
     |     |                          |          
     |     |                          |          h -> hight
   h |   1 |                          | 3        w -> width
     |     |                          |
     |     |                          |
     -     |--------------------------|            
                        2
     
           '--------------------------'
                         w

The boundary conditions:
    1: u = 0, nhFwtFS x n = 0
    2: v = 0, nhFwtFS x n = 0
    3: p = 0
    4: p = 0

"""

# --- Imports ---
from mpi4py import MPI
import numpy as np
import typing

import dolfinx.fem as dfem
import dolfinx.mesh as dmesh
import ufl

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

from .mesh_generation import create_geometry_rectangle


# --- Setter function ---
def setup_calculation(
    equation_type: EquationType,
    primary_variables: PrimaryVariables,
    material: typing.Type[AbstractMaterialPoro],
    rhohatS: float,
    domain_geometry: typing.Optional[typing.List[float]] = None,
    sdisc_fetype: typing.Optional[FeSpaces] = None,
    sdisc_discont_volfrac: typing.Optional[bool] = False,
    sdisc_eorder: typing.Optional[typing.List[int]] = None,
    sdisc_nelmt: typing.Optional[typing.List[int]] = None,
    initcond_nhSt0S: typing.Optional[float] = 0.5,
    scale_output: typing.Optional[bool] = False,
    time_function_growth: typing.Optional[typing.Callable] = None,
    allow_outflow: typing.Optional[bool] = True,
    traction_top: typing.Optional[float] = None,
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

    if not allow_outflow:
        if traction_top is None:
            raise ValueError("Traction BC required for well-posed problem!")

    # Specify growth rate
    material.set_volumetric_term(VolumeTerms.growth_rate, rhohatS, time_function_growth)

    # The mesh
    domain = create_geometry_rectangle(domain_geometry, sdisc_nelmt)

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
    if primary_variables == PrimaryVariables.upn:
        # Initialise essential boundary conditions
        fem_problem.initialise_essential_bc(
            DirichletGrowthUPN(
                fem_problem.domain,
                fem_problem.solution_space.function_space,
                fem_problem.solution_space.sub_function_spaces,
                allow_outflow,
            )
        )

        # Set traction BC
        if traction_top is not None:
            # Set traction BC on top surface
            normal = ufl.FacetNormal(fem_problem.domain.mesh)
            fem_problem.initialise_natural_bc(traction_top * normal, 4, 0)
    elif primary_variables == PrimaryVariables.uvpn:
        raise NotImplementedError("Not implemented yet.")
    else:
        raise ValueError("Unknown formulation type.")

    # Set initial conditions
    if primary_variables == PrimaryVariables.upn:
        fem_problem.initialise_solution([0, initcond_nhSt0S], [0, 2])
    elif primary_variables == PrimaryVariables.uvpn:
        fem_problem.initialise_solution([0, initcond_nhSt0S], [0, 3])
    else:
        raise ValueError("Unknown formulation type.")

    return fem_problem


# --- The Dirichlet boundary conditions ---
class DirichletGrowthUPN(DirichletBC):
    def __init__(
        self,
        domain: Domain,
        V: dfem.FunctionSpace,
        V_sub: typing.List[dfem.FunctionSpace],
        allow_outflow: bool,
    ):
        # Set identifier for outflow
        self.allow_outflow = allow_outflow

        # Default constructor
        super().__init__(domain, V, V_sub)

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

        # Boundary 2: No vertical displacement
        facets = fct_fkts.indices[fct_fkts.values == 2]
        dofs = dfem.locate_dofs_topological(
            (V.sub(0).sub(1), V_sub[0].sub(1)), 1, facets
        )
        self.list.append(dfem.dirichletbc(self.uD[0], dofs, V.sub(0)))

        # --- Pressure BCs
        if self.allow_outflow:
            # Boundary 3: Outflow
            facets = fct_fkts.indices[fct_fkts.values == 3]
            dofs = dfem.locate_dofs_topological((V.sub(1), V_sub[1]), 1, facets)
            self.list.append(dfem.dirichletbc(self.uD[1], dofs, V.sub(1)))

            # Boundary 4: Outflow
            facets = fct_fkts.indices[fct_fkts.values == 4]
            dofs = dfem.locate_dofs_topological((V.sub(1), V_sub[1]), 1, facets)
            self.list.append(dfem.dirichletbc(self.uD[1], dofs, V.sub(1)))
