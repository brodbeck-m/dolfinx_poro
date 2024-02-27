"""
Testcase: Terzaghi problem

Solves a (quasi 1D) consolidation problem on a rectangular domain, 
where a traction boundary is applied on the drained top boundary.

"""

# --- Imports ---
import typing

import dolfinx.fem as dfem
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
    q_top: float,
    domain_hight: typing.Optional[float] = 1.0,
    sdisc_fetype: typing.Optional[FeSpaces] = None,
    sdisc_discont_volfrac: typing.Optional[bool] = False,
    sdisc_eorder: typing.Optional[typing.List[int]] = None,
    sdisc_nelmt: typing.Optional[typing.List[int]] = None,
    initcond_nhSt0S: typing.Optional[float] = 0.5,
    scale_output: typing.Optional[bool] = False,
):
    # Check input
    if sdisc_fetype is None:
        sdisc_fetype = set_default_spaces(primary_variables)

    if sdisc_eorder is None:
        sdisc_eorder = set_default_orders(primary_variables, sdisc_fetype)

    if sdisc_nelmt is None:
        sdisc_nelmt = [9, 72]

    # Set volume fraction (if it is no field-quantity)
    if primary_variables == PrimaryVariables.up:
        material.set_volumetric_term(VolumeTerms.nhSt0S, initcond_nhSt0S)

    # The mesh
    domain = create_geometry_rectangle([domain_hight / 5, domain_hight], sdisc_nelmt)

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
            DirichletTerzaghiUPN(
                fem_problem.domain,
                fem_problem.solution_space.function_space,
                fem_problem.solution_space.sub_function_spaces,
            )
        )

        # Set traction BC on top surface
        fem_problem.initialise_natural_bc(
            -q_top * ufl.FacetNormal(fem_problem.domain.mesh), 4, 0
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
class DirichletTerzaghiUPN(DirichletBC):
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
        # Left: No horizontal displacement
        facets = fct_fkts.indices[fct_fkts.values == 1]
        dofs = dfem.locate_dofs_topological(
            (V.sub(0).sub(0), V_sub[0].sub(0)), 1, facets
        )
        self.list.append(dfem.dirichletbc(self.uD[0], dofs, V.sub(0)))

        # Bottom: No vertical displacement
        facets = fct_fkts.indices[fct_fkts.values == 2]
        dofs = dfem.locate_dofs_topological(
            (V.sub(0).sub(1), V_sub[0].sub(1)), 1, facets
        )
        self.list.append(dfem.dirichletbc(self.uD[0], dofs, V.sub(0)))

        # Right: No horizontal displacement
        facets = fct_fkts.indices[fct_fkts.values == 3]
        dofs = dfem.locate_dofs_topological(
            (V.sub(0).sub(0), V_sub[0].sub(0)), 1, facets
        )
        self.list.append(dfem.dirichletbc(self.uD[0], dofs, V.sub(0)))

        # --- Pressure BCs
        # Right: Outflow
        facets = fct_fkts.indices[fct_fkts.values == 4]
        dofs = dfem.locate_dofs_topological((V.sub(1), V_sub[1]), 1, facets)
        self.list.append(dfem.dirichletbc(self.uD[1], dofs, V.sub(1)))
