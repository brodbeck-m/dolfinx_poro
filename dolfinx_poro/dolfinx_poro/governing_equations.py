# --- Includes ---
import typing

import basix
import dolfinx.fem as dfem
import dolfinx.mesh as dmesh
import ufl

from .fem import SolutionSpace
from .fem.finite_element_problem import FemProblem

from .poroelasticity import EquationType, PrimaryVariables, FeSpaces, FieldVariables
from .governing_equations_biot import (
    goveq_biot_nomex_up,
    goveq_biot_mex_upn,
    goveq_biot_nomex_uppt,
    goveq_biot_mex_upptn,
    goveq_biot_nomex_uvp,
    goveq_biot_mex_uvpn,
)
from .governing_equations_lTPM import goveq_lTPM_nomex_up, goveq_lTPM_mex_upn
from .governing_equations_TPM import (
    goveq_TPM_nomex_up,
    goveq_TPM_mex_upn,
    goveq_TPM_nomex_uvp,
    goveq_TPM_mex_uvpn,
)


# --- Set finite-element discretisation ---
def set_discretisation(
    equation_type: EquationType,
    primary_variables: PrimaryVariables,
    domain_mesh: dmesh.Mesh,
    fe_spaces: FeSpaces,
    element_order: typing.List[int],
    nhS_is_dg: typing.Optional[bool] = False,
):
    # Check if problem is linear
    if equation_type == EquationType.Biot:
        if (
            primary_variables == PrimaryVariables.up
            or primary_variables == PrimaryVariables.uppt
            or primary_variables == PrimaryVariables.uvp
        ):
            is_linear = True
        elif (
            primary_variables == PrimaryVariables.upn
            or primary_variables == PrimaryVariables.upptn
            or primary_variables == PrimaryVariables.uvpn
        ):
            is_linear = False
        else:
            raise ValueError("Unknown formulation type.")
    else:
        is_linear = False

    # Set FunctionSpace
    if primary_variables == PrimaryVariables.up:
        return set_discretisation_up(domain_mesh, fe_spaces, element_order, is_linear)
    elif primary_variables == PrimaryVariables.upn:
        return set_discretisation_upn(
            domain_mesh, fe_spaces, element_order, is_linear, nhS_is_dg
        )
    elif primary_variables == PrimaryVariables.uppt:
        return set_discretisation_uppt(domain_mesh, fe_spaces, element_order, is_linear)
    elif primary_variables == PrimaryVariables.upptn:
        return set_discretisation_upptn(
            domain_mesh, fe_spaces, element_order, is_linear, nhS_is_dg
        )
    elif primary_variables == PrimaryVariables.uvp:
        return set_discretisation_uvp(domain_mesh, fe_spaces, element_order, is_linear)
    elif primary_variables == PrimaryVariables.uvpn:
        return set_discretisation_uvpn(
            domain_mesh, fe_spaces, element_order, is_linear, nhS_is_dg
        )
    else:
        raise ValueError("Unknown formulation type.")


def set_default_spaces(primary_variables: PrimaryVariables):
    if (
        primary_variables == PrimaryVariables.up
        or primary_variables == PrimaryVariables.upn
    ):
        return FeSpaces.up_TH
    elif (
        primary_variables == PrimaryVariables.uppt
        or primary_variables == PrimaryVariables.upptn
    ):
        return FeSpaces.uppt_TH
    elif (
        primary_variables == PrimaryVariables.uvp
        or primary_variables == PrimaryVariables.uvpn
    ):
        return FeSpaces.uvp_RT
    else:
        raise ValueError("Unknown formulation type.")


def set_default_orders(primary_variables: PrimaryVariables, fe_spaces: FeSpaces):
    if (primary_variables == PrimaryVariables.up) or (
        primary_variables == PrimaryVariables.upn
    ):
        if fe_spaces == FeSpaces.up_EO:
            print("Warning: Not Inf-Sub stable!")
            return [2, 2, 1]
        elif fe_spaces == FeSpaces.up_TH:
            return [2, 1, 1]
        elif fe_spaces == FeSpaces.up_mini:
            return [1, 1, 1]
        else:
            raise ValueError("Unknown fe-space")
    if (primary_variables == PrimaryVariables.uppt) or (
        primary_variables == PrimaryVariables.upptn
    ):
        if fe_spaces == FeSpaces.uppt_TH:
            return [2, 1, 1, 1]
        elif fe_spaces == FeSpaces.uppt_mini:
            return [1, 1, 1, 1]
        else:
            raise ValueError("Unknown fe-space")
    elif (primary_variables == PrimaryVariables.uvp) or (
        primary_variables == PrimaryVariables.uvpn
    ):
        if fe_spaces == FeSpaces.uvp_RT or fe_spaces == FeSpaces.uvp_BDM:
            return [2, 1, 0, 1]
        else:
            raise ValueError("Unknown fe-space")


# --- Finite-element space for u-p(-n) formulation
def set_spaces_up_base(
    domain_mesh: dmesh.Mesh, fe_spaces: FeSpaces, element_order: typing.List[int]
):
    # Set function spaces
    if (fe_spaces == FeSpaces.up_EO) or (fe_spaces == FeSpaces.up_TH):
        Pu = ufl.VectorElement("Lagrange", domain_mesh.ufl_cell(), element_order[0])
    elif fe_spaces == FeSpaces.up_mini:
        # Create bubble enrichment
        lagrange_ufl = basix.ufl.element("Lagrange", domain_mesh.basix_cell(), 1)
        bubble_ufl = basix.ufl.element("bubble", domain_mesh.basix_cell(), 3)

        # Create enriched element
        Pu = ufl.VectorElement(ufl.EnrichedElement(lagrange_ufl, bubble_ufl))
    else:
        raise ValueError("Unknown element type.")

    Pp = ufl.FiniteElement("Lagrange", domain_mesh.ufl_cell(), element_order[1])

    return Pu, Pp


def set_discretisation_up(
    domain_mesh: dmesh.Mesh,
    fe_spaces: FeSpaces,
    element_order: typing.List[int],
    is_linear: bool,
):
    # Set function spaces
    Pu, Pp = set_spaces_up_base(domain_mesh, fe_spaces, element_order)

    V_up = dfem.FunctionSpace(domain_mesh, ufl.MixedElement(Pu, Pp))

    return SolutionSpace(V_up, ["u", "p"], True, is_linear)


def set_discretisation_upn(
    domain_mesh: dmesh.Mesh,
    fe_spaces: FeSpaces,
    element_order: typing.List[int],
    is_linear: bool,
    nhS_is_dg: bool,
):
    # Set function spaces
    Pu, Pp = set_spaces_up_base(domain_mesh, fe_spaces, element_order)

    if nhS_is_dg:
        Pn = ufl.FiniteElement("DG", domain_mesh.ufl_cell(), element_order[2])
    else:
        Pn = ufl.FiniteElement("CG", domain_mesh.ufl_cell(), element_order[2])

    V_upn = dfem.FunctionSpace(domain_mesh, ufl.MixedElement(Pu, Pp, Pn))

    return SolutionSpace(V_upn, ["u", "p", "nhS"], True, is_linear)


# --- Finite-element space for u-p-pt(-n) formulation
def set_spaces_uppt_base(
    domain_mesh: dmesh.Mesh, fe_spaces: FeSpaces, element_order: typing.List[int]
):
    # Set function spaces
    if fe_spaces == FeSpaces.uppt_TH:
        Pu = ufl.VectorElement("Lagrange", domain_mesh.ufl_cell(), element_order[0])
        Ppt = ufl.FiniteElement("Lagrange", domain_mesh.ufl_cell(), element_order[2])
    elif fe_spaces == FeSpaces.uppt_mini:
        # Create bubble enrichment
        lagrange_ufl = basix.ufl.element("Lagrange", domain_mesh.basix_cell(), 1)
        bubble_ufl = basix.ufl.element("bubble", domain_mesh.basix_cell(), 3)

        # Create enriched element (displacement)
        Pu = ufl.VectorElement(ufl.EnrichedElement(lagrange_ufl, bubble_ufl))

        # Element for total pressure
        Ppt = ufl.FiniteElement("Lagrange", domain_mesh.ufl_cell(), 1)
    else:
        raise ValueError("Unknown fe-space.")

    Pp = ufl.FiniteElement("Lagrange", domain_mesh.ufl_cell(), element_order[1])

    return Pu, Pp, Ppt


def set_discretisation_uppt(
    domain_mesh: dmesh.Mesh,
    fe_spaces: FeSpaces,
    element_order: typing.List[int],
    is_linear: bool,
):
    # Set function spaces
    Pu, Pp, Ppt = set_spaces_uppt_base(domain_mesh, fe_spaces, element_order)

    V_upp = dfem.FunctionSpace(domain_mesh, ufl.MixedElement(Pu, Pp, Ppt))

    return SolutionSpace(V_upp, ["u", "p", "pt"], True, is_linear)


def set_discretisation_upptn(
    domain_mesh: dmesh.Mesh,
    fe_spaces: FeSpaces,
    element_order: typing.List[int],
    is_linear: bool,
    nhS_is_dg: bool,
):
    # Set function spaces
    Pu, Pp, Ppt = set_spaces_uppt_base(domain_mesh, fe_spaces, element_order)

    if nhS_is_dg:
        Pn = ufl.FiniteElement("DG", domain_mesh.ufl_cell(), element_order[3])
    else:
        Pn = ufl.FiniteElement("CG", domain_mesh.ufl_cell(), element_order[3])

    V_uppn = dfem.FunctionSpace(domain_mesh, ufl.MixedElement(Pu, Pp, Ppt, Pn))

    return SolutionSpace(V_uppn, ["u", "p", "pt", "nhS"], True, is_linear)


# --- Finite-element space for u-vD-p(-n) formulation
def set_spaces_uvp_base(
    domain_mesh: dmesh.Mesh, fe_spaces: FeSpaces, element_order: typing.List[int]
):
    # Set function spaces
    Pu = ufl.VectorElement("Lagrange", domain_mesh.ufl_cell(), element_order[0])
    Pp = ufl.FiniteElement("DG", domain_mesh.ufl_cell(), element_order[2])

    if fe_spaces == FeSpaces.uvp_RT:
        P_vD = ufl.FiniteElement("RT", domain_mesh.ufl_cell(), element_order[1])
    elif fe_spaces == FeSpaces.uvp_BDM:
        P_vD = ufl.VectorElement("BDM", domain_mesh.ufl_cell(), element_order[1])
    else:
        raise ValueError("Unknown element type.")

    return Pu, P_vD, Pp


def set_discretisation_uvp(
    domain_mesh: dmesh.Mesh,
    fe_spaces: FeSpaces,
    element_order: typing.List[int],
    is_linear: bool,
):
    # Set function spaces
    Pu, Pvd, Pp = set_spaces_uvp_base(domain_mesh, fe_spaces, element_order)
    V_mixed = dfem.FunctionSpace(domain_mesh, ufl.MixedElement(Pu, Pvd, Pp))

    return SolutionSpace(V_mixed, ["u", "vD", "p"], True, is_linear)


def set_discretisation_uvpn(
    domain_mesh: dmesh.Mesh,
    fe_spaces: FeSpaces,
    element_order: typing.List[int],
    is_linear: bool,
    nhS_is_dg: bool,
):
    # Set function spaces
    Pu, Pvd, Pp = set_spaces_uvp_base(domain_mesh, fe_spaces, element_order)

    if nhS_is_dg:
        Pn = ufl.FiniteElement("DG", domain_mesh.ufl_cell(), element_order[3])
    else:
        Pn = ufl.FiniteElement("CG", domain_mesh.ufl_cell(), element_order[3])

    V_mixed = dfem.FunctionSpace(domain_mesh, ufl.MixedElement(Pu, Pvd, Pp, Pn))

    return SolutionSpace(V_mixed, ["u", "vD", "p", "nhS"], True, is_linear)


# --- Set weak forms ---
def set_weakform(
    equation_type: EquationType,
    primary_variables: PrimaryVariables,
    problem: FemProblem,
):
    if equation_type == EquationType.Biot:
        if primary_variables == PrimaryVariables.up:
            goveq_biot_nomex_up(problem)
        elif primary_variables == PrimaryVariables.upn:
            goveq_biot_mex_upn(problem)
        elif primary_variables == PrimaryVariables.uppt:
            goveq_biot_nomex_uppt(problem)
        elif primary_variables == PrimaryVariables.upptn:
            goveq_biot_mex_upptn(problem)
        elif primary_variables == PrimaryVariables.uvp:
            goveq_biot_nomex_uvp(problem)
        elif primary_variables == PrimaryVariables.uvpn:
            goveq_biot_mex_uvpn(problem)
        else:
            raise ValueError("Unknown formulation type.")
    elif equation_type == EquationType.lTPM or equation_type == EquationType.lTPM_Bluhm:
        if primary_variables == PrimaryVariables.up:
            goveq_lTPM_nomex_up(problem, equation_type)
        elif primary_variables == PrimaryVariables.upn:
            goveq_lTPM_mex_upn(problem, equation_type)
        else:
            raise ValueError("Unknown formulation type.")
    elif equation_type == EquationType.TPM:
        if primary_variables == PrimaryVariables.up:
            goveq_TPM_nomex_up(problem)
        elif primary_variables == PrimaryVariables.upn:
            goveq_TPM_mex_upn(problem)
        elif primary_variables == PrimaryVariables.uvp:
            goveq_TPM_nomex_uvp(problem)
        elif primary_variables == PrimaryVariables.uvpn:
            goveq_TPM_mex_uvpn(problem)
        else:
            raise ValueError("Unknown formulation type.")
    else:
        raise ValueError("Unknown equation type.")


# --- Non-dimensionalisation ---
def scale_primal_variables(primary_variables: PrimaryVariables, problem: FemProblem):
    if primary_variables == PrimaryVariables.up:
        scaling = problem.material.rescale(
            [FieldVariables.time, FieldVariables.displacement, FieldVariables.pressure]
        )
    elif primary_variables == PrimaryVariables.upn:
        scaling = problem.material.rescale(
            [
                FieldVariables.time,
                FieldVariables.displacement,
                FieldVariables.pressure,
                FieldVariables.volume_fraction,
            ]
        )
    else:
        raise ValueError("Unknown formulation type.")

    # Apply scaling
    problem.set_scaling_pvar(scaling[1:], scaling[0])
