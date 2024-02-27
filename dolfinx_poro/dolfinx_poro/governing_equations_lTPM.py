# --- Imports ---
import typing

import ufl

from .fem.finite_element_problem import FemProblem
from .poroelasticity import EquationType, MatPar, NondimPar, VolumeTerms


# --- Poroelasticity without mass exchange ---
def goveq_lTPM_base(
    problem: FemProblem, type: EquationType, nhStS0S: typing.Optional[typing.Any] = None
):
    # --- Getters
    # Classes
    solution_space = problem.solution_space
    material = problem.material

    # Material parameters
    lhS = material.get_matpar_ufl(MatPar.lhS)
    mhS = material.get_matpar_ufl(MatPar.mhS)
    rhohSR = material.get_matpar_ufl(MatPar.rhohSR)
    khSt0S = material.get_matpar_ufl(MatPar.khSt0S)
    rhohFR = material.get_matpar_ufl(MatPar.rhohFR)
    mhFR = material.get_matpar_ufl(MatPar.mhFR)

    # Non-dimensional parameters
    pi_1 = material.get_pi_ufl(NondimPar.pi_1)
    pi_2 = material.get_pi_ufl(NondimPar.pi_2)
    pi_3 = material.get_pi_ufl(NondimPar.pi_3)

    # Volumetric terms
    body_force = material.get_volumetric_term_ufl(VolumeTerms.body_force)

    # --- Primal variables
    u = solution_space.trial_fkt[0]
    p = solution_space.trial_fkt[1]

    u_n = solution_space.uh_n[0]

    # --- Test functions
    v_u = solution_space.test_fkt[0]
    v_p = solution_space.test_fkt[1]

    # --- Kinematics
    # The deformation gradient
    JtS = 1 + ufl.div(u)

    # Linearized Green-Lagrange strain
    EtS = ufl.sym(ufl.grad(u))

    # Solid velocity
    vtS_t_dt = u - u_n

    # --- Constitutives
    if type == EquationType.lTPM:
        h1 = JtS * ufl.Identity(len(u)) - 2 * EtS
    else:
        h1 = JtS * ufl.Identity(len(u))

    # Stress
    PhSE = 2.0 * mhS * EtS + pi_1 * lhS * ufl.div(u) * ufl.Identity(len(u))
    P = PhSE - p * h1

    # Fluid flux
    ktD = problem.ufl_dt * (khSt0S / mhFR)
    if body_force is not None:
        raise NotImplementedError("Body force not implemented!")
    else:
        nhFwtFS0S_t_dt = ktD * ufl.dot(ufl.grad(p), h1)

    # --- Governing equations
    # Volume integrator
    dv = problem.domain.dv

    # Linerisation of the divergence of the velocity
    if type == EquationType.lTPM:
        div_vtS_t_dt = ufl.div(vtS_t_dt)
    else:
        div_vtS_t_dt = JtS * ufl.div(vtS_t_dt)

    # Balance equations
    res_BLM = ufl.inner(P, ufl.sym(ufl.grad(v_u))) * dv
    res_BMO = (div_vtS_t_dt * v_p + ufl.inner(nhFwtFS0S_t_dt, ufl.grad(v_p))) * dv

    if body_force is not None:
        # Check in initial volume fraction is set
        if nhStS0S is None:
            raise ValueError("Initial volume fraction required!")

        raise NotImplementedError("Body force not implemented!")

    return res_BLM, res_BMO


# --- u-p formulation
def goveq_lTPM_nomex_up(problem: FemProblem, type: EquationType):
    # --- Getters
    # Volumetric terms
    nhStS0S = problem.material.get_volumetric_term_ufl(VolumeTerms.nhSt0S)

    # --- The weak form
    res_BLM, res_BMO = goveq_lTPM_base(problem, type, nhStS0S)

    # Set predators for RHS
    problem.prefactors_natural_bc[0] = -1.0
    problem.prefactors_natural_bc[1] = -1.0

    # Add volumetric contributions of weak form
    problem.weak_form += res_BLM + res_BMO


# --- Poroelasticity with mass exchange ---
# --- u-p-n formulation ---
def goveq_lTPM_mex_upn(problem: FemProblem, type: EquationType):
    # --- Getters
    # Classes
    solution_space = problem.solution_space
    material = problem.material

    # Material parameters
    rhohSR = material.get_matpar_ufl(MatPar.rhohSR)
    rhohFR = material.get_matpar_ufl(MatPar.rhohFR)

    # Non-dimensional parameters
    pi_2 = material.get_pi_ufl(NondimPar.pi_2)
    pi_4 = material.get_pi_ufl(NondimPar.pi_4)

    # Volumetric terms
    rhohathS = material.get_volumetric_term_ufl(VolumeTerms.growth_rate)

    # --- Primal variables
    u = solution_space.trial_fkt[0]
    nhS = solution_space.trial_fkt[2]

    u_n = solution_space.uh_n[0]
    nhS_n = solution_space.uh_n[2]

    # --- Test functions
    v_p = solution_space.test_fkt[1]
    v_n = solution_space.test_fkt[2]

    # --- Kinematics
    # Deformation Gradient
    JtS = 1 + ufl.div(u)

    # Velocities
    vtS_t_dt = u - u_n
    DnhSDt_t_dt = nhS - nhS_n

    # --- Governing equations
    dv = problem.domain.dv

    res_BLM, res_BMO = goveq_lTPM_base(problem, type, JtS * nhS)

    if type == EquationType.lTPM:
        res_BMS = (JtS * DnhSDt_t_dt + nhS * ufl.div(vtS_t_dt)) * v_n * dv
    else:
        res_BMS = (DnhSDt_t_dt + nhS * ufl.div(vtS_t_dt)) * v_n * dv

    if rhohathS is not None:
        if type == EquationType.lTPM:
            pi_t_rhatS_t_dt = JtS * (pi_4 / pi_2) * problem.ufl_dt * rhohathS
            res_BMS -= (1 / rhohSR) * pi_t_rhatS_t_dt * v_n * dv
            res_BMO -= ((1 / rhohSR) - pi_2 * (1 / rhohFR)) * pi_t_rhatS_t_dt * v_p * dv
        else:
            pi_t_rhatS_t_dt = (pi_4 / pi_2) * problem.ufl_dt * rhohathS
            res_BMS -= (1 / rhohSR) * pi_t_rhatS_t_dt * v_n * dv
            res_BMO -= (
                JtS * ((1 / rhohSR) - pi_2 * (1 / rhohFR)) * pi_t_rhatS_t_dt * v_p * dv
            )

    # Set predators for RHS
    problem.prefactors_natural_bc[0] = -1.0
    problem.prefactors_natural_bc[1] = -1.0

    # Add volumetric contributions of weak form
    problem.weak_form += res_BLM + res_BMO + res_BMS
