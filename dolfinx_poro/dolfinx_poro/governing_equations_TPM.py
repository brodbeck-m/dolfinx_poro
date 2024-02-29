# --- Imports ---
import typing

import ufl

from .fem.finite_element_problem import FemProblem
from .poroelasticity import MatPar, NondimPar, VolumeTerms


# --- Base model ---
def goveq_TPM_up_base(problem: FemProblem, nhStS0S: typing.Optional[typing.Any] = None):
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
    I = ufl.Identity(len(u))

    # The deformation gradient
    FtS = I + ufl.grad(u)
    JtS = ufl.det(FtS)

    inv_tFtS = ufl.inv(ufl.transpose(FtS))

    # Right Cauchy-Green deformation tensor
    CtS = ufl.dot(ufl.transpose(FtS), FtS)

    # Solid velocity
    vtS_t_dt = u - u_n
    divxtS_t_dt = JtS * ufl.inner(ufl.grad(vtS_t_dt), inv_tFtS)

    # --- Constitutives
    # Stress
    PhSE = (
        mhS * ufl.dot((I - ufl.inv(CtS)), inv_tFtS)
        + pi_1 * lhS * ufl.ln(JtS) * inv_tFtS
    )
    P = PhSE - JtS * p * inv_tFtS

    # Fluid flux
    ktD = problem.ufl_dt * (khSt0S / mhFR)
    nhFwtFS0S_t_dt = JtS * ktD * ufl.dot(ufl.grad(p), ufl.inv(CtS))

    if body_force is not None:
        nhFwtFS0S_t_dt -= JtS * ktD * pi_3 * rhohFR * ufl.dot(body_force, inv_tFtS)

    # --- Governing equations
    dv = problem.domain.dv

    res_BLM = ufl.inner(P, ufl.grad(v_u)) * dv
    res_BMO = (divxtS_t_dt * v_p + ufl.inner(nhFwtFS0S_t_dt, ufl.grad(v_p))) * dv

    if body_force is not None:
        # Check in initial volume fraction is set
        if nhStS0S is None:
            raise ValueError("Initial volume fraction required!")

        # Evaluate real density
        density = rhohFR + nhStS0S * (pi_2 * rhohSR - rhohFR)

        # Add volumetric forces to BLM
        res_BLM -= JtS * (pi_3 * density * ufl.inner(body_force, v_u)) * dv

    return res_BLM, res_BMO


def goveq_TPM_upv_base(
    problem: FemProblem, nhStS0S: typing.Optional[typing.Any] = None
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

    ktD = khSt0S / mhFR

    # Non-dimensional parameters
    pi_1 = material.get_pi_ufl(NondimPar.pi_1)
    pi_2 = material.get_pi_ufl(NondimPar.pi_2)
    pi_3 = material.get_pi_ufl(NondimPar.pi_3)

    # Volumetric terms
    body_force = material.get_volumetric_term_ufl(VolumeTerms.body_force)

    # --- Primal variables
    u = solution_space.trial_fkt[0]
    vD = solution_space.trial_fkt[1]
    p = solution_space.trial_fkt[2]

    u_n = solution_space.uh_n[0]

    # --- Test functions
    v_u = solution_space.test_fkt[0]
    v_vD = solution_space.test_fkt[1]
    v_p = solution_space.test_fkt[2]

    # --- Kinematics
    I = ufl.Identity(len(u))

    # The deformation gradient
    FtS = I + ufl.grad(u)
    JtS = ufl.det(FtS)

    inv_tFtS = ufl.inv(ufl.transpose(FtS))

    # Right Cauchy-Green deformation tensor
    CtS = ufl.dot(ufl.transpose(FtS), FtS)

    # Solid velocity
    vtS_t_dt = u - u_n
    divxtS_t_dt = JtS * ufl.inner(ufl.grad(vtS_t_dt), inv_tFtS)

    # --- Constitutives
    # Stress
    PhSE = (
        mhS * ufl.dot((I - ufl.inv(CtS)), inv_tFtS)
        + pi_1 * lhS * ufl.ln(JtS) * inv_tFtS
    )
    P = PhSE - JtS * p * inv_tFtS

    # --- Governing equations
    dv = problem.domain.dv
    dt = problem.ufl_dt

    res_BLM = ufl.inner(P, ufl.grad(v_u)) * dv
    res_BMO = (divxtS_t_dt * v_p + dt * ufl.div(vD) * v_p) * dv
    res_vD = (
        dt
        * ((1 / (JtS * ktD)) * ufl.inner(ufl.dot(vD, CtS), v_vD) - p * ufl.div(v_vD))
        * dv
    )

    if body_force is not None:
        # Check in initial volume fraction is set
        if nhStS0S is None:
            raise ValueError("Initial volume fraction required!")

        # Evaluate the real density
        density = rhohFR + nhStS0S * (pi_2 * rhohSR - rhohFR)

        # Add volumetric forces to BLM
        res_BLM -= JtS * (pi_3 * density * ufl.inner(body_force, v_u)) * dv

        # Add volumetric contribution of seepage velocity
        res_vD -= dt * pi_3 * rhohFR * ufl.dot(body_force, FtS)

    return res_BLM, res_BMO, res_vD


# --- Poroelasticity without mass exchange ---
# --- u-p formulation
def goveq_TPM_nomex_up(problem: FemProblem):
    # --- Getters
    # Volumetric terms
    nhStS0S = problem.material.get_volumetric_term_ufl(VolumeTerms.nhSt0S)

    # --- The weak form
    res_BLM, res_BMO = goveq_TPM_up_base(problem, nhStS0S)

    # Set predators for RHS
    problem.prefactors_natural_bc[0] = -1.0
    problem.prefactors_natural_bc[1] = -1.0

    # Add volumetric contributions of weak form
    problem.weak_form += res_BLM + res_BMO


# --- u-vD-p formulation
def goveq_TPM_nomex_uvp(problem: FemProblem):
    # --- Getters
    # Volumetric terms
    nhStS0S = problem.material.get_volumetric_term_ufl(VolumeTerms.nhSt0S)

    # --- The weak form
    res_BLM, res_BMO, res_vD = goveq_TPM_upv_base(problem, nhStS0S)

    # Set predators for RHS
    problem.prefactors_natural_bc[0] = -1.0
    problem.prefactors_natural_bc[1] = 0.0
    problem.prefactors_natural_bc[2] = -problem.ufl_dt

    # Add volumetric contributions of weak form
    problem.weak_form += res_BLM + res_BMO + res_vD


# --- Poroelasticity with mass exchange ---
# --- u-p-n formulation ---
def goveq_TPM_mex_upn(problem: FemProblem):
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
    I = ufl.Identity(len(u))

    # The deformation gradient
    FtS = I + ufl.grad(u)
    JtS = ufl.det(FtS)

    # Velocities
    vtS_t_dt = u - u_n
    divxtS_t_dt = JtS * ufl.inner(ufl.grad(vtS_t_dt), ufl.inv(ufl.transpose(FtS)))
    DnhSDt_t_dt = nhS - nhS_n

    # --- Governing equations
    dv = problem.domain.dv

    res_BLM, res_BMO = goveq_TPM_up_base(problem, JtS * nhS)
    res_BMS = (DnhSDt_t_dt + nhS * divxtS_t_dt) * v_n * dv

    if rhohathS is not None:
        pi_t_rht_t_dt = (pi_4 / pi_2) * problem.ufl_dt * rhohathS

        res_BMO -= JtS * ((1 / rhohSR) - pi_2 * (1 / rhohFR)) * pi_t_rht_t_dt * v_p * dv
        res_BMS -= (1 / rhohSR) * pi_t_rht_t_dt * v_n * dv

    # Set predators for RHS
    problem.prefactors_natural_bc[0] = -1.0
    problem.prefactors_natural_bc[1] = -1.0

    # Add volumetric contributions of weak form
    problem.weak_form += res_BLM + res_BMO + res_BMS


# --- u-vD-p-n formulation ---
def goveq_TPM_mex_uvpn(problem: FemProblem):
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
    nhS = solution_space.trial_fkt[3]

    u_n = solution_space.uh_n[0]
    nhS_n = solution_space.uh_n[3]

    # --- Test functions
    v_p = solution_space.test_fkt[2]
    v_n = solution_space.test_fkt[3]

    # --- Kinematics
    I = ufl.Identity(len(u))

    # The deformation gradient
    FtS = I + ufl.grad(u)
    JtS = ufl.det(FtS)

    # Velocities
    vtS_t_dt = u - u_n
    divxtS_t_dt = ufl.inner(ufl.grad(vtS_t_dt), ufl.inv(ufl.transpose(FtS)))

    DnhSDt_t_dt = nhS - nhS_n

    # --- Governing equations
    dv = problem.domain.dv

    res_BLM, res_BMO, res_vD = goveq_TPM_upv_base(problem, nhS)
    res_BMS = (DnhSDt_t_dt + nhS * divxtS_t_dt) * v_n * dv

    if rhohathS is not None:
        pi_t_rht_t_dt = (pi_4 / pi_2) * problem.ufl_dt * rhohathS

        res_BMO -= JtS * ((1 / rhohSR) - pi_2 * (1 / rhohFR)) * pi_t_rht_t_dt * v_p * dv
        res_BMS -= (1 / rhohSR) * pi_t_rht_t_dt * v_n * dv

    # Set predators for RHS
    problem.prefactors_natural_bc[0] = -1.0
    problem.prefactors_natural_bc[1] = 0.0
    problem.prefactors_natural_bc[2] = -problem.ufl_dt

    # Add volumetric contributions of weak form
    problem.weak_form += res_BLM + res_BMO + res_vD + res_BMS
