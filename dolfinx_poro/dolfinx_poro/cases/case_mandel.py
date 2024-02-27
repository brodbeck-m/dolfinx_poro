"""
Testcase: Mandel problem

Solves the Mandel problem of por-elasticity. 
In order to ensure the horizontal top surface, 
the traction BC is replaced by an (analytic) 
displacement (see [1]).

The domain:

                   4 
     -     |---------------|
     |     |               |          
     |     |               |          h -> hight
   h |   1 |               | 3        w -> width
     |     |               |
     |     |               |
     -     |---------------|            
                   2
     
           '---------------'
                   a

The boundary conditions:
    1: u = 0, nhFwtFS x n = 0
    2: v = 0, nhFwtFS x n = 0
    3: p = 0
    4: (u, v) = (e_ext, v_ext), nhFwtFS x n = 0

[1] van Duijn C. J. and Mikelic A.: Mathematical proof of 
    the mandel-cryer effect in poroelasticity. (2021)
"""

# --- Imports ---
from enum import Enum
import numpy as np
from scipy import integrate
import typing

import dolfinx
import dolfinx.fem as dfem
import ufl

from dolfinx_poro import (
    EquationType,
    PrimaryVariables,
    FeSpaces,
    set_default_spaces,
    set_default_orders,
    set_discretisation,
    set_weakform,
    scale_primal_variables,
)

from dolfinx_poro.poroelasticity import NondimPar
from dolfinx_poro.material import AbstractMaterialPoro
from dolfinx_poro.fem import (
    Domain,
    DirichletFunction,
    DirichletBC,
    set_finite_element_problem,
)

from .mesh_generation import create_geometry_rectangle


# --- Setter function ---
def setup_calculation(
    equation_type: EquationType,
    primary_variables: PrimaryVariables,
    material: typing.Type[AbstractMaterialPoro],
    pif_top: float,
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
        domain_geometry = [0.5, 0.5]

    if sdisc_fetype is None:
        sdisc_fetype = set_default_spaces(primary_variables)

    if sdisc_eorder is None:
        sdisc_eorder = set_default_orders(primary_variables, sdisc_fetype)

    if sdisc_nelmt is None:
        sdisc_nelmt = [8, 40]

    if not material.nondimensional_form:
        raise ValueError("Mandel problem only in non-dimensional form supported")

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
    dirichlet_function = TopDisplMandel(pif_top, material.get_pi(NondimPar.pi_1))

    if (primary_variables == PrimaryVariables.upn) or (
        primary_variables == PrimaryVariables.upptn
    ):
        # Initialise essential boundary conditions
        fem_problem.initialise_essential_bc(
            DirichletMandelUPN(
                fem_problem.domain,
                fem_problem.solution_space.function_space,
                fem_problem.solution_space.sub_function_spaces,
                dirichlet_function,
            )
        )
    else:
        raise ValueError("Unknown formulation type.")

    # Set initial conditions
    if primary_variables == PrimaryVariables.upn:
        fem_problem.initialise_solution([0, initcond_nhSt0S], [0, 2])
    elif primary_variables == PrimaryVariables.upptn:
        fem_problem.initialise_solution([0, 0, initcond_nhSt0S], [1, 2, 3])
    elif primary_variables == PrimaryVariables.uvpn:
        fem_problem.initialise_solution([0, initcond_nhSt0S], [0, 3])
    else:
        raise ValueError("Unknown formulation type.")

    return fem_problem


# --- The Dirichlet boundary conditions ---
class TopDisplMandel(DirichletFunction):
    def __init__(
        self,
        pi_f: typing.Optional[float] = None,
        pi_1: typing.Optional[float] = None,
    ):
        # Default constructor
        super().__init__(0, True)

        # Initialise analytic solution
        self.analytic_solution = AnalyticSolution(pi_1, pi_f, n_coeffs=500)

    def __call__(self, x):
        # Evaluate top displacement
        u_top = 0.01

        # Return the external displacement
        uD = np.zeros((2, x.shape[1]), dtype=dolfinx.default_scalar_type)
        uD[0] = 0
        uD[1] = -u_top

        return uD


class DirichletMandelUPN(DirichletBC):
    def __init__(
        self,
        domain: Domain,
        V: dfem.FunctionSpace,
        V_sub: typing.List[dfem.FunctionSpace],
        bfunc_utop: TopDisplMandel,
    ):
        # Default constructor
        super().__init__(domain, V, V_sub)

        # Initialise analytic solution
        self.initialise_dirichlet_values(V_sub[0], dirichlet_function=bfunc_utop)

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
        self.list.append(dfem.dirichletbc(self.uD[1], dofs, V.sub(0)))

        # Boundary 2: No vertical displacement
        facets = fct_fkts.indices[fct_fkts.values == 2]
        dofs = dfem.locate_dofs_topological(
            (V.sub(0).sub(1), V_sub[0].sub(1)), 1, facets
        )
        self.list.append(dfem.dirichletbc(self.uD[1], dofs, V.sub(0)))

        # Boundary 4: Prescribe analytic solution
        facets = fct_fkts.indices[fct_fkts.values == 4]
        dofs = dfem.locate_dofs_topological((V.sub(0), V_sub[0]), 1, facets)
        self.list.append(dfem.dirichletbc(self.uD[0], dofs, V.sub(0)))

        # --- Pressure BCs
        # Boundary 3: Outflow
        facets = fct_fkts.indices[fct_fkts.values == 3]
        dofs = dfem.locate_dofs_topological((V.sub(1), V_sub[1]), 1, facets)
        self.list.append(dfem.dirichletbc(self.uD[2], dofs, V.sub(1)))


# --- Analytic solution ---
class AnalyticResults(Enum):
    time = 0
    u_max = 1
    p_max = 2
    mass_solid = 3
    mass_total = 4
    outflow = 5
    error = 6


class AnalyticFields(Enum):
    displacement = 0
    pressure = 1
    volume_fraction = 2
    stress = 3
    seep_velocity = 4


class AnalyticSolution:
    def __init__(
        self,
        pi_1: float,
        pi_f: float,
        z: typing.Optional[np.ndarray] = None,
        n_coeffs: typing.Optional[int] = 100,
    ) -> None:
        # Counters
        self.n_points = z.shape[0]
        self.n_coeffs = n_coeffs

        # Points
        if z is not None:
            self.position = z
        else:
            raise ("Advanced quadrature rule not implemented")

        # Source term BMS
        self.pi_1 = pi_1
        self.pi_f = pi_f

        # --- Coefficients

        # --- Results
        # Field quantities
        self.field_var = None
        self.results_mass = None

    # --- Time dependent part of the Fourier coefficients ---
    def calculate_timefactors(self, time: float) -> None:
        raise NotImplementedError

    # --- Outflow over top-surface ---
    def calculate_outflux(self, time: float) -> float:
        raise NotImplementedError

    # --- Solutions ---
    # Volume faction
    def evaluate_nhS(
        self,
        nhSt0S: float,
        times: typing.Union[typing.List, np.ndarray],
    ):
        raise NotImplementedError

    # Top displacement
    def evaluate_uz_top(self, time: float):
        raise NotImplementedError

    # Evaluate all quantities
    def evaluate_solution(
        self,
        nhSt0S: float,
        times: typing.Union[typing.List, np.ndarray],
        evaluate_mass_error: typing.Optional[float] = False,
        out_name: typing.Optional[str] = None,
        supress_output: typing.Optional[bool] = False,
    ) -> None:
        # Initialise storage
        self.field_var = np.zeros((self.n_points, 1 + len(times), 5))

        if evaluate_mass_error:
            initial_mass = 1 + (self.pi_2 - 1) * nhSt0S
            self.results_mass = np.zeros((len(times) + 1, 7))

            # Set initial mass
            self.results_mass[0, AnalyticResults.mass_solid.value] = self.pi_2 * nhSt0S
            self.results_mass[0, AnalyticResults.mass_total.value] = initial_mass

        # Numerical integration of outflow
        # TODO - Add calculation of outflow

        # Evaluate solution
        header = "z"

        for n, time in enumerate(times):
            # Evaluate (time dependent) coefficients
            self.calculate_timefactors()

            # Evaluate displacement
            self.field_var[:, n + 1, 0] = np.dot(self.u_z, self.pn)

            # Evaluate pressure
            self.field_var[:, n + 1, 1] = np.dot(self.p_z, self.pn)

            # Evaluate volume fraction
            # TODO - Add evaluation of volume fraction

            # Evaluate total stress
            # TODO - Add total stress

            # Evaluate seepage velocity
            self.field_var[:, n + 1, 4] = np.multiply(
                1 / (1 + np.dot(self.dudz_z, self.pn)),
                -np.dot(self.dpdz_z, self.pn),
            )

            # Evaluate mass error
            if evaluate_mass_error:
                # Set time to output
                self.results_mass[n + 1, AnalyticResults.time.value] = time

                # Evaluate displacement at x=1
                if np.isclose(self.position[-1], 1):
                    self.results_mass[n + 1, AnalyticResults.u_max.value] = (
                        self.field_var[-1, n + 1, AnalyticFields.displacement.value]
                    )
                else:
                    raise ValueError("x=1 not in position array!")

                # Evaluate pressure at x=0
                if np.isclose(self.position[0], 0):
                    self.results_mass[n + 1, AnalyticResults.p_max.value] = (
                        self.field_var[0, n + 1, AnalyticFields.pressure.value]
                    )
                else:
                    raise ValueError("z=1 not in position array!")

                # Evaluate solid mass
                # TODO - Implement evaluation of solid mass
                self.results_mass[n + 1, AnalyticResults.mass_solid.value] = 0

                # Evaluate total mass
                # TODO - Implement evaluation of total mass
                self.results_mass[n + 1, AnalyticResults.mass_total.value] = 0

                # Evaluate outflow
                #  TODO - Implement evaluation of outflow
                self.results_mass[n + 1, AnalyticResults.outflow.value] = 0

                # Evaluate mass error (internal + outflow is initial!)
                # Outflow is positive when it leaves the domain!
                diff_mass = (
                    self.results_mass[n + 1, AnalyticResults.mass_total.value]
                    + self.results_mass[n + 1, AnalyticResults.outflow.value]
                )
                self.results_mass[n + 1, AnalyticResults.error.value] = 100 * (
                    diff_mass / initial_mass - 1
                )

            # Update header
            header += "," + str(time)

        # Export to file
        if not supress_output:
            # Set default name for output
            if out_name is None:
                out_name = "mandel-analytic_solution"

            # Export primal field variables
            for i, pvar in enumerate(["u", "p", "nhS"]):
                # Store positions
                self.field_var[:, 0, i] = self.position

                # Export results
                np.savetxt(
                    out_name + "_pvar-" + pvar + ".csv",
                    self.field_var[:, :, i],
                    delimiter=",",
                    header=header,
                )

            # Export mass error
            if evaluate_mass_error:
                np.savetxt(
                    out_name + "_error-mass.csv",
                    self.results_mass,
                    delimiter=",",
                    header="time, u_max, p_max, mass_solid, mass_total, outflux, error [%]",
                )
