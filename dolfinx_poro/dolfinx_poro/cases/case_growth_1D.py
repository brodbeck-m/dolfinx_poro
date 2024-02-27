"""
Testcase: 1D growth problem

Solves a 1D problem on a rectangular domain. Quasi 1D BCs are applied,
outflow is allowed on surface 4.

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
                   w

The boundary conditions:
    1: u = 0, nhFwtFS x n = 0
    2: v = 0, nhFwtFS x n = 0
    3: u = 0, nhFwtFS x n = 0
    4: p = 0

"""

# --- Imports ---
from enum import Enum
import numpy as np
from scipy import integrate
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
        domain_geometry = [0.2, 1]

    if sdisc_fetype is None:
        sdisc_fetype = set_default_spaces(primary_variables)

    if sdisc_eorder is None:
        sdisc_eorder = set_default_orders(primary_variables, sdisc_fetype)

    if sdisc_nelmt is None:
        sdisc_nelmt = [8, 40]

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
    if (primary_variables == PrimaryVariables.upn) or (
        primary_variables == PrimaryVariables.upptn
    ):
        # Initialise essential boundary conditions
        fem_problem.initialise_essential_bc(
            DirichletGrowth1DUPN(
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
        # Initialise essential boundary conditions
        fem_problem.initialise_essential_bc(
            DirichletGrowth1DUVPN(
                fem_problem.domain,
                fem_problem.solution_space.function_space,
                fem_problem.solution_space.sub_function_spaces,
                allow_outflow,
            )
        )

        # Initialise natural boundary conditions
        if traction_top is not None:
            # Set traction BC on top surface
            normal = ufl.FacetNormal(fem_problem.domain.mesh)
            fem_problem.initialise_natural_bc(traction_top * normal, 4, 0)

            # Weak pressure BC on top neglected as p is equal zero!
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
class DirichletGrowth1DUPN(DirichletBC):
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

        # Boundary 3: No horizontal displacement
        facets = fct_fkts.indices[fct_fkts.values == 3]
        dofs = dfem.locate_dofs_topological(
            (V.sub(0).sub(0), V_sub[0].sub(0)), 1, facets
        )
        self.list.append(dfem.dirichletbc(self.uD[0], dofs, V.sub(0)))

        # --- Pressure BCs
        if self.allow_outflow:
            # Boundary 4: Outflow
            facets = fct_fkts.indices[fct_fkts.values == 4]
            dofs = dfem.locate_dofs_topological((V.sub(1), V_sub[1]), 1, facets)
            self.list.append(dfem.dirichletbc(self.uD[1], dofs, V.sub(1)))


class DirichletGrowth1DUVPN(DirichletBC):
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

        # Boundary 3: No horizontal displacement
        facets = fct_fkts.indices[fct_fkts.values == 3]
        dofs = dfem.locate_dofs_topological(
            (V.sub(0).sub(0), V_sub[0].sub(0)), 1, facets
        )
        self.list.append(dfem.dirichletbc(self.uD[0], dofs, V.sub(0)))

        # --- Flow BCs
        # Boundary 1: No outflow
        facets = fct_fkts.indices[fct_fkts.values == 1]
        dofs = dfem.locate_dofs_topological((V.sub(1), V_sub[1]), 1, facets)
        self.list.append(dfem.dirichletbc(self.uD[1], dofs, V.sub(1)))

        # Boundary 2: No outflow
        facets = fct_fkts.indices[fct_fkts.values == 2]
        dofs = dfem.locate_dofs_topological((V.sub(1), V_sub[1]), 1, facets)
        self.list.append(dfem.dirichletbc(self.uD[1], dofs, V.sub(1)))

        # Boundary 3: No outflow
        facets = fct_fkts.indices[fct_fkts.values == 3]
        dofs = dfem.locate_dofs_topological((V.sub(1), V_sub[1]), 1, facets)
        self.list.append(dfem.dirichletbc(self.uD[1], dofs, V.sub(1)))

        if not self.allow_outflow:
            # Boundary 4: Outflow
            facets = fct_fkts.indices[fct_fkts.values == 4]
            dofs = dfem.locate_dofs_topological((V.sub(1), V_sub[1]), 1, facets)
            self.list.append(dfem.dirichletbc(self.uD[1], dofs, V.sub(1)))


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
        pi_2: float,
        pi_4: float,
        pi_5: float,
        z: typing.Optional[np.ndarray] = None,
        end_time_growth: typing.Optional[float] = None,
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

        # End time growth-process (thereafter only consolidation)
        if end_time_growth is None:
            self.end_time_growth = 0
        else:
            self.end_time_growth = end_time_growth

        # Source term BMS
        self.pi_1 = pi_1
        self.pi_2 = pi_2
        self.pi_4 = pi_4
        self.pi_5 = pi_5

        # --- Coefficients
        # Initialise storage
        self.Nn = (pi_1 + 2) * ((0.5 * np.pi * (1 + 2 * np.arange(n_coeffs))) ** 2)

        self.pn = np.zeros(n_coeffs)
        self.pndt = np.zeros(n_coeffs)

        self.p_z = np.zeros((self.n_points, n_coeffs))
        self.u_z = np.zeros((self.n_points, n_coeffs))
        self.dpdz_z = np.zeros((self.n_points, n_coeffs))
        self.dudz_z = np.zeros((self.n_points, n_coeffs))

        # Set (constant) values
        # (Reverse z-axis to compare with dolfinx_poro)
        h_1 = 2 * (1 - pi_2) * (pi_4 * pi_5 / pi_2)
        h_2 = h_1 / (pi_1 + 2)

        fctrn = 0.5 * np.pi * (1 + 2 * np.arange(n_coeffs))
        fctrn_pm2 = 1 / (fctrn**2)
        fctrn_pm3 = 1 / (fctrn**3)
        fctrn_pm4 = 1 / (fctrn**4)

        for i in range(self.n_points):
            z_i = -z[i] + 1

            self.p_z[i, :] = h_1 * np.multiply(fctrn_pm3, np.sin(fctrn * z_i))
            self.dpdz_z[i, :] = -h_1 * np.multiply(fctrn_pm2, np.cos(fctrn * z_i))

            self.u_z[i, :] = h_2 * np.multiply(fctrn_pm4, np.cos(fctrn * z_i))
            self.dudz_z[i, :] = h_2 * np.multiply(fctrn_pm3, np.sin(fctrn * z_i))

        # --- Results
        # Field quantities
        self.field_var = None
        self.results_mass = None

    # --- Fourier coefficients of the pressure field ---
    def calculate_pn(self, time: float) -> None:
        if time - self.end_time_growth <= 0:
            self.pn[:] = 1 - np.exp(-self.Nn * time)
        else:
            dt_c = time - self.end_time_growth
            self.pn[:] = np.exp(-self.Nn * dt_c) - np.exp(-self.Nn * time)

    def calculate_pndt(self, time: float) -> None:
        if time - self.end_time_growth <= 0:
            self.pndt[:] = np.multiply(self.Nn, np.exp(-self.Nn * time))
        else:
            dt_c = time - self.end_time_growth
            self.pndt[:] = np.multiply(
                self.Nn, np.exp(-self.Nn * time) - np.exp(-self.Nn * dt_c)
            )

    # --- Outflow over top-surface ---
    def calculate_outflux(self, time: float) -> float:
        h1 = 2 * (1 - self.pi_2) * ((self.pi_4 * self.pi_5) / self.pi_2)

        if time - self.end_time_growth <= 0:
            return h1 * np.sum(
                np.multiply(
                    (1 / self.Nn) * (self.pi_1 + 2), 1 - np.exp(-self.Nn * time)
                )
            )
        else:
            dt_c = time - self.end_time_growth

            return h1 * np.sum(
                np.multiply(
                    (1 / self.Nn) * (self.pi_1 + 2),
                    np.exp(-self.Nn * dt_c) - np.exp(-self.Nn * time),
                )
            )

    def calculate_outflow(self, time: float):
        h1 = (
            2
            * (1 - self.pi_2)
            * ((self.pi_4 * self.pi_5) / self.pi_2)
            / (self.pi_1 + 2)
        )

        if time - self.end_time_growth <= 0:
            return h1 * np.sum(
                np.multiply(
                    (1 / (self.Nn**2)) * ((self.pi_1 + 2) ** 2),
                    np.exp(-self.Nn * time) + self.Nn * time - 1,
                )
            )
        else:
            dt_c = time - self.end_time_growth

            return h1 * np.sum(
                np.multiply(
                    (1 / (self.Nn**2)) * ((self.pi_1 + 2) ** 2),
                    np.exp(-self.Nn * time)
                    - np.exp(-self.Nn * dt_c)
                    + self.Nn * self.end_time_growth,
                )
            )

    # --- Solutions ---
    # Volume faction
    def calculate_dndt(self, time: float, nhS: np.ndarray):
        # Evaluate time-derivative of p
        self.calculate_pndt(time)

        if time - self.end_time_growth < 0:
            # Source term
            source = self.pi_4 * self.pi_5 / self.pi_2

            return (
                np.multiply(-nhS / (self.pi_1 + 2), np.dot(self.p_z, self.pndt))
                + source
            )
        else:
            return np.multiply(-nhS / (self.pi_1 + 2), np.dot(self.p_z, self.pndt))

    def evaluate_nhS(
        self,
        nhSt0S: float,
        times: typing.Union[typing.List, np.ndarray],
        end_time_growth: typing.Optional[float] = None,
    ):
        # Reset growth time
        if end_time_growth is not None:
            self.end_time_growth = end_time_growth

        # Solve ODE for nhS (during growth)
        times_growth = [t for t in times if t < self.end_time_growth]
        times_growth.append(self.end_time_growth)
        results_nhS_dg = integrate.solve_ivp(
            self.calculate_dndt,
            (0, self.end_time_growth),
            y0=nhSt0S * np.ones(self.n_points),
            method="DOP853",
            rtol=1e-12,
            atol=1e-12,
            t_eval=times_growth,
            vectorized=False,
        )

        # Solve ODE for nhS (after growth)
        times_after_growth = [self.end_time_growth]
        times_after_growth.extend([t for t in times if t > self.end_time_growth])

        if len(times_after_growth) > 0:
            results_nhS_pg = integrate.solve_ivp(
                self.calculate_dndt,
                (self.end_time_growth, times[-1]),
                y0=results_nhS_dg.y[:, -1],
                method="DOP853",
                rtol=1e-12,
                atol=1e-12,
                t_eval=times_after_growth,
                vectorized=False,
            )

            return [results_nhS_dg, results_nhS_pg]
        else:
            return [results_nhS_dg]

    # Evaluate all quantities
    def evaluate_solution(
        self,
        nhSt0S: float,
        times: typing.Union[typing.List, np.ndarray],
        end_time_growth: typing.Optional[float] = None,
        evaluate_mass_error: typing.Optional[float] = False,
        out_name: typing.Optional[str] = None,
        supress_output: typing.Optional[bool] = False,
        map_seepage_velocity: typing.Optional[bool] = True,
    ) -> None:
        # Reset growth time
        if end_time_growth is not None:
            self.end_time_growth = end_time_growth

        # Initialise storage
        self.field_var = np.zeros((self.n_points, 1 + len(times), 5))

        if evaluate_mass_error:
            initial_mass = 1 + (self.pi_2 - 1) * nhSt0S
            self.results_mass = np.zeros((len(times) + 1, 7))

            # Set initial mass
            self.results_mass[0, AnalyticResults.mass_solid.value] = self.pi_2 * nhSt0S
            self.results_mass[0, AnalyticResults.mass_total.value] = initial_mass

        # Solve ODE for nhS (during growth)
        results_nhS = self.evaluate_nhS(nhSt0S, times)

        # Evaluate solution
        header = "z"
        if np.any(np.isclose(times, self.end_time_growth)):
            ntimes_growth = results_nhS[0].y.shape[1] - 1
        else:
            ntimes_growth = results_nhS[0].y.shape[1] - 2

        for n, time in enumerate(times):
            # Evaluate (time dependent) coefficients
            self.calculate_pn(time)

            # Evaluate displacement
            self.field_var[:, n + 1, 0] = np.dot(self.u_z, self.pn)

            # Evaluate pressure
            self.field_var[:, n + 1, 1] = np.dot(self.p_z, self.pn)

            # Evaluate volume fraction
            if (time < self.end_time_growth) or np.isclose(time, self.end_time_growth):
                self.field_var[:, n + 1, 2] = results_nhS[0].y[:, n]
            else:
                id = n - ntimes_growth
                self.field_var[:, n + 1, 2] = results_nhS[1].y[:, id]

            # Evaluate total stress
            # TODO - Add total stress

            # Evaluate seepage velocity
            if map_seepage_velocity:
                self.field_var[:, n + 1, 4] = np.multiply(
                    1 / (1 + np.dot(self.dudz_z, self.pn)),
                    -np.dot(self.dpdz_z, self.pn),
                )
            else:
                self.field_var[:, n + 1, 4] = -np.dot(self.dpdz_z, self.pn)

            # Evaluate mass error
            if evaluate_mass_error:
                # Set time to output
                self.results_mass[n + 1, AnalyticResults.time.value] = time

                # Evaluate displacement at z=1
                if np.isclose(self.position[-1], 1):
                    self.results_mass[
                        n + 1, AnalyticResults.u_max.value
                    ] = self.field_var[-1, n + 1, AnalyticFields.displacement.value]
                else:
                    raise ValueError("z=1 not in position array!")

                # Evaluate pressure at z=0
                if np.isclose(self.position[0], 0):
                    self.results_mass[
                        n + 1, AnalyticResults.p_max.value
                    ] = self.field_var[0, n + 1, AnalyticFields.pressure.value]
                else:
                    raise ValueError("z=1 not in position array!")

                # Evaluate solid mass
                self.results_mass[
                    n + 1, AnalyticResults.mass_solid.value
                ] = integrate.simpson(
                    np.multiply(
                        1 + (self.field_var[:, n + 1, 1] / (self.pi_1 + 2)),
                        self.pi_2 * self.field_var[:, n + 1, 2],
                    ),
                    self.position,
                )

                # Evaluate total mass
                self.results_mass[
                    n + 1, AnalyticResults.mass_total.value
                ] = integrate.simpson(
                    np.multiply(
                        1 + (self.field_var[:, n + 1, 1] / (self.pi_1 + 2)),
                        1 + (self.pi_2 - 1) * self.field_var[:, n + 1, 2],
                    ),
                    self.position,
                )

                # Evaluate outflow
                self.results_mass[
                    n + 1, AnalyticResults.outflow.value
                ] = self.calculate_outflow(time)

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
                out_name = "growth_1d-analytic_solution"

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
