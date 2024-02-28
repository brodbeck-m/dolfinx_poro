"""
Testcase: Terzaghi problem

Solves a (quasi 1D) consolidation problem on a rectangular domain, 
where a traction boundary is applied on the drained top boundary.

"""

# --- Imports ---
from enum import Enum
import numpy as np
import typing

import dolfinx
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
from dolfinx_poro.fem import (
    Domain,
    DirichletBC,
    DirichletFunction,
    set_finite_element_problem,
)

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
    elif primary_variables == PrimaryVariables.usigpv_ls1:
        # Initialise essential boundary conditions
        fem_problem.initialise_essential_bc(
            DirichletTerzaghiUSigPW(
                fem_problem.domain,
                fem_problem.solution_space.function_space,
                fem_problem.solution_space.sub_function_spaces,
                -q_top,
            )
        )
    else:
        raise ValueError("Unknown formulation type.")

    # Set initial conditions
    if primary_variables == PrimaryVariables.up:
        fem_problem.initialise_solution([0], [0])
    elif primary_variables == PrimaryVariables.upn:
        fem_problem.initialise_solution([0, initcond_nhSt0S], [0, 2])
    elif primary_variables == PrimaryVariables.usigpv_ls1:
        fem_problem.initialise_solution([0], [0])

    return fem_problem


# --- The Dirichlet boundary conditions ---
# --- u_p_n resp. u_p_pt_n formulation
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


# --- 4-field formulation based on u, sigma, p and nhFwtFS
class TractionTop_Terzaghi4Field(DirichletFunction):
    def __init__(self, subspace: int, is_timedependent: bool, traction_top: float):
        super().__init__(subspace, is_timedependent)

        # The traction on the top boundary
        self.traction_top = traction_top

    def __call__(self, x):
        traction = np.zeros((2, x.shape[1]), dtype=dolfinx.default_scalar_type)
        traction[0] = 0
        traction[1] = self.traction_top

        return traction


class DirichletTerzaghiUSigPW(DirichletBC):
    def __init__(
        self,
        domain: Domain,
        V: dfem.FunctionSpace,
        V_sub: typing.List[dfem.FunctionSpace],
        traction_top: float,
    ):
        # The traction on the top boundary
        self.traction_top = traction_top

        # Constructor of base class
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
        self.initialise_dirichlet_values(
            V_sub[1],
            dirichlet_function=TractionTop_Terzaghi4Field(1, False, self.traction_top),
        )

        self.initialise_dirichlet_values(V_sub[3], const_value=0.0, id_subspace=3)
        self.initialise_dirichlet_values(V_sub[4], const_value=0.0, id_subspace=4)

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

        # --- Traction BCs
        # Left
        facets = fct_fkts.indices[fct_fkts.values == 1]
        dofs = dfem.locate_dofs_topological((V.sub(2), V_sub[2]), 1, facets)
        self.list.append(dfem.dirichletbc(self.uD[1], dofs, V.sub(2)))

        # Bottom
        facets = fct_fkts.indices[fct_fkts.values == 2]
        dofs = dfem.locate_dofs_topological((V.sub(1), V_sub[1]), 1, facets)
        self.list.append(dfem.dirichletbc(self.uD[1], dofs, V.sub(1)))

        # Right
        facets = fct_fkts.indices[fct_fkts.values == 3]
        dofs = dfem.locate_dofs_topological((V.sub(2), V_sub[2]), 1, facets)
        self.list.append(dfem.dirichletbc(self.uD[1], dofs, V.sub(2)))

        # Top: Traction in y direction
        facets = fct_fkts.indices[fct_fkts.values == 4]

        dofs = dfem.locate_dofs_topological((V.sub(1), V_sub[1]), 1, facets)
        self.list.append(dfem.dirichletbc(self.uD[1], dofs, V.sub(1)))

        dofs = dfem.locate_dofs_topological((V.sub(2), V_sub[2]), 1, facets)
        self.list.append(dfem.dirichletbc(self.uD[2], dofs, V.sub(2)))

        # --- Pressure BCs
        # Top: Outflow
        facets = fct_fkts.indices[fct_fkts.values == 4]
        dofs = dfem.locate_dofs_topological((V.sub(3), V_sub[3]), 1, facets)
        self.list.append(dfem.dirichletbc(self.uD[3], dofs, V.sub(3)))

        # --- Flow BCs
        # Left: No outflow
        facets = fct_fkts.indices[fct_fkts.values == 1]
        dofs = dfem.locate_dofs_topological((V.sub(4), V_sub[4]), 1, facets)
        self.list.append(dfem.dirichletbc(self.uD[4], dofs, V.sub(4)))

        # Bottom No outflow
        facets = fct_fkts.indices[fct_fkts.values == 2]
        dofs = dfem.locate_dofs_topological((V.sub(4), V_sub[4]), 1, facets)
        self.list.append(dfem.dirichletbc(self.uD[4], dofs, V.sub(4)))

        # Right: No outflow
        facets = fct_fkts.indices[fct_fkts.values == 3]
        dofs = dfem.locate_dofs_topological((V.sub(4), V_sub[4]), 1, facets)
        self.list.append(dfem.dirichletbc(self.uD[4], dofs, V.sub(4)))


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
    seep_velocity = 3
    stress = 4


class AnalyticSolution:
    def __init__(
        self,
        pi_1: float,
        pi_2: float,
        geom_L: float,
        bc_qtop: float,
        z: typing.Optional[np.ndarray] = None,
        n_coeffs: typing.Optional[int] = 100,
    ) -> None:
        # Counters
        self.n_points = z.shape[0]
        self.n_coeffs = n_coeffs

        # Points
        self.position = z

        # Source term BMS
        self.pi_1 = pi_1
        self.pi_2 = pi_2

        # --- Coefficients
        Lhm1 = 1 / geom_L
        h1 = 1 / (2 + pi_1)
        f0 = bc_qtop

        # Initialise storage
        self.Nn = (pi_1 + 2) * (
            (0.5 * Lhm1 * np.pi * (1 + 2 * np.arange(n_coeffs))) ** 2
        )

        self.pn = np.zeros(n_coeffs)
        self.pndt = np.zeros(n_coeffs)

        self.p_z = np.zeros((self.n_points, n_coeffs))
        self.u_z = np.zeros((self.n_points, n_coeffs))
        self.uf_z = np.zeros(self.n_points)
        self.dpdz_z = np.zeros((self.n_points, n_coeffs))
        self.dudz_z = np.zeros((self.n_points, n_coeffs))

        # Set (constant) values
        # (Reverse z-axis to compare with dolfinx_poro)
        fctrn = 0.5 * Lhm1 * np.pi * (1 + 2 * np.arange(n_coeffs))
        mfctrn_pm1 = (2 * f0) / (fctrn * geom_L)
        mfctrn_pm2 = (2 * f0 * Lhm1) / (fctrn**2)

        for i in range(self.n_points):
            z_i = -z[i] + geom_L

            self.p_z[i, :] = np.multiply(mfctrn_pm1, np.sin(fctrn * z_i))
            self.dpdz_z[i, :] = -2 * f0 * Lhm1 * np.cos(fctrn * z_i)

            self.u_z[i, :] = -h1 * np.multiply(mfctrn_pm2, np.cos(fctrn * z_i))
            self.dudz_z[i, :] = -h1 * np.multiply(mfctrn_pm1, np.sin(fctrn * z_i))

        self.uf_z = -h1 * f0 * z[:]

        # --- Results
        # Field quantities
        self.field_var = None
        self.results_mass = None

    # --- Fourier coefficients of the pressure field ---
    def calculate_pn(self, time: float) -> None:
        self.pn[:] = np.exp(-self.Nn * time)

    # --- Outflow over top-surface ---
    def calculate_outflux(self, time: float) -> float:
        raise NotImplementedError

    def calculate_outflow(self, time: float):
        raise NotImplementedError

    # --- Solutions ---
    def evaluate_solution(
        self,
        nhSt0S: float,
        times: typing.Union[typing.List, np.ndarray],
        evaluate_mass_error: typing.Optional[float] = False,
        out_name: typing.Optional[str] = None,
        supress_output: typing.Optional[bool] = False,
        map_seepage_velocity: typing.Optional[bool] = True,
    ) -> None:
        # Initialise storage
        self.field_var = np.zeros((self.n_points, 1 + len(times), 5))

        if evaluate_mass_error:
            raise NotImplementedError

        # Evaluate solution
        header = "z"

        for n, time in enumerate(times):
            # Evaluate (time dependent) coefficients
            self.calculate_pn(time)

            # Evaluate displacement
            self.field_var[:, n + 1, 0] = self.uf_z - np.dot(self.u_z, self.pn)

            # Evaluate pressure
            self.field_var[:, n + 1, 1] = np.dot(self.p_z, self.pn)

            # Evaluate volume fraction
            self.field_var[:, n + 1, 2] = nhSt0S / (1 + np.dot(self.dudz_z, self.pn))

            # Evaluate total stress
            # TODO - Add total stress

            # Evaluate seepage velocity
            if map_seepage_velocity:
                self.field_var[:, n + 1, 3] = np.multiply(
                    1 / (1 + np.dot(self.dudz_z, self.pn)),
                    -np.dot(self.dpdz_z, self.pn),
                )
            else:
                self.field_var[:, n + 1, 3] = -np.dot(self.dpdz_z, self.pn)

            # Evaluate mass error
            if evaluate_mass_error:
                # Set time to output
                self.results_mass[n + 1, AnalyticResults.time.value] = time

                # Evaluate displacement at z=1
                if np.isclose(self.position[-1], 1):
                    self.results_mass[n + 1, AnalyticResults.u_max.value] = (
                        self.field_var[-1, n + 1, AnalyticFields.displacement.value]
                    )
                else:
                    raise ValueError("z=1 not in position array!")

                # Evaluate pressure at z=0
                if np.isclose(self.position[0], 0):
                    self.results_mass[n + 1, AnalyticResults.p_max.value] = (
                        self.field_var[0, n + 1, AnalyticFields.pressure.value]
                    )
                else:
                    raise ValueError("z=1 not in position array!")

                # Evaluate solid mass

                # Evaluate total mass

                # Evaluate outflow

                # Evaluate mass error (internal + outflow is initial!)
                # Outflow is positive when it leaves the domain!

                raise NotImplementedError

            # Update header
            header += "," + str(time)

        # Export to file
        if not supress_output:
            # Set default name for output
            if out_name is None:
                out_name = "growth_1d-analytic_solution"

            # Export primal field variables
            for i, pvar in enumerate(["u", "p", "nhS", "nhFwtFS"]):
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
