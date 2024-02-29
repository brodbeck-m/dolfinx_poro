# --- Imports ---
from enum import Enum
import math
from petsc4py import PETSc
import typing

import basix
import dolfinx.fem as dfem
import dolfinx.fem.petsc as dfem_petsc
import dolfinx.io
import dolfinx.nls.petsc as dnls_petsc
import ufl
from mpi4py import MPI

from .auxiliaries import set_femconstant
from .domain import Domain
from .solution_space import SolutionSpace
from .abstract_material import AbstractMaterial
from .boundary_conditions import DirichletBC
from .post_processor import PostProcessor


# --- Setter function ---
def set_finite_element_problem(
    domain: Domain,
    solution_space: SolutionSpace,
    material: typing.Type[AbstractMaterial],
    equation_type: typing.Optional[typing.Type[Enum]] = None,
    primary_variables: typing.Optional[typing.Type[Enum]] = None,
    finite_elmt_spaces: typing.Optional[typing.Type[Enum]] = None,
):
    # Initialise material parameters
    material.initialise_parameters(domain.mesh)

    # Initialise finite-element problem
    if solution_space.is_linear:
        return FemProblemLinear(
            domain,
            solution_space,
            material,
            equation_type,
            primary_variables,
            finite_elmt_spaces,
        )
    else:
        return FemProblemNonLinear(
            domain,
            solution_space,
            material,
            equation_type,
            primary_variables,
            finite_elmt_spaces,
        )


# --- The FiniteElementProblem ---
class FemProblem:
    def __init__(
        self,
        domain: Domain,
        solution_space: SolutionSpace,
        material: typing.Type[AbstractMaterial],
        equation_type: typing.Optional[typing.Type[Enum]] = None,
        primary_variables: typing.Optional[typing.Type[Enum]] = None,
        finite_elmt_spaces: typing.Optional[typing.Type[Enum]] = None,
    ):
        # Problem specification
        self.equation_type = equation_type
        self.primary_variables = primary_variables
        self.finite_elmt_spaces = finite_elmt_spaces

        # The Domain
        self.domain = domain

        # The SolutionSpace
        self.solution_space = solution_space

        # The Material
        self.material = material

        # The (essential) boundary conditions
        self.dirichlet_bc = None

        # The natural boundary conditions
        self.time_dependent_natural_bc = False

        self.prefactors_natural_bc = self.solution_space.num_subspaces * [1.0]
        self.time_functions_natural_bc = []
        self.time_functions_ufl_natural_bc = []

        # The residual
        self.weak_form = 0

        # The solution time
        self.time = 0.0
        self.end_time = 1.0
        self.dt = 1.0

        self.ufl_time = set_femconstant(self.domain.mesh, self.time)
        self.ufl_dt = set_femconstant(self.domain.mesh, self.dt)

        # The calculation time
        self.calculation_time = 0.0
        self.num_iterations = 0

        # The ParaView output
        self.scale_output = False
        self.scaling_pvars = self.solution_space.num_subspaces * [1.0]
        self.scaling_time = 1.0

        self.paraview_files = []
        self.paraview_outpfunction = []

        self.pfields_require_interpolation = False
        self.intplt_functions = []
        self.intplt_info = []

    # --- Initialisation ---
    def initialise_essential_bc(self, dirichlet_bc: DirichletBC):
        self.dirichlet_bc = dirichlet_bc

    def initialise_natural_bc(
        self,
        value: typing.Any,
        boundary_id: int,
        subspace: typing.Optional[int] = 0,
        time_function: typing.Optional[typing.Callable] = None,
    ):
        # Initialise time function
        if time_function is not None:
            self.time_dependent_natural_bc = True

            self.time_functions_natural_bc.append(time_function)
            self.time_functions_ufl_natural_bc.append(
                set_femconstant(self.domain.mesh, 1.0)
            )

            tval = self.time_functions_ufl_natural_bc[-1]
        else:
            tval = 1.0

        # Set natural BC
        self.weak_form += (
            self.prefactors_natural_bc[subspace]
            * tval
            * ufl.inner(value, self.solution_space.test_fkt[subspace])
            * self.domain.ds(boundary_id)
        )

    def initialise_solution(
        self,
        uh_t0: typing.List[typing.Union[float, typing.Callable]],
        subspace: typing.List[int],
    ):
        # Check input
        if len(uh_t0) != len(subspace):
            raise ValueError("Length of initial conditions and subspace do not match")

        # Set initial conditions
        for u0, i in zip(uh_t0, subspace):
            # Set values
            if callable(u0):
                # Workaround see https://fenicsproject.discourse.group/t/error-interpolating-on-sub-of-vector-function/10313
                raise NotImplementedError("General conditions not implemented")
            else:
                self.solution_space.uh_n[i].x.array[:] = u0

        # Initailise load ramps
        if self.material.is_time_dependent:
            self.material.update_time(0.0)

        # Set identifier
        self.solution_space.initial_conditions_set = True

    def initialise_solver(residual: typing.Any):
        raise NotImplementedError("Solver initialisation not implemented")

    # --- Set scaling parameters ---
    def set_scaling_pvar(
        self, scaling_primary_variables: typing.List[float], scaling_time: float
    ):
        # Check input
        if len(scaling_primary_variables) != self.solution_space.num_subspaces:
            raise ValueError("Wrong shape of scaling parameters")

        # Set scaling parameters
        self.scaling_pvars = scaling_primary_variables
        self.scaling_time = scaling_time

        # Set identifier
        self.scale_output = True

    # --- Solve problem ---

    def solve_equation_system():
        raise NotImplementedError("Solution method not implemented")

    def solve_problem(
        self,
        dt: float,
        t_end: float,
        output_paraview: bool = False,
        output_name: str = None,
        post_processors: typing.Optional[
            typing.List[typing.Type[PostProcessor]]
        ] = None,
        simulation_series: typing.Optional[typing.List[int]] = None,
    ):
        # Initialise time discretisation
        self.time = 0.0
        self.ufl_time.value = 0.0

        self.dt = dt
        self.end_time = t_end
        self.ufl_dt.value = self.dt

        # Initialise solver
        self.initialise_solver()

        # Initialise post-processing
        if output_paraview:
            self.initialise_paraview(output_name)

        if post_processors is None:
            requires_post_processing = False
        else:
            requires_post_processing = True
            for post_proc in post_processors:
                post_proc.initialise_storage(self.end_time, self.dt)

        # Initialise output
        if simulation_series is None:
            simulation_series = [1, 1]

        # Time loop
        duration_solve = 0.0
        num_time_steps = 0

        for num_time_steps in range(1, math.ceil(self.end_time / self.dt) + 1):
            # Update time
            self.update_time()

            # Solve equation system
            duration_solve -= MPI.Wtime()
            n_iter = self.solve_equation_system()
            duration_solve += MPI.Wtime()

            self.num_iterations += n_iter

            print(
                "Phys. Time {:.4f}, Calc. Time {:.4f}, Num. Iter. {}, Sim. {}/{}".format(
                    self.time,
                    duration_solve,
                    n_iter,
                    simulation_series[0],
                    simulation_series[1],
                )
            )

            # History update
            if self.scale_output:
                self.solution_space.scale_solution(self.scaling_pvars)
            else:
                self.solution_space.update_history_field()

            # ParaView output
            if output_paraview:
                self.output_paraview()

            # Post-processing
            if requires_post_processing:
                for post_proc in post_processors:
                    post_proc(
                        self.time * self.scaling_time,
                        self.dt * self.scaling_time,
                        num_time_steps,
                    )

            # Undo scaling of history fields
            if self.scale_output:
                self.solution_space.update_history_field()

        if output_paraview:
            self.close_paraview()

        if requires_post_processing:
            for post_proc in post_processors:
                post_proc.write_data(output_name)

        # Store solution time
        self.calculation_time = duration_solve

    # --- Output ---
    def initialise_paraview(self, basename):
        # Reinitialise output
        self.paraview_files = []

        # Check input
        if basename is None:
            raise ValueError("Output name not specified")

        # Initialise output files
        count_intpl = 0

        for i, (uh, pvar_name) in enumerate(
            zip(self.solution_space.uh_n, self.solution_space.name_pvars)
        ):
            # Set output file
            filename = basename + "_pvar-" + pvar_name + ".bp"

            # Check if FE space requires projection
            if (
                uh.function_space.element.basix_element.family != basix.ElementFamily.P
                or uh.function_space.element.basix_element.discontinuous
            ):
                # Set id for interpolation
                self.pfields_require_interpolation = True
                self.intplt_info.append([i, count_intpl])

                # Create function_space into which the solution is interpolated
                # FIXME - Change spaces to DG if supported with VTX
                degree = uh.function_space.element.basix_element.degree

                if (uh.function_space.num_sub_spaces > 1) or (
                    uh.function_space.element.basix_element.value_size > 1
                ):
                    V = dfem.VectorFunctionSpace(self.domain.mesh, ("CG", 1))
                    self.intplt_functions.append(dfem.Function(V))
                else:
                    V = dfem.FunctionSpace(self.domain.mesh, ("CG", 1))
                    self.intplt_functions.append(dfem.Function(V))

                # Initialise output file
                self.paraview_files.append(
                    dolfinx.io.VTXWriter(
                        self.domain.mesh.comm,
                        filename,
                        [self.intplt_functions[-1]],
                        engine="BP4",
                    )
                )

                # Increment counter for interpolated functions
                count_intpl += 1
            else:
                # Initialise output file
                self.paraview_files.append(
                    dolfinx.io.VTXWriter(
                        self.domain.mesh.comm, filename, [uh], engine="BP4"
                    )
                )

    def output_paraview(self):
        # Scale time
        scaled_time = self.time * self.scaling_time

        if self.pfields_require_interpolation:
            for info in self.intplt_info:
                self.intplt_functions[info[1]].interpolate(
                    self.solution_space.uh_n[info[0]]
                )

        # Write output
        for outfile in self.paraview_files:
            # Write output
            outfile.write(scaled_time)

    def close_paraview(self):
        for outfile in self.paraview_files:
            outfile.close()

    def update_time(self):
        # Update current time
        self.time += self.dt
        self.ufl_time.value = self.time

        # Update load ramps
        if self.material.is_time_dependent:
            self.material.update_time(self.time)

        # Update boundary conditions
        if self.dirichlet_bc.is_timedependent:
            self.dirichlet_bc.update_time(self.time)

        if self.time_dependent_natural_bc:
            for fct, fct_ufl in zip(
                self.time_functions_natural_bc, self.time_functions_ufl_natural_bc
            ):
                fct_ufl.value = fct(self.time)


class FemProblemLinear(FemProblem):
    def __init__(
        self,
        domain: Domain,
        solution_space: SolutionSpace,
        material_parameters: typing.Any,
        equation_type: typing.Optional[typing.Type[Enum]] = None,
        primary_variables: typing.Optional[typing.Type[Enum]] = None,
        finite_elmt_spaces: typing.Optional[typing.Type[Enum]] = None,
    ):
        # Check if problem is linear
        if not solution_space.is_linear:
            raise ValueError("Non-linear solution-space initialised")

        # Constructor of the base class
        super().__init__(
            domain,
            solution_space,
            material_parameters,
            equation_type,
            primary_variables,
            finite_elmt_spaces,
        )

        # The equation system
        self.a = None
        self.l = None

        # The equation system
        self.A = None
        self.L = None

        # The problem solver
        self.solver = None

    # --- Initialisation ---
    def initialise_solver(self):
        # Extract (bi-)linear form
        self.a = dfem.form(ufl.lhs(self.weak_form))
        self.l = dfem.form(ufl.rhs(self.weak_form))

        # Assemble system matrix
        self.A = dfem_petsc.assemble_matrix(self.a, bcs=self.dirichlet_bc.list)
        self.A.assemble()

        # Initialise RHS
        self.L = dfem_petsc.create_vector(self.l)

        # Initialise solver
        self.solver = PETSc.KSP().create(MPI.COMM_WORLD)
        self.solver.setOperators(self.A)
        self.solver.setTolerances(rtol=1e-10, atol=1e-10)

        # Configure mumps
        self.solver.setType(PETSc.KSP.Type.PREONLY)
        pc = self.solver.getPC()
        pc.setType(PETSc.PC.Type.LU)
        pc.setFactorSolverType("mumps")

    # --- Solve problem ---

    def solve_equation_system(self):
        # Re-initialise RHS
        with self.L.localForm() as loc_L:
            loc_L.set(0)

        # Reassemble RHS
        dfem.petsc.assemble_vector(self.L, self.l)

        # Apply boundary conditions
        dfem.apply_lifting(self.L, [self.a], [self.dirichlet_bc.list])
        self.L.ghostUpdate(
            addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE
        )

        dfem.set_bc(self.L, self.dirichlet_bc.list)

        # Solve
        self.solver(self.L, self.solution_space.uh.vector)
        self.solution_space.uh.x.scatter_forward()

        return 1


class FemProblemNonLinear(FemProblem):
    def __init__(
        self,
        domain: Domain,
        solution_space: SolutionSpace,
        material_parameters: typing.Any,
        equation_type: typing.Optional[typing.Type[Enum]] = None,
        primary_variables: typing.Optional[typing.Type[Enum]] = None,
        finite_elmt_spaces: typing.Optional[typing.Type[Enum]] = None,
    ):
        # Constructor of the base class
        super().__init__(
            domain,
            solution_space,
            material_parameters,
            equation_type,
            primary_variables,
            finite_elmt_spaces,
        )

        # The problem solver
        self.nl_problem = None
        self.solver = None

    # --- Initialisation ---
    def initialise_solver(self):
        # Initialise non-linear problem
        self.nl_problem = dfem_petsc.NonlinearProblem(
            self.weak_form, self.solution_space.uh, self.dirichlet_bc.list
        )

        # Initialise newton solver
        self.solver = dnls_petsc.NewtonSolver(self.domain.mesh.comm, self.nl_problem)
        self.solver.atol = 1e-10
        self.solver.rtol = 1e-10
        self.solver.convergence_criterion = "incremental"

        # Configure mumps
        self.solver.krylov_solver.setType(PETSc.KSP.Type.PREONLY)
        pc = self.solver.krylov_solver.getPC()
        pc.setType(PETSc.PC.Type.LU)
        pc.setFactorSolverType("mumps")

    # --- Solve problem ---

    def solve_equation_system(self):
        # Solve
        num_its, converged = self.solver.solve(self.solution_space.uh)
        assert converged

        return num_its
