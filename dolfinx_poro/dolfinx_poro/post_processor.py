# --- Imports ---
from enum import Enum
import math
from mpi4py import MPI
import numpy as np
import typing

import dolfinx.fem as dfem
import dolfinx.mesh as dmesh
import ufl

from .fem import Domain, SolutionSpace
from .fem.abstract_material import AbstractMaterial
from .fem.post_processor import PostProcessor
from .poroelasticity import (
    EquationType,
    PrimaryVariables,
    MatPar,
    NondimPar,
    VolumeTerms,
)


# --- Mass evaluation for poro-elastic problem ---
class MassResult(Enum):
    time = 0
    volume = 1
    volume_solid = 2
    ux_max = 3
    uy_max = 4
    uz_max = 5
    p_max = 6
    mass_ref = 7
    mass = 8
    mass_solid = 9
    mass_flux_surface = 10
    error = 11


class EvaluateMass(PostProcessor):
    def __init__(
        self,
        domain: Domain,
        solution_space: SolutionSpace,
        material: typing.Type[AbstractMaterial],
        equation_type: typing.Type[Enum],
        primary_variables: typing.Type[Enum],
        only_nth_step: typing.Optional[int] = 1,
        output_results: typing.Optional[bool] = True,
        flux_surfaces: typing.Optional[typing.List[int]] = None,
    ):
        # --- Check input
        # Check if solution is initialised
        if solution_space.initial_conditions_set is False:
            raise ValueError("Solution space has to be initialised first!")

        # Check if unscaled domain is considered
        if not np.isclose(material.l_ref, 1.0):
            raise ValueError("Post-processing of mass only on unscaled domains!")

        # Call basic constructor
        super().__init__(
            domain, solution_space, material, only_nth_step, output_results
        )

        # --- Identifier
        self.mass_is_constant = False

        if flux_surfaces is None:
            self.mass_is_constant = True

        # --- Initialise output
        # The results
        self.initial_mass = None
        self.results = None
        self.fluxes = None

        # The evaluated integrals
        self.integral_total_volume = None
        self.integral_solid_volume = None
        self.integral_total_mass = None
        self.integral_solid_mass = None
        self.integral_mass_flux = None

        # History array
        self.flux_n = 0.0
        self.outflow = 0.0

        self.initialise_integral_evaluation(
            equation_type, primary_variables, flux_surfaces=flux_surfaces
        )

    # --- Initialisation ---
    def initialise_integral_evaluation(
        self,
        equation_type: typing.Type[Enum],
        primary_variables: typing.Type[Enum],
        flux_surfaces: typing.Optional[typing.List[int]] = None,
    ):
        # The primal variables
        nhS = None
        if (
            primary_variables == PrimaryVariables.up
            or primary_variables == PrimaryVariables.uppt
        ):
            u = self.solution_space.uh_n[0]
            p = self.solution_space.uh_n[1]
        elif primary_variables == PrimaryVariables.upn:
            u = self.solution_space.uh_n[0]
            p = self.solution_space.uh_n[1]
            nhS = self.solution_space.uh_n[2]
        elif primary_variables == PrimaryVariables.upptn:
            u = self.solution_space.uh_n[0]
            p = self.solution_space.uh_n[1]
            nhS = self.solution_space.uh_n[3]
        elif primary_variables == PrimaryVariables.uvp:
            u = self.solution_space.uh_n[0]
            p = self.solution_space.uh_n[2]
        elif primary_variables == PrimaryVariables.uvpn:
            u = self.solution_space.uh_n[0]
            p = self.solution_space.uh_n[2]
            nhS = self.solution_space.uh_n[3]
        else:
            raise ValueError("Unsupported primal variables!")

        # Required material parameters
        rhohSR = self.material.get_matpar_ufl(MatPar.rhohSR)
        rhohFR = self.material.get_matpar_ufl(MatPar.rhohFR)
        pi_2 = self.material.get_pi_ufl(NondimPar.pi_2)

        # Required mappings
        FtS = ufl.Identity(len(u)) + ufl.grad(u)
        JtS = ufl.det(FtS)

        # Set nhS if it is no primal variable
        if nhS is None:
            nhS0S = self.material.get_volumetric_term_ufl(VolumeTerms.nhSt0S)
            nhS = nhS0S / JtS

        # The volume integrals
        self.integral_total_volume = dfem.form(JtS * ufl.dx)
        self.integral_solid_volume = dfem.form(nhS * JtS * ufl.dx)

        density_solid = nhS * pi_2 * rhohSR
        density_mixture = (1 - nhS) * rhohFR + nhS * pi_2 * rhohSR

        self.integral_solid_mass = dfem.form(JtS * density_solid * ufl.dx)
        self.integral_total_mass = dfem.form(JtS * density_mixture * ufl.dx)

        # Surface integrals
        if not self.mass_is_constant:
            # Additional material parameters
            khSt0S = self.material.get_matpar_ufl(MatPar.khSt0S)
            rhohFR = self.material.get_matpar_ufl(MatPar.rhohFR)
            mhFR = self.material.get_matpar_ufl(MatPar.mhFR)

            ktD = khSt0S / mhFR

            # Flux definition
            # TODO: Implement flux with body-force
            if (
                primary_variables == PrimaryVariables.up
                or primary_variables == PrimaryVariables.upn
                or primary_variables == PrimaryVariables.uppt
                or primary_variables == PrimaryVariables.upptn
            ):
                CtS = ufl.dot(ufl.transpose(FtS), FtS)
                flux = JtS * ktD * ufl.dot(-ufl.grad(p), ufl.inv(CtS))
            elif (
                primary_variables == PrimaryVariables.uvp
                or primary_variables == PrimaryVariables.uvpn
            ):
                if equation_type == EquationType.Biot:
                    CtS = ufl.dot(ufl.transpose(FtS), FtS)
                    flux = JtS * ufl.dot(self.solution_space.uh_n[1], ufl.inv(CtS))
                elif (
                    equation_type == EquationType.lTPM
                    or equation_type == EquationType.lTPM_Bluhm
                ):
                    print("Warning: Mass flux (Ref. Config.) can not be evaluated!")
                    flux = self.solution_space.uh_n[1]
                else:
                    flux = self.solution_space.uh_n[1]

            else:
                raise ValueError("Unsupported primal variables!")

            # Facet normal
            normal = ufl.FacetNormal(self.domain.mesh)

            # Surface integrals
            integral_mass_flux = ufl.inner(rhohFR * flux, normal) * self.domain.ds(
                flux_surfaces[0]
            )

            if len(flux_surfaces) > 1:
                for srf in flux_surfaces[1:]:
                    integral_mass_flux += ufl.inner(flux, normal) * self.domain.ds(srf)

            self.integral_mass_flux = dfem.form(integral_mass_flux)

    def initialise_storage(self, time_end: float, dt: float):
        # Number of time steps
        n_dt = math.ceil(time_end / dt)

        # Number of evaluations
        n_eval = math.floor(n_dt / self.only_nth_step) + 1

        if self.only_nth_step > 1:
            n_eval += 1

        # Resize storage
        self.results = np.zeros((n_eval, 12))
        self.fluxes = np.zeros((n_eval, 2))

        # Reset history values
        self.flux_n = 0.0
        self.outflow = 0.0

        # Calculate initial volume/ mass
        (
            self.results[0, MassResult.volume.value],
            self.results[0, MassResult.volume_solid.value],
            self.results[0, MassResult.mass.value],
            self.results[0, MassResult.mass_solid.value],
            self.results[0, MassResult.mass_flux_surface.value],
        ) = self.evaluate_integrals(dt=0)

        self.results[0, MassResult.mass_ref.value] = self.results[
            0, MassResult.mass.value
        ]

        # Store initial mass
        self.initial_mass = self.results[0, MassResult.mass_ref.value]

    # --- Evaluation ---
    def evaluate_integrals(self, dt):
        total_volume = self.domain.mesh.comm.allreduce(
            dfem.assemble_scalar(self.integral_total_volume), op=MPI.SUM
        )
        solid_volume = self.domain.mesh.comm.allreduce(
            dfem.assemble_scalar(self.integral_solid_volume), op=MPI.SUM
        )
        mass = self.domain.mesh.comm.allreduce(
            dfem.assemble_scalar(self.integral_total_mass), op=MPI.SUM
        )
        solid_mass = self.domain.mesh.comm.allreduce(
            dfem.assemble_scalar(self.integral_solid_mass), op=MPI.SUM
        )

        mass_flux = 0.0
        if not self.mass_is_constant:
            # Evaluate flux integral
            flux_np1 = self.domain.mesh.comm.allreduce(
                dfem.assemble_scalar(self.integral_mass_flux), op=MPI.SUM
            )

            # Calculate outflow
            mass_flux = 0.5 * (flux_np1 + self.flux_n) * dt

            # Set history data
            self.flux_n = flux_np1

        return total_volume, solid_volume, mass, solid_mass, mass_flux

    def __call__(self, time: float, dt: float, num_timestep: int):
        if ((num_timestep % self.only_nth_step) == 0) or (num_timestep == 1):
            # Storage position
            pos = int(num_timestep / self.only_nth_step)

            if self.only_nth_step > 1:
                pos += 1

            # Set time
            self.results[pos, MassResult.time.value] = time

            # Evaluate integrals
            (
                self.results[pos, MassResult.volume.value],
                self.results[pos, MassResult.volume_solid.value],
                self.results[pos, MassResult.mass.value],
                self.results[pos, MassResult.mass_solid.value],
                outflow,
            ) = self.evaluate_integrals(dt)

            self.outflow += outflow
            self.results[pos, MassResult.mass_flux_surface.value] += self.outflow

            # Evaluate maximal displacement
            res = MassResult.ux_max.value
            for i in range(0, self.domain.mesh.topology.dim):
                u_i = self.solution_space.uh.sub(0).sub(i).collapse().x.array
                self.results[pos, res + i] = max(u_i.min(), u_i.max(), key=abs)

            # Evaluate maximal pressure
            p = self.solution_space.uh.sub(1).collapse().x.array
            self.results[pos, MassResult.p_max.value] = max(p.min(), p.max(), key=abs)

            # Store out-flux
            self.fluxes[pos, 0] = time
            self.fluxes[pos, 1] = self.flux_n

            # Evaluate error
            if self.mass_is_constant:
                # Set correct mass
                self.results[pos, MassResult.mass_ref.value] = self.initial_mass

                # Evaluate error
                mass_error = (
                    self.results[pos, MassResult.mass.value] - self.initial_mass
                )
                self.results[pos, MassResult.error.value] = (
                    mass_error / self.initial_mass
                )
            else:
                # Set correct mass
                mass_ref = (
                    self.initial_mass
                    - self.results[pos, MassResult.mass_flux_surface.value]
                )

                # Evaluate error
                mass_error = self.results[pos, MassResult.mass.value] - mass_ref

                self.results[pos, MassResult.mass_ref.value] = mass_ref
                self.results[pos, MassResult.error.value] = (
                    mass_error / self.initial_mass
                ) * 100

    # --- Output ---
    def write_data(self, out_name: typing.Optional[str] = None):
        if (out_name is not None) and self.output_results:
            # Remove unused storage
            self.results = self.results[~np.all(self.results == 0, axis=1)]

            # Set output name
            file_name = out_name + "_eval-mass.csv"
            header = (
                "time, volume, volume_solid, ux_max, uy_max, uz_max, p_max,"
                " mass_ref, mass, mass_solid, mass_flux_surface, error [%]"
            )

            np.savetxt(
                file_name,
                self.results,
                delimiter=",",
                header=header,
            )
