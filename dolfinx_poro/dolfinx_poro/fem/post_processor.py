# --- Imports ---
import math
import numpy as np
import typing

import dolfinx.geometry as dgeom
import dolfinx.mesh as dmesh

from .domain import Domain
from .solution_space import SolutionSpace
from .abstract_material import AbstractMaterial


# --- Base class for post processing ---
class PostProcessor:
    def __init__(
        self,
        domain: Domain,
        solution_space: SolutionSpace,
        material: typing.Type[AbstractMaterial],
        only_nth_step: int,
    ):
        # The domain
        self.domain = domain

        # The solution space
        self.solution_space = solution_space

        # The material
        self.material = material

        # Evaluate only after every n_th time step
        self.only_nth_step = only_nth_step

    def initialise_storage(self, time_end: float, dt: float):
        raise NotImplementedError

    def __call__(self, time: float, dt: float, n_dt: int):
        raise NotImplementedError

    def write_data(self, out_name: str):
        raise NotImplementedError


# --- Evaluate primary variables over line ---
class Line:
    def __init__(
        self,
        num_points: int,
        name: str,
        points: typing.Optional[np.ndarray] = None,
        start: typing.Optional[np.ndarray] = None,
        end: typing.Optional[np.ndarray] = None,
    ):
        if points is None:
            # Check input
            assert start.shape == (3,)
            assert end.shape == (3,)

            # Store start/end point
            self.start = start
            self.end = end

            self.points = None
        else:
            # Check input
            assert points.shape[2] == 3

            # Store point coordinates
            self.points = points

            self.start = None
            self.end = None

        # Name
        self.name = name

        # Number of evaluation points
        self.num_points = num_points

    def create_evaulation_points(self):
        if self.points is None:
            return np.linspace(self.start, self.end, self.num_points)
        else:
            return self.points


class PlotOverLine(PostProcessor):
    def __init__(
        self,
        domain: Domain,
        solution_space: SolutionSpace,
        material: typing.Type[AbstractMaterial],
        line: Line,
        only_nth_step: typing.Optional[int] = 1,
        scaling_position: typing.Optional[float] = 1.0,
    ):
        # Call basic constructor
        super().__init__(domain, solution_space, material, only_nth_step)

        # The Line over which the solution is evaluated
        self.line = line

        # Scaling factor for the positions
        self.scaling_position = scaling_position

        # --- Initialise output
        # The block-size of each output-quantity
        self.out_bs = []

        # The header of the output files
        self.out_header = ["x", "y", "z"]

        # The output fiels
        self.out_files = []
        self.out_nfiles = 0

        for V, pvar in zip(
            self.solution_space.sub_function_spaces, self.solution_space.name_pvars
        ):
            # Block size of the field
            bs = V.dofmap.bs * V.element.basix_element.value_size

            # Basename of output-file
            basename = "_" + self.line.name + "_pvar-" + pvar

            # Handel vector-valued fields
            if bs > 1:
                self.out_bs.append(bs)
                for i in range(0, bs):
                    self.out_files.append(basename + self.out_header[i] + ".csv")
            else:
                self.out_bs.append(bs)
                self.out_files.append(basename + ".csv")

            # Append counter
            self.out_nfiles += bs

        # The results
        self.results = None

        # The data for point evaluations
        self.eval_points = []
        self.eval_cells = []

        self.initialise_point_evaluation(domain.mesh, material.l_ref)

    def initialise_point_evaluation(self, domain_mesh: dmesh.Mesh, l_ref: float):
        list_points = self.line.create_evaulation_points()

        # The search tree
        bb_tree = dgeom.bb_tree(domain_mesh, domain_mesh.topology.dim)

        # Find cells whose bounding-box collide with the the points
        cell_candidates = dgeom.compute_collisions_points(bb_tree, list_points)

        # Choose one of the cells that contains the point
        points_on_proc = []
        colliding_cells = dgeom.compute_colliding_cells(
            domain_mesh, cell_candidates, list_points
        )

        for i, point in enumerate(list_points):
            if len(colliding_cells.links(i)) > 0:
                points_on_proc.append(point)
                self.eval_cells.append(colliding_cells.links(i)[0])

        self.eval_points = np.array(points_on_proc, dtype=np.float64)

    def initialise_storage(self, time_end: float, dt: float):
        # Number of time steps
        n_dt = math.ceil(time_end / dt)

        # Number of evaluations
        n_eval = math.floor(n_dt / self.only_nth_step) + 1

        # Resize storage
        self.results = np.zeros((self.line.num_points, n_eval + 3, self.out_nfiles))

        # Set spacial positions
        list_points = self.line.create_evaulation_points()

        for i in range(0, self.out_nfiles):
            self.results[:, :3, i] = list_points * self.scaling_position

    def __call__(self, time: float, dt: float, num_timestep: int):
        if ((num_timestep % self.only_nth_step) == 0) or (num_timestep == 1):
            # Set time
            self.out_header.append("{:.4e}".format(time))

            # Evaluate field quantities
            h1 = int(num_timestep / self.only_nth_step) + 3
            h2 = 0

            for uh, bs in zip(self.solution_space.uh_n, self.out_bs):
                self.results[:, h1, h2 : h2 + bs] = uh.eval(
                    self.eval_points, self.eval_cells
                )

                h2 += bs

    def write_data(self, out_name: typing.Optional[str] = None):
        if out_name is not None:
            # Create header for output
            header = ",".join(self.out_header)

            # Output data for different field
            for i, file in enumerate(self.out_files):
                file_name = out_name + file
                np.savetxt(
                    file_name, self.results[:, :, i], delimiter=",", header=header
                )
