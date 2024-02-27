# --- Imports ---
import numpy as np
import typing

import dolfinx.fem as dfem

from .domain import Domain


# --- The DirichletValues ---
class DirichletFunction:
    def __init__(self, subspace: int, is_timedependent: bool):
        # Id of Subspace on which the function is applied
        self.subspace = subspace

        # Physical time
        self.is_timedependent = is_timedependent
        self.time = 0.0

    def __call__(self, x):
        raise NotImplementedError

    def update_time(self, time: float):
        self.time = time


# --- The collection of all essential boundary conditions ---
class DirichletBC:
    def __init__(
        self,
        domain: Domain,
        V: dfem.FunctionSpace,
        V_sub: typing.List[dfem.FunctionSpace],
    ):
        # Id if transient bcs are considered
        self.is_timedependent = False

        # The Dirichlet values
        self.dirichlet_functions = []

        # The Dirichlet DOFs
        self.uD = []

        # The DirichletBC objects
        self.list = []

        # Number of boundary conditions
        self.n_bc = 0

        # --- Set BCs
        self.set_dirichletbc(domain, V, V_sub)

        # Check time-dependency
        for func in self.dirichlet_functions:
            if func.is_timedependent:
                self.is_timedependent = True

        # Update number of BCs
        self.n_bc = len(self.dirichlet_functions)

    def initialise_dirichlet_values(
        self,
        sub_function_space: dfem.FunctionSpace,
        dirichlet_function: typing.Optional[DirichletFunction] = None,
        const_value: typing.Optional[float] = None,
        id_subspace: typing.Optional[int] = None,
    ):
        # Set DirichletFunction
        if dirichlet_function is not None:
            self.dirichlet_functions.append(dirichlet_function)
        else:
            # Check input
            if id_subspace is None:
                raise ValueError("Id of the subspace required!")

            # Set (static) default DirichletFunction
            self.dirichlet_functions.append(DirichletFunction(id_subspace, False))

        # Set storage for boundary DOFs
        self.uD.append(dfem.Function(sub_function_space))

        # Set DOFs for static BCs
        if not self.dirichlet_functions[-1].is_timedependent:
            if const_value is None:
                self.uD[-1].interpolate(self.dirichlet_functions[-1])
            else:
                self.uD[-1].x.array[:] = const_value

    def set_dirichletbc(
        self,
        domain: Domain,
        V: dfem.FunctionSpace,
        V_sub: typing.List[dfem.FunctionSpace],
    ):
        raise NotImplementedError

    def update_time(self, time: float):
        if self.is_timedependent:
            for func, uD in zip(self.dirichlet_functions, self.uD):
                if func.is_timedependent:
                    # time update
                    func.update_time(time)

                    # Update boundary function
                    uD.interpolate(func)
