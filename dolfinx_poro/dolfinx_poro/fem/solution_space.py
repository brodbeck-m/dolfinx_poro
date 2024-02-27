# --- Imports ---
import typing

import dolfinx.fem as dfem
import ufl


# --- The SolutionSpace ---
class SolutionSpace:
    def __init__(
        self,
        function_space: dfem.FunctionSpace,
        name_pvars: typing.List[str],
        problem_is_timedependent: bool,
        problem_is_linear: bool,
    ):
        # Identifiers
        self.is_linear = problem_is_linear
        self.initial_conditions_set = False

        # Name of the primal variables
        self.name_pvars = name_pvars

        # The FunctionSpace
        self.function_space = function_space

        # Handling of mixed spaces
        if (
            self.function_space.element.num_sub_elements > 1
            and self.function_space.dofmap.index_map_bs == 1
        ):
            self.is_mixed = True
            self.num_subspaces = self.function_space.element.num_sub_elements
        else:
            self.is_mixed = False
            self.num_subspaces = 1

        # Subspaces
        self.sub_function_spaces = []

        if self.is_mixed:
            for i in range(self.num_subspaces):
                self.sub_function_spaces.append(
                    self.function_space.sub(i).collapse()[0]
                )
        else:
            self.sub_function_spaces.append(self.function_space)

        # Solution
        self.uh = dfem.Function(self.function_space)

        # History fields
        self.uh_n = []
        self.space_to_subspace = []

        if problem_is_timedependent:
            if self.is_mixed:
                for i in range(self.num_subspaces):
                    # Extract subspace
                    V_sub, space_to_sub = self.function_space.sub(i).collapse()

                    # Create storage of history fields
                    self.uh_n.append(dfem.Function(V_sub))
                    self.space_to_subspace.append(space_to_sub)

                    # Set name of function
                    self.uh_n[-1].name = self.name_pvars[i]
                else:
                    self.uh_n.append(dfem.Function(self.function_space))
                    self.space_to_subspace.append(None)
                    self.uh_n[0].name = self.name_pvars[0]

        # Trial- and test-functions
        self.trial_fkt = []
        self.test_fkt = []

        if self.is_mixed:
            if self.is_linear:
                trial_function = ufl.TrialFunctions(self.function_space)
            else:
                trial_function = ufl.split(self.uh)

            for u, v in zip(
                trial_function,
                ufl.TestFunctions(self.function_space),
            ):
                self.trial_fkt.append(u)
                self.test_fkt.append(v)
        else:
            if self.is_linear:
                self.trial_fkt.append(ufl.TrialFunction(self.function_space))
            else:
                self.trial_fkt.append(self.uh)

            self.test_fkt.append(ufl.TestFunction(self.function_space))

    def scale_solution(self, scaling: typing.List[float]):
        for uh_n, map, factor in zip(self.uh_n, self.space_to_subspace, scaling):
            uh_n.x.array[:] = factor * self.uh.x.array[map]

    def update_history_field(self):
        for uh_n, map in zip(self.uh_n, self.space_to_subspace):
            uh_n.x.array[:] = self.uh.x.array[map]
