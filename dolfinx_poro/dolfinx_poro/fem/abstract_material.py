# --- Imports ---
from enum import Enum
import typing

import dolfinx.fem as dfem
import dolfinx.mesh as dmesh

from .auxiliaries import set_femconstant


# --- An abstract material definition ---
class AbstractMaterial:
    def __init__(
        self,
        num_matpar: int,
        num_nondim: int,
        num_volumetric_terms: int,
        nondimensional_form: typing.Optional[bool] = False,
        reference_length: typing.Optional[float] = 1.0,
    ):
        # The reference length
        self.l_ref = reference_length

        # Required identifiers
        self.material_parameters_set = False
        self.reference_parameters_set = False

        self.volumetric_term_set = [False] * num_volumetric_terms

        self.nondimensional_form = nondimensional_form
        self.is_time_dependent = False

        # Material parameters
        self.material_parameters = [1.0] * num_matpar
        self.material_parameters_ref = [1.0] * num_matpar
        self.material_parameters_ufl = []

        # Dimensionless parameters
        self.pi = [1.0] * num_nondim
        self.pi_ufl = []

        # Volumetric sources
        self.volumetric_terms = [None] * num_volumetric_terms
        self.volumetric_terms_ufl = []

        self.time_functions = [lambda time: 1.0] * num_volumetric_terms
        self.time_functions_ufl = []

    # --- Initialise ufl-values ---
    def initialise_parameters(self, domain_mesh: dmesh.Mesh):
        # Remove all prior ufl-values
        self.material_parameters_ufl = []
        self.pi_ufl = []
        self.volumetric_terms_ufl = []
        self.time_functions_ufl = []

        # Decide wether the parameters have to be scaled
        if self.nondimensional_form:
            scaling = self.material_parameters_ref
        else:
            scaling = [1.0] * len(self.material_parameters_ref)

        # Initialise material parameters
        for par, par_ref in zip(self.material_parameters, scaling):
            if isinstance(par, (int, float)):
                self.material_parameters_ufl.append(
                    set_femconstant(domain_mesh, par / par_ref)
                )
            else:
                raise ValueError("Spacial material parameters not supported")

        for pi in self.pi:
            self.pi_ufl.append(set_femconstant(domain_mesh, pi))

        # Initialise volumetric terms
        for source in self.volumetric_terms:
            if source is None:
                self.volumetric_terms_ufl.append(None)
            elif isinstance(source, (int, float)):
                self.volumetric_terms_ufl.append(set_femconstant(domain_mesh, source))
            elif isinstance(source, list):
                if len(source) == 2:
                    self.volumetric_terms_ufl.append(
                        set_femconstant(domain_mesh, (source[0], source[1]))
                    )
                elif len(source) == 3:
                    self.volumetric_terms_ufl.append(
                        set_femconstant(domain_mesh, (source[0], source[1], source[2]))
                    )
                else:
                    raise ValueError("Unsupported vector-length of volumetric term")
            else:
                raise ValueError("Spacial volumetric sources not supported")

            # Initialise time functions
            self.time_functions_ufl.append(set_femconstant(domain_mesh, 1.0))

    # --- Setter functions ---
    def set_material_parameters(
        self,
        mat_par: typing.Type[Enum],
        list_values: typing.List[float],
    ):
        # Set identifier
        self.material_parameters_set = True

        # Set values
        for param, value in zip(mat_par, list_values):
            self.material_parameters[param.value] = value

    def set_reference_parameters(
        self,
        mat_par: typing.Type[Enum],
        list_values: typing.List[float],
    ):
        # Set identifier
        self.reference_parameters_set = True

        # Set values
        for param, value in zip(mat_par, list_values):
            self.material_parameters_ref[param.value] = value

    def set_nondimensional_parameters(
        self, nondim_par: typing.Type[Enum], list_values: typing.List[float]
    ):
        # Check if material is non-dimensional
        if not self.nondimensional_form:
            raise ValueError("Material is not of non-dimensional type.")

        # Set values
        for param, value in zip(nondim_par, list_values):
            self.pi[param.value] = value

    def set_volumetric_term(
        self,
        name: typing.Type[Enum],
        value: typing.Any,
        time_function: typing.Optional[typing.Callable] = None,
    ):
        # Set identifier
        self.volumetric_term_set[name.value] = True

        # Set values
        self.volumetric_terms[name.value] = value

        # Set time function
        if time_function is not None:
            self.is_time_dependent = True
            self.time_functions[name.value] = time_function

    # --- Update material parameters ---
    def update_material_parameters(
        self,
        mat_par: typing.Type[Enum],
        list_values: typing.List[float],
        list_values_ref: typing.Optional[typing.List[float]] = None,
    ):
        # Check if material-parameters are set
        if not self.material_parameters_set:
            raise ValueError("No material-parameters set!")

        # Check if reference values have to be updated
        if list_values_ref is not None:
            self.set_reference_parameters(mat_par, list_values_ref)

        # Decide wether the parameters have to be scaled
        if self.nondimensional_form:
            scaling = self.material_parameters_ref
        else:
            scaling = [1.0] * len(self.material_parameters_ref)

        # Reset parameters
        for param, value, value_ref in zip(mat_par, list_values, scaling):
            # Reset value
            self.material_parameters[param.value] = value

            # Reset ufl
            if isinstance(value, (int, float)):
                self.material_parameters_ufl[param.value].value = value / value_ref
            else:
                raise ValueError("Spacial material parameters not supported")

    def update_nondimensional_parameters(
        self, nondim_par: typing.Type[Enum], list_values: typing.List[float]
    ):
        # Check if material-parameters are set
        if not self.nondimensional_form:
            raise ValueError("Material not in non-dimensional form")

        # Reset parameters
        for param, value in zip(nondim_par, list_values):
            # Reset value
            self.pi[param.value] = value

            # Reset ufl
            self.pi_ufl[param.value].value = value

    def update_volumetric_term(
        self,
        vol_sources: typing.List[typing.Type[Enum]],
        list_values: typing.List[float],
    ):
        for source, value in zip(vol_sources, list_values):
            # Reset value
            if self.volumetric_term_set[source.value]:
                self.volumetric_terms[source.value] = value
            else:
                raise ValueError("Volumetric can not be updated!")

            # Reset ufl
            if isinstance(value, (int, float)):
                self.material_parameters_ufl[source.value].value = value
            elif isinstance(value, list):
                if len(value) == 2:
                    self.material_parameters_ufl[source.value].value[0] = value[0]
                    self.material_parameters_ufl[source.value].value[1] = value[1]
                elif len(value) == 3:
                    self.material_parameters_ufl[source.value].value[0] = value[0]
                    self.material_parameters_ufl[source.value].value[1] = value[1]
                    self.material_parameters_ufl[source.value].value[2] = value[2]
                else:
                    raise ValueError("Unsupported vector-length of volumetric term")
            else:
                raise ValueError("Spacial volumetric sources not supported")

    # --- Time update ---
    def update_time(self, time: float):
        if self.is_time_dependent:
            for fct, fct_ufl in zip(self.time_functions, self.time_functions_ufl):
                fct_ufl.value = fct(time)

    # --- Getter functions ---
    def get_matpar(self, name: typing.Type[Enum]):
        return self.material_parameters[name.value]

    def get_matpar_ref(self, name: typing.Type[Enum]):
        return self.material_parameters_ref[name.value]

    def get_matpar_ufl(self, name: typing.Type[Enum]):
        return self.material_parameters_ufl[name.value]

    def get_pi(self, name: typing.Type[Enum]):
        return self.pi[name.value]

    def get_pi_ufl(self, name: typing.Type[Enum]):
        return self.pi_ufl[name.value]

    def get_volumetric_term(
        self, name: typing.Type[Enum], time: typing.Optional[float] = None
    ):
        if self.volumetric_term_set[name.value]:
            vol_term = self.volumetric_terms[name.value]

            if time is None:
                return vol_term
            else:
                return vol_term * self.time_functions[name.value](time)
        else:
            return None

    def get_volumetric_term_ufl(self, name: typing.Type[Enum]):
        if self.volumetric_term_set[name.value]:
            return (
                self.time_functions_ufl[name.value]
                * self.volumetric_terms_ufl[name.value]
            )
        else:
            return None
