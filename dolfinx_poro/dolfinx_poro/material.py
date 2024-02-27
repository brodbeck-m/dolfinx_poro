# --- Imports ---
import typing

import dolfinx.fem as dfem

from .fem.finite_element_problem import AbstractMaterial
from .poroelasticity import FieldVariables, MatPar, NondimPar


# --- The set of material parameters ---
class AbstractMaterialPoro(AbstractMaterial):
    def __init__(self, l_ref: float, nondimensional_form: bool):
        # Constructor of base class
        super().__init__(
            6,
            4,
            3,
            nondimensional_form=nondimensional_form,
            reference_length=l_ref,
        )

    # --- Setter functions ---
    def set_material_parameters(
        self,
        params: typing.List[typing.Any],
        params_ref: typing.Optional[typing.List[float]] = None,
    ):
        # Check if reference parameters are required
        reference_parameters_required = False

        for param in params:
            if not isinstance(param, (int, float)):
                if params_ref is None:
                    raise ValueError("Reference parameters required.")
                else:
                    reference_parameters_required = True

        # Set parameters
        super().set_material_parameters(MatPar, params)

        # Set reference parameters
        if reference_parameters_required:
            if params_ref is None:
                raise ValueError("No reference parameters given!")

            super().set_reference_parameters(MatPar, params_ref)
        else:
            super().set_reference_parameters(MatPar, params)

    def set_nondimensional_parameters(
        self, pi_1: float, pi_2: float, pi_3: float, pi_4: float
    ):
        # Set dimensionless numbers
        super().set_nondimensional_parameters(NondimPar, [pi_1, pi_2, pi_3, pi_4])

    # --- Non-dimensionalisation ---
    def rescale(self, list_quantities: typing.List[FieldVariables]):
        # Check if required quantities are set
        if not (self.reference_parameters_set):
            raise ValueError("Reference material-parameters not specified.")

        # Calculate list of dimensionless factors
        list_dimfactors = []

        mhS = self.get_matpar_ref(MatPar.mhS)
        khSt0S = self.get_matpar_ref(MatPar.khSt0S)
        mhFR = self.get_matpar_ref(MatPar.mhFR)

        for quantity in list_quantities:
            if quantity == FieldVariables.displacement:
                list_dimfactors.append(self.l_ref)
            elif quantity == FieldVariables.pressure:
                list_dimfactors.append(mhS)
            elif quantity == FieldVariables.volume_fraction:
                list_dimfactors.append(1.0)
            elif quantity == FieldVariables.stress:
                list_dimfactors.append(mhS)
            elif quantity == FieldVariables.seep_velocity:
                list_dimfactors.append((khSt0S * mhS) / (mhFR * self.l_ref))
            elif quantity == FieldVariables.time:
                list_dimfactors.append((mhFR * (self.l_ref**2)) / (khSt0S * mhS))

        if self.nondimensional_form:
            return list_dimfactors
        else:
            return [1.0 / dimfactor for dimfactor in list_dimfactors]


class Material(AbstractMaterialPoro):
    def __init__(
        self,
        lhS: typing.Union[float, dfem.Function],
        mhS: typing.Union[float, dfem.Function],
        rhohSR: typing.Union[float, dfem.Function],
        khSt0S: typing.Union[float, dfem.Function],
        rhohFR: typing.Union[float, dfem.Function],
        mhFR: typing.Union[float, dfem.Function],
        l_ref: typing.Optional[float] = 1.0,
        reference_parameters: typing.Optional[typing.List[float]] = None,
    ):
        # Call constructor of base class
        super().__init__(l_ref, False)

        # Set material parameters
        self.set_material_parameters(
            [lhS, mhS, rhohSR, khSt0S, rhohFR, mhFR], reference_parameters
        )


class NondimensionalMaterial(AbstractMaterialPoro):
    def __init__(
        self,
        pi_1: float,
        pi_2: float,
        pi_3: float,
        pi_4: typing.Optional[float] = 0.0,
        l_ref: typing.Optional[float] = 1.0,
    ):
        # Call constructor of base class
        super().__init__(l_ref, True)

        # Set dimensionless parameters
        self.set_nondimensional_parameters(pi_1, pi_2, pi_3, pi_4)

    def set_material_parameters(
        self,
        lhS: typing.Union[float, dfem.Function],
        mhS: typing.Union[float, dfem.Function],
        rhohSR: typing.Union[float, dfem.Function],
        khSt0S: typing.Union[float, dfem.Function],
        rhohFR: typing.Union[float, dfem.Function],
        mhFR: typing.Union[float, dfem.Function],
        reference_parameters: typing.Optional[typing.List[float]] = None,
    ):
        # Set material parameters (made nondimensional by division with reference parameters)
        self.set_material_parameters(
            [lhS, mhS, rhohSR, khSt0S, rhohFR, mhFR],
            reference_parameters=reference_parameters,
        )
