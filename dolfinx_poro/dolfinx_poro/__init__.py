# Characterisation of the problem
from .poroelasticity import (
    EquationType,
    PrimaryVariables,
    FeSpaces,
    FieldVariables,
    VolumeTerms,
    MatPar,
    NondimPar,
)

# The material class
from .material import Material, NondimensionalMaterial

# The (discretised) weak form
from .governing_equations import (
    set_discretisation,
    set_default_spaces,
    set_default_orders,
    set_weakform,
    scale_primal_variables,
)
