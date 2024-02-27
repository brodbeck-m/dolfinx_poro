# --- Imports ---
from enum import Enum

# --- Characterization of a poro-elastic problem ---


class EquationType(Enum):
    Biot = 0
    lTPM = 1
    lTPM_Bluhm = 2
    TPM = 3


class PrimaryVariables(Enum):
    up = 1
    upn = 2
    uppt = 3
    upptn = 4
    uvp = 5
    uvpn = 6
    usigpv_ls1 = 7


class FeSpaces(Enum):
    up_EO = 0
    up_TH = 1
    up_mini = 2
    up_EG = 3
    uppt_TH = 4
    uppt_mini = 5
    uvp_RT = 6
    uvp_BDM = 7
    ls_1_RT = 8
    ls_1_BDM = 9


class FieldVariables(Enum):
    displacement = 1
    pressure = 2
    volume_fraction = 3
    stress = 4
    seep_velocity = 5
    time = 6


class MatPar(Enum):
    lhS = 0
    mhS = 1
    rhohSR = 2
    khSt0S = 3
    rhohFR = 4
    mhFR = 5


class NondimPar(Enum):
    pi_1 = 0
    pi_2 = 1
    pi_3 = 2
    pi_4 = 3


class VolumeTerms(Enum):
    body_force = 0
    growth_rate = 1
    nhSt0S = 2
