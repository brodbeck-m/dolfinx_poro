"""
Testcase: Terzaghi problem
"""

# --- Imports ---
import numpy as np

from dolfinx_poro import (
    EquationType,
    PrimaryVariables,
    FeSpaces,
    NondimensionalMaterial,
)
from dolfinx_poro.cases.case_mandel import setup_calculation
from dolfinx_poro.fem.post_processor import Line, PlotOverLine
from dolfinx_poro.post_processor import EvaluateMass, MassResult

# --- The Terzaghi problem ---
# Material
pi_1 = 1

# Equation
equation_type = EquationType.Biot
primary_variables = PrimaryVariables.up

# Spacial discretization
sdisc_nelmt = [40, 20]

sdisc_fetype = FeSpaces.up_TH
sdisc_eorder = [3, 2]

# Temporal discretization
tdisc_dt = 0.05
tdisc_tend = 1

# Geometry and boundary conditions
geom_length = [2.0, 1.0]
bc_loadF = 1.0

load_as_constant_traction = True
load_with_ramp = True

ramp_time = 0.5

# Initial conditions
nhSt0S = 0.5

# Output
output_name = "mandel-ndim"
output_dimensionless = False


# --- Perform calculation ---
# The growth function
def time_function(time_ramp, time):
    if time < time_ramp:
        return time / time_ramp
    else:
        return 1.0


# The FemProblem
if load_with_ramp:
    time_function_load = lambda time: time_function(ramp_time, time)
else:
    time_function_load = None

fem_problem = setup_calculation(
    equation_type,
    primary_variables,
    NondimensionalMaterial(pi_1, 0.0, 0.0, l_ref=1.0),
    bc_loadF,
    constant_traction_bc=load_as_constant_traction,
    time_function_load=time_function_load,
    domain_geometry=geom_length,
    sdisc_fetype=sdisc_fetype,
    sdisc_nelmt=sdisc_nelmt,
    sdisc_eorder=sdisc_eorder,
    initcond_nhSt0S=nhSt0S,
    scale_output=output_dimensionless,
)

# Postprocessing
plot_over_line = PlotOverLine(
    fem_problem.domain,
    fem_problem.solution_space,
    fem_problem.material,
    Line(
        250,
        "diagonal",
        start=np.array([0.0, geom_length[1], 0.0]),
        end=np.array([geom_length[0], 0.0, 0.0]),
    ),
    only_nth_step=10,
)

# Solve problem
fem_problem.solve_problem(
    tdisc_dt,
    tdisc_tend,
    output_paraview=True,
    output_name=output_name,
)
