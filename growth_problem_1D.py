"""
Testcase: Growth
"""

# --- Imports ---
import numpy as np

from dolfinx_poro import (
    EquationType,
    PrimaryVariables,
    FeSpaces,
    NondimensionalMaterial,
)

from dolfinx_poro.cases.case_growth_1D import setup_calculation
from dolfinx_poro.fem.post_processor import PlotOverLine, Line
from dolfinx_poro.post_processor import EvaluateMass

# --- The footing problem ---
# Non-dimensional parameters
pi_1 = 0.25
pi_2 = 0.2
pi_4 = 1
pi_5 = 10

# Initial condition
nhSt0S_init = 0.2

# Growth time
volmean_nhS_end = 0.75
time_growth = ((pi_2 * nhSt0S_init) / (pi_4 * pi_5)) * (
    volmean_nhS_end / nhSt0S_init - 1
)

# Equation
equation_type = EquationType.TPM
primary_variables = PrimaryVariables.upn

# Spacial discretization
sdisc_nelmt = [2, 40]

sdisc_fetype = FeSpaces.up_TH
sdisc_eorder = [3, 2, 2]

# Temporal discretization
tdisc_tend = time_growth
tdisc_dt = time_growth / 100

# Output
output_name = "output-growth-1D_pi45-4d1-NoOut"

# --- Perform calculation ---
# The material
material = NondimensionalMaterial(pi_1, pi_2, 0.0, pi_4, l_ref=1.0)


# The growth function
def growth_function(time_growth, time):
    if time < time_growth or np.isclose(time, time_growth):
        return 1.0
    else:
        return 0.0


# The FemProblem
fem_problem = setup_calculation(
    equation_type,
    primary_variables,
    material,
    pi_5,
    domain_geometry=[0.1, 1],
    sdisc_fetype=sdisc_fetype,
    sdisc_discont_volfrac=False,
    sdisc_eorder=sdisc_eorder,
    sdisc_nelmt=sdisc_nelmt,
    initcond_nhSt0S=nhSt0S_init,
    scale_output=False,
    allow_outflow=False,
    traction_top=0.0,
    time_function_growth=lambda time: growth_function(time_growth, time),
)

mass_evaluator = EvaluateMass(
    fem_problem.domain,
    fem_problem.solution_space,
    material,
    equation_type,
    primary_variables,
    flux_surfaces=[4],
)

plot_over_line = PlotOverLine(
    fem_problem.domain,
    fem_problem.solution_space,
    fem_problem.material,
    Line(
        100,
        "centerline",
        start=np.array([0.1, 0.0, 0.0]),
        end=np.array([0.1, 1, 0.0]),
    ),
    only_nth_step=20,
)

# Solve problem
fem_problem.solve_problem(
    tdisc_dt,
    tdisc_tend,
    output_paraview=True,
    output_name=output_name,
    post_processors=[mass_evaluator, plot_over_line],
)
