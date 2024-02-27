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
from dolfinx_poro.cases.case_terzaghi import setup_calculation
from dolfinx_poro.fem.post_processor import Line, PlotOverLine

# --- The Terzaghi problem ---
# Material
pi_1 = 1

# Equation
equation_type = EquationType.Biot
primary_variables = PrimaryVariables.usigpv_ls1

# Spacial discretization
sdisc_nelmt = [3, 40]

sdisc_fetype = FeSpaces.ls_1_BDM
sdisc_eorder = [1, 1, 1, 1]

# Temporal discretization
tdisc_dt = 0.0001
tdisc_tend = 10 * tdisc_dt

# Boundary conditions
bc_qtop = 0.1

# Initial conditions
nhSt0S = 0.5

# Output
output_name = "output-terzaghi_nondim-ls"
output_dimensionless = False

# --- Perform calculation ---
# The FemProblem
fem_problem = setup_calculation(
    equation_type,
    primary_variables,
    NondimensionalMaterial(pi_1, 0.0, 0.0, l_ref=1.0),
    bc_qtop,
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
        "centerline",
        start=np.array([0.2, 0.0, 0.0]),
        end=np.array([0.2, 1, 0.0]),
    ),
    only_nth_step=10,
)

# Solve problem
fem_problem.solve_problem(
    tdisc_dt,
    tdisc_tend,
    output_paraview=True,
    output_name=output_name,
    post_processors=[plot_over_line],
)
