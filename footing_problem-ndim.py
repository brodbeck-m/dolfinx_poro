"""
Testcase: Footing problem
"""

# --- Imports ---
import numpy as np

from dolfinx_poro import (
    EquationType,
    PrimaryVariables,
    FeSpaces,
    NondimensionalMaterial,
)
from dolfinx_poro.fem.post_processor import Line, PlotOverLine

from dolfinx_poro.cases.case_footing import setup_calculation

# --- The footing problem ---
# Material
pi_1 = 1.0

# Domain
domain_width = 8
domain_hight = 4

# Equation
equation_type = EquationType.Biot
primary_variables = PrimaryVariables.up

# Spatial discretization
sdisc_fespaces = FeSpaces.up_TH

sdisc_nelmt = [60, 30]
sdisc_eorder = [2, 1]

# Temporal discretization
tdisc_dt = 0.01
tdisc_tend = 100 * tdisc_dt

# Boundary conditions
bc_qtop = 0.1

# Initial conditions
nhSt0S = 0.5

# Output
output_name = "footing-ndim_up"

# --- Perform calculation ---
# The FemProblem
fem_problem = setup_calculation(
    equation_type,
    primary_variables,
    NondimensionalMaterial(pi_1, 0.0, 0.0, l_ref=1.0),
    bc_qtop,
    domain_geometry=[domain_width, domain_hight],
    sdisc_fetype=sdisc_fespaces,
    sdisc_eorder=sdisc_eorder,
    sdisc_nelmt=sdisc_nelmt,
    initcond_nhSt0S=nhSt0S,
)

# Postprocessing
plot_over_line = PlotOverLine(
    fem_problem.domain,
    fem_problem.solution_space,
    fem_problem.material,
    Line(
        250,
        "diagonal",
        start=np.array([0.0, domain_hight, 0.0]),
        end=np.array([domain_width, 0.0, 0.0]),
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
