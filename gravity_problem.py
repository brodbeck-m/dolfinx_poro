"""
Testcase: Gravity
"""
# --- Imports ---
import numpy as np

from dolfinx_poro import EquationType, PrimaryVariables, FeSpaces, Material
from dolfinx_poro.fem.post_processor import Line, PlotOverLine

from dolfinx_poro.cases.case_gravity import setup_calculation

# --- The footing problem ---
# Material
mat_lhS = 4.0e6
mat_mhS = 4.0e6
mat_rhohSR = 1.0
mat_khSt0S = 1.0e-7
mat_rhohFR = 2.0
mat_mhFR = 1

# Domain
domain_width = 5
domain_hight = 10
l_ref = 10.0

# Equation
equation_type = EquationType.Biot
primary_variables = PrimaryVariables.up

# Spacial discretization
sdisc_nelmt = [36, 72]

sdisc_fetype = FeSpaces.up_TH
sdisc_eorder = [2, 1]

# Temporal discretization
tdisc_dt = 0.05
tdisc_tend = 20 * tdisc_dt

# Boundary conditions
body_force = [0, -1.0e5]

# Initial conditions
nhSt0S = 0.5

# Output
output_name = "output-gravity"
output_dimensionless = True

# --- Perform calculation ---
material = Material(
    mat_lhS, mat_mhS, mat_rhohSR, mat_khSt0S, mat_rhohFR, mat_mhFR, l_ref
)

# The FemProblem
fem_problem = setup_calculation(
    equation_type,
    primary_variables,
    material,
    body_force,
    domain_geometry=[domain_width, domain_hight],
    sdisc_fetype=sdisc_fetype,
    sdisc_eorder=sdisc_eorder,
    sdisc_nelmt=sdisc_nelmt,
    initcond_nhSt0S=nhSt0S,
    scale_output=output_dimensionless,
)

# Postprocessing
if output_dimensionless:
    scale_x = 1 / material.l_ref
else:
    scale_x = 1

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
    scaling_position=scale_x,
)

# Solve problem
fem_problem.solve_problem(
    tdisc_dt,
    tdisc_tend,
    output_paraview=True,
    output_name=output_name,
    post_processors=[plot_over_line],
)
