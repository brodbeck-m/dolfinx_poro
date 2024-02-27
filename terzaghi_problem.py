"""
Testcase: Terzaghi problem
"""
# --- Imports ---
import numpy as np

from dolfinx_poro import EquationType, PrimaryVariables, FeSpaces, Material
from dolfinx_poro.cases.case_terzaghi import setup_calculation
from dolfinx_poro.fem.post_processor import Line, PlotOverLine

# --- The Terzaghi problem ---
# Material
mat_EhS = 1.0e7
mat_nuhS = 0.25
mat_rhohSR = 1
mat_khSt0S = 1.0e-6
mat_rhohFR = 1
mat_mhFR = 1

# Domain
domain_hight = 5.0
l_ref = 1.0

# Equation
equation_type = EquationType.Biot
primary_variables = PrimaryVariables.up

# Spacial discretization
sdisc_nelmt = [9, 72]

sdisc_fetype = FeSpaces.up_TH
sdisc_eorder = [2, 1]

# Temporal discretization
tdisc_dt = 0.005
tdisc_tend = 300 * tdisc_dt

# Boundary conditions
bc_qtop = 10000

# Initial conditions
nhSt0S = 0.5

# Output
output_name = "output-terzaghi"
output_dimensionless = False

# --- Perform calculation ---
# The material parameters
mat_lhs = (mat_nuhS * mat_EhS) / ((1 + mat_nuhS) * (1 - 2 * mat_nuhS))
mat_mhs = mat_EhS / (2 * (1 + mat_nuhS))

material = Material(
    mat_lhs, mat_mhs, mat_rhohSR, mat_khSt0S, mat_rhohFR, mat_mhFR, l_ref
)

# The FemProblem
fem_problem = setup_calculation(
    equation_type,
    primary_variables,
    material,
    bc_qtop,
    domain_hight=domain_hight,
    sdisc_fetype=sdisc_fetype,
    sdisc_nelmt=sdisc_nelmt,
    sdisc_eorder=sdisc_eorder,
    initcond_nhSt0S=nhSt0S,
    scale_output=output_dimensionless,
)

# Postprocessing
domain_width = domain_hight / 5

plot_over_line = PlotOverLine(
    fem_problem.domain,
    fem_problem.solution_space,
    fem_problem.material,
    Line(
        250,
        "centerline",
        start=np.array([domain_width, 0.0, 0.0]),
        end=np.array([domain_width, domain_hight, 0.0]),
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
