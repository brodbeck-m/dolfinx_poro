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
from dolfinx_poro.cases.case_terzaghi import setup_calculation, AnalyticSolution
from dolfinx_poro.fem.post_processor import Line, PlotOverLine

# --- The Terzaghi problem ---
# Material
pi_1 = 1

# Equation
equation_type = EquationType.Biot
primary_variables = PrimaryVariables.usigpv_ls1

# Spacial discretization
sdisc_nelmt = [6, 160]

sdisc_fetype = FeSpaces.ls_1_BDM
sdisc_eorder = [1, 1, 1, 1]

# Temporal discretization
tdisc_dt = 0.3 / 320
tdisc_tend = 0.3

# Geometry and boundary conditions
geom_L = 5.0
bc_qtop = 0.1

# Initial conditions
nhSt0S = 0.5

# Output
output_name = "terzaghi-ndim"
output_dimensionless = False

# --- Perform calculation ---
# The FemProblem
fem_problem = setup_calculation(
    equation_type,
    primary_variables,
    NondimensionalMaterial(pi_1, 0.0, 0.0, l_ref=1.0),
    bc_qtop,
    domain_hight=geom_L,
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
        start=np.array([geom_L / 5, 0.0, 0.0]),
        end=np.array([geom_L, geom_L, 0.0]),
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

# Analytic solution
outname_anlt = output_name + "-analytic"
anlt_sol = AnalyticSolution(
    pi_1, 1.0, geom_L, bc_qtop, z=np.linspace(0, geom_L, 50), n_coeffs=1000
)

anlt_sol.evaluate_solution(
    0.5,
    np.r_[0 : tdisc_tend + tdisc_dt : tdisc_dt],
    out_name=outname_anlt,
    map_seepage_velocity=False,
)
