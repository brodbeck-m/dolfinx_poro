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

from dolfinx_poro.cases.case_growth import setup_calculation
from dolfinx_poro.post_processor import EvaluateMass

# --- The footing problem ---
# Non-dimensional parameters
pi_1 = 10
pi_2 = 0.2
pi_4 = 1

nhSt0S = 0.2
nondim_rhohatS = 10

# Domain
domain_width = 1
domain_hight = 1

# Equation
equation_type = EquationType.Biot
primary_variables = PrimaryVariables.upn

# Spacial discretization
sdisc_nelmt = [36, 36]

sdisc_fetype = FeSpaces.up_TH
sdisc_eorder = [2, 1, 1]

# Temporal discretization
tdisc_dt = 0.011 / 100
tdisc_tend = 0.011

# Output
output_name = "output-growth"

# --- Perform calculation ---
material = NondimensionalMaterial(pi_1, pi_2, 0.0, pi_4, l_ref=1.0)

# The FemProblem
fem_problem = setup_calculation(
    equation_type,
    primary_variables,
    material,
    nondim_rhohatS,
    domain_geometry=[1.0, 1.0],
    sdisc_fetype=sdisc_fetype,
    sdisc_discont_volfrac=False,
    sdisc_eorder=sdisc_eorder,
    sdisc_nelmt=sdisc_nelmt,
    initcond_nhSt0S=nhSt0S,
    scale_output=False,
    allow_outflow=True,
    traction_top=0.0,
)

mass_evaluator = EvaluateMass(
    fem_problem.domain,
    fem_problem.solution_space,
    material,
    equation_type,
    primary_variables,
)

# Solve problem
fem_problem.solve_problem(
    tdisc_dt,
    tdisc_tend,
    output_paraview=True,
    output_name=output_name,
    post_processors=[mass_evaluator],
)
