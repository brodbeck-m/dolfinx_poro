# --- Imports ---
import typing

from dolfinx import default_scalar_type
import dolfinx.fem as dfem
import dolfinx.mesh as dmesh


# --- Definition of dolfinX constant ---
def set_femconstant(mesh: dmesh.Mesh, value: float):
    return dfem.Constant(mesh, default_scalar_type(value))
