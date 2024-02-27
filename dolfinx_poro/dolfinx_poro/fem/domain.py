# --- Imports ---
import typing

import dolfinx.mesh as dmesh
import ufl


# --- The Domain ---
class Domain:
    def __init__(
        self,
        mesh: dmesh.Mesh,
        facet_fkts: typing.Any,
        ds: typing.Any,
        quadrature_degree: typing.Optional[int] = None,
        dv: typing.Optional[typing.Any] = None,
    ):
        # Mesh
        self.mesh = mesh

        # Facet functions
        self.facet_functions = facet_fkts

        # Integrators
        self.ds = ds

        if dv is None:
            if quadrature_degree is None:
                self.dv = ufl.dx
            else:
                self.dv = ufl.dx(degree=quadrature_degree)
        else:
            self.dv = dv
