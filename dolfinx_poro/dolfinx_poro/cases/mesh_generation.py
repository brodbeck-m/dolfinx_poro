# --- Imports ---
from mpi4py import MPI
import numpy as np
from typing import List

import dolfinx.mesh as dmesh
from dolfinx.mesh import CellType, DiagonalType, create_rectangle
import ufl

from dolfinx_poro.fem import Domain


def create_geometry_rectangle(
    l_domain: List[float],
    n_elmt: List[int],
    diagonal: DiagonalType = DiagonalType.left,
):
    # --- Create mesh
    mesh = create_rectangle(
        MPI.COMM_WORLD,
        [np.array([0, 0]), np.array([l_domain[0], l_domain[1]])],
        [n_elmt[0], n_elmt[1]],
        cell_type=CellType.triangle,
        diagonal=diagonal,
    )
    tol = 1.0e-14
    boundaries = [
        (1, lambda x: np.isclose(x[0], 0)),
        (2, lambda x: np.isclose(x[1], 0)),
        (3, lambda x: np.isclose(x[0], l_domain[0])),
        (4, lambda x: np.isclose(x[1], l_domain[1])),
    ]

    facet_indices, facet_markers = [], []
    for marker, locator in boundaries:
        facets = dmesh.locate_entities(mesh, 1, locator)
        facet_indices.append(facets)
        facet_markers.append(np.full(len(facets), marker))

    facet_indices = np.array(np.hstack(facet_indices), dtype=np.int32)
    facet_markers = np.array(np.hstack(facet_markers), dtype=np.int32)
    sorted_facets = np.argsort(facet_indices)
    facet_tag = dmesh.meshtags(
        mesh, 1, facet_indices[sorted_facets], facet_markers[sorted_facets]
    )
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tag)

    return Domain(mesh, facet_tag, ds)
