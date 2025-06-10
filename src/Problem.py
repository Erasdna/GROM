from mpi4py import MPI
from dolfinx import fem, default_scalar_type
import ufl
from dolfinx.fem.petsc import LinearProblem
from pathlib import Path
from dolfinx import fem
import numpy as np
import adios4dolfinx as a4d


class Problem:
    def __init__(self, mesher) -> None:
        self.mesher = mesher
        self.V = self._init_functionspace(self.mesher.domain)

    def save(self, f: fem.Function, results_file: Path):
        # Write the mesh with tags
        self.mesher.save(results_file)
        # Write the function
        a4d.write_function(results_file, f)

    def load(self, filename: Path):
        # Load mesh
        domain,_,_ = self.mesher.load(filename)
        u = fem.Function(self._init_functionspace(domain))
        a4d.read_function(filename, u)
        return u

    def _solve(self):
        pass

    def solve(self, *args, **kwargs):
        return self._solve(*args, **kwargs)

    def _init_functionspace(self, brain):
        pass
    
    def _compute_volumes(self):
        self.volumes = {"volume": dict(), "areas": dict()}

        # Sometimes the order of the surface tags are switched between pial and ventricle. The pial surface should always be bigger
        a1 = self.mesher.domain.comm.allreduce(
            fem.assemble_scalar(fem.form(1 * self.ds(2)))
        )
        a2 = self.mesher.domain.comm.allreduce(
            fem.assemble_scalar(fem.form(1 * self.ds(3)))
        )

        if a1 > a2:
            pia = a1
            self.pial_ds = self.ds(2)
            ventricle = a2
            self.ventricle_ds = self.ds(3)
            self.facet_map = {"pial" : 2, "ventricle" : 3}
        else:
            pia = a2
            self.pial_ds = self.ds(3)
            ventricle = a1
            self.ventricle_ds = self.ds(2)
            self.facet_map = {"pial" : 3, "ventricle" : 2}

        volume = fem.assemble_scalar(fem.form(1 * self.dx))


        # reduce across processes
        self.volumes["areas"]["pia"] = pia
        self.volumes["areas"]["ventricles"] = ventricle
        self.volumes["volume"]["brain"] = self.mesher.domain.comm.allreduce(volume)
