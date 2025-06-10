from dolfinx.io import XDMFFile
from mpi4py import MPI
from src.svmtk_utils import (
    repair,
    create_volume_mesh,
    convert_subdomains,
    remesh_surface,
)
import SVMTK as svmtk
from pathlib import Path
from dolfinx import mesh
import adios4dolfinx as a4d
import numpy as np
import ufl


class Mesher:
    def __init__(self, mesh_config) -> None:
        self.mesh_config = mesh_config
        self._mesh()

    def _mesh(self):
        savepath = Path(self.mesh_config.save.path)
        savepath.mkdir(parents=True, exist_ok=True)

        meshfile_xdmf = self.mesh_config.save.xdmf
        tmp_mesh = self.mesh_config.save.tmp
        facet_file = self.mesh_config.save.facets
        if MPI.COMM_WORLD.Get_rank() == 0 and (
            not Path(meshfile_xdmf).is_file() or self.mesh_config.mesh.remesh
        ):
            # Make mesh and save to xdmf
            surfaces, smap, remove = self._make_subdomains()
            domain = create_volume_mesh(
                surfaces=surfaces,
                resolution=self.mesh_config.mesh.resolution,
                smap=smap,
            )

            if remove is not None:
                for tag in remove:
                    domain.remove_subdomain(tag)

            domain.save(tmp_mesh)
            convert_subdomains(tmp_mesh, meshfile_xdmf, facet_file)
        self.domain, self.cell_tags, self.facet_tags = self.load_xdmf(meshfile_xdmf, facet_file)

    def _make_subdomains(self):
        pass

    def load_xdmf(self, infile, facet_infile):
        with XDMFFile(MPI.COMM_WORLD, infile, "r") as xdmf:
            domain = xdmf.read_mesh(name="Grid")
            try:
                ct = xdmf.read_meshtags(domain, name="Grid")
            except RuntimeError:
                ct = xdmf.read_meshtags(domain, name="mesh_tags")

        with XDMFFile(MPI.COMM_WORLD, facet_infile, "r") as xdmf:
            domain.topology.create_connectivity(
                domain.topology.dim - 1, domain.topology.dim
            )
            try:
                ft = xdmf.read_meshtags(domain, name="Grid")
            except RuntimeError:
                ft = xdmf.read_meshtags(domain, name="mesh_tags")
        return domain, ct, ft

    def load(self, meshfile):
        domain = a4d.read_mesh(meshfile, comm=MPI.COMM_WORLD)
        cell_tags = a4d.read_meshtags(meshfile, domain, "cell_tags")
        facet_tags = a4d.read_meshtags(meshfile, domain, "facet_tags")
        return domain, cell_tags, facet_tags

    def save(self, filename):
        a4d.write_mesh(filename, self.domain)
        a4d.write_meshtags(
            filename, self.domain, self.cell_tags, meshtag_name="cell_tags"
        )
        a4d.write_meshtags(
            filename, self.domain, self.facet_tags, meshtag_name="facet_tags"
        )

    def ds(self) -> ufl.Measure:
        return ufl.Measure("ds", domain=self.domain, subdomain_data=self.facet_tags)

    def dx(self) -> ufl.Measure:
        return ufl.Measure("dx", domain=self.domain, subdomain_data=self.cell_tags)


class LoadMesher(Mesher):
    def __init__(self, mesh_file : Path) -> None:
        self.domain, self.cell_tags, self.facet_tags = self.load(Path(mesh_file))

            

# Create a shell with a hollow core to be removed
class HollowShell(Mesher):
    def _make_subdomains(self):
        surfs = self._make_surf()
        smap = svmtk.SubdomainMap()
        smap.add("10", 1)
        smap.add("11", 2)
        return surfs, smap, [2]

    def _make_surf(self):
        return [
            repair(svmtk.Surface(self.mesh_config.ellipsoid.outer)),
            repair(svmtk.Surface(self.mesh_config.ellipsoid.inner)),
        ]

# Use the right + left side merge for the meshing of the pial. Removing the ventricles corresponds to making a hollow core
class PialVentricleMesher(HollowShell):
    def _make_surf(self):
        ventricles = repair(svmtk.Surface(self.mesh_config.brain.ventricles))
        ventricles.isotropic_remeshing(1.0, 3, False)
        return [
            self._combine_lh_rh(),
            ventricles,
        ]

    def _combine_lh_rh(self):
        if self.mesh_config.brain.pial.use_full:
            pial = repair(svmtk.Surface(self.mesh_config.brain.pial.pial))
            pial = remesh_surface(pial, 2, 3)
        else:
            lh = repair(svmtk.Surface(self.mesh_config.brain.pial.lh))
            rh = repair(svmtk.Surface(self.mesh_config.brain.pial.rh))
            print("Separate overlapping surfaces")
            svmtk.separate_overlapping_surfaces(lh, rh)
            print("Separate close surfaces")
            svmtk.separate_close_surfaces(lh, rh)
            lh = remesh_surface(lh, 1.0, 3)
            rh = remesh_surface(rh, 1.0, 3)
            pial = svmtk.union_partially_overlapping_surfaces(rh, lh)
            pial = repair(pial)
        return pial
