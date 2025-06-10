from omegaconf import OmegaConf
from src.Mesher import PialVentricleMesher, LoadMesher
from src.steady_MPET import SteadyMPET
from src.threecomp import threecomp
from pathlib import Path
from dolfinx.io import XDMFFile
from dolfinx import fem
import argparse
import sys
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate meshes with various resolutions from a config file"
    )
    parser.add_argument("config", type=str, help="Path to config file")
    parser.add_argument("sub", type=str, help="Brainmask prefix")
    parser.add_argument("resolution", type=int, help="Mesh resolution")
    parser.add_argument("problem", type=str, help="Problem type")

    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    cfg.brainmask.name = args.sub[:7]
    cfg.mesh.mesh_config.mesh.resolution = args.resolution
    cfg.problem = args.problem

    if cfg.mesh.mesh_config.mesh.remesh:
        mesher = PialVentricleMesher(cfg.mesh.mesh_config)
    else:
        mesher = LoadMesher(Path(cfg.mesh.mesh_config.mesh.read_mesh))

    if cfg.problem == "MPET":
        p = SteadyMPET(mesher)
    elif cfg.problem == "threecomp":
        p = threecomp(mesher)
    else:
        print("Invalid problem type...exiting")
        exit()

    sys.stdout.flush()

    # Solve the problem
    f = p.solve()
    print(f.x.array.shape)
    print("Done")
    sys.stdout.flush()
    # Save function:
    saving = Path(cfg.solve.folder)
    saving.mkdir(parents=True, exist_ok=True)
    a4d_file = saving.joinpath(cfg.solve.save_state)
    
    if cfg.solve.plotting.save_plotting:
        with XDMFFile(
            p.mesher.domain.comm,
            saving.joinpath(cfg.solve.plotting.plot).with_suffix(".xdmf"),
            "w",
        ) as xdmf:
            xdmf.write_mesh(f.function_space.mesh)
            if cfg.problem == "MPET":
                tmp_space = fem.functionspace(f.function_space.mesh, ("Lagrange", 1))
                tmp_f = fem.Function(tmp_space)
                for i in range(f.function_space.num_sub_spaces):
                    tmp_f.interpolate(f.sub(i))
                    xdmf.write_function(tmp_f, t=i)
            elif cfg.problem == "threecomp":
                tmp_space = fem.functionspace(f.function_space.mesh, ("Lagrange", 1))
                tmp_f = fem.Function(tmp_space)
                for i in range(f.function_space.num_sub_spaces):
                    tmp_f.interpolate(f.sub(i))
                    xdmf.write_function(tmp_f, t=i)
            else:
                xdmf.write_function(f)

    p.save(f, a4d_file)
