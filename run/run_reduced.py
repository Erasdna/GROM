import h5py
from omegaconf import OmegaConf
from src.Mesher import LoadMesher
from src.steady_MPET import SteadyMPET
from src.threecomp import threecomp
from pathlib import Path
import argparse
import sys
import numpy as np
from src.fenics_utils import save_h5

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate meshes with various resolutions from a config file"
    )
    parser.add_argument("config", type=str, help="Path to config file")
    parser.add_argument("sub", type=str, help="Brainmask prefix")
    parser.add_argument("resolution", type=int, help="Mesh resolution")
    parser.add_argument("problem", type=str, help="Problem type")
    parser.add_argument(
        "basis_folder", type=str, help="Path to folder containing reduced basis"
    )
    parser.add_argument(
        "basis_name", type=str, help="Name of the reduced basis"
    )
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    cfg.brainmask.name = args.sub[:7]
    cfg.mesh.mesh_config.mesh.resolution = args.resolution
    cfg.problem = args.problem
    basis_folder = Path(args.basis_folder)
    basis_name = args.basis_name

    mesher = LoadMesher(Path(cfg.mesh.mesh_config.mesh.read_mesh))

    if cfg.problem == "MPET":
        p = SteadyMPET(mesher)
    elif cfg.problem == "threecomp":
        p = threecomp(mesher)
    else:
        print("Invalid problem type...exiting")
        exit()

    print("Solving high fidelity problem")
    fom = p.solve()
    print("Finished solving high fidelity problem")

    sys.stdout.flush()

    # Load the basis
    with h5py.File(basis_folder.joinpath("basis.h5"), "r") as h5:
        U = h5[basis_name + "/U"][()]

    with h5py.File(basis_folder.joinpath("reduced.h5"), "a") as save_result:
        try: 
            save_result.create_group(basis_name)
        except:
            pass 
        
    print("Basis shape: ", U.shape)
    # Basis sanity check
    fom_norm = np.linalg.norm(fom.x.array[:])
    print(
        "Relative error full basis: ",
        np.linalg.norm(fom.x.array[:] - U @ (U.T @ fom.x.array[:])) / fom_norm,
    )

    for b in [1,4,9]: #range(1,U.shape[1]):
        # Solve the problem
        basis = U[:, : b + 1]
        print("Solving problem for basis of size: ", b + 1)
        red = p.solve(basis=basis)
        print("Finished solving")

        print(
            "Relative error: ",
            np.linalg.norm(fom.x.array[:] - red.x.array[:]) / fom_norm,
        )
        err_dict = p.compute_error(fom,red)
        with h5py.File(basis_folder.joinpath("reduced.h5"), "a") as save_result:
            save_result = save_result[basis_name]
            try:
                g = save_result.create_group(str(b + 1))
            except:
                g = save_result[str(b + 1)]

            # Compute H1 norm, L2 norm:
            save_h5(g, [red.x.array[:],err_dict["all"]["L2"],err_dict["all"]["H1"]], ["coefs","L2","H1"])
            
            try:
                g = g.create_group("compartments")
            except:
                g = g["compartments"]
            
            for (comp, dict) in err_dict["compartments"].items():
                try:
                    sub_g = g.create_group(comp)
                except:
                    sub_g = g[comp]
                    
                save_h5(sub_g, dict.values(), dict.keys()) 
                

