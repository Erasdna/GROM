import h5py
import nibabel as nib
from mpi4py import MPI
from pathlib import Path
import adios4dolfinx as a4d
from src.Mesher import LoadMesher
from omegaconf import OmegaConf
from src.steady_MPET import SteadyMPET
from src.threecomp import threecomp
import argparse
from dolfinx import fem, io
from src.mesh_deformation import deform_mesh, deform_function
from src.fenics_utils import save_h5

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process image and simulation results")
    parser.add_argument("config", type=str, help="Path to config file")
    parser.add_argument(
        "file", type=str, help="Path to brainmask folder"
    )  # i.e source folder
    parser.add_argument("Sub", type=str, help="Target subject name")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    cfg.target_name = args.Sub
    image_to_warp = Path(args.file)

    # Establish baseline paths
    warp_path = Path(cfg.warp.path)
    simulation_path = Path(cfg.simulation.path)

    # Find target image and load
    target_image_path = Path(cfg.target.image)
    target_affine = nib.load(target_image_path).affine

    # Find target simulation states file
    target_a4d = Path(cfg.target.simulation)
    target_mesher = LoadMesher(target_a4d)
    # Setup solver class to get the correct function spaces
    if cfg.simulation.problem == "MPET":
        p = SteadyMPET(target_mesher)
    elif cfg.simulation.problem == "threecomp":
        p = threecomp(target_mesher)
    else:
        print("Invalid problem type...exiting")
        exit()

    target_V = p.V
    target_solution = fem.Function(target_V)
    # Load target states
    a4d.read_function(target_a4d, target_solution)

    # Load measures
    dx = target_mesher.dx()
    pial_ds = p.pial_ds
    ventricle_ds = p.ventricle_ds

    print("Process original images and warped solutions")
    # Process original images and warped solutions
    file = image_to_warp.joinpath(image_to_warp.stem + "_brainmask.mgz")
    filename = file.stem[:7]
    # warped_image_path = warp_path.joinpath(file.stem, cfg.warp.warped)
    if filename != target_image_path.stem[:7]:  # and warped_image_path.is_file():
        save_path = Path(cfg.saving.path)
        save_path.mkdir(parents=True, exist_ok=True)
        h5_file = Path(cfg.saving.h5_file)

        # Read states which will be warped
        source_states = Path(cfg.simulation.path).joinpath(
            filename, cfg.simulation.original_resolution, filename + "-states.bp"
        )
        source_mesh = a4d.read_mesh(source_states, MPI.COMM_WORLD)
        source_V = p._init_functionspace(source_mesh)
        source_f = fem.Function(source_V)
        a4d.read_function(source_states, source_f)

        # Load affine and deform the mesh according to the warp
        source_affine = nib.load(file).affine
        new_coords, _ = deform_mesh(
            source_mesh=source_mesh,
            source_affine=source_affine,
            target_affine=target_affine,
            affine_path=warp_path.joinpath(file.stem, cfg.warp.affine),
            warp_path=warp_path.joinpath(file.stem, cfg.warp.warp),
            invert_affine=False,
        )

        source_f.function_space.mesh.geometry.x[:] = new_coords

        deform_mesh_path = (
            Path(cfg.base)
            .joinpath("drawn_figures", filename, filename + "_deformed")
            .with_suffix(".xdmf")
        )
        with io.XDMFFile(MPI.COMM_WORLD, deform_mesh_path, "w") as xdmf:
            xdmf.write_mesh(source_f.function_space.mesh)

        new_f, hausdorff = deform_function(source_f, target_mesher.domain, p.V)
        # Save some statistics
        err_dict = p.compute_error(target_solution, new_f)

        not_available = True
        while not_available:
            try:
                with h5py.File(h5_file, mode="a") as f:
                    # Save the target state
                    try:
                        t = f.create_group("target")
                    except:
                        t = f["target"]

                    # for k in p.index_convention.keys():
                    try:
                        t["coefs"] = target_solution.x.array[:]
                    except:
                        t["coefs"][()] = target_solution.x.array[:]

                    # Save the index maps
                    try:
                        t = f.create_group("maps")
                    except:
                        t = f["maps"]

                    for k in p.index_convention.keys():
                        try:
                            t[k] = p.maps[p.index_convention[k]]
                        except:
                            t[k][()] = p.maps[p.index_convention[k]]

                    try:
                        f.create_group("warped")
                    except:
                        pass

                    try:
                        g = f["warped"].create_group(filename)
                    except:
                        g = f["warped/" + filename]

                    try:
                        g["coefs"] = new_f.x.array[:]
                    except:
                        g["coefs"][()] = new_f.x.array[:]
                        not_available = False
            except:
                print("Could not open file... Trying again")
                pass

        # p.compute_error(target_solution, corrected_f)
        not_available = True
        while not_available:
            try:
                with h5py.File(h5_file, "a") as f:
                    g = f[
                        "warped/" + filename
                    ]  # Created in the previous opening of the file

                    save_h5(
                        g,
                        [err_dict["all"]["L2"], err_dict["all"]["H1"], hausdorff],
                        ["L2", "H1", "hausdorff"],
                    )

                    try:
                        g = g.create_group("compartments")
                    except:
                        g = g["compartments"]

                    for comp, dict in err_dict["compartments"].items():
                        try:
                            sub_g = g.create_group(comp)
                        except:
                            sub_g = g[comp]

                        save_h5(sub_g, dict.values(), dict.keys())
                not_available = False
            except:
                print("Could not open file... Trying again")
                pass
