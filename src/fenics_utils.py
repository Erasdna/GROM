from dolfinx import fem, io
import numpy as np
import ufl
from mpi4py import MPI
import adios4dolfinx as a4d
from pathlib import Path
import dolfinx
from nibabel.affines import apply_affine
import nibabel as nib


def a4d_to_XDMF(
    a4d_file: Path, xdmf_file: Path | list[Path], mesher, functionspace_init
):
    # Read the mesh:
    domain, cell_tags, facet_tags = mesher.load(a4d_file)
    f = fem.Function(functionspace_init(domain))
    a4d.read_function(a4d_file, f)

    for f_sub, file in zip(f.split(), xdmf_file):
        with io.XDMFFile(MPI.COMM_WORLD, file, "w") as xdmf:
            xdmf.write_mesh(domain)
            xdmf.write_function(f_sub)


def save_h5(h5_group, data, names):
    for d, name in zip(data, names):
        try:
            h5_group[name] = d
        except:
            h5_group[name][()] = d


def norm(expr, norm_type, comm):
    def sq(expr):
        return ufl.inner(expr, expr)

    if norm_type == "L2":
        loc_form = fem.assemble_scalar(fem.form(sq(expr) * ufl.dx))
    elif norm_type == "H1":
        loc_form = fem.assemble_scalar(
            fem.form((sq(expr) + sq(ufl.grad(expr))) * ufl.dx)
        )
    return np.sqrt(comm.allreduce(loc_form))


# Adapted from: https://github.com/scientificcomputing/scifem/blob/main/src/scifem/eval.py
# to allow for better handling op points outside the domain
def evaluate_function(u: dolfinx.fem.Function, points, padding=0):
    mesh = u.function_space.mesh
    u.x.scatter_forward()
    comm = mesh.comm
    points = np.array(points, dtype=np.float64)
    assert (
        len(points.shape) == 2
    ), f"Expected points to have shape (num_points, dim), got {points.shape}"
    num_points = points.shape[0]
    extra_dim = 3 - mesh.geometry.dim

    # Append zeros to points if the mesh is not 3D
    if extra_dim > 0:
        points = np.hstack((points, np.zeros((points.shape[0], extra_dim))))

    bb_tree = dolfinx.geometry.bb_tree(mesh, mesh.topology.dim, padding=padding)
    # Find cells whose bounding-box collide with the the points
    potential_colliding_cells = dolfinx.geometry.compute_collisions_points(
        bb_tree, points
    )
    # Choose one of the cells that contains the point
    adj = dolfinx.geometry.compute_colliding_cells(
        mesh, potential_colliding_cells, points
    )
    indices = np.flatnonzero(adj.offsets[1:] - adj.offsets[:-1])
    cells = adj.array[adj.offsets[indices]]
    points_on_proc = points[indices]

    ret_values = np.full(points.shape[0], np.nan)
    if len(indices) == 0:
        for i in range(points.shape[0]):
            cells = potential_colliding_cells.links(i)
            if len(cells) > 0:
                midpoints = dolfinx.mesh.compute_midpoints(mesh, 3, cells)
                vals = u.eval(midpoints, cells)
                dist = np.linalg.norm(points[i, :] - midpoints, axis=-1)
                weight = np.exp(-0.5 * (dist / np.min(dist)) ** 2)
                ret_values[i] = np.sum(vals.T * weight) / np.sum(weight)
    else:
        values = u.eval(points_on_proc, cells)
        ret_values[indices] = values.flatten()
    return ret_values


def save_function_as_image(f : fem.Function, names : list, img : nib.nifti1.Nifti1Image, save_loc : Path):
    
    save_loc.mkdir(parents=True, exist_ok=True)
    affine = img.header.get_vox2ras_tkr()
    coords = np.argwhere(img.get_fdata() > 0)
    ras_coords = apply_affine(affine, coords)

    for sub, name in zip(f.split(), names):
        print(name)
        vals = evaluate_function(sub, ras_coords)
        new_img_data = np.zeros_like(img.get_fdata())
        new_img_data[*coords.T] = vals
        new_img = nib.nifti1.Nifti1Image(new_img_data, img.affine, img.header)
        nib.save(new_img, save_loc.joinpath(name).with_suffix(".nii"))
        
