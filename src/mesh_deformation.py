import numpy as np
from nibabel.affines import apply_affine
from dolfinx.mesh import Mesh, compute_midpoints
from dolfinx.fem import Function, functionspace
from dolfinx import geometry
from pathlib import Path
from scipy.io import loadmat
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import map_coordinates
import nibabel as nib
from src.fenics_utils import evaluate_function

def make_affine(mat):
    aff = np.eye(4, 4)
    aff[:3, :3] = mat["AffineTransform_double_3_3"][:9].reshape(3, 3)

    aff[:3, -1] = (
        (mat["AffineTransform_double_3_3"][9:] + mat["fixed"])
        - aff[:3, :3] @ mat["fixed"]
    )[:, 0]

    return aff


def deform_mesh(
    source_mesh: Mesh,
    source_affine,
    target_affine,
    affine_path: Path,
    warp_path: Path,
    invert_affine=True,
):
    affine = make_affine(loadmat(affine_path))

    print(affine_path)
    if invert_affine:
        affine = np.linalg.inv(affine)

    # Load the warp:
    nonlinear_warp = nib.load(warp_path).get_fdata()

    # Voxel -> RAS space is always the same matrix
    vox2ras = np.array(
        [
            [-1.0, 0.0, 0.0, 128.0],
            [0.0, 0.0, 1.0, -128.0],
            [0.0, -1.0, 0.0, 128.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    # The nibabel convention inverts sign of first and second coordinates
    sign_correction = np.eye(4, 4)
    sign_correction[0, 0] = -1
    sign_correction[1, 1] = -1
    source_vox2itk = sign_correction @ source_affine
    target_vox2itk = sign_correction @ target_affine

    source_coords_RAS = source_mesh.geometry.x

    # Convert from RAS space to itk physical space using the source image
    source_coords_vox = apply_affine(np.linalg.inv(vox2ras), source_coords_RAS)
    source_coords_physical = apply_affine(source_vox2itk, source_coords_vox)

    # Apply affine registration to the points in physical space:
    affine_coords_physical = apply_affine(affine, source_coords_physical)
    # Convert to voxel coordinates to sample the nonlinear registration. Note that we use the target physical coordinates:
    affine_coords_vox = apply_affine(
        np.linalg.inv(target_vox2itk), affine_coords_physical
    )

    u = np.zeros_like(affine_coords_physical)

    if invert_affine:
        roi = affine_coords_vox.T
    else:
        roi = source_coords_vox.T

    for i in range(3):
        u[..., i] = map_coordinates(nonlinear_warp[..., 0, i], roi, order=1)

    # Apply the warp in physical space
    nonlinear_coords_physical = affine_coords_physical + u

    # Go back to RAS space:
    nonlinear_coords_vox = apply_affine(
        np.linalg.inv(target_vox2itk), nonlinear_coords_physical
    )
    nonlinear_coords_RAS = apply_affine(vox2ras, nonlinear_coords_vox)

    return nonlinear_coords_RAS, u


def deform_function(
    source_function: Function,
    target_mesh: Mesh,
    target_functionspace: functionspace,
):
    source_mesh = source_function.function_space.mesh
    target_f = Function(target_functionspace)
    #points = target_mesh.geometry.x[:]

    for i in range(len(source_function.split())):

        V0, maps = target_functionspace.sub(i).collapse()
        points = V0.tabulate_dof_coordinates()
        
        maps = np.array(maps)
        f = source_function.sub(i)
        ids = np.arange(len(maps))
        padding = 0
        while len(ids) > 0:
            values = evaluate_function(f, points[ids], padding=padding)
            target_f.x.array[maps[ids]] = values.flatten()

            ids = np.argwhere(np.isnan(target_f.x.array[maps])).flatten()
            padding += 1

    hausdorff = directed_hausdorff(source_mesh.geometry.x, target_mesh.geometry.x)
    return target_f, hausdorff
