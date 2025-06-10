import SVMTK as svmtk
import meshio
from pathlib import Path


def create_volume_mesh(surfaces, resolution=16, smap=None):
    # Load input file
    # surface = svmtk.Surface(stlfile)

    # Generate the volume mesh
    if smap is not None:
        domain = svmtk.Domain(surfaces, smap)
    else:
        domain = svmtk.Domain(surfaces)

    domain.create_mesh(resolution)

    # Write the mesh to the output file
    return domain


def smoothen_surface(surface, n=1, eps=1.0, preserve_volume=True):
    # Load input STL file
    # surface = svmtk.Surface(stl_input)

    # Smooth using Taubin smoothing
    # if volume should be preserved,
    # otherwise use Laplacian smoothing
    if preserve_volume:
        surface.smooth_taubin(n)
    else:
        surface.smooth_laplacian(eps, n)

    return surface


def remesh_surface(surface, L, n, do_not_move_boundary_edges=False):

    # Load input STL file
    # surface = svmtk.Surface(stl_input)

    # Remesh surface
    surface.isotropic_remeshing(L, n, do_not_move_boundary_edges)

    # Save remeshed STL surface
    return surface
    # surface.save(output)


def repair(surface):
    # Import the STL surface
    # surf = svmtk.Surface(stl_file)

    # Find and fill holes
    surface.fill_holes()

    # Separate narrow gaps
    # Default argument is -0.33.
    surface.separate_narrow_gaps(-0.25)
    return surface


# medit files are the default file types made by the SVMTk
def convert(meshFile, saveFile, fileFormatAccess="medit:ref"):
    msh = meshio.read(meshFile)

    for cell in msh.cells:
        if cell.type == "triangle":
            triangle_cells = cell.data
        elif cell.type == "tetra":
            tetra_cells = cell.data

    for key in msh.cell_data_dict[fileFormatAccess].keys():
        if key == "triangle":
            triangle_data = msh.cell_data_dict[fileFormatAccess][key]
        elif key == "tetra":
            tetra_data = msh.cell_data_dict[fileFormatAccess][key]
    tetra_mesh = meshio.Mesh(points=msh.points, cells={"tetra": tetra_cells})
    triangle_mesh = meshio.Mesh(
        points=msh.points,
        cells=[("triangle", triangle_cells)],
        cell_data={"name_to_read": [triangle_data]},
    )

    print("")
    print(f"Writing FEniCS compatible mesh files")
    print(
        "You will need both of these files to be present when calling XDMFFile(...) in FEniCS"
    )
    print(
        "---> Saving *tetrahedral* mesh conversion (for triangular meshes, use SVMTk-tri-mesh-to-FEniCS-XDMF.py)"
    )
    meshio.write(saveFile, tetra_mesh)


def convert_subdomains(meshFile, cell_mesh_name, facets_mesh_name):
    msh = meshio.read(meshFile)

    # try:
    tmp_dir = Path(".tmp")
    tmp_dir.mkdir(exist_ok=True)

    cell_mesh = meshio.Mesh(
        points=msh.points,
        cells={"tetra": msh.cells_dict["tetra"]},
        cell_data={"subdomains": [msh.cell_data_dict["medit:ref"]["tetra"]]},
    )

    meshio.write(cell_mesh_name, cell_mesh)

    facet_mesh = meshio.Mesh(
        points=msh.points,
        cells={"triangle": msh.cells_dict["triangle"]},
        cell_data={"patches": [msh.cell_data_dict["medit:ref"]["triangle"]]},
    )
    meshio.write(facets_mesh_name, facet_mesh)
