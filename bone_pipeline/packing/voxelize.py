"""Voxelize an STL mesh into a 3D boolean array at given voxel spacing."""

import numpy as np
from pathlib import Path
import trimesh


def voxelize_stl(stl_path, spacing):
    """Load an STL file and convert to a 3D voxel grid.

    Parameters
    ----------
    stl_path : str or Path
        Path to the STL file.
    spacing : array-like of float
        Voxel spacing (z, y, x) in mm.

    Returns
    -------
    voxels : 3D bool ndarray
        Binary voxel representation of the shape.
    """
    stl_path = Path(stl_path)
    if not stl_path.exists():
        raise FileNotFoundError(f"STL file not found: {stl_path}")

    mesh = trimesh.load(stl_path, force='mesh')

    pitch = float(min(spacing))
    voxelized = mesh.voxelized(pitch)
    matrix = voxelized.matrix

    target_spacing = np.array(spacing, dtype=float)
    current_spacing = np.array([pitch, pitch, pitch])
    scale = current_spacing / target_spacing

    if not np.allclose(scale, 1.0, atol=0.05):
        from scipy.ndimage import zoom
        matrix = zoom(matrix.astype(float), scale, order=0) > 0.5

    return matrix.astype(bool)
