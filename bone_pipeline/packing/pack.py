"""
Mixed specimen packing into bone regions.

Given a region mask (cortical or cancellous) and a set of specimen STL files,
finds the arrangement that packs the maximum number of specimens (all types
mixed) into the available volume without overlap.

Approach:
  1. Voxelize each STL specimen shape at the scan's voxel spacing
  2. Compute distance transform of the region mask (depth into interior)
  3. Generate candidate positions sorted by distance (deepest first)
  4. For each candidate, try all specimen types at multiple orientations
  5. Place the first shape that fits, subtract from available volume, repeat
  6. Return placed specimens with positions, orientations, and voxel masks
"""

import numpy as np
from pathlib import Path
from scipy import ndimage
from scipy.spatial.transform import Rotation

from .voxelize import voxelize_stl


def pack_specimens(region_mask, spacing, stl_paths, orientations_per_shape=6,
                   min_depth_mm=0.5, candidate_stride=None):
    """Pack multiple specimen types into a bone region.

    Parameters
    ----------
    region_mask : 3D bool ndarray
        Binary mask of the region to pack into.
    spacing : tuple of float
        Voxel spacing (z, y, x) in mm.
    stl_paths : list of str or Path
        Paths to STL files defining specimen shapes.
    orientations_per_shape : int
        Number of orientations to test per shape (default 6: axis-aligned).
    min_depth_mm : float
        Minimum distance from region boundary for placement (mm).
    candidate_stride : int or None
        Step size for candidate position grid. None = auto (based on
        smallest specimen dimension).

    Returns
    -------
    dict with keys:
        'placements' : list of dicts, each with:
            'shape_index' : int, index into stl_paths
            'shape_name'  : str, STL filename stem
            'position'    : (z, y, x) voxel coordinates of placement origin
            'orientation' : int, orientation index
            'mask'        : 3D bool ndarray, voxel mask of placed specimen
            'volume_mm3'  : float
        'total_specimens' : int
        'per_shape_count' : dict of {shape_name: count}
        'remaining_mask'  : 3D bool ndarray, unused region
        'packing_efficiency' : float, fraction of region filled
    """
    spacing = np.array(spacing, dtype=float)
    voxel_vol = float(np.prod(spacing))

    shapes = _load_and_prepare_shapes(stl_paths, spacing,
                                      orientations_per_shape)

    if candidate_stride is None:
        min_dim = min(
            min(s['extent']) for s in shapes
        )
        candidate_stride = max(1, min_dim // 2)

    dist = ndimage.distance_transform_edt(region_mask, sampling=spacing)
    min_depth_vox = min_depth_mm / float(np.mean(spacing))

    available = region_mask.copy()
    placements = []
    per_shape_count = {s['name']: 0 for s in shapes}

    candidates = _generate_candidates(dist, min_depth_mm, candidate_stride)

    print(f"Packing {len(shapes)} shape types into region "
          f"({np.sum(region_mask) * voxel_vol:.0f} mm³)")
    print(f"  {len(candidates)} candidate positions, "
          f"stride={candidate_stride}")

    for cz, cy, cx in candidates:
        if not available[cz, cy, cx]:
            continue
        if dist[cz, cy, cx] < min_depth_mm:
            continue

        placed = False
        for shape in shapes:
            for oi, oriented in enumerate(shape['orientations']):
                if _try_place(available, oriented, cz, cy, cx):
                    mask = _stamp(available.shape, oriented, cz, cy, cx)
                    available[mask] = False

                    placements.append({
                        'shape_index': shape['index'],
                        'shape_name': shape['name'],
                        'position': (cz, cy, cx),
                        'orientation': oi,
                        'mask': mask,
                        'volume_mm3': float(np.sum(mask)) * voxel_vol,
                    })
                    per_shape_count[shape['name']] += 1
                    placed = True
                    break
            if placed:
                break

    region_vol = float(np.sum(region_mask)) * voxel_vol
    placed_vol = sum(p['volume_mm3'] for p in placements)
    efficiency = placed_vol / region_vol if region_vol > 0 else 0.0

    print(f"  Placed {len(placements)} specimens "
          f"({placed_vol:.0f}/{region_vol:.0f} mm³, "
          f"{100*efficiency:.1f}% efficiency)")
    for name, count in per_shape_count.items():
        print(f"    {name}: {count}")

    return {
        'placements': placements,
        'total_specimens': len(placements),
        'per_shape_count': per_shape_count,
        'remaining_mask': available,
        'packing_efficiency': efficiency,
    }


def _load_and_prepare_shapes(stl_paths, spacing, n_orientations):
    """Load STL files and create oriented voxel templates."""
    shapes = []
    rotations = _generate_orientations(n_orientations)

    for i, path in enumerate(stl_paths):
        path = Path(path)
        base_voxels = voxelize_stl(path, spacing)

        orientations = []
        for rot in rotations:
            rotated = _rotate_voxel_template(base_voxels, rot)
            if rotated is not None:
                orientations.append(rotated)

        if not orientations:
            orientations = [base_voxels]

        extent = base_voxels.shape
        shapes.append({
            'index': i,
            'name': path.stem,
            'orientations': orientations,
            'extent': extent,
            'volume_voxels': int(np.sum(base_voxels)),
        })

        print(f"  Shape '{path.stem}': {extent} voxels, "
              f"{len(orientations)} orientations")

    shapes.sort(key=lambda s: s['volume_voxels'], reverse=True)
    return shapes


def _generate_orientations(n):
    """Generate a set of distinct rotations for specimen orientation search."""
    if n <= 1:
        return [np.eye(3)]

    rotations = [np.eye(3)]

    axis_rotations = [
        Rotation.from_euler('x', 90, degrees=True).as_matrix(),
        Rotation.from_euler('y', 90, degrees=True).as_matrix(),
        Rotation.from_euler('z', 90, degrees=True).as_matrix(),
        Rotation.from_euler('x', 90, degrees=True).as_matrix()
        @ Rotation.from_euler('y', 90, degrees=True).as_matrix(),
        Rotation.from_euler('x', 90, degrees=True).as_matrix()
        @ Rotation.from_euler('z', 90, degrees=True).as_matrix(),
    ]

    for r in axis_rotations[:n - 1]:
        rotations.append(r)

    return rotations


def _rotate_voxel_template(template, rotation_matrix):
    """Rotate a voxel template by a rotation matrix.

    Re-voxelizes by mapping each output voxel back to the input.
    """
    if np.allclose(rotation_matrix, np.eye(3)):
        return template

    coords = np.argwhere(template)
    if len(coords) == 0:
        return None

    center = coords.mean(axis=0)
    centered = coords - center
    rotated = (rotation_matrix @ centered.T).T

    rotated -= rotated.min(axis=0)
    rotated = np.round(rotated).astype(int)

    shape = tuple(rotated.max(axis=0) + 1)
    out = np.zeros(shape, dtype=bool)
    for r in rotated:
        out[r[0], r[1], r[2]] = True

    out = ndimage.binary_fill_holes(out)
    return out


def _generate_candidates(dist, min_depth_mm, stride):
    """Generate candidate placement positions sorted by depth (deepest first)."""
    valid = dist >= min_depth_mm
    coords = np.argwhere(valid)

    if stride > 1:
        mask = np.all(coords % stride == 0, axis=1)
        coords = coords[mask]

    if len(coords) == 0:
        return coords

    depths = dist[coords[:, 0], coords[:, 1], coords[:, 2]]
    order = np.argsort(-depths)
    return coords[order]


def _try_place(available, template, cz, cy, cx):
    """Check if a specimen template fits at the given position."""
    tz, ty, tx = template.shape
    az, ay, ax = available.shape

    ez, ey, ex = cz + tz, cy + ty, cx + tx
    if ez > az or ey > ay or ex > ax:
        return False

    region = available[cz:ez, cy:ey, cx:ex]
    return np.all(region[template])


def _stamp(vol_shape, template, cz, cy, cx):
    """Create a full-volume mask for a placed specimen."""
    tz, ty, tx = template.shape
    mask = np.zeros(vol_shape, dtype=bool)
    mask[cz:cz+tz, cy:cy+ty, cx:cx+tx] = template
    return mask
