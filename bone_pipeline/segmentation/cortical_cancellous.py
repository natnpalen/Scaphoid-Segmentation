"""
Cortical vs. cancellous bone segmentation using adaptive thresholding
and morphological endosteal boundary detection.

Approach (morphological escalator):
  1. Otsu threshold on the bone's HU values to get bone mask
  2. Periosteal boundary = outer surface of bone mask
  3. Iterative morphological closing with increasing kernel radius
     fills trabecular pores, leaving only the medullary cavity
  4. Endosteal boundary = inner surface of closed result
  5. Cortical = between periosteal and endosteal boundaries
  6. Cancellous = interior to endosteal boundary
"""

import numpy as np
from scipy import ndimage
from skimage.filters import threshold_otsu
from skimage.morphology import ball


def segment_cortical_cancellous(volume, bone_mask, spacing,
                                closing_radii_mm=None,
                                min_cortical_thickness_mm=0.3):
    """Segment a single bone into cortical and cancellous regions.

    Parameters
    ----------
    volume : 3D ndarray
        HU values for the full scan (or cropped region).
    bone_mask : 3D bool ndarray
        Binary mask of the bone to segment.
    spacing : tuple of float
        Voxel spacing (z, y, x) in mm.
    closing_radii_mm : list of float or None
        Radii in mm for iterative morphological closing.
        Default: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    min_cortical_thickness_mm : float
        Minimum cortical thickness to preserve (mm).

    Returns
    -------
    dict with keys:
        'cortical_mask'   : 3D bool ndarray
        'cancellous_mask' : 3D bool ndarray
        'endosteal_mask'  : 3D bool ndarray (the inner boundary surface)
        'periosteal_mask' : 3D bool ndarray (the outer boundary surface)
        'bone_threshold'  : float (Otsu threshold in HU)
        'cortical_volume_mm3'   : float
        'cancellous_volume_mm3' : float
    """
    if closing_radii_mm is None:
        closing_radii_mm = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

    voxel_vol_mm3 = float(np.prod(spacing))
    mean_spacing = float(np.mean(spacing))

    bone_hu = volume[bone_mask]
    bone_thresh = threshold_otsu(bone_hu)

    dense_bone = bone_mask & (volume >= bone_thresh)

    filled = _morphological_escalator(dense_bone, bone_mask, spacing,
                                      closing_radii_mm)

    periosteal = bone_mask.copy()

    erosion_r = max(1, int(round(min_cortical_thickness_mm / mean_spacing)))
    eroded_periosteal = ndimage.binary_erosion(periosteal,
                                               structure=ball(erosion_r))

    endosteal_interior = filled & eroded_periosteal

    endosteal_interior = ndimage.binary_erosion(
        endosteal_interior, structure=ball(erosion_r))
    endosteal_interior = ndimage.binary_dilation(
        endosteal_interior, structure=ball(erosion_r))

    cancellous_mask = (~filled) & bone_mask & (~dense_bone)

    cancellous_mask = cancellous_mask | (bone_mask & (~filled))

    cancellous_mask = _keep_interior_cancellous(cancellous_mask, bone_mask,
                                                spacing)

    cortical_mask = bone_mask & (~cancellous_mask)

    cortical_vol = float(np.sum(cortical_mask)) * voxel_vol_mm3
    cancellous_vol = float(np.sum(cancellous_mask)) * voxel_vol_mm3

    print(f"Segmentation: threshold={bone_thresh:.0f} HU")
    print(f"  Cortical:   {cortical_vol:.1f} mm³ "
          f"({100*cortical_vol/(cortical_vol+cancellous_vol):.1f}%)")
    print(f"  Cancellous: {cancellous_vol:.1f} mm³ "
          f"({100*cancellous_vol/(cortical_vol+cancellous_vol):.1f}%)")

    return {
        'cortical_mask': cortical_mask,
        'cancellous_mask': cancellous_mask,
        'endosteal_mask': endosteal_interior,
        'periosteal_mask': periosteal,
        'bone_threshold': bone_thresh,
        'cortical_volume_mm3': cortical_vol,
        'cancellous_volume_mm3': cancellous_vol,
    }


def _morphological_escalator(dense_bone, bone_mask, spacing, radii_mm):
    """Iterative morphological closing to fill trabecular pores.

    Progressively closes the dense bone mask with increasing kernel radii.
    At each step, the result is intersected with the bone mask to stay
    within the periosteal boundary.
    """
    mean_spacing = float(np.mean(spacing))
    result = dense_bone.copy()

    for radius_mm in sorted(radii_mm):
        radius_vox = max(1, int(round(radius_mm / mean_spacing)))
        selem = ball(radius_vox)

        closed = ndimage.binary_closing(result, structure=selem)
        closed = ndimage.binary_fill_holes(closed)

        result = closed & bone_mask

    return result


def _keep_interior_cancellous(cancellous_mask, bone_mask, spacing):
    """Remove cancellous regions that touch the periosteal surface.

    True cancellous bone is interior; small patches at the surface are
    likely noise or thin cortical regions misclassified.
    """
    mean_spacing = float(np.mean(spacing))

    surface_thickness = max(1, int(round(0.5 / mean_spacing)))
    eroded = ndimage.binary_erosion(bone_mask,
                                    structure=ball(surface_thickness))
    surface_band = bone_mask & (~eroded)

    labeled, n = ndimage.label(cancellous_mask)
    if n == 0:
        return cancellous_mask

    cleaned = cancellous_mask.copy()
    for i in range(1, n + 1):
        component = labeled == i
        surface_overlap = np.sum(component & surface_band)
        total_voxels = np.sum(component)
        if surface_overlap / max(total_voxels, 1) > 0.5:
            cleaned[component] = False

    return cleaned
