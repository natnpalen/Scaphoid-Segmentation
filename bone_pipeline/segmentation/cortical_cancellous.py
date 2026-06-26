"""
Cortical vs. cancellous bone segmentation.

Two complementary approaches combined:
  1. Density-based: Otsu threshold within the bone separates high-density
     (cortical) from low-density (cancellous) voxels
  2. Geometry-based: The cortical shell is the outer layer of the bone;
     erosion peels it off to reveal the cancellous interior

The final classification uses both: cortical = voxels in the outer shell
OR high-density voxels; cancellous = interior voxels that are low-density.
"""

import numpy as np
from scipy import ndimage
from skimage.filters import threshold_otsu
from skimage.morphology import ball


def segment_cortical_cancellous(volume, bone_mask, spacing,
                                cortical_thickness_mm=None):
    """Segment a single bone into cortical and cancellous regions.

    Parameters
    ----------
    volume : 3D ndarray
        HU values for the full scan.
    bone_mask : 3D bool ndarray
        Binary mask of the bone to segment.
    spacing : tuple of float
        Voxel spacing (z, y, x) in mm.
    cortical_thickness_mm : float or None
        Estimated cortical shell thickness in mm. If None, auto-estimated
        from the bone's distance transform.

    Returns
    -------
    dict with keys:
        'cortical_mask'   : 3D bool ndarray
        'cancellous_mask' : 3D bool ndarray
        'bone_threshold'  : float (Otsu threshold in HU)
        'cortical_thickness_mm' : float (used/estimated thickness)
        'cortical_volume_mm3'   : float
        'cancellous_volume_mm3' : float
    """
    voxel_vol_mm3 = float(np.prod(spacing))
    mean_spacing = float(np.mean(spacing))

    # --- Density-based classification ---
    bone_hu_raw = volume[bone_mask]
    bone_hu = bone_hu_raw[(bone_hu_raw > -100) & (bone_hu_raw < 3000)]

    geometry_only = False
    if len(bone_hu) < 100:
        print("  Warning: too few valid bone voxels for Otsu — "
              "using geometry-only segmentation")
        bone_thresh = 0.0
        geometry_only = True
    else:
        bone_thresh = threshold_otsu(bone_hu)
        if bone_thresh < 50:
            print(f"  Warning: Otsu threshold {bone_thresh:.0f} HU is "
                  "suspiciously low — using geometry-only segmentation")
            geometry_only = True

    if geometry_only:
        high_density = np.zeros_like(bone_mask)
        low_density = bone_mask.copy()
    else:
        high_density = bone_mask & (volume >= bone_thresh)
        low_density = bone_mask & (volume < bone_thresh)

    print(f"  Density threshold (Otsu): {bone_thresh:.0f} HU")
    print(f"    High-density voxels: "
          f"{np.sum(high_density) * voxel_vol_mm3:.0f} mm³")
    print(f"    Low-density voxels:  "
          f"{np.sum(low_density) * voxel_vol_mm3:.0f} mm³")

    # --- Geometry-based: find the cortical shell via distance transform ---
    dist = ndimage.distance_transform_edt(bone_mask, sampling=spacing)

    if cortical_thickness_mm is None:
        cortical_thickness_mm = _estimate_cortical_thickness(
            dist, high_density, spacing)

    print(f"  Cortical thickness: {cortical_thickness_mm:.2f} mm")

    shell_mask = bone_mask & (dist <= cortical_thickness_mm)
    interior_mask = bone_mask & (dist > cortical_thickness_mm)

    # --- Combined classification ---
    # Cortical: in the outer shell OR high-density anywhere in the bone
    # Cancellous: interior AND low-density
    cortical_mask = shell_mask | high_density
    cancellous_mask = interior_mask & low_density

    # Clean up small isolated cancellous patches
    cancellous_mask = _remove_small_components(
        cancellous_mask, min_volume_mm3=1.0, voxel_vol_mm3=voxel_vol_mm3)

    # Anything in bone_mask not cancellous is cortical
    cortical_mask = bone_mask & (~cancellous_mask)

    cortical_vol = float(np.sum(cortical_mask)) * voxel_vol_mm3
    cancellous_vol = float(np.sum(cancellous_mask)) * voxel_vol_mm3
    total = cortical_vol + cancellous_vol

    if total > 0:
        print(f"  Cortical:   {cortical_vol:.1f} mm³ "
              f"({100 * cortical_vol / total:.1f}%)")
        print(f"  Cancellous: {cancellous_vol:.1f} mm³ "
              f"({100 * cancellous_vol / total:.1f}%)")
    else:
        print("  Warning: no bone volume detected")

    return {
        'cortical_mask': cortical_mask,
        'cancellous_mask': cancellous_mask,
        'bone_threshold': bone_thresh,
        'cortical_thickness_mm': cortical_thickness_mm,
        'cortical_volume_mm3': cortical_vol,
        'cancellous_volume_mm3': cancellous_vol,
    }


def _estimate_cortical_thickness(dist, high_density_mask, spacing):
    """Estimate cortical thickness from the distance transform.

    Looks at how deep into the bone high-density voxels extend from the
    surface. The cortical shell thickness is estimated as the depth at
    which the fraction of high-density voxels drops below 50%.
    """
    max_dist = float(dist.max())
    if max_dist == 0:
        return float(np.mean(spacing))

    # Sample density fraction at increasing depths
    n_bins = max(10, int(max_dist / float(np.min(spacing))))
    edges = np.linspace(0, max_dist, n_bins + 1)

    for i in range(len(edges) - 1):
        band = (dist >= edges[i]) & (dist < edges[i + 1])
        band_voxels = np.sum(band)
        if band_voxels == 0:
            continue
        high_frac = np.sum(band & high_density_mask) / band_voxels
        if high_frac < 0.5:
            thickness = edges[i]
            return max(thickness, float(np.mean(spacing)))

    # If high-density throughout, use 20% of max depth
    return max(0.2 * max_dist, float(np.mean(spacing)))


def _remove_small_components(mask, min_volume_mm3, voxel_vol_mm3):
    """Remove connected components smaller than min_volume_mm3."""
    labeled, n = ndimage.label(mask)
    if n == 0:
        return mask

    cleaned = mask.copy()
    for i in range(1, n + 1):
        component = labeled == i
        vol = np.sum(component) * voxel_vol_mm3
        if vol < min_volume_mm3:
            cleaned[component] = False

    return cleaned
