"""
Bone separation module: isolates individual bones from a multi-bone CT scan.

Pipeline:
  1. Load DICOM series into a 3D HU volume
  2. Two-pass thresholding: exclude metal, then Otsu on tissue
  3. Morphological closing to bridge trabecular gaps within bones
  4. Connected component labeling to identify distinct objects
  5. Classify components as bone vs. lead tag by size and HU
  6. Associate each tag with its nearest bone by proximity
  7. Return labeled bone volumes with tag-based identifiers
"""

import numpy as np
from pathlib import Path
from scipy import ndimage
from skimage.filters import threshold_otsu
from skimage.measure import regionprops
from skimage.morphology import ball

from .dicom_io import load_dicom_series


def separate_bones(dicom_folder, tag_hu_min=1500, min_bone_volume_mm3=200.0,
                   metal_hu_cap=3000, closing_radius_mm=2.0):
    """Separate individual bones from a multi-bone DICOM CT scan.

    Parameters
    ----------
    dicom_folder : str or Path
        Path to folder containing DICOM files for one scan.
    tag_hu_min : float
        Minimum mean HU to consider a component a lead tag (default 1500).
    min_bone_volume_mm3 : float
        Minimum volume in mm^3 for a component to be considered a bone.
    metal_hu_cap : float
        HU values above this are excluded from Otsu thresholding.
    closing_radius_mm : float
        Morphological closing radius in mm. Bridges internal gaps
        (trabecular voids, marrow spaces) so each bone stays as one
        connected component.

    Returns
    -------
    dict with keys:
        'volume'    : 3D ndarray of HU values (original volume)
        'spacing'   : tuple of (z, y, x) voxel spacing in mm
        'bones'     : list of dicts, each with:
            'label'    : int, component label
            'mask'     : 3D bool ndarray, mask for this bone
            'tag_id'   : int or None, associated tag label
            'tag_dist' : float or None, distance to nearest tag (mm)
            'bbox'     : tuple, bounding box slice objects
            'volume_mm3' : float
            'mean_hu'  : float
    """
    dicom_folder = Path(dicom_folder)
    volume, spacing = load_dicom_series(dicom_folder)

    if volume.ndim != 3:
        raise ValueError(
            f"Expected 3D volume, got shape {volume.shape}. "
            f"Check DICOM series selection.")

    voxel_vol_mm3 = float(np.prod(spacing))
    mean_spacing = float(np.mean(spacing))

    # --- Thresholding ---
    # Exclude air (< -500) AND metal tags (> metal_hu_cap) from Otsu
    tissue = volume[(volume > -500) & (volume < metal_hu_cap)]
    if len(tissue) == 0:
        tissue = volume[volume > -500]
    if len(tissue) == 0:
        tissue = volume.ravel()

    bone_thresh = threshold_otsu(tissue)
    bone_mask = volume > bone_thresh

    print(f"  Otsu threshold: {bone_thresh:.0f} HU "
          f"(metal capped at {metal_hu_cap})")
    print(f"  Raw mask volume: {np.sum(bone_mask) * voxel_vol_mm3:.0f} mm³")

    # --- Morphological closing to bridge trabecular gaps ---
    closing_r_vox = max(1, int(round(closing_radius_mm / mean_spacing)))
    selem = ball(closing_r_vox)
    print(f"  Closing radius: {closing_radius_mm} mm "
          f"({closing_r_vox} voxels), bridging internal gaps...")

    bone_mask = ndimage.binary_closing(bone_mask, structure=selem)
    bone_mask = ndimage.binary_fill_holes(bone_mask)

    print(f"  Closed mask volume: {np.sum(bone_mask) * voxel_vol_mm3:.0f} mm³")

    # --- Connected component labeling ---
    labeled, n_components = ndimage.label(bone_mask)
    print(f"  {n_components} components after closing")

    props = regionprops(labeled, intensity_image=volume)

    bones = []
    tags = []
    small_count = 0

    for prop in props:
        vol_mm3 = prop.area * voxel_vol_mm3
        mean_hu = float(prop.intensity_mean)

        if mean_hu >= tag_hu_min and vol_mm3 < min_bone_volume_mm3:
            tags.append({
                'label': prop.label,
                'centroid': np.array(prop.centroid) * np.array(spacing),
                'mask': labeled == prop.label,
                'volume_mm3': vol_mm3,
                'mean_hu': mean_hu,
            })
        elif vol_mm3 >= min_bone_volume_mm3:
            bones.append({
                'label': prop.label,
                'centroid': np.array(prop.centroid) * np.array(spacing),
                'mask': labeled == prop.label,
                'bbox': prop.bbox,
                'volume_mm3': vol_mm3,
                'mean_hu': mean_hu,
            })
        else:
            small_count += 1

    if small_count > 0:
        print(f"  Filtered out {small_count} small components "
              f"(< {min_bone_volume_mm3} mm³)")

    # --- Tag-to-bone association ---
    for bone in bones:
        bone['tag_id'] = None
        bone['tag_dist'] = None

    if tags and bones:
        bone_centroids = np.array([b['centroid'] for b in bones])
        for tag in tags:
            dists = np.linalg.norm(bone_centroids - tag['centroid'], axis=1)
            nearest_idx = int(np.argmin(dists))
            nearest_dist = float(dists[nearest_idx])

            current = bones[nearest_idx].get('tag_dist')
            if current is None or nearest_dist < current:
                bones[nearest_idx]['tag_id'] = tag['label']
                bones[nearest_idx]['tag_dist'] = nearest_dist

    bones.sort(key=lambda b: b['volume_mm3'], reverse=True)

    print(f"Found {len(bones)} bones and {len(tags)} tags in scan")
    for i, bone in enumerate(bones):
        tag_str = f"tag {bone['tag_id']}" if bone['tag_id'] else "no tag"
        print(f"  Bone {i+1}: {bone['volume_mm3']:.1f} mm³, "
              f"mean HU {bone['mean_hu']:.0f}, {tag_str}")

    return {
        'volume': volume,
        'spacing': spacing,
        'bones': bones,
    }
