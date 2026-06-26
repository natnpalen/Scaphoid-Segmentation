"""
Bone separation module: isolates individual bones from a multi-bone CT scan.

Pipeline:
  1. Load DICOM series into a 3D HU volume
  2. Two-pass thresholding: first exclude metal tags, then Otsu on bone
  3. Connected component labeling to identify distinct objects
  4. Classify components as bone vs. lead tag by size and HU
  5. Associate each tag with its nearest bone by proximity
  6. Return labeled bone volumes with tag-based identifiers
"""

import numpy as np
from pathlib import Path
from scipy import ndimage
from skimage.filters import threshold_otsu
from skimage.measure import regionprops

from .dicom_io import load_dicom_series


def separate_bones(dicom_folder, tag_hu_min=1500, min_bone_volume_mm3=500.0,
                   metal_hu_cap=3000):
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
        HU values above this are excluded from Otsu thresholding to prevent
        metal tags from skewing the bone threshold upward.

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

    # Exclude air (< -500) AND metal tags (> metal_hu_cap) from Otsu
    # so dense tags don't pull the threshold above bone tissue
    tissue = volume[(volume > -500) & (volume < metal_hu_cap)]
    if len(tissue) == 0:
        tissue = volume[volume > -500]
    if len(tissue) == 0:
        tissue = volume.ravel()

    bone_thresh = threshold_otsu(tissue)

    # Bone mask includes everything above Otsu threshold (bone + tags)
    bone_mask = volume > bone_thresh
    bone_mask = ndimage.binary_fill_holes(bone_mask)

    labeled, n_components = ndimage.label(bone_mask)
    print(f"  Otsu threshold: {bone_thresh:.0f} HU "
          f"(metal capped at {metal_hu_cap}), "
          f"{n_components} components found")

    props = regionprops(labeled, intensity_image=volume)

    bones = []
    tags = []

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
