"""
Bone separation — faithful port of the scaphoid pipeline's segmentation
strategy (+segment/run_segmentation.m).

Pipeline stages:
  1. Specimen isolation (non-air, all significant components)
  2. Marker detection with Gaussian falloff artifact weighting
  3. Adaptive constraint sweep: Loose/Medium/Strict (HU-offset, grad-percentile)
     pairs each produce a candidate via morphological reconstruction through
     OR-logic allow regions.  Candidates scored by 90th-percentile surface HU.
  4. Morphological closing to bridge trabecular gaps
  5. Boundary-band cling with deep-interior protection (>=1 mm always kept)
  6. Edge-backed perimeter prune
  7. Conservative boundary carve (multi-condition near air)
  8. Final boundary carve
  9. Connected components -> individual bones
 10. Tag-to-bone association by proximity
"""

import numpy as np
from pathlib import Path
from scipy import ndimage
from skimage.measure import regionprops
from skimage.morphology import ball, opening

from .dicom_io import load_dicom_series


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def separate_bones(dicom_folder, tag_hu_min=1200, min_bone_volume_mm3=200.0,
                   closing_radius_mm=2.0):
    """Separate individual bones from a multi-bone DICOM CT scan.

    Uses the same multi-stage strategy as the scaphoid MATLAB pipeline:
    adaptive constraint sweep with OR-logic allow regions, morphological
    reconstruction from dense core seeds, boundary-band cling with deep-
    interior protection, and multi-pass perimeter carving.

    Parameters
    ----------
    dicom_folder : str or Path
        Folder containing DICOM files for one scan.
    tag_hu_min : float
        HU threshold for lead tag detection (default 1200).
    min_bone_volume_mm3 : float
        Minimum volume for a component to count as a bone.
    closing_radius_mm : float
        Morphological closing radius (mm) for bridging trabecular gaps.

    Returns
    -------
    dict  with keys 'volume', 'spacing', 'bones' (list of bone dicts).
    """
    dicom_folder = Path(dicom_folder)
    volume, spacing = load_dicom_series(dicom_folder)

    if volume.ndim != 3:
        raise ValueError(f"Expected 3-D volume, got shape {volume.shape}.")

    vol = volume.astype(np.float32)
    voxel_vol = float(np.prod(spacing))
    mean_sp = float(np.mean(spacing))
    conn26 = np.ones((3, 3, 3), dtype=bool)

    # === Stage 1: Specimen isolation ===
    print("  Stage 1: Specimen isolation...")
    specimen = _isolate_specimen(vol, spacing)
    print(f"    Specimen: {np.sum(specimen) * voxel_vol:.0f} mm³")

    # === Stage 2: Marker & artifact detection ===
    print("  Stage 2: Marker & artifact detection...")
    # Detect tags in the FULL volume — they may be in separate non-air
    # components from the bones (the scaphoid pipeline only had one bone,
    # so restricting to specimen was fine; for multi-bone it drops tags).
    lead_mask = vol > tag_hu_min
    marker_mask = lead_mask.copy()
    if np.any(lead_mask):
        near_lead = ndimage.binary_dilation(lead_mask, structure=ball(2))
        flags = (vol >= 200) & (vol <= 700) & near_lead
        marker_mask = marker_mask | flags

    if np.any(marker_mask):
        d_mm = ndimage.distance_transform_edt(~marker_mask, sampling=spacing)
        artifact_w = np.exp(-(d_mm / 3.0) ** 2)
    else:
        artifact_w = np.zeros_like(vol)

    marker_dil = ndimage.binary_dilation(marker_mask, structure=ball(2))

    tag_components = _find_tags(lead_mask, spacing, voxel_vol)
    print(f"    Found {len(tag_components)} metal tags")

    # === Stage 3: Adaptive thresholds (anchored to softMed) ===
    print("  Stage 3: Adaptive bone detection...")

    non_bg = vol[specimen & (vol > -300)]
    softMed = float(np.median(non_bg)) if len(non_bg) > 0 else -300.0

    vals_mask = specimen & ~marker_dil & (vol > -300) & (vol < 2000)
    vals = vol[vals_mask]
    if len(vals) == 0:
        vals = vol[specimen & (vol > -300)]

    G = _gradient_magnitude(vol, spacing)

    core_thr = max(280, min(700, float(np.percentile(vals, 94))))
    core = specimen & ~marker_dil & (vol > core_thr)

    if not np.any(core):
        for p in [92, 90, 88, 86, 84]:
            alt = max(220, min(650, float(np.percentile(vals, p))))
            c_alt = specimen & ~marker_dil & (vol > alt)
            if np.sum(c_alt) > 2000:
                core, core_thr = c_alt, alt
                print(f"    Core relaxed to p{p} (thr={alt:.0f})")
                break

    print(f"    softMed={softMed:.0f}, core_thr={core_thr:.0f}, "
          f"core={np.sum(core) * voxel_vol:.0f} mm³")

    # === Stage 4: Constraint sweep ===
    constraints = [
        ("Loose",  110, 85),
        ("Medium", 135, 89),
        ("Strict", 160, 92),
    ]

    best_mask = None
    best_score = -np.inf
    best_name = ""

    specimen_G = G[specimen]

    for name, hu_off, g_pct in constraints:
        hu_floor = max(70, min(220, softMed + hu_off))
        g_thr = float(np.percentile(specimen_G, g_pct))

        allow = specimen & ~marker_dil & (
            (vol > hu_floor) | (G > g_thr))

        if np.any(core):
            cand = _reconstruct(core, allow)
        else:
            cand = allow

        cand = _remove_tiny(cand, 500)
        score = _perimeter_score(cand, vol, conn26)

        print(f"    {name}: hu_floor={hu_floor:.0f}, "
              f"vol={np.sum(cand) * voxel_vol:.0f} mm³, "
              f"perim90={score:.1f}")

        if score > best_score:
            best_score, best_mask, best_name = score, cand, name

    print(f"    Winner: {best_name} (perim90={best_score:.1f})")
    bone_mask = best_mask if best_mask is not None else np.zeros_like(
        specimen, dtype=bool)

    # Relaxed retry if surface looks too low
    if best_score < 150 or not np.any(bone_mask):
        print("    Low score — trying relaxed constraints...")
        for name, hu_off, g_pct in [("RelaxA", 80, 80), ("RelaxB", 95, 83)]:
            hu_floor = max(50, min(200, softMed + hu_off))
            g_thr = float(np.percentile(specimen_G, g_pct))
            allow = specimen & ~marker_dil & (
                (vol > hu_floor) | (G > g_thr))
            cand = _reconstruct(core, allow) if np.any(core) else allow
            cand = _remove_tiny(cand, 500)
            score = _perimeter_score(cand, vol, conn26)
            if score > best_score or not np.any(bone_mask):
                bone_mask, best_score = cand, score
                print(f"    Relaxed winner: {name} (perim90={score:.1f})")

    # Last-ditch fallback
    if not np.any(bone_mask):
        print("    Fallback: simple threshold grow...")
        t_lo = max(130, min(240, softMed + 140))
        allowed = specimen & (vol > t_lo) & ~marker_dil
        seed_dil = ndimage.binary_dilation(core, structure=ball(1))
        bone_mask = _reconstruct(seed_dil, allowed) if np.any(core) else allowed
        bone_mask = _remove_tiny(bone_mask, 200)

    # Morphological closing to bridge trabecular gaps
    close_r = max(1, int(round(closing_radius_mm / mean_sp)))
    bone_mask = ndimage.binary_closing(bone_mask, structure=ball(close_r))
    bone_mask = ndimage.binary_fill_holes(bone_mask)
    bone_mask = bone_mask & specimen

    print(f"    After reconstruction + closing: "
          f"{np.sum(bone_mask) * voxel_vol:.0f} mm³")

    # === Stage 5: Boundary refinement (matching MATLAB) ===
    print("  Stage 5: Boundary refinement...")

    # 5a. Boundary-band cling with interior protection
    bone_mask = _boundary_cling(bone_mask, vol, G, spacing, vals,
                                softMed, conn26)

    # 5b. Edge-backed perimeter prune
    bone_mask = _edge_prune(bone_mask, vol, G, softMed, conn26)

    # 5c. Conservative boundary carve
    bone_mask = _boundary_carve(bone_mask, vol, G, spacing, softMed,
                                core, conn26)

    # 5d. Final boundary carve
    bone_mask = _final_carve(bone_mask, vol, softMed, conn26)

    # Final cleanup (no keep-largest — we want multiple bones)
    bone_mask = opening(bone_mask, ball(1))
    bone_mask = ndimage.binary_closing(bone_mask, structure=ball(1))
    bone_mask = ndimage.binary_fill_holes(bone_mask)

    min_vox = max(200, int(min_bone_volume_mm3 / voxel_vol / 2))
    bone_mask = _remove_tiny(bone_mask, min_vox)

    final_vol = np.sum(bone_mask) * voxel_vol
    print(f"    Final bone volume: {final_vol:.0f} mm³")

    # === Stage 6: Split into individual bones ===
    print("  Stage 6: Splitting into individual bones...")
    labeled, n_comp = ndimage.label(bone_mask)
    props = regionprops(labeled, intensity_image=volume)

    bones = []
    for prop in props:
        vol_mm3 = prop.area * voxel_vol
        mean_hu = float(prop.intensity_mean)
        if mean_hu > tag_hu_min:
            continue
        if vol_mm3 >= min_bone_volume_mm3:
            bones.append({
                "label": prop.label,
                "centroid": np.array(prop.centroid) * np.array(spacing),
                "mask": labeled == prop.label,
                "bbox": prop.bbox,
                "volume_mm3": vol_mm3,
                "mean_hu": mean_hu,
            })

    _associate_tags(bones, tag_components)
    bones.sort(key=lambda b: b["volume_mm3"], reverse=True)

    print(f"\nFound {len(bones)} bones and {len(tag_components)} tags in scan")
    for i, b in enumerate(bones):
        tag = f"tag {b['tag_id']}" if b.get("tag_id") else "no tag"
        print(f"  Bone {i + 1}: {b['volume_mm3']:.1f} mm³, "
              f"mean HU {b['mean_hu']:.0f}, {tag}")

    return {"volume": volume, "spacing": spacing, "bones": bones}


# ---------------------------------------------------------------------------
# Helpers — specimen & markers
# ---------------------------------------------------------------------------

def _isolate_specimen(vol, spacing):
    """All non-air material after generous closing and hole-filling.

    The MATLAB scaphoid pipeline takes only the largest non-air component
    because it segments a single bone.  For multi-bone scans the bones are
    separated by air, so we keep ALL significant non-air components.  A
    generous closing (2 mm radius) bridges internal trabecular porosity so
    each bone forms a single connected region.
    """
    non_air = vol > -500
    r = max(2, int(round(2.0 / np.mean(spacing))))
    non_air = ndimage.binary_closing(non_air, structure=ball(r))
    non_air = ndimage.binary_fill_holes(non_air)
    voxel_vol = float(np.prod(spacing))
    min_vox = max(100, int(50.0 / voxel_vol))
    non_air = _remove_tiny(non_air, min_vox)
    return non_air


def _find_tags(lead_mask, spacing, voxel_vol):
    if not np.any(lead_mask):
        return []
    labeled, n = ndimage.label(lead_mask)
    tags = []
    for prop in regionprops(labeled):
        tags.append({
            "label": prop.label,
            "centroid": np.array(prop.centroid) * np.array(spacing),
            "volume_mm3": prop.area * voxel_vol,
        })
    return tags


def _associate_tags(bones, tags):
    for b in bones:
        b["tag_id"] = None
        b["tag_dist"] = None
    if not tags or not bones:
        return
    centroids = np.array([b["centroid"] for b in bones])
    for tag in tags:
        dists = np.linalg.norm(centroids - tag["centroid"], axis=1)
        idx = int(np.argmin(dists))
        d = float(dists[idx])
        if bones[idx]["tag_dist"] is None or d < bones[idx]["tag_dist"]:
            bones[idx]["tag_id"] = tag["label"]
            bones[idx]["tag_dist"] = d


# ---------------------------------------------------------------------------
# Helpers — core operations
# ---------------------------------------------------------------------------

def _gradient_magnitude(vol, spacing):
    """3-D gradient magnitude with anisotropic spacing correction."""
    gz = ndimage.sobel(vol, axis=0) / spacing[0]
    gy = ndimage.sobel(vol, axis=1) / spacing[1]
    gx = ndimage.sobel(vol, axis=2) / spacing[2]
    return np.sqrt(gz ** 2 + gy ** 2 + gx ** 2)


def _reconstruct(seed, mask):
    """Binary morphological reconstruction (imreconstruct equivalent).

    Keeps all connected components of *mask* that contain at least one
    *seed* voxel.  O(n) via connected-component labelling — orders of
    magnitude faster than iterative dilation.
    """
    labeled, n = ndimage.label(mask)
    if n == 0:
        return np.zeros_like(mask, dtype=bool)
    seed_labels = set(np.unique(labeled[seed & mask]).tolist()) - {0}
    if not seed_labels:
        return np.zeros_like(mask, dtype=bool)
    return np.isin(labeled, list(seed_labels))


def _perimeter(mask, conn26):
    """Foreground voxels with at least one background 26-neighbour."""
    return mask & ~ndimage.binary_erosion(mask, structure=conn26)


def _perimeter_score(mask, vol, conn26):
    """90th-percentile HU on the perimeter (MATLAB scoring criterion)."""
    if not np.any(mask):
        return -np.inf
    perim = _perimeter(mask, conn26)
    if not np.any(perim):
        return -np.inf
    return float(np.percentile(vol[perim], 90))


def _remove_tiny(mask, min_voxels):
    """Remove connected components smaller than *min_voxels*."""
    labeled, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = ndimage.sum(mask, labeled, range(1, n + 1))
    keep = np.zeros(n + 1, dtype=bool)
    for i in range(n):
        if sizes[i] >= min_voxels:
            keep[i + 1] = True
    return keep[labeled]


# ---------------------------------------------------------------------------
# Helpers — boundary refinement (faithful to MATLAB lines 207-264)
# ---------------------------------------------------------------------------

def _boundary_cling(bone_mask, vol, G, spacing, vals, softMed, conn26):
    """Boundary-band cling with deep-interior protection.

    Deep interior (>=1 mm from surface) is unconditionally preserved.
    The surface band (0-1 mm) is replaced with only the voxels that are
    connected to dense surface seeds through an OR-logic support region.

    MATLAB reference: +segment/run_segmentation.m lines 207-224.
    """
    D_in = ndimage.distance_transform_edt(bone_mask, sampling=spacing)
    deep = D_in >= 1.0
    band = bone_mask & ~deep

    if not np.any(band):
        return bone_mask

    core_thr = max(240, min(650, float(np.percentile(vals, 92))))
    core_seed = band & (vol > core_thr)

    hu_support = max(60, min(180, softMed + 90))
    g_thr = float(np.percentile(G, 80))
    support = band & ((vol > hu_support) | (G > g_thr))

    if np.any(core_seed) and np.any(support):
        cling = _reconstruct(core_seed, support)
    elif np.any(support):
        cling = support
    else:
        cling = band

    result = deep | cling
    result = _remove_tiny(result, 200)
    result = ndimage.binary_closing(result, structure=ball(1))
    result = ndimage.binary_fill_holes(result)
    return result


def _edge_prune(bone_mask, vol, G, softMed, conn26):
    """Edge-backed perimeter prune.

    Remove surface-band voxels that are BOTH low-HU AND low-gradient.

    MATLAB reference: lines 226-236.
    """
    perim = _perimeter(bone_mask, conn26)
    band1 = ndimage.binary_dilation(perim, structure=ball(1))

    T_hu = max(160, min(340, softMed + 190))
    T_g = float(np.percentile(G, 70))

    kill = band1 & (vol < T_hu) & (G < T_g)
    if not np.any(kill):
        return bone_mask

    result = bone_mask.copy()
    result[kill] = False
    result = _remove_tiny(result, 200)
    result = ndimage.binary_closing(result, structure=ball(1))
    result = ndimage.binary_fill_holes(result)
    return result


def _boundary_carve(bone_mask, vol, G, spacing, softMed, core, conn26):
    """Conservative boundary carve near air.

    Remove perimeter-band voxels that are low-HU, near air, and not
    protected by dense core.

    MATLAB reference: lines 239-256.
    """
    perim = _perimeter(bone_mask, conn26)
    band1 = ndimage.binary_dilation(perim, structure=ball(1))
    outer1 = ndimage.binary_dilation(bone_mask, structure=ball(1)) & ~bone_mask

    protected = (ndimage.binary_dilation(core, structure=ball(2))
                 if np.any(core) else np.zeros_like(bone_mask))

    T_hi = max(170, min(360, softMed + 210))
    T_lo = T_hi - 80

    air_near = outer1 & ndimage.binary_dilation(vol < -300, structure=ball(1))

    kill = (band1
            & (vol < T_lo)
            & ndimage.binary_dilation(air_near, structure=ball(1))
            & ~protected)

    if not np.any(kill):
        return bone_mask

    result = bone_mask.copy()
    result[kill] = False
    result = _remove_tiny(result, 200)
    result = ndimage.binary_closing(result, structure=ball(1))
    result = ndimage.binary_fill_holes(result)
    return result


def _final_carve(bone_mask, vol, softMed, conn26):
    """Final boundary carve: remove weak perimeter voxels.

    MATLAB reference: lines 259-264.
    """
    perim = _perimeter(bone_mask, conn26)
    band = ndimage.binary_dilation(perim, structure=ball(1))

    hu_floor = max(180, min(400, softMed + 220))
    kill = band & (vol < hu_floor)

    if not np.any(kill):
        return bone_mask

    result = bone_mask.copy()
    result[kill] = False
    return result
