# Bone Segmentation Pipeline

Automated segmentation of multiple excised-in-air bones from CT scans. Designed for cadaveric hand/wrist specimens (metacarpals, carpals) with embedded lead identification markers. Segments individual bones, classifies cortical vs. cancellous tissue, optionally packs mechanical test specimens, and exports masks and meshes.

## MATLAB Version

- MATLAB R2020b or later (tested on R2024a+)

## Required Toolboxes

| Toolbox | Used For |
|---------|----------|
| **Image Processing Toolbox** | `bwdist`, `bwconncomp`, `regionprops3`, `imclose`, `imopen`, `imerode`, `imdilate`, `imfill`, `imreconstruct`, `bwareaopen`, `bwperim`, `strel`, `imgradient3`, `imsegfmm`, `imresize3`, `isosurface`, `smooth3`, `mat2gray`, `gradientweight`, `imdiffusefilt`, `stlread`, `reducepatch`, `niftiwrite`, `niftiinfo`, `dicominfo`, `dicomread` |
| **Parallel Computing Toolbox** *(optional)* | `parfor` — used to process bones in parallel during cortical/cancellous segmentation and specimen packing. Pipeline works without it (falls back to serial `for` loops). |

## How to Run

1. Open MATLAB and navigate to `Bone_Pipeline/src/`
2. Edit `run_scan.m` — set `dicomFolder` to your DICOM series folder and `stlFolder` to your specimen STL folder
3. Run:
   ```matlab
   run_scan
   ```

The pipeline is self-contained — no dependencies on other folders in the repository.

---

## Pipeline Overview

The pipeline runs 6 stages in sequence:

### Stage 1: DICOM Loading

**File:** `+dicom/series_load.m`

Loads a CT series from a folder of DICOM files. Uses direct file enumeration (reads each file's header individually) rather than MATLAB's `dicomCollection`/`dicomreadVolume`, which fail on scanner-exported files that lack standard extensions or complete metadata.

- Finds all valid DICOM files recursively under the given folder
- Reads headers, sorts slices by Image Position Patient (IPP) projection along the slice normal
- Applies rescale slope/intercept to produce Hounsfield Unit (HU) values
- Extracts voxel spacing, orientation vectors, and coordinate transforms
- Filters to the dominant image dimensions when the folder contains multiple series (e.g. CT volume + scout/localizer images)
- Optionally resamples to isotropic voxel spacing and applies edge-preserving diffusion smoothing

**Output:** Dataset struct (`ds`) with the HU volume, spacing, origin, direction vectors, and voxel-to-world coordinate transforms.

### Stage 2: Bone Separation

**File:** `+bone/separate_bones.m`

Isolates individual bones from a multi-bone scan. Adapted from the scaphoid pipeline's seed-and-grow approach for multiple bones.

**Sub-stages:**

1. **Marker detection and artifact field** — Identifies lead identification markers (HU > 3000) and their attached flag tabs. Grows marker masks iteratively from lead cores into adjacent high-HU voxels (> 400 HU), limited to 8 dilation steps to prevent leaking into bone. Builds a Gaussian artifact weight field that decays with distance from markers, used to down-weight artifact-contaminated regions during FMM growth.

2. **Seed point finding** — Locates one seed per bone by:
   - Thresholding the volume to non-air (HU > -300), excluding a buffer around markers
   - Finding connected components and merging fragments within 5 mm (bridges gaps created by marker exclusion)
   - Scoring components by sphericity, elongation, and volume
   - Rejecting air pockets (mean HU < 50) and small components (< `MinBoneVolMM3`)
   - Placing each seed at the deepest interior point (maximum of the distance transform)

3. **Per-seed FMM growth** — For each seed, grows the bone mask using Fast Marching Method (FMM) with artifact-weighted speeds:
   - Crops to a local ROI (10 mm margin around the source component) for speed
   - Builds an allow region: flood-fills from high-HU cores through moderate-HU/high-gradient voxels (catches cancellous bone), excluding a buffer around markers (5 mm from lead, 2 mm from full marker mask)
   - Constructs FMM weight map from bone-tissue probability, edge weights, and artifact decay
   - Runs `imsegfmm` and selects the best threshold via adaptive sweep (9 thresholds from 0.14 to 0.42), scored by boundary gradient alignment and interior HU
   - Post-processes: shell sealing (morphological close + fill), marker carving, boundary refinement (cling-prune-carve chain), surface tissue scrub, small blob removal

4. **Non-bone rejection and tag association** — Rejects objects with mean HU < 50 (air pockets, soft tissue). Associates each lead marker assembly with its nearest bone by surface-to-surface distance. Sorts bones by volume (largest first).

### Stage 3: Cortical / Cancellous Segmentation

**File:** `+bone/cortical_cancellous.m`

Segments each bone mask into cortical (dense outer shell) and cancellous (spongy interior) regions using gradient-based depth profiling.

- Computes anisotropic-aware distance from bone surface
- Finds the bone's principal axis via PCA and divides it into axial slabs (4 mm wide, 3-12 slabs depending on bone length)
- For each slab, builds a depth-vs-HU profile (0.20 mm depth bins) and detects the cortical-cancellous transition as the depth of steepest negative HU gradient
- Classifies bone shape as "elongated" (aspect ratio > 2, e.g. metacarpals — cortical cap 2.5 mm) or "compact" (e.g. carpals — cortical cap 1.2 mm)
- Interpolates transition depths across slabs with 3-point moving average smoothing
- Classifies each voxel as cortical if it is shallower than the local transition depth AND above the local HU threshold
- Cleans up with morphological closing

**Output:** Cortical mask, cancellous mask, and info struct with per-slab profiles, transition depths, volumes, and cortical fraction.

### Stage 4: Specimen Packing

**File:** `+bone/pack_specimens.m`

Packs mechanical test specimens (Bend, Compression, Punch, Shear) into the cortical and cancellous regions of each bone. This stage is slow and can be disabled with `PackSpecimens = false`.

- Loads specimen STL meshes and generates bone-axis-aligned rotations (configurable number of orientations)
- For each rotation: rotates mesh vertices, then voxelizes the rotated mesh using slice-by-slice ray casting
- Packs each region (cortical and cancellous separately) in two phases:
  1. **Priority phase:** places one of each specimen type
  2. **Greedy phase:** fills remaining space with best-fitting specimens (up to 50 additional)
- Placement scoring uses 3D convolution on a cropped bounding box ROI: overlap fraction (must be >= 95%) plus average depth from region surface (prefers interior placements)
- Records placed mesh vertices in volume coordinates for visualization

### Stage 5: Visualization

**File:** `+bone/visualize_results.m`

Generates up to 3 interactive 3D figures:

1. **Bone Separation Overview** — Color-coded bones with volume/HU/tag labels
2. **Cortical / Cancellous Segmentation** — Translucent cortical shells over solid cancellous interiors, with cortical fraction and depth labels
3. **Specimen Packing** — Transparent bone shells with color-coded placed specimens

Figures are saved as PNG to the output directory when `SaveOutputs` is enabled.

### Stage 6: Output Saving

Writes results to `bone_pipeline_outputs/<series_name>/<timestamp>/` next to the DICOM folder (or to `OutputDir` if specified).

**Per-scan outputs:**
- `pipeline_results.mat` — full results struct (separation, segmentation, packing)
- `pipeline_summary.txt` — human-readable text summary

**Per-bone outputs:**
- `bone_XX_mask.nii.gz` — binary bone mask (NIfTI, compressed)
- `bone_XX_cortical.nii.gz` — cortical region mask
- `bone_XX_cancellous.nii.gz` — cancellous region mask
- `bone_XX_hu.nii.gz` — HU values within bone (non-bone voxels set to -3000)
- `bone_XX_voxelized.stl` — voxel-accurate bone mesh (light Gaussian smoothing, binary STL)
- `bone_XX_smooth.stl` — anatomical bone mesh (heavier Gaussian smoothing, 15 iterations of Laplacian mesh smoothing, decimated to 30% of faces, binary STL)

---

## Pipeline Options

Set these as name-value pairs in the `run_bone_pipeline()` call inside `run_scan.m`:

| Option | Default | Description |
|--------|---------|-------------|
| `PackSpecimens` | `true` | Run specimen packing stage (slow — disable for faster runs) |
| `SaveOutputs` | `true` | Export MAT, NIfTI, and STL files |
| `ShowViewer` | `true` | Show interactive 3D visualization figures |
| `PackingOrientations` | `6` | Number of rotations to try per specimen shape |
| `PackingMinDepthMM` | `0.5` | Minimum depth from region surface for specimen placement |
| `TagHUMin` | `1200` | HU threshold for metal tag detection |
| `MinBoneVolMM3` | `500` | Minimum bone component volume (mm^3) to keep |
| `ClosingRadiusMM` | `3.0` | Morphological closing radius for bone masks |
| `ArtifactSigmaMM` | `3.0` | Gaussian sigma for marker artifact weight decay |
| `TargetIsoMM` | `[]` (disabled) | Target isotropic voxel size (mm) for resampling |
| `Smoothing` | `false` | Apply edge-preserving diffusion smoothing to the HU volume |
| `OutputDir` | `''` (auto) | Output directory (auto-creates timestamped subfolder if empty) |

---

## File Structure

```
Bone_Pipeline/
  REQUIREMENTS.md          ← this file
  src/
    run_scan.m             ← entry point: set paths and run
    run_bone_pipeline.m    ← main pipeline orchestrator (6 stages)
    +dicom/
      series_load.m        ← DICOM CT series loader
    +bone/
      separate_bones.m     ← multi-bone separation (FMM-based)
      cortical_cancellous.m ← cortical/cancellous segmentation
      pack_specimens.m     ← mechanical specimen packing
      visualize_results.m  ← 3D visualization
    +meshing/
      write_stl_binary.m   ← binary STL file writer
    +utils/
      parse_opts.m         ← name-value option parser
```

---

## Input Requirements

### DICOM Folder
- Must contain a single CT series (or multiple series — the loader auto-selects the dominant image size)
- Scanner-exported files without `.dcm` extensions are supported
- Files must have valid DICOM headers with at minimum: pixel data, `Rows`, `Columns`, `RescaleSlope`, `RescaleIntercept`
- `ImagePositionPatient` is used for slice ordering when available; falls back to `InstanceNumber`

### STL Folder (for specimen packing)
- Must contain one or more of: `Bend.STL`, `Compression.STL`, `Punch.STL`, `Shear.STL`
- Files should be in mm units, centered roughly at origin
- Case-insensitive extension matching (`.STL` or `.stl`)
