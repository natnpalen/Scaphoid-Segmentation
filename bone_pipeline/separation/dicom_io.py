"""DICOM series loading with proper HU conversion and spatial metadata."""

import numpy as np
from pathlib import Path
import SimpleITK as sitk


def load_dicom_series(dicom_folder):
    """Load a DICOM series from a folder and return HU volume + spacing.

    Selects the series with the most slices (skipping scouts/localizers).
    Handles multi-component (RGB) images by converting to grayscale.
    Applies rescale slope/intercept for proper HU values.

    Parameters
    ----------
    dicom_folder : str or Path
        Path to folder containing DICOM files.

    Returns
    -------
    volume : 3D ndarray (float32)
        Hounsfield Unit values, shape (Z, Y, X).
    spacing : tuple of float
        Voxel spacing in mm as (z_spacing, y_spacing, x_spacing).
    """
    dicom_folder = str(Path(dicom_folder))

    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(dicom_folder)

    if not series_ids:
        raise FileNotFoundError(
            f"No DICOM series found in {dicom_folder}")

    best_id = series_ids[0]
    best_count = 0

    if len(series_ids) > 1:
        print(f"Found {len(series_ids)} DICOM series, selecting best...")
        for sid in series_ids:
            fnames = reader.GetGDCMSeriesFileNames(dicom_folder, sid)
            count = len(fnames)
            print(f"  Series {sid[:20]}...: {count} files")
            if count > best_count:
                best_count = count
                best_id = sid
        print(f"  Selected series with {best_count} files")

    file_names = reader.GetGDCMSeriesFileNames(dicom_folder, best_id)
    reader.SetFileNames(file_names)
    reader.MetaDataDictionaryArrayUpdateOn()

    image = reader.Execute()

    n_components = image.GetNumberOfComponentsPerPixel()
    if n_components > 1:
        print(f"  Multi-component image ({n_components} channels), "
              f"converting to scalar...")
        if n_components == 3:
            arr = sitk.GetArrayFromImage(image)
            gray = (0.2989 * arr[..., 0] +
                    0.5870 * arr[..., 1] +
                    0.1140 * arr[..., 2])
            volume = gray.astype(np.float32)
        else:
            extractor = sitk.VectorIndexSelectionCastImageFilter()
            extractor.SetIndex(0)
            scalar_image = extractor.Execute(image)
            volume = sitk.GetArrayFromImage(scalar_image).astype(np.float32)
    else:
        volume = sitk.GetArrayFromImage(image).astype(np.float32)

    _apply_rescale(volume, reader, file_names)

    sitk_spacing = image.GetSpacing()
    spacing = (sitk_spacing[2], sitk_spacing[1], sitk_spacing[0])

    print(f"Loaded DICOM: {volume.shape} voxels, "
          f"spacing {spacing[0]:.3f}×{spacing[1]:.3f}×{spacing[2]:.3f} mm, "
          f"HU range [{volume.min():.0f}, {volume.max():.0f}]")

    return volume, spacing


def _apply_rescale(volume, reader, file_names):
    """Apply DICOM rescale slope/intercept if present in metadata."""
    try:
        slope_str = reader.GetMetaData(0, '0028|1053')
        intercept_str = reader.GetMetaData(0, '0028|1052')
        slope = float(slope_str.strip())
        intercept = float(intercept_str.strip())

        if slope != 1.0 or intercept != 0.0:
            volume *= slope
            volume += intercept
            print(f"  Applied rescale: slope={slope}, intercept={intercept}")
    except (RuntimeError, ValueError):
        pass
