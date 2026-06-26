"""DICOM series loading with proper HU conversion and spatial metadata."""

import numpy as np
from pathlib import Path
import SimpleITK as sitk


def load_dicom_series(dicom_folder):
    """Load a DICOM series from a folder and return HU volume + spacing.

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

    if len(series_ids) > 1:
        print(f"Warning: {len(series_ids)} DICOM series found, using first")

    file_names = reader.GetGDCMSeriesFileNames(dicom_folder, series_ids[0])
    reader.SetFileNames(file_names)
    reader.MetaDataDictionaryArrayUpdateOn()

    image = reader.Execute()

    volume = sitk.GetArrayFromImage(image).astype(np.float32)

    sitk_spacing = image.GetSpacing()
    spacing = (sitk_spacing[2], sitk_spacing[1], sitk_spacing[0])

    print(f"Loaded DICOM: {volume.shape} voxels, "
          f"spacing {spacing[0]:.3f}×{spacing[1]:.3f}×{spacing[2]:.3f} mm, "
          f"HU range [{volume.min():.0f}, {volume.max():.0f}]")

    return volume, spacing
