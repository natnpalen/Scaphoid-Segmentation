"""Export segmented bone regions and specimen volumes as DICOM series."""

import numpy as np
from pathlib import Path
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import generate_uid, ExplicitVRLittleEndian
from pydicom.sequence import Sequence as DicomSequence
import datetime


def export_dicom_series(volume, mask, spacing, output_dir, series_desc,
                        reference_dicom_dir=None):
    """Export a masked volume region as a DICOM series.

    Parameters
    ----------
    volume : 3D ndarray
        HU values (full volume or cropped).
    mask : 3D bool ndarray
        Region mask — only voxels where mask=True are preserved.
    spacing : tuple of float
        Voxel spacing (z, y, x) in mm.
    output_dir : str or Path
        Directory to write DICOM files into.
    series_desc : str
        DICOM Series Description tag value.
    reference_dicom_dir : str or Path or None
        If provided, copies patient/study metadata from this DICOM series.

    Returns
    -------
    output_dir : Path
        Path to the written DICOM series.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    masked_volume = np.where(mask, volume, -1024.0).astype(np.float32)

    ref_meta = _load_reference_metadata(reference_dicom_dir)

    study_uid = ref_meta.get('study_uid', generate_uid())
    series_uid = generate_uid()
    frame_of_ref_uid = ref_meta.get('frame_of_ref_uid', generate_uid())

    now = datetime.datetime.now()
    date_str = now.strftime('%Y%m%d')
    time_str = now.strftime('%H%M%S.%f')

    n_slices = masked_volume.shape[0]

    for z in range(n_slices):
        slice_data = masked_volume[z]

        intercept = -1024.0
        slope = 1.0
        pixel_data = np.clip(slice_data - intercept, 0, 65535)
        pixel_data = pixel_data.astype(np.uint16)

        file_path = output_dir / f"slice_{z:04d}.dcm"
        file_meta = pydicom.dataset.FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

        ds = FileDataset(str(file_path), {}, file_meta=file_meta,
                         preamble=b"\x00" * 128)

        ds.PatientName = ref_meta.get('patient_name', 'BonePipeline')
        ds.PatientID = ref_meta.get('patient_id', 'BP001')

        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
        ds.FrameOfReferenceUID = frame_of_ref_uid

        ds.StudyDate = ref_meta.get('study_date', date_str)
        ds.SeriesDate = date_str
        ds.ContentDate = date_str
        ds.StudyTime = ref_meta.get('study_time', time_str)
        ds.SeriesTime = time_str

        ds.Modality = 'CT'
        ds.SeriesDescription = series_desc
        ds.Manufacturer = 'BonePipeline'

        ds.Rows = slice_data.shape[0]
        ds.Columns = slice_data.shape[1]
        ds.PixelSpacing = [float(spacing[1]), float(spacing[2])]
        ds.SliceThickness = float(spacing[0])
        ds.SpacingBetweenSlices = float(spacing[0])

        ds.ImagePositionPatient = [0.0, 0.0, float(z * spacing[0])]
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.InstanceNumber = z + 1
        ds.SliceLocation = float(z * spacing[0])

        ds.RescaleIntercept = str(intercept)
        ds.RescaleSlope = str(slope)
        ds.RescaleType = 'HU'

        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = 'MONOCHROME2'
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.PixelData = pixel_data.tobytes()

        ds.save_as(str(file_path))

    print(f"  Wrote {n_slices} DICOM slices to {output_dir}")
    return output_dir


def export_all_results(volume, spacing, bones, segmentations, packings,
                       output_root, reference_dicom_dir=None):
    """Export the full pipeline results as organized DICOM series.

    Output structure:
        output_root/
        ├── bone_01/
        │   ├── cortical/
        │   │   ├── region/          # Cortical region DICOM
        │   │   └── specimens/
        │   │       ├── specimen_001/ # Individual shape DICOMs
        │   │       └── ...
        │   └── cancellous/
        │       ├── region/
        │       └── specimens/
        │           ├── specimen_001/
        │           └── ...
        └── bone_02/
            └── ...
    """
    output_root = Path(output_root)

    for bone_idx, bone in enumerate(bones):
        bone_name = f"bone_{bone_idx + 1:02d}"
        bone_dir = output_root / bone_name

        seg = segmentations[bone_idx]

        for region_name, region_mask in [('cortical', seg['cortical_mask']),
                                         ('cancellous', seg['cancellous_mask'])]:
            region_dir = bone_dir / region_name / 'region'
            export_dicom_series(
                volume, region_mask, spacing, region_dir,
                f"{bone_name}_{region_name}",
                reference_dicom_dir=reference_dicom_dir,
            )

            packing_key = f"{bone_idx}_{region_name}"
            if packing_key in packings:
                packing = packings[packing_key]
                for pi, placement in enumerate(packing['placements']):
                    spec_dir = (bone_dir / region_name / 'specimens' /
                                f"specimen_{pi + 1:03d}_{placement['shape_name']}")
                    export_dicom_series(
                        volume, placement['mask'], spacing, spec_dir,
                        f"{bone_name}_{region_name}_{placement['shape_name']}_{pi+1}",
                        reference_dicom_dir=reference_dicom_dir,
                    )

    print(f"\nAll results exported to {output_root}")
    return output_root


def _load_reference_metadata(dicom_dir):
    """Extract patient/study metadata from a reference DICOM series."""
    meta = {}
    if dicom_dir is None:
        return meta

    dicom_dir = Path(dicom_dir)
    dcm_files = list(dicom_dir.glob('*.dcm')) + list(dicom_dir.glob('*.IMA'))
    if not dcm_files:
        return meta

    try:
        ds = pydicom.dcmread(str(dcm_files[0]), stop_before_pixels=True)
        meta['patient_name'] = str(getattr(ds, 'PatientName', ''))
        meta['patient_id'] = str(getattr(ds, 'PatientID', ''))
        meta['study_uid'] = str(getattr(ds, 'StudyInstanceUID', ''))
        meta['study_date'] = str(getattr(ds, 'StudyDate', ''))
        meta['study_time'] = str(getattr(ds, 'StudyTime', ''))
        meta['frame_of_ref_uid'] = str(
            getattr(ds, 'FrameOfReferenceUID', ''))
    except Exception:
        pass

    return meta
