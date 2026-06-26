"""
Bone segmentation pipeline — main orchestrator.

Runs the full pipeline:
  1. Separate bones from multi-bone CT scan
  2. Segment each bone into cortical and cancellous regions
  3. Pack mechanical testing specimens into each region
  4. Export all results as DICOM series
  5. Generate diagnostic visualizations
"""

import argparse
import json
from pathlib import Path

from .separation import separate_bones
from .segmentation import segment_cortical_cancellous
from .packing import pack_specimens
from .export.dicom_export import export_all_results
from .visualization import (visualize_separation, visualize_segmentation,
                             visualize_hu_histogram,
                             visualize_full_volume_histogram)


def run_pipeline(config):
    """Run the full bone segmentation and packing pipeline.

    Parameters
    ----------
    config : dict
        Pipeline configuration with keys:
            'dicom_folder'  : str, path to input DICOM series
            'output_dir'    : str, path for output
            'stl_shapes'    : list of str, paths to specimen STL files
            'bone_indices'  : list of int or None (default: all)
            'tag_hu_min'    : float (default 1500)
            'min_bone_volume_mm3' : float (default 200)
            'closing_radius_mm'   : float (default 2.0)
            'cortical_thickness_mm' : float or None (auto)
            'orientations_per_shape' : int (default 6)
            'packing_stride'      : int or None (auto)
            'min_depth_mm'        : float (default 0.5)

    Returns
    -------
    results : dict with all pipeline outputs
    """
    dicom_folder = config['dicom_folder']
    output_dir = Path(config['output_dir'])
    stl_paths = [Path(p) for p in config['stl_shapes']]

    output_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = output_dir / 'diagnostics'
    viz_dir.mkdir(parents=True, exist_ok=True)

    # --- Stage 1: Bone separation ---
    print("=" * 60)
    print("STAGE 1: Bone Separation")
    print("=" * 60)

    sep_result = separate_bones(
        dicom_folder,
        tag_hu_min=config.get('tag_hu_min', 1500),
        min_bone_volume_mm3=config.get('min_bone_volume_mm3', 200.0),
        closing_radius_mm=config.get('closing_radius_mm', 2.0),
    )

    volume = sep_result['volume']
    spacing = sep_result['spacing']
    bones = sep_result['bones']

    # Diagnostic: full volume histogram
    print("\nGenerating volume histogram...")
    visualize_full_volume_histogram(volume, output_dir=viz_dir)

    bone_indices = config.get('bone_indices')
    if bone_indices is not None:
        bones = [bones[i] for i in bone_indices if i < len(bones)]

    # Diagnostic: separation overview
    print("Generating separation overview...")
    visualize_separation(volume, spacing, bones, output_dir=viz_dir)

    # --- Stage 2: Cortical/cancellous segmentation ---
    print("\n" + "=" * 60)
    print("STAGE 2: Cortical/Cancellous Segmentation")
    print("=" * 60)

    segmentations = []
    for i, bone in enumerate(bones):
        print(f"\nProcessing bone {i + 1}/{len(bones)} "
              f"({bone['volume_mm3']:.0f} mm³)...")
        seg = segment_cortical_cancellous(
            volume, bone['mask'], spacing,
            cortical_thickness_mm=config.get('cortical_thickness_mm'),
        )
        segmentations.append(seg)

        # Diagnostic: per-bone histogram and segmentation view
        visualize_hu_histogram(volume, bone['mask'], bone_index=i,
                               output_dir=viz_dir)
        visualize_segmentation(volume, spacing, bone['mask'],
                                seg['cortical_mask'], seg['cancellous_mask'],
                                bone_index=i, output_dir=viz_dir)

    # --- Stage 3: Specimen packing ---
    print("\n" + "=" * 60)
    print("STAGE 3: Specimen Packing")
    print("=" * 60)

    packings = {}
    for i, (bone, seg) in enumerate(zip(bones, segmentations)):
        for region_name, region_mask in [
            ('cortical', seg['cortical_mask']),
            ('cancellous', seg['cancellous_mask']),
        ]:
            print(f"\nPacking into bone {i+1} {region_name}...")
            packing = pack_specimens(
                region_mask, spacing, stl_paths,
                orientations_per_shape=config.get(
                    'orientations_per_shape', 6),
                min_depth_mm=config.get('min_depth_mm', 0.5),
                candidate_stride=config.get('packing_stride'),
            )
            packings[f"{i}_{region_name}"] = packing

    # --- Stage 4: DICOM export ---
    print("\n" + "=" * 60)
    print("STAGE 4: DICOM Export")
    print("=" * 60)

    export_all_results(
        volume, spacing, bones, segmentations, packings,
        output_dir, reference_dicom_dir=dicom_folder,
    )

    # --- Summary ---
    _print_summary(bones, segmentations, packings)

    print(f"\nDiagnostic images saved to {viz_dir}")

    return {
        'separation': sep_result,
        'segmentations': segmentations,
        'packings': packings,
        'output_dir': output_dir,
    }


def _print_summary(bones, segmentations, packings):
    """Print a final summary of the pipeline results."""
    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)

    for i, (bone, seg) in enumerate(zip(bones, segmentations)):
        print(f"\nBone {i + 1}:")
        print(f"  Volume:     {bone['volume_mm3']:.1f} mm³")
        print(f"  Cortical:   {seg['cortical_volume_mm3']:.1f} mm³")
        print(f"  Cancellous: {seg['cancellous_volume_mm3']:.1f} mm³")

        for region_name in ['cortical', 'cancellous']:
            key = f"{i}_{region_name}"
            if key in packings:
                p = packings[key]
                print(f"  {region_name.capitalize()} specimens: "
                      f"{p['total_specimens']} total, "
                      f"{100*p['packing_efficiency']:.1f}% fill")
                for name, count in p['per_shape_count'].items():
                    print(f"    {name}: {count}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Bone segmentation and specimen packing pipeline')
    parser.add_argument('--dicom', required=True,
                        help='Path to DICOM folder')
    parser.add_argument('--output', required=True,
                        help='Output directory')
    parser.add_argument('--shapes', required=True, nargs='+',
                        help='Paths to specimen STL files')
    parser.add_argument('--config', default=None,
                        help='JSON config file (overrides CLI args)')
    parser.add_argument('--bones', default=None, type=int, nargs='+',
                        help='Bone indices to process (0-based, default: all)')
    parser.add_argument('--orientations', default=6, type=int,
                        help='Orientations per shape (default: 6)')
    parser.add_argument('--stride', default=None, type=int,
                        help='Candidate grid stride (default: auto)')
    parser.add_argument('--min-depth', default=0.5, type=float,
                        help='Min depth from boundary in mm (default: 0.5)')

    args = parser.parse_args()

    if args.config:
        with open(args.config) as f:
            config = json.load(f)
    else:
        config = {
            'dicom_folder': args.dicom,
            'output_dir': args.output,
            'stl_shapes': args.shapes,
            'bone_indices': args.bones,
            'orientations_per_shape': args.orientations,
            'packing_stride': args.stride,
            'min_depth_mm': args.min_depth,
        }

    run_pipeline(config)


if __name__ == '__main__':
    main()
