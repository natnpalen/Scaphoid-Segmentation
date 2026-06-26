"""Visualization tools for pipeline diagnostics."""

import numpy as np
from pathlib import Path

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def visualize_separation(volume, spacing, bones, tags=None, output_dir=None,
                          n_slices=9):
    """Show axial slices with bone masks overlaid in different colors.

    Parameters
    ----------
    volume : 3D ndarray
        HU volume.
    spacing : tuple
        Voxel spacing (z, y, x) in mm.
    bones : list of dict
        Bone results from separate_bones().
    tags : list of dict or None
        Tag results (if available).
    output_dir : str or Path or None
        If given, saves figure to this directory.
    n_slices : int
        Number of evenly-spaced axial slices to show.
    """
    if not HAS_MPL:
        print("  matplotlib not installed, skipping visualization")
        return

    nz = volume.shape[0]
    slice_indices = np.linspace(0, nz - 1, n_slices, dtype=int)

    ncols = 3
    nrows = (n_slices + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    axes = axes.ravel()

    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan',
              'magenta', 'yellow']
    bone_colors = colors[:len(bones)]

    vmin = np.percentile(volume, 1)
    vmax = np.percentile(volume, 99)

    for idx, (ax, sl) in enumerate(zip(axes, slice_indices)):
        ax.imshow(volume[sl], cmap='gray', vmin=vmin, vmax=vmax,
                  aspect=spacing[2] / spacing[1])
        ax.set_title(f"Slice {sl} (z={sl * spacing[0]:.1f} mm)")

        for bi, bone in enumerate(bones):
            mask_sl = bone['mask'][sl]
            if np.any(mask_sl):
                overlay = np.ma.masked_where(~mask_sl,
                                             np.ones_like(mask_sl, float))
                ax.imshow(overlay, cmap=ListedColormap([bone_colors[bi]]),
                          alpha=0.3, aspect=spacing[2] / spacing[1])

        ax.axis('off')

    for ax in axes[len(slice_indices):]:
        ax.axis('off')

    fig.suptitle(f"Bone Separation: {len(bones)} bones found", fontsize=14)
    plt.tight_layout()

    if output_dir:
        path = Path(output_dir) / 'separation_overview.png'
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(path), dpi=150, bbox_inches='tight')
        print(f"  Saved separation overview to {path}")

    plt.close(fig)


def visualize_segmentation(volume, spacing, bone_mask, cortical_mask,
                            cancellous_mask, bone_index=0, output_dir=None,
                            n_slices=9):
    """Show axial slices with cortical (red) and cancellous (blue) overlaid.

    Parameters
    ----------
    volume : 3D ndarray
    spacing : tuple
    bone_mask : 3D bool ndarray
    cortical_mask : 3D bool ndarray
    cancellous_mask : 3D bool ndarray
    bone_index : int
    output_dir : str or Path or None
    n_slices : int
    """
    if not HAS_MPL:
        print("  matplotlib not installed, skipping visualization")
        return

    # Find the z-range that contains this bone
    z_has_bone = np.any(bone_mask, axis=(1, 2))
    z_indices = np.where(z_has_bone)[0]
    if len(z_indices) == 0:
        print("  No bone voxels found, skipping visualization")
        return

    z_min, z_max = z_indices[0], z_indices[-1]
    slice_indices = np.linspace(z_min, z_max, min(n_slices, z_max - z_min + 1),
                                dtype=int)

    ncols = 3
    nrows = (len(slice_indices) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    axes = axes.ravel()

    # Crop to bone bounding box for better view
    yx_has_bone = np.any(bone_mask, axis=0)
    y_indices = np.where(np.any(yx_has_bone, axis=1))[0]
    x_indices = np.where(np.any(yx_has_bone, axis=0))[0]
    pad = 10
    y_min = max(0, y_indices[0] - pad)
    y_max = min(volume.shape[1], y_indices[-1] + pad)
    x_min = max(0, x_indices[0] - pad)
    x_max = min(volume.shape[2], x_indices[-1] + pad)

    vmin = np.percentile(volume[bone_mask], 1) if np.any(bone_mask) else -500
    vmax = np.percentile(volume[bone_mask], 99) if np.any(bone_mask) else 2000

    for idx, ax in enumerate(axes):
        if idx >= len(slice_indices):
            ax.axis('off')
            continue

        sl = slice_indices[idx]
        img = volume[sl, y_min:y_max, x_min:x_max]
        ax.imshow(img, cmap='gray', vmin=vmin, vmax=vmax,
                  aspect=spacing[2] / spacing[1])

        cort_sl = cortical_mask[sl, y_min:y_max, x_min:x_max]
        canc_sl = cancellous_mask[sl, y_min:y_max, x_min:x_max]

        if np.any(cort_sl):
            overlay = np.ma.masked_where(~cort_sl,
                                         np.ones_like(cort_sl, float))
            ax.imshow(overlay, cmap=ListedColormap(['red']),
                      alpha=0.35, aspect=spacing[2] / spacing[1])

        if np.any(canc_sl):
            overlay = np.ma.masked_where(~canc_sl,
                                         np.ones_like(canc_sl, float))
            ax.imshow(overlay, cmap=ListedColormap(['blue']),
                      alpha=0.35, aspect=spacing[2] / spacing[1])

        ax.set_title(f"Slice {sl}")
        ax.axis('off')

    fig.suptitle(f"Bone {bone_index + 1}: Cortical (red) / Cancellous (blue)",
                 fontsize=14)
    plt.tight_layout()

    if output_dir:
        path = Path(output_dir) / f'segmentation_bone_{bone_index + 1:02d}.png'
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(path), dpi=150, bbox_inches='tight')
        print(f"  Saved segmentation view to {path}")

    plt.close(fig)


def visualize_hu_histogram(volume, bone_mask, bone_index=0, output_dir=None):
    """Show the HU histogram within a bone, with Otsu threshold marked.

    Useful for diagnosing thresholding issues.
    """
    if not HAS_MPL:
        return

    from skimage.filters import threshold_otsu

    bone_hu = volume[bone_mask]
    if len(bone_hu) == 0:
        return

    thresh = threshold_otsu(bone_hu)

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.hist(bone_hu, bins=200, color='steelblue', alpha=0.7, edgecolor='none')
    ax.axvline(thresh, color='red', linewidth=2, linestyle='--',
               label=f'Otsu threshold: {thresh:.0f} HU')
    ax.set_xlabel('HU')
    ax.set_ylabel('Voxel count')
    ax.set_title(f'Bone {bone_index + 1}: HU distribution '
                 f'({len(bone_hu)} voxels)')
    ax.legend()
    plt.tight_layout()

    if output_dir:
        path = Path(output_dir) / f'histogram_bone_{bone_index + 1:02d}.png'
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(path), dpi=150, bbox_inches='tight')
        print(f"  Saved histogram to {path}")

    plt.close(fig)


def visualize_full_volume_histogram(volume, output_dir=None):
    """Show the full volume HU histogram to diagnose thresholding."""
    if not HAS_MPL:
        return

    from skimage.filters import threshold_otsu

    non_air = volume[volume > -500]
    if len(non_air) == 0:
        return

    # Also compute Otsu with and without metal cap
    non_air_capped = non_air[non_air < 3000]
    thresh_capped = threshold_otsu(non_air_capped) if len(non_air_capped) > 0 else 0
    thresh_raw = threshold_otsu(non_air)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(non_air, bins=300, color='steelblue', alpha=0.7,
                 edgecolor='none')
    axes[0].axvline(thresh_raw, color='red', linewidth=2, linestyle='--',
                    label=f'Otsu (raw): {thresh_raw:.0f} HU')
    axes[0].axvline(thresh_capped, color='green', linewidth=2, linestyle='--',
                    label=f'Otsu (capped): {thresh_capped:.0f} HU')
    axes[0].set_xlabel('HU')
    axes[0].set_ylabel('Voxel count')
    axes[0].set_title('Full histogram (HU > -500)')
    axes[0].legend()

    # Zoomed view on the bone region
    bone_range = non_air[(non_air > -200) & (non_air < 2000)]
    if len(bone_range) > 0:
        axes[1].hist(bone_range, bins=200, color='steelblue', alpha=0.7,
                     edgecolor='none')
        axes[1].axvline(thresh_capped, color='green', linewidth=2,
                        linestyle='--',
                        label=f'Otsu (capped): {thresh_capped:.0f} HU')
        axes[1].set_xlabel('HU')
        axes[1].set_ylabel('Voxel count')
        axes[1].set_title('Zoomed: -200 to 2000 HU')
        axes[1].legend()

    plt.tight_layout()

    if output_dir:
        path = Path(output_dir) / 'volume_histogram.png'
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(path), dpi=150, bbox_inches='tight')
        print(f"  Saved volume histogram to {path}")

    plt.close(fig)
