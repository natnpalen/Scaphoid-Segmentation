function [cortical, cancellous, info] = cortical_cancellous(ds, bone_mask, opts)
% CORTICAL_CANCELLOUS  Segment a bone mask into cortical and cancellous regions.
%
%   [cortical, cancellous, info] = bone.cortical_cancellous(ds, bone_mask, opts)
%
% Uses a combination of distance-from-surface and adaptive HU thresholding
% (Otsu's method on in-bone voxels) to classify cortical vs cancellous.
% The cortical shell is identified as the dense outer layer; cancellous is
% the less-dense interior.
%
% Inputs
%   ds        : dataset struct from dicom.series_load
%   bone_mask : logical 3D mask of one bone
%   opts      : pipeline options struct
%
% Outputs
%   cortical  : logical 3D mask (cortical bone)
%   cancellous: logical 3D mask (cancellous bone)
%   info      : struct with .otsu_threshold, .cortical_thickness_mm,
%               .cortical_volume_mm3, .cancellous_volume_mm3, .cortical_fraction

vol = double(ds.HU);
spacing = ds.spacing;
voxel_vol = prod(spacing);

% ---- Distance from bone surface (in mm) ----
D_mm = bwdist(~bone_mask) .* mean(spacing);

% ---- In-bone HU values ----
hu_bone = vol(bone_mask);

% ---- Adaptive threshold via Otsu ----
hu_min = min(hu_bone);
hu_max = max(hu_bone);
if hu_max <= hu_min
    cortical = bone_mask;
    cancellous = false(size(bone_mask));
    info = struct('otsu_threshold', NaN, 'cortical_thickness_mm', NaN, ...
        'cortical_volume_mm3', sum(cortical(:))*voxel_vol, ...
        'cancellous_volume_mm3', 0, 'cortical_fraction', 1.0);
    return;
end

% Normalize to [0, 1] for Otsu
hu_norm = (hu_bone - hu_min) / (hu_max - hu_min);
otsu_level = graythresh(hu_norm);
otsu_hu = otsu_level * (hu_max - hu_min) + hu_min;

% ---- Estimate cortical thickness from density-vs-depth profile ----
max_depth = max(D_mm(bone_mask));
n_bins = max(10, round(max_depth / 0.1));
depth_edges = linspace(0, max_depth, n_bins + 1);
depth_centers = (depth_edges(1:end-1) + depth_edges(2:end)) / 2;

mean_hu_by_depth = zeros(n_bins, 1);
for b = 1:n_bins
    in_bin = bone_mask & (D_mm >= depth_edges(b)) & (D_mm < depth_edges(b+1));
    if any(in_bin(:))
        mean_hu_by_depth(b) = mean(vol(in_bin));
    else
        mean_hu_by_depth(b) = NaN;
    end
end

% Find where density drops below Otsu threshold (cortical → cancellous)
valid = ~isnan(mean_hu_by_depth);
if any(valid)
    first_below = find(valid & (mean_hu_by_depth < otsu_hu), 1, 'first');
    if ~isempty(first_below) && first_below > 1
        cortical_thickness = depth_centers(first_below);
    else
        cortical_thickness = max_depth * 0.3;
    end
else
    cortical_thickness = max_depth * 0.3;
end

% Clamp to reasonable range
cortical_thickness = max(0.3, min(cortical_thickness, max_depth * 0.6));

% ---- Classification: combine depth AND density ----
% Cortical = (near surface) AND (dense), or very dense anywhere
is_outer = D_mm <= cortical_thickness;
is_dense = vol >= otsu_hu;

cortical = bone_mask & (is_outer & is_dense);

% Interior voxels that are also very dense should be cortical
% (e.g. cortical bridges, endosteal surfaces)
very_dense_thr = otsu_hu + 0.3 * (hu_max - otsu_hu);
cortical = cortical | (bone_mask & (vol >= very_dense_thr));

cancellous = bone_mask & ~cortical;

% ---- Morphological cleanup ----
se = strel('sphere', 1);
cortical = imclose(cortical, se);
cancellous = bone_mask & ~cortical;

% ---- Output info ----
info = struct();
info.otsu_threshold = otsu_hu;
info.cortical_thickness_mm = cortical_thickness;
info.cortical_volume_mm3 = sum(cortical(:)) * voxel_vol;
info.cancellous_volume_mm3 = sum(cancellous(:)) * voxel_vol;
info.cortical_fraction = sum(cortical(:)) / max(1, sum(bone_mask(:)));
info.depth_profile = struct('depth_mm', depth_centers, ...
                            'mean_hu', mean_hu_by_depth);
end
