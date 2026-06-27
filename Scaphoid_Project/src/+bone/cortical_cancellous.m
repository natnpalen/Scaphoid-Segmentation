function [cortical, cancellous, info] = cortical_cancellous(ds, bone_mask, opts)
% CORTICAL_CANCELLOUS  Segment a bone mask into cortical and cancellous regions.
%
%   [cortical, cancellous, info] = bone.cortical_cancellous(ds, bone_mask, opts)

vol = double(ds.HU);
spacing = ds.spacing;
voxel_vol = prod(spacing);

% ---- Distance from bone surface ----
% Use anisotropic distance: rescale volume to isotropic grid for bwdist
iso_spacing = min(spacing);
scale = spacing / iso_spacing;
if any(abs(scale - 1) > 0.01)
    bone_mask_iso = imresize3(uint8(bone_mask), round(size(bone_mask) .* scale), 'nearest') > 0;
    D_iso = bwdist(~bone_mask_iso) * iso_spacing;
    D_mm = imresize3(single(D_iso), size(bone_mask), 'linear');
    D_mm = double(D_mm);
else
    D_mm = bwdist(~bone_mask) * iso_spacing;
end

% ---- In-bone HU values (exclude air for Otsu) ----
bone_tissue_mask = bone_mask & (vol > -200);
hu_bone = vol(bone_tissue_mask);

n_tissue = sum(bone_tissue_mask(:));
n_total = sum(bone_mask(:));
fprintf('      Tissue voxels: %d / %d (%.0f%%)\n', n_tissue, n_total, 100*n_tissue/max(1,n_total));

if isempty(hu_bone) || numel(hu_bone) < 10
    cortical = bone_mask;
    cancellous = false(size(bone_mask));
    info = struct('otsu_threshold', NaN, 'cortical_thickness_mm', NaN, ...
        'cortical_volume_mm3', sum(cortical(:))*voxel_vol, ...
        'cancellous_volume_mm3', 0, 'cortical_fraction', 1.0);
    return;
end

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

hu_norm = (hu_bone - hu_min) / (hu_max - hu_min);
otsu_level = graythresh(hu_norm);
otsu_hu = otsu_level * (hu_max - hu_min) + hu_min;

fprintf('      HU range: [%.0f, %.0f], Otsu: %.0f HU\n', hu_min, hu_max, otsu_hu);

% ---- Estimate cortical thickness from depth profile ----
max_depth = max(D_mm(bone_tissue_mask));
fprintf('      Max depth: %.2f mm\n', max_depth);

n_bins = max(10, round(max_depth / 0.2));
depth_edges = linspace(0, max_depth, n_bins + 1);
depth_centers = (depth_edges(1:end-1) + depth_edges(2:end)) / 2;

mean_hu_by_depth = zeros(n_bins, 1);
for b = 1:n_bins
    in_bin = bone_tissue_mask & (D_mm >= depth_edges(b)) & (D_mm < depth_edges(b+1));
    if any(in_bin(:))
        mean_hu_by_depth(b) = mean(vol(in_bin));
    else
        mean_hu_by_depth(b) = NaN;
    end
end

% Log first few depth bins for diagnostics
fprintf('      Depth profile (first bins): ');
for b = 1:min(6, n_bins)
    if ~isnan(mean_hu_by_depth(b))
        fprintf('%.1fmm:%.0f ', depth_centers(b), mean_hu_by_depth(b));
    end
end
fprintf('\n');

% Find cortical-cancellous transition
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

cortical_thickness = max(0.3, min(cortical_thickness, max_depth * 0.6));

% ---- Classification ----
is_outer = D_mm <= cortical_thickness;
is_dense = vol >= otsu_hu;

cortical = bone_mask & (is_outer & is_dense);

very_dense_thr = otsu_hu + 0.3 * (hu_max - otsu_hu);
cortical = cortical | (bone_mask & (vol >= very_dense_thr));

cancellous = bone_mask & ~cortical;

% Morphological cleanup
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
