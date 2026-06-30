function [cortical, cancellous, info] = cortical_cancellous(ds, bone_mask, opts)
% CORTICAL_CANCELLOUS  Segment a bone mask into cortical and cancellous regions.
%
%   [cortical, cancellous, info] = bone.cortical_cancellous(ds, bone_mask, opts)

vol = double(ds.HU);
spacing = ds.spacing;
voxel_vol = prod(spacing);

% ---- Distance from bone surface (anisotropic-aware) ----
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

% ---- In-bone tissue voxels ----
bone_tissue_mask = bone_mask & (vol > -200);
n_tissue = sum(bone_tissue_mask(:));
n_total = sum(bone_mask(:));
% Tissue count stored in info struct

if n_tissue < 10
    cortical = bone_mask;
    cancellous = false(size(bone_mask));
    info = make_empty_info(cortical, cancellous, voxel_vol, bone_mask);
    return;
end

% ---- Principal axis via PCA ----
[r, c, s] = ind2sub(size(bone_mask), find(bone_tissue_mask));
coords_mm = [r(:)*spacing(1), c(:)*spacing(2), s(:)*spacing(3)];
centroid = mean(coords_mm, 1);
coords_centered = coords_mm - centroid;
[V, ~] = eig(coords_centered' * coords_centered);
principal_axis = V(:, 3)';

axial_pos = coords_centered * principal_axis';
axial_min = min(axial_pos);
axial_max = max(axial_pos);
axial_length = axial_max - axial_min;

axial_map = zeros(size(bone_mask));
linear_idx = find(bone_tissue_mask);
axial_map(linear_idx) = axial_pos;

% axial_length stored in info

% ---- Determine slab count ----
SLAB_WIDTH_MM = 4.0;
n_slabs = max(3, round(axial_length / SLAB_WIDTH_MM));
n_slabs = min(n_slabs, 12);
slab_edges = linspace(axial_min, axial_max, n_slabs + 1);
slab_centers = (slab_edges(1:end-1) + slab_edges(2:end)) / 2;

% slab count stored in info

% ---- Depth bins for profiling ----
max_depth = max(D_mm(bone_tissue_mask));
DEPTH_BIN_MM = 0.20;
n_depth_bins = max(10, round(max_depth / DEPTH_BIN_MM));
depth_edges = linspace(0, max_depth, n_depth_bins + 1);
depth_centers = ((depth_edges(1:end-1) + depth_edges(2:end)) / 2)';
bin_width = depth_edges(2) - depth_edges(1);

% depth info stored in info struct

% ---- Anatomical cortical depth limits ----
% Metacarpal shaft: ~1.5-2.5mm, near joints: ~0.5-1.0mm
% Carpal bones: ~0.3-0.8mm
MIN_CORTICAL_DEPTH = 0.25;
MAX_CORTICAL_DEPTH_LONG = 2.5;   % for elongated bones (metacarpals)
MAX_CORTICAL_DEPTH_SHORT = 1.2;  % for compact bones (carpals)

% Classify bone shape by aspect ratio
elongation = axial_length / max(1, max_depth * 2);
if elongation > 2.0
    max_cortical_cap = MAX_CORTICAL_DEPTH_LONG;
    bone_shape = 'elongated';
else
    max_cortical_cap = MAX_CORTICAL_DEPTH_SHORT;
    bone_shape = 'compact';
end
% shape classification stored in info

% ---- Per-slab depth profiles and gradient-based transition detection ----
slab_surface_peak = zeros(n_slabs, 1);
slab_transition_depth = zeros(n_slabs, 1);
slab_threshold = zeros(n_slabs, 1);
slab_profile = zeros(n_depth_bins, n_slabs);

for si = 1:n_slabs
    in_slab = bone_tissue_mask & ...
        (axial_map >= slab_edges(si)) & (axial_map < slab_edges(si+1));

    if sum(in_slab(:)) < 20
        slab_surface_peak(si) = NaN;
        slab_transition_depth(si) = NaN;
        slab_threshold(si) = NaN;
        continue;
    end

    % Build depth-vs-HU profile for this slab
    profile = NaN(n_depth_bins, 1);
    for di = 1:n_depth_bins
        in_bin = in_slab & (D_mm >= depth_edges(di)) & (D_mm < depth_edges(di+1));
        if sum(in_bin(:)) >= 3
            profile(di) = mean(vol(in_bin));
        end
    end
    slab_profile(:, si) = profile;

    % Surface peak: max HU of the outermost populated bins (within 0.5mm)
    surface_bins = depth_centers <= 0.5;
    surface_vals = profile(surface_bins & ~isnan(profile));
    if isempty(surface_vals)
        valid_profile = profile(~isnan(profile));
        if isempty(valid_profile)
            slab_surface_peak(si) = NaN;
            slab_transition_depth(si) = NaN;
            slab_threshold(si) = NaN;
            continue;
        end
        slab_surface_peak(si) = valid_profile(1);
    else
        slab_surface_peak(si) = max(surface_vals);
    end

    % ---- Gradient-based transition detection ----
    % Find the depth where HU drops most steeply (max negative gradient).
    % This is the cortical-cancellous boundary: the steepest density change.
    valid = ~isnan(profile);
    valid_idx = find(valid);

    trans_depth = max_cortical_cap;
    if numel(valid_idx) >= 3
        % Compute gradient (HU change per mm)
        grad = zeros(n_depth_bins, 1);
        for di = 2:n_depth_bins
            if valid(di) && valid(di-1)
                grad(di) = (profile(di) - profile(di-1)) / bin_width;
            end
        end

        % Look for steepest negative gradient within anatomical range
        search_range = valid & (depth_centers >= MIN_CORTICAL_DEPTH) & ...
                       (depth_centers <= max_cortical_cap) & (grad < 0);
        if any(search_range)
            candidates = find(search_range);
            [~, best] = min(grad(candidates));
            inflection_bin = candidates(best);
            trans_depth = depth_centers(inflection_bin);
        else
            % Fallback: first bin where HU < 70% of surface peak
            fallback_thr = slab_surface_peak(si) * 0.70;
            for di = 2:n_depth_bins
                if valid(di) && profile(di) < fallback_thr && ...
                        depth_centers(di) >= MIN_CORTICAL_DEPTH
                    trans_depth = depth_centers(di);
                    break;
                end
            end
        end
    end

    slab_transition_depth(si) = max(MIN_CORTICAL_DEPTH, min(trans_depth, max_cortical_cap));

    % Threshold: HU at the transition depth (what density defines the boundary)
    trans_bin = find(depth_centers >= slab_transition_depth(si), 1);
    if ~isempty(trans_bin) && valid(trans_bin)
        slab_threshold(si) = max(80, profile(trans_bin));
    else
        slab_threshold(si) = max(80, slab_surface_peak(si) * 0.50);
    end

    % (slab details stored in info struct for inspection)
end

% ---- Interpolate NaN slabs from neighbors ----
valid_slabs = ~isnan(slab_surface_peak);
if ~any(valid_slabs)
    cortical = bone_mask;
    cancellous = false(size(bone_mask));
    info = make_empty_info(cortical, cancellous, voxel_vol, bone_mask);
    return;
end

if sum(valid_slabs) < n_slabs
    xi = 1:n_slabs;
    vi = find(valid_slabs);
    slab_surface_peak = interp1(vi, slab_surface_peak(valid_slabs), xi, 'nearest', 'extrap')';
    slab_transition_depth = interp1(vi, slab_transition_depth(valid_slabs), xi, 'nearest', 'extrap')';
    slab_threshold = interp1(vi, slab_threshold(valid_slabs), xi, 'nearest', 'extrap')';
end

% ---- Smooth parameters across slabs (3-point moving average) ----
if n_slabs >= 3
    kernel = [0.25; 0.5; 0.25];
    slab_transition_depth = conv(slab_transition_depth, kernel, 'same');
    slab_threshold = conv(slab_threshold, kernel, 'same');
    % Fix edge normalization (kernel doesn't sum to 1 at boundaries)
    slab_transition_depth(1) = slab_transition_depth(1) / 0.75;
    slab_transition_depth(end) = slab_transition_depth(end) / 0.75;
    slab_threshold(1) = slab_threshold(1) / 0.75;
    slab_threshold(end) = slab_threshold(end) / 0.75;
end

% ---- Vectorized classification ----
cortical = false(size(bone_mask));
bone_idx = find(bone_tissue_mask);
bone_axial = axial_map(bone_idx);
bone_depth = D_mm(bone_idx);
bone_hu = vol(bone_idx);

% Precompute per-voxel local depth and threshold via interpolation
local_depths = zeros(numel(bone_idx), 1);
local_thrs = zeros(numel(bone_idx), 1);

for vi = 1:numel(bone_idx)
    ap = bone_axial(vi);

    si = find(slab_edges(1:end-1) <= ap & slab_edges(2:end) > ap, 1);
    if isempty(si)
        if ap <= slab_edges(1), si = 1; else, si = n_slabs; end
    end

    slab_mid = slab_centers(si);
    if si > 1 && ap < slab_mid
        frac = min(1, max(0, (slab_mid - ap) / (slab_centers(si) - slab_centers(si-1))));
        local_depths(vi) = slab_transition_depth(si) * (1-frac) + slab_transition_depth(si-1) * frac;
        local_thrs(vi) = slab_threshold(si) * (1-frac) + slab_threshold(si-1) * frac;
    elseif si < n_slabs && ap >= slab_mid
        frac = min(1, max(0, (ap - slab_mid) / (slab_centers(si+1) - slab_centers(si))));
        local_depths(vi) = slab_transition_depth(si) * (1-frac) + slab_transition_depth(si+1) * frac;
        local_thrs(vi) = slab_threshold(si) * (1-frac) + slab_threshold(si+1) * frac;
    else
        local_depths(vi) = slab_transition_depth(si);
        local_thrs(vi) = slab_threshold(si);
    end
end

is_cortical = (bone_depth <= local_depths) & (bone_hu >= local_thrs);
cortical(bone_idx(is_cortical)) = true;
cancellous = bone_mask & ~cortical;

% ---- Morphological cleanup: close small gaps in cortical shell ----
se = strel('sphere', 1);
cortical = imclose(cortical, se);
cancellous = bone_mask & ~cortical;

% ---- Global summary stats ----
cortical_vol = sum(cortical(:)) * voxel_vol;
cancellous_vol = sum(cancellous(:)) * voxel_vol;
cortical_frac = sum(cortical(:)) / max(1, n_total);

mean_trans_depth = mean(slab_transition_depth);
mean_threshold = mean(slab_threshold);
% final stats stored in info struct

% ---- Output info ----
info = struct();
info.method = 'gradient_inflection';
info.bone_shape = bone_shape;
info.elongation = elongation;
info.n_slabs = n_slabs;
info.axial_length_mm = axial_length;
info.slab_centers = slab_centers;
info.slab_surface_peak = slab_surface_peak;
info.slab_transition_depth = slab_transition_depth;
info.slab_threshold = slab_threshold;
info.slab_profiles = slab_profile;
info.depth_bin_centers = depth_centers;
info.mean_cortical_depth_mm = mean_trans_depth;
info.mean_threshold_hu = mean_threshold;
info.cortical_volume_mm3 = cortical_vol;
info.cancellous_volume_mm3 = cancellous_vol;
info.cortical_fraction = cortical_frac;
end


function info = make_empty_info(cortical, cancellous, voxel_vol, bone_mask)
    info = struct('method', 'fallback', 'bone_shape', 'unknown', ...
        'elongation', 0, 'n_slabs', 0, ...
        'axial_length_mm', 0, ...
        'mean_cortical_depth_mm', NaN, 'mean_threshold_hu', NaN, ...
        'cortical_volume_mm3', sum(cortical(:))*voxel_vol, ...
        'cancellous_volume_mm3', sum(cancellous(:))*voxel_vol, ...
        'cortical_fraction', sum(cortical(:)) / max(1, sum(bone_mask(:))));
end
