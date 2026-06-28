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
fprintf('      Tissue voxels: %d / %d (%.0f%%)\n', n_tissue, n_total, 100*n_tissue/max(1,n_total));

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

% Project onto principal axis -> axial position
axial_pos = coords_centered * principal_axis';
axial_min = min(axial_pos);
axial_max = max(axial_pos);
axial_length = axial_max - axial_min;

% Build full-volume axial position map for bone voxels
axial_map = zeros(size(bone_mask));
linear_idx = find(bone_tissue_mask);
axial_map(linear_idx) = axial_pos;

fprintf('      Bone length: %.1f mm along principal axis\n', axial_length);

% ---- Determine slab count ----
SLAB_WIDTH_MM = 4.0;
n_slabs = max(3, round(axial_length / SLAB_WIDTH_MM));
n_slabs = min(n_slabs, 12);
slab_edges = linspace(axial_min, axial_max, n_slabs + 1);
slab_centers = (slab_edges(1:end-1) + slab_edges(2:end)) / 2;

fprintf('      Slabs: %d (%.1f mm each)\n', n_slabs, axial_length / n_slabs);

% ---- Depth bins for profiling ----
max_depth = max(D_mm(bone_tissue_mask));
DEPTH_BIN_MM = 0.25;
n_depth_bins = max(8, round(max_depth / DEPTH_BIN_MM));
depth_edges = linspace(0, max_depth, n_depth_bins + 1);
depth_centers = (depth_edges(1:end-1) + depth_edges(2:end)) / 2;

fprintf('      Max depth: %.2f mm (%d depth bins)\n', max_depth, n_depth_bins);

% ---- Per-slab depth profiles and transition detection ----
SURFACE_DEPTH_MM = 0.5;
TRANSITION_FRAC = 0.50;
MIN_CORTICAL_DEPTH = 0.25;
MAX_CORTICAL_DEPTH = min(3.0, max_depth * 0.6);

slab_surface_peak = zeros(n_slabs, 1);
slab_transition_depth = zeros(n_slabs, 1);
slab_threshold = zeros(n_slabs, 1);
slab_profile = zeros(n_depth_bins, n_slabs);

fprintf('      Slab profiles:\n');
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

    % Surface peak: mean HU of the outermost populated bins (within 0.5mm)
    surface_bins = depth_centers <= SURFACE_DEPTH_MM;
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

    % Transition threshold: fraction of surface peak
    local_thr = slab_surface_peak(si) * TRANSITION_FRAC;
    slab_threshold(si) = max(local_thr, 80);

    % Find transition depth: first depth where profile drops below threshold
    trans_depth = MAX_CORTICAL_DEPTH;
    valid = ~isnan(profile);
    for di = 2:n_depth_bins
        if valid(di) && profile(di) < slab_threshold(si)
            trans_depth = depth_centers(di);
            break;
        end
    end
    slab_transition_depth(si) = max(MIN_CORTICAL_DEPTH, min(trans_depth, MAX_CORTICAL_DEPTH));

    fprintf('        Slab %d: surface=%.0f HU, thr=%.0f HU, depth=%.2f mm\n', ...
        si, slab_surface_peak(si), slab_threshold(si), slab_transition_depth(si));
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
    slab_transition_depth(1) = slab_transition_depth(1) / (0.5 + 0.25);
    slab_transition_depth(end) = slab_transition_depth(end) / (0.5 + 0.25);
    slab_threshold(1) = slab_threshold(1) / (0.5 + 0.25);
    slab_threshold(end) = slab_threshold(end) / (0.5 + 0.25);
end

% ---- Classify each voxel ----
cortical = false(size(bone_mask));
bone_idx = find(bone_tissue_mask);
bone_axial = axial_map(bone_idx);
bone_depth = D_mm(bone_idx);
bone_hu = vol(bone_idx);

% Assign each voxel to nearest slab and interpolate parameters
for vi = 1:numel(bone_idx)
    ap = bone_axial(vi);

    % Find enclosing slab (clamp to range)
    si = find(slab_edges(1:end-1) <= ap & slab_edges(2:end) > ap, 1);
    if isempty(si)
        if ap <= slab_edges(1)
            si = 1;
        else
            si = n_slabs;
        end
    end

    % Interpolate between this slab and neighbor for smooth transitions
    slab_mid = slab_centers(si);
    if si > 1 && ap < slab_mid
        frac = (slab_mid - ap) / (slab_centers(si) - slab_centers(si-1));
        frac = min(1, max(0, frac));
        local_depth = slab_transition_depth(si) * (1 - frac) + slab_transition_depth(si-1) * frac;
        local_thr = slab_threshold(si) * (1 - frac) + slab_threshold(si-1) * frac;
    elseif si < n_slabs && ap >= slab_mid
        frac = (ap - slab_mid) / (slab_centers(si+1) - slab_centers(si));
        frac = min(1, max(0, frac));
        local_depth = slab_transition_depth(si) * (1 - frac) + slab_transition_depth(si+1) * frac;
        local_thr = slab_threshold(si) * (1 - frac) + slab_threshold(si+1) * frac;
    else
        local_depth = slab_transition_depth(si);
        local_thr = slab_threshold(si);
    end

    if bone_depth(vi) <= local_depth && bone_hu(vi) >= local_thr
        cortical(bone_idx(vi)) = true;
    end
end

cancellous = bone_mask & ~cortical;

% ---- Morphological cleanup ----
se = strel('sphere', 1);
cortical = imclose(cortical, se);
cancellous = bone_mask & ~cortical;

% ---- Global summary stats ----
cortical_vol = sum(cortical(:)) * voxel_vol;
cancellous_vol = sum(cancellous(:)) * voxel_vol;
cortical_frac = sum(cortical(:)) / max(1, n_total);

mean_trans_depth = mean(slab_transition_depth);
mean_threshold = mean(slab_threshold);
fprintf('      Mean cortical depth: %.2f mm, mean threshold: %.0f HU\n', ...
    mean_trans_depth, mean_threshold);
fprintf('      Cortical: %.0f mm^3 (%.0f%%) | Cancellous: %.0f mm^3\n', ...
    cortical_vol, cortical_frac*100, cancellous_vol);

% ---- Output info ----
info = struct();
info.method = 'adaptive_depth_profile';
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
    info = struct('method', 'fallback', 'n_slabs', 0, ...
        'axial_length_mm', 0, ...
        'mean_cortical_depth_mm', NaN, 'mean_threshold_hu', NaN, ...
        'cortical_volume_mm3', sum(cortical(:))*voxel_vol, ...
        'cancellous_volume_mm3', sum(cancellous(:))*voxel_vol, ...
        'cortical_fraction', sum(cortical(:)) / max(1, sum(bone_mask(:))));
end
