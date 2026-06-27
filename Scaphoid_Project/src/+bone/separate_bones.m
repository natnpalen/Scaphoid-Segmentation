function result = separate_bones(ds, opts)
% SEPARATE_BONES  Isolate individual bones from a multi-bone excised-in-air CT scan.
%
%   result = bone.separate_bones(ds, opts)
%
% For excised specimens scanned in air, bones are separated by air gaps.
% Strategy:
%   1. Find non-air voxels with minimal closing (preserve natural gaps)
%   2. Remove lead cores (HU>1200) before component analysis
%   3. Connected components → each is either a bone or tag collar
%   4. Classify by HU distribution and volume
%   5. Per-bone refinement using scaphoid pipeline's protected-core approach

vol = double(ds.HU);
spacing = ds.spacing;
voxel_vol = prod(spacing);

% ---- Stage 1: Marker detection ----
fprintf('  [Separate] Stage 1: Marker detection...\n');
lead_core = vol > opts.TagHUMin;
[marker_mask, artifact_w] = marker_and_artifact_maps(vol, opts.MarkerRangeHU, ...
    opts.ArtifactSigmaMM, spacing);

n_lead = sum(lead_core(:));
n_marker = sum(marker_mask(:));
fprintf('    Lead core voxels: %d (%.1f mm^3)\n', n_lead, n_lead*voxel_vol);
fprintf('    Marker mask voxels: %d (%.1f mm^3)\n', n_marker, n_marker*voxel_vol);

% Count real tags (lead clusters > 2 mm^3)
CC_tags = bwconncomp(lead_core, 26);
min_tag_vox = max(5, round(2.0 / voxel_vol));
real_tags = {};
for i = 1:CC_tags.NumObjects
    if numel(CC_tags.PixelIdxList{i}) >= min_tag_vox
        [rr, cc, ss] = ind2sub(size(vol), CC_tags.PixelIdxList{i});
        tag = struct();
        tag.label = numel(real_tags) + 1;
        tag.centroid_mm = [mean(rr) mean(cc) mean(ss)] .* spacing;
        tag.volume_mm3 = numel(CC_tags.PixelIdxList{i}) * voxel_vol;
        real_tags{end+1} = tag; %#ok<AGROW>
    end
end
fprintf('    Real tags: %d (each > %.0f mm^3)\n', numel(real_tags), 2.0);
for t = 1:numel(real_tags)
    fprintf('      Tag %d: %.1f mm^3 at [%.1f %.1f %.1f] mm\n', ...
        t, real_tags{t}.volume_mm3, real_tags{t}.centroid_mm);
end

% ---- Stage 2: Find objects (bones + tag collars) ----
fprintf('  [Separate] Stage 2: Finding objects...\n');

% Non-air mask with small closing (1mm) to fill surface porosity only
non_air = vol > -500;
non_air = physical_close(non_air, 1.0, spacing);
fprintf('    Non-air voxels: %d (%.0f mm^3)\n', sum(non_air(:)), sum(non_air(:))*voxel_vol);

% Remove lead cores BEFORE connectivity analysis — this prevents
% tags from bridging to nearby bones through the lead material
non_air_no_lead = non_air & ~imdilate(lead_core, strel('sphere', 1));
fprintf('    After removing lead+1vox buffer: %d voxels (%.0f mm^3)\n', ...
    sum(non_air_no_lead(:)), sum(non_air_no_lead(:))*voxel_vol);

% Connected components — each is naturally a bone or tag collar
CC = bwconncomp(non_air_no_lead, 26);
fprintf('    Connected components: %d\n', CC.NumObjects);

% Sort by volume for logging
comp_vols = cellfun(@numel, CC.PixelIdxList) * voxel_vol;
[~, vol_order] = sort(comp_vols, 'descend');

% ---- Stage 3: Classify components ----
fprintf('  [Separate] Stage 3: Classifying components...\n');
bones = {};
small_count = 0;
tag_collar_count = 0;

for ii = 1:CC.NumObjects
    i = vol_order(ii);
    comp = false(size(vol));
    comp(CC.PixelIdxList{i}) = true;
    comp_vol = sum(comp(:)) * voxel_vol;

    % Skip tiny components
    if comp_vol < opts.MinBoneVolMM3
        small_count = small_count + 1;
        continue;
    end

    % HU statistics of this component
    hu_vals = vol(comp);
    n_vox = numel(hu_vals);
    mean_hu_all = mean(hu_vals);
    hu_bone_vals = hu_vals(hu_vals > -200);
    if ~isempty(hu_bone_vals)
        mean_hu_bone = mean(hu_bone_vals);
    else
        mean_hu_bone = mean_hu_all;
    end

    % Fraction of voxels near lead (within 3 voxels of lead core)
    lead_dilated_3 = imdilate(lead_core, strel('sphere', 3));
    near_lead_frac = sum(comp(:) & lead_dilated_3(:)) / n_vox;

    % Fraction of voxels in marker HU range (200-700)
    in_marker_range = sum(hu_vals >= opts.MarkerRangeHU(1) & hu_vals <= opts.MarkerRangeHU(2)) / n_vox;

    % Fraction very bright (> 800 HU, tag/artifact material)
    bright_frac = sum(hu_vals > 800) / n_vox;

    % Dense bone fraction (> 200 HU)
    dense_frac = sum(hu_vals > 200) / n_vox;

    fprintf('    Component %d: %.0f mm^3, mean HU %.0f (bone-only %.0f)\n', ...
        i, comp_vol, mean_hu_all, mean_hu_bone);
    fprintf('      near_lead=%.0f%%, marker_range=%.0f%%, bright=%.0f%%, dense=%.0f%%\n', ...
        near_lead_frac*100, in_marker_range*100, bright_frac*100, dense_frac*100);

    % Classification: tag collar if mostly near lead AND mostly in marker HU range
    if near_lead_frac > 0.5 && in_marker_range > 0.3
        fprintf('      -> TAG COLLAR (near lead + marker HU range)\n');
        tag_collar_count = tag_collar_count + 1;
        continue;
    end

    % Also reject if very bright (likely tag fragment that survived lead removal)
    if bright_frac > 0.30
        fprintf('      -> TAG FRAGMENT (%.0f%% > 800 HU)\n', bright_frac*100);
        tag_collar_count = tag_collar_count + 1;
        continue;
    end

    % Must have some actual bone tissue
    if dense_frac < 0.02
        fprintf('      -> NOISE (%.1f%% dense)\n', dense_frac*100);
        continue;
    end

    fprintf('      -> BONE CANDIDATE\n');

    % ---- Per-bone refinement ----
    % Small closing to seal surface pores
    refined = physical_close(comp, 1.0, spacing);
    refined = refined & non_air;

    % Per-slice 2D fill to capture enclosed marrow cavities
    for z = 1:size(refined, 3)
        sl = refined(:,:,z);
        if any(sl(:))
            refined(:,:,z) = imfill(sl, 'holes');
        end
    end
    refined = imfill(refined, 'holes');

    % Scaphoid-style marker cleanup: remove marker voxels at boundary,
    % protect interior (imerode + restore core)
    core = imerode(refined, strel('sphere', 1));
    marker_near = imdilate(marker_mask, strel('sphere', 1));
    refined = (refined & ~marker_near) | core;

    % Keep only the largest connected piece
    CC_ref = bwconncomp(refined, 26);
    if CC_ref.NumObjects > 1
        comp_sizes = cellfun(@numel, CC_ref.PixelIdxList);
        [~, largest] = max(comp_sizes);
        refined = false(size(vol));
        refined(CC_ref.PixelIdxList{largest}) = true;
    end

    refined_vol = sum(refined(:)) * voxel_vol;
    if refined_vol < opts.MinBoneVolMM3
        fprintf('      -> Too small after refinement (%.0f mm^3)\n', refined_vol);
        continue;
    end

    % HU of refined bone (bone-tissue voxels only)
    bone_vals = vol(refined & (vol > -200));
    if ~isempty(bone_vals)
        refined_hu = mean(bone_vals);
    else
        refined_hu = mean(vol(refined));
    end

    % Centroid
    [rr, cc, ss] = ind2sub(size(refined), find(refined));
    centroid_mm = [mean(rr), mean(cc), mean(ss)] .* spacing;

    % Bounding box
    bbox = [min(rr) min(cc) min(ss) max(rr) max(cc) max(ss)];

    bone_info = struct();
    bone_info.mask = refined;
    bone_info.label = i;
    bone_info.centroid_mm = centroid_mm;
    bone_info.volume_mm3 = refined_vol;
    bone_info.mean_hu = refined_hu;
    bone_info.dense_fraction = dense_frac;
    bone_info.bbox = bbox;
    bone_info.tag_id = [];
    bone_info.tag_dist = Inf;

    bones{end+1} = bone_info; %#ok<AGROW>

    fprintf('      Final: %.0f -> %.0f mm^3, mean HU %.0f\n', ...
        comp_vol, refined_vol, refined_hu);
end

fprintf('    Filtered: %d small (< %.0f mm^3), %d tag collars/fragments\n', ...
    small_count, opts.MinBoneVolMM3, tag_collar_count);

% ---- Stage 4: Tag association ----
fprintf('  [Separate] Stage 4: Tag association...\n');
bones = associate_tags(bones, real_tags);

% Sort by volume (largest first)
vols = cellfun(@(b) b.volume_mm3, bones);
[~, order] = sort(vols, 'descend');
bones = bones(order);

fprintf('\n  Found %d bones and %d tags in scan\n', numel(bones), numel(real_tags));
for i = 1:numel(bones)
    b = bones{i};
    if ~isempty(b.tag_id)
        tag_str = sprintf('tag %d (%.1f mm away)', b.tag_id, b.tag_dist);
    else
        tag_str = 'no tag';
    end
    fprintf('    Bone %d: %.1f mm^3, mean HU %.0f, centroid [%.1f %.1f %.1f] mm, %s\n', ...
        i, b.volume_mm3, b.mean_hu, b.centroid_mm, tag_str);
end

% Build specimen mask (union of all bones, for visualization)
specimen = false(size(vol));
for i = 1:numel(bones)
    specimen = specimen | bones{i}.mask;
end

result = struct();
result.bones = bones;
result.specimen = specimen;
result.marker_mask = marker_mask;
result.artifact_weight = artifact_w;
result.n_tags = numel(real_tags);
end


% =========================================================================
function closed = physical_close(mask, radius_mm, spacing)
    radius_vox = ceil(radius_mm ./ spacing);
    [Y, X, Z] = ndgrid(-radius_vox(1):radius_vox(1), ...
                        -radius_vox(2):radius_vox(2), ...
                        -radius_vox(3):radius_vox(3));
    dist_mm_sq = (Y*spacing(1)).^2 + (X*spacing(2)).^2 + (Z*spacing(3)).^2;
    se = strel(dist_mm_sq <= radius_mm^2);
    closed = imclose(mask, se);
end


function [marker_mask, artifact_w] = marker_and_artifact_maps(HU, marker_range, sigma_mm, spacing)
    lead = HU > 1200;
    flags = (HU >= marker_range(1) & HU <= marker_range(2)) & ...
            imdilate(lead, strel('sphere', 2));
    marker_mask = lead | flags;

    d_vox = bwdist(marker_mask);
    d_mm = d_vox * mean(spacing);
    artifact_w = exp(-(d_mm / sigma_mm).^2);
end


function bones = associate_tags(bones, tags)
    if isempty(tags) || isempty(bones), return; end
    centroids = zeros(numel(bones), 3);
    for i = 1:numel(bones)
        centroids(i,:) = bones{i}.centroid_mm;
    end

    fprintf('    Tag-bone distances (mm):\n');
    for t = 1:numel(tags)
        dists = vecnorm(centroids - tags{t}.centroid_mm, 2, 2);
        [d, idx] = min(dists);
        fprintf('      Tag %d -> Bone %d: %.1f mm\n', t, idx, d);
        if isempty(bones{idx}.tag_id) || d < bones{idx}.tag_dist
            bones{idx}.tag_id = tags{t}.label;
            bones{idx}.tag_dist = d;
        end
    end
end
