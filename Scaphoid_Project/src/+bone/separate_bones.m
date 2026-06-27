function result = separate_bones(ds, opts)
% SEPARATE_BONES  Isolate individual bones from a multi-bone excised-in-air CT scan.
%
%   result = bone.separate_bones(ds, opts)
%
% For excised specimens scanned in air, the non-air non-tag region IS the
% bone.  Uses the scaphoid pipeline's proven marker detection strategy:
% lead cores (HU>1200) plus flag collars (HU 200-700 near lead), with
% Gaussian artifact weighting.

vol = double(ds.HU);
spacing = ds.spacing;
voxel_vol = prod(spacing);

% ---- Stage 1: Specimen isolation ----
fprintf('  [Separate] Stage 1: Specimen isolation...\n');
specimen = isolate_specimen(vol, spacing, opts.ClosingRadiusMM);
fprintf('    Specimen: %.0f mm^3 (%d voxels)\n', ...
    sum(specimen(:))*voxel_vol, sum(specimen(:)));

% ---- Stage 2: Marker & artifact detection (scaphoid pipeline approach) ----
fprintf('  [Separate] Stage 2: Marker & artifact detection...\n');
lead_core = vol > opts.TagHUMin;
[marker_mask, artifact_w] = marker_and_artifact_maps(vol, opts.MarkerRangeHU, ...
    opts.ArtifactSigmaMM, spacing);

% Extended marker mask: also include bright voxels (700-1200 HU) near lead
% that aren't caught by the standard collar range
bright_near_lead = (vol >= 700 & vol <= 1200) & imdilate(lead_core, strel('sphere', 4));
marker_mask_ext = marker_mask | bright_near_lead;

% Dilate for exclusion — generous to prevent tag-bone merging
marker_excl = imdilate(marker_mask_ext, strel('sphere', 5));

% Count actual lead tags (real tags are big enough, not streak artifacts)
CC_tags = bwconncomp(lead_core, 26);
min_tag_vox = max(5, round(2.0 / voxel_vol));  % at least 2 mm^3
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
fprintf('    Marker mask: %d voxels, %d real tags (>%.0f mm^3)\n', ...
    sum(marker_mask(:)), numel(real_tags), 2.0);

% ---- Stage 3: Bone envelope detection ----
fprintf('  [Separate] Stage 3: Bone envelope detection...\n');

% Bone = specimen minus markers and their proximity
bone_region = specimen & ~marker_excl;

% Split into connected components
CC = bwconncomp(bone_region, 26);
fprintf('    Raw bone region: %.0f mm^3, %d components\n', ...
    sum(bone_region(:))*voxel_vol, CC.NumObjects);

% ---- Stage 4: Per-bone fill & validation ----
fprintf('  [Separate] Stage 4: Per-bone fill & validation...\n');
bones = {};
small_count = 0;

for i = 1:CC.NumObjects
    comp = false(size(vol));
    comp(CC.PixelIdxList{i}) = true;
    comp_vol = sum(comp(:)) * voxel_vol;

    if comp_vol < opts.MinBoneVolMM3
        small_count = small_count + 1;
        continue;
    end

    % Must have some dense bone tissue (not just noise)
    n_dense = sum(vol(comp) > 200);
    dense_frac = n_dense / max(1, sum(comp(:)));
    if dense_frac < 0.02
        fprintf('    Component %d: %.0f mm^3 — skipped (%.1f%% dense)\n', ...
            i, comp_vol, dense_frac*100);
        continue;
    end

    % Mean HU must be plausible for bone (not air/noise)
    mean_hu_raw = mean(vol(comp));
    if mean_hu_raw < -100
        fprintf('    Component %d: %.0f mm^3 — skipped (mean HU %.0f, not bone)\n', ...
            i, comp_vol, mean_hu_raw);
        continue;
    end

    % Reject if significant fraction is very bright (likely tag material)
    n_very_bright = sum(vol(comp) > 1000);
    bright_frac = n_very_bright / max(1, sum(comp(:)));
    if bright_frac > 0.10
        fprintf('    Component %d: %.0f mm^3 — skipped (%.0f%% > 1000 HU, likely tag)\n', ...
            i, comp_vol, bright_frac*100);
        continue;
    end

    % Per-slice 2D fill captures enclosed marrow cavities
    filled = comp;
    for z = 1:size(filled, 3)
        sl = filled(:,:,z);
        if any(sl(:))
            filled(:,:,z) = imfill(sl, 'holes');
        end
    end
    filled = imfill(filled, 'holes');

    % Stay within specimen AND away from markers
    filled = filled & specimen & ~marker_excl;

    filled_vol = sum(filled(:)) * voxel_vol;
    filled_hu = mean(vol(filled));

    % Centroid in mm
    [rr, cc, ss] = ind2sub(size(filled), find(filled));
    centroid_vox = [mean(rr), mean(cc), mean(ss)];
    centroid_mm = centroid_vox .* spacing;

    % Bounding box
    bbox = [min(rr) min(cc) min(ss) max(rr) max(cc) max(ss)];

    bone_info = struct();
    bone_info.mask = filled;
    bone_info.label = i;
    bone_info.centroid_mm = centroid_mm;
    bone_info.volume_mm3 = filled_vol;
    bone_info.mean_hu = filled_hu;
    bone_info.dense_fraction = dense_frac;
    bone_info.bbox = bbox;
    bone_info.tag_id = [];
    bone_info.tag_dist = Inf;

    bones{end+1} = bone_info; %#ok<AGROW>

    fprintf('    Bone: %.0f -> %.0f mm^3 (fill +%.0f), mean HU %.0f, dense %.0f%%\n', ...
        comp_vol, filled_vol, filled_vol - comp_vol, filled_hu, dense_frac*100);
end

if small_count > 0
    fprintf('    Filtered %d small components (< %.0f mm^3)\n', ...
        small_count, opts.MinBoneVolMM3);
end

% ---- Stage 5: Tag association ----
associate_tags(bones, real_tags);

% Sort by volume (largest first)
vols = cellfun(@(b) b.volume_mm3, bones);
[~, order] = sort(vols, 'descend');
bones = bones(order);

fprintf('\n  Found %d bones and %d tags in scan\n', numel(bones), numel(real_tags));
for i = 1:numel(bones)
    b = bones{i};
    if ~isempty(b.tag_id)
        tag_str = sprintf('tag %d', b.tag_id);
    else
        tag_str = 'no tag';
    end
    fprintf('    Bone %d: %.1f mm^3, mean HU %.0f, %s\n', ...
        i, b.volume_mm3, b.mean_hu, tag_str);
end

result = struct();
result.bones = bones;
result.specimen = specimen;
result.marker_mask = marker_mask;
result.artifact_weight = artifact_w;
result.n_tags = numel(real_tags);
end


% =========================================================================
function specimen = isolate_specimen(vol, spacing, closing_radius_mm)
    non_air = vol > -500;
    non_air = physical_close(non_air, closing_radius_mm, spacing);
    non_air = imfill(non_air, 'holes');

    voxel_vol = prod(spacing);
    min_vox = max(100, round(50.0 / voxel_vol));

    CC = bwconncomp(non_air, 26);
    for i = 1:CC.NumObjects
        if numel(CC.PixelIdxList{i}) < min_vox
            non_air(CC.PixelIdxList{i}) = false;
        end
    end
    specimen = non_air;
end


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
    % Scaphoid pipeline's marker detection strategy:
    % Lead cores (HU > 1200) plus flag collars (HU in marker_range near lead)
    lead = HU > 1200;
    flags = (HU >= marker_range(1) & HU <= marker_range(2)) & ...
            imdilate(lead, strel('sphere', 2));
    marker_mask = lead | flags;

    % Gaussian distance falloff for artifact weighting
    d_vox = bwdist(marker_mask);
    d_mm = d_vox * mean(spacing);
    artifact_w = exp(-(d_mm / sigma_mm).^2);
end


function associate_tags(bones, tags)
    if isempty(tags) || isempty(bones), return; end
    centroids = zeros(numel(bones), 3);
    for i = 1:numel(bones)
        centroids(i,:) = bones{i}.centroid_mm;
    end
    for t = 1:numel(tags)
        dists = vecnorm(centroids - tags{t}.centroid_mm, 2, 2);
        [d, idx] = min(dists);
        if isempty(bones{idx}.tag_id) || d < bones{idx}.tag_dist
            bones{idx}.tag_id = tags{t}.label;
            bones{idx}.tag_dist = d;
        end
    end
end
