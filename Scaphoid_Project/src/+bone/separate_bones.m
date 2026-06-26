function result = separate_bones(ds, opts)
% SEPARATE_BONES  Isolate individual bones from a multi-bone excised-in-air CT scan.
%
%   result = bone.separate_bones(ds, opts)
%
% For excised specimens scanned in air, the non-air non-tag region IS the
% bone.  This function finds the specimen envelope, removes lead tags, and
% splits into connected components — one per bone.
%
% Inputs
%   ds   : dataset struct from dicom.series_load
%   opts : options struct with fields TagHUMin, MinBoneVolMM3, ClosingRadiusMM
%
% Output  result  struct with fields:
%   .bones      : cell array of structs, one per bone
%   .specimen   : logical 3D mask of all non-air material
%   .n_tags     : number of detected tags

vol = double(ds.HU);
spacing = ds.spacing;
voxel_vol = prod(spacing);

% ---- Stage 1: Specimen isolation ----
fprintf('  [Separate] Stage 1: Specimen isolation...\n');
specimen = isolate_specimen(vol, spacing, opts.ClosingRadiusMM);
fprintf('    Specimen: %.0f mm^3 (%d voxels)\n', ...
    sum(specimen(:))*voxel_vol, sum(specimen(:)));

% ---- Stage 2: Tag detection ----
fprintf('  [Separate] Stage 2: Tag detection...\n');
lead_mask = vol > opts.TagHUMin;
tags = find_tags(lead_mask, spacing, voxel_vol);
fprintf('    Found %d metal tags\n', numel(tags));

if any(lead_mask(:))
    % Anisotropy-aware distance: build ellipsoidal exclusion
    tag_dist_mm = aniso_distance_mm(lead_mask, spacing);
    tag_exclusion = tag_dist_mm < 1.0;
else
    tag_exclusion = false(size(vol));
end

% ---- Stage 3: Bone envelope detection ----
fprintf('  [Separate] Stage 3: Bone envelope detection...\n');
bone_region = specimen & ~tag_exclusion;
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

    mean_hu = mean(vol(comp));
    if mean_hu > opts.TagHUMin
        fprintf('    Component %d: %.0f mm^3 — skipped (mean HU %.0f, likely tag)\n', ...
            i, comp_vol, mean_hu);
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
    filled = filled & specimen;

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
associate_tags(bones, tags);

% Sort by volume (largest first)
vols = cellfun(@(b) b.volume_mm3, bones);
[~, order] = sort(vols, 'descend');
bones = bones(order);

fprintf('\n  Found %d bones and %d tags in scan\n', numel(bones), numel(tags));
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
result.n_tags = numel(tags);
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
    % Morphological closing with a true spherical structuring element
    % that accounts for anisotropic voxel spacing.
    radius_vox = ceil(radius_mm ./ spacing);
    [Y, X, Z] = ndgrid(-radius_vox(1):radius_vox(1), ...
                        -radius_vox(2):radius_vox(2), ...
                        -radius_vox(3):radius_vox(3));
    dist_mm_sq = (Y*spacing(1)).^2 + (X*spacing(2)).^2 + (Z*spacing(3)).^2;
    se = strel(dist_mm_sq <= radius_mm^2);
    closed = imclose(mask, se);
end


function d_mm = aniso_distance_mm(BW, spacing)
    % Approximate anisotropic Euclidean distance in mm.
    % Rescale to isotropic, compute bwdist, scale back.
    iso = min(spacing);
    scale = spacing ./ iso;
    sz_new = round(size(BW) .* scale);
    BW_iso = imresize3(uint8(BW), sz_new, 'nearest') > 0;
    d_iso = bwdist(BW_iso) * iso;
    d_mm = imresize3(single(d_iso), size(BW), 'linear');
end


function tags = find_tags(lead_mask, spacing, voxel_vol)
    tags = {};
    if ~any(lead_mask(:)), return; end
    CC = bwconncomp(lead_mask, 26);
    for i = 1:CC.NumObjects
        comp = false(size(lead_mask));
        comp(CC.PixelIdxList{i}) = true;
        [rr, cc, ss] = ind2sub(size(lead_mask), CC.PixelIdxList{i});
        tag = struct();
        tag.label = i;
        tag.cc_idx = i;
        tag.centroid_mm = [mean(rr) mean(cc) mean(ss)] .* spacing;
        tag.volume_mm3 = numel(CC.PixelIdxList{i}) * voxel_vol;
        tags{end+1} = tag; %#ok<AGROW>
    end
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
