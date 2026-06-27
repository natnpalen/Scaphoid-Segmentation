function result = separate_bones(ds, opts)
% SEPARATE_BONES  Isolate individual bones from a multi-bone excised-in-air CT scan.
%
%   result = bone.separate_bones(ds, opts)
%
% Uses the scaphoid pipeline's seed-and-grow approach adapted for multiple
% bones.  Instead of subtracting tags, each bone is GROWN from a seed point
% using FMM with artifact-weighted speeds.  Tags are never included because
% growth is suppressed near marker assemblies.
%
% Algorithm:
%   1. Detect markers and build artifact weight field (Gaussian falloff)
%   2. Find seed points: one per bone (deepest interior of compact components)
%   3. Per-seed FMM growth with adaptive threshold scoring
%   4. Post-processing: marker carve with protected interior, boundary refine

vol = double(ds.HU);
spacing = ds.spacing;
voxel_vol = prod(spacing);

% ---- Stage 1: Markers and artifact field ----
fprintf('  [Separate] Stage 1: Marker detection & artifact field...\n');
[marker_mask, artifact_w] = marker_and_artifact_maps(vol, opts.MarkerRangeHU, ...
    opts.ArtifactSigmaMM, spacing);
metal = vol > opts.TagHUMin;

fprintf('    Marker mask: %d voxels (%.0f mm^3)\n', sum(marker_mask(:)), sum(marker_mask(:))*voxel_vol);
fprintf('    Metal (HU>%d): %d voxels (%.0f mm^3)\n', opts.TagHUMin, sum(metal(:)), sum(metal(:))*voxel_vol);

% Identify individual marker assemblies
CC_metal = bwconncomp(metal, 26);
min_tag_vox = max(5, round(2.0 / voxel_vol));
real_tags = {};
for i = 1:CC_metal.NumObjects
    if numel(CC_metal.PixelIdxList{i}) >= min_tag_vox
        [rr, cc, ss] = ind2sub(size(vol), CC_metal.PixelIdxList{i});
        tag = struct();
        tag.label = numel(real_tags) + 1;
        tag.centroid_mm = [mean(rr) mean(cc) mean(ss)] .* spacing;
        tag.volume_mm3 = numel(CC_metal.PixelIdxList{i}) * voxel_vol;
        real_tags{end+1} = tag; %#ok<AGROW>
    end
end
fprintf('    Marker assemblies: %d\n', numel(real_tags));
for t = 1:numel(real_tags)
    fprintf('      Marker %d: %.1f mm^3 at [%.1f %.1f %.1f] mm\n', ...
        t, real_tags{t}.volume_mm3, real_tags{t}.centroid_mm);
end

% ---- Stage 2: Find seed points (one per bone) ----
fprintf('  [Separate] Stage 2: Finding bone seed points...\n');
seeds = find_bone_seeds(vol, marker_mask, spacing, opts.MinBoneVolMM3);
fprintf('    Found %d seed points\n', numel(seeds));
for si = 1:numel(seeds)
    fprintf('      Seed %d: [%d %d %d] vox, component %.0f mm^3, score %.2f\n', ...
        si, seeds{si}.ijk, seeds{si}.comp_vol_mm3, seeds{si}.score);
end

if isempty(seeds)
    warning('No bone seeds found.');
    result = struct('bones', {{}}, 'specimen', false(size(vol)), ...
        'marker_mask', marker_mask, 'artifact_weight', artifact_w, 'n_tags', numel(real_tags));
    return;
end

% ---- Stage 3: Grow each bone from its seed using FMM ----
fprintf('  [Separate] Stage 3: Growing bones from seeds (FMM)...\n');

% Compute gradient once (shared across all seeds)
G = imgradient3(vol);

% Soft-tissue median and non-marker HU stats (same as scaphoid pipeline)
softMed = median(vol(vol > -300), 'omitnan');
if ~isfinite(softMed), softMed = -300; end
vals = vol(~marker_mask & vol > -300 & vol < 2000);
if isempty(vals), vals = vol(isfinite(vol)); end

% Core threshold (94th percentile of non-marker HU, clamped to [280, 700])
core_thr = max(280, min(700, prctile(vals, 94)));
core = vol > core_thr;

% Specimen mask (largest non-air component)
specimen = build_specimen_mask(vol, spacing);

bones = {};
all_bone_masks = false(size(vol));

for si = 1:numel(seeds)
    seed_ijk = seeds{si}.ijk;
    fprintf('    Bone %d: seed [%d %d %d]...\n', si, seed_ijk);

    seedMask = false(size(vol));
    seedMask(seed_ijk(1), seed_ijk(2), seed_ijk(3)) = true;

    % Build FMM weight map (scaphoid pipeline approach)
    W = build_fmm_weights(vol, seedMask, artifact_w, opts);

    % Apply gradient and HU-based allow region
    HU_ALLOW_MIN = max(70, min(220, softMed + 110));
    gThr = prctile(G(:), 85);
    maskR = (vol > HU_ALLOW_MIN) | (G > gThr);

    % Reconstruct from core through allowed region within specimen
    if any(core(:))
        allow = imreconstruct(core, maskR) & specimen;
    else
        allow = maskR & specimen;
    end

    % Zero weight outside allowed region
    W(~allow) = eps;

    % Robust normalize
    Lo = prctile(W(:), 1); Hi = prctile(W(:), 99);
    W = min(max((W - Lo) / max(eps, (Hi - Lo)), 0), 1);
    W = max(W, eps);

    % Run FMM
    th0 = min(max(0.01, mean(W(seedMask)) * 0.5), 0.99);
    [~, D] = imsegfmm(W, seedMask, th0);

    % Adaptive threshold sweep (9 thresholds from 0.14 to 0.42)
    mask_bone = adaptive_fmm_threshold(D, vol, G, softMed, specimen);

    % Don't overlap with already-assigned bone voxels
    mask_bone = mask_bone & ~all_bone_masks;

    % Post-processing (scaphoid approach)
    mask_bone = postprocess_bone(mask_bone, marker_mask, vol, G, softMed, core, spacing);

    if ~any(mask_bone(:))
        fprintf('      -> empty after post-processing, skipped\n');
        continue;
    end

    bone_vol = sum(mask_bone(:)) * voxel_vol;
    if bone_vol < opts.MinBoneVolMM3
        fprintf('      -> too small after post-processing (%.0f mm^3), skipped\n', bone_vol);
        continue;
    end

    all_bone_masks = all_bone_masks | mask_bone;

    % HU stats (tissue voxels only)
    tissue_vals = vol(mask_bone & vol > -200);
    if ~isempty(tissue_vals)
        bone_hu = mean(tissue_vals);
    else
        bone_hu = mean(vol(mask_bone));
    end

    % Centroid
    [rr, cc, ss] = ind2sub(size(mask_bone), find(mask_bone));
    centroid_mm = [mean(rr), mean(cc), mean(ss)] .* spacing;
    bbox = [min(rr) min(cc) min(ss) max(rr) max(cc) max(ss)];

    bone_info = struct();
    bone_info.mask = mask_bone;
    bone_info.label = si;
    bone_info.centroid_mm = centroid_mm;
    bone_info.volume_mm3 = bone_vol;
    bone_info.mean_hu = bone_hu;
    bone_info.dense_fraction = sum(vol(mask_bone) > 200) / max(1, sum(mask_bone(:)));
    bone_info.bbox = bbox;
    bone_info.tag_id = [];
    bone_info.tag_dist = Inf;

    bones{end+1} = bone_info; %#ok<AGROW>
    fprintf('      Final: %.0f mm^3, mean HU %.0f, dense %.0f%%\n', ...
        bone_vol, bone_hu, bone_info.dense_fraction*100);
end

% ---- Stage 4: Tag association ----
fprintf('  [Separate] Stage 4: Tag association...\n');
bones = associate_tags(bones, real_tags);

% Sort by volume (largest first)
vols = cellfun(@(b) b.volume_mm3, bones);
[~, order] = sort(vols, 'descend');
bones = bones(order);

fprintf('\n  Found %d bones and %d markers\n', numel(bones), numel(real_tags));
for i = 1:numel(bones)
    b = bones{i};
    if ~isempty(b.tag_id)
        tag_str = sprintf('marker %d (%.1f mm)', b.tag_id, b.tag_dist);
    else
        tag_str = 'no marker';
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
%  SEED FINDING (adapted from scaphoid proposeScaphoidSeed for multi-bone)
% =========================================================================
function seeds = find_bone_seeds(vol, marker_mask, spacing, min_vol_mm3)
    voxel_vol = prod(spacing);
    sz = size(vol);

    % Non-air, excluding markers + 2-voxel buffer + border voxels
    bw = vol > -300;
    bw = bw & ~imdilate(marker_mask, strel('sphere', 2));

    border = false(sz);
    border([1 end],:,:) = true;
    border(:,[1 end],:) = true;
    border(:,:,[1 end]) = true;
    bw = bw & ~imdilate(border, strel('sphere', 1));

    bw = bwareaopen(bw, max(200, round(min_vol_mm3 / voxel_vol)));

    CC = bwconncomp(bw, 26);
    if CC.NumObjects == 0
        seeds = {};
        return;
    end

    % Score each component by sphericity, elongation, and volume
    % (same scoring as scaphoid pipeline)
    scores = zeros(CC.NumObjects, 1);
    comp_vols = zeros(CC.NumObjects, 1);

    for n = 1:CC.NumObjects
        M = false(sz);
        M(CC.PixelIdxList{n}) = true;
        V_vox = numel(CC.PixelIdxList{n});
        comp_vols(n) = V_vox * voxel_vol;

        try
            stats = regionprops3(M, 'Volume', 'SurfaceArea', 'PrincipalAxisLength');
            V = double(stats.Volume(1));
            A = double(stats.SurfaceArea(1));
            pa = stats.PrincipalAxisLength(1, :);
            elong = max(pa) / max(1e-6, min(pa));
            sph = (pi^(1/3)) * ((6*max(V, eps))^(2/3)) / max(A, eps);
            sph = max(0, min(1, sph));
            scores(n) = (0.6*sph + 0.4*(1/elong)) * log1p(V);
        catch
            scores(n) = log1p(V_vox);
        end
    end

    % Keep components that could be bones (volume > threshold)
    % Sort by score descending
    [~, order] = sort(scores, 'descend');

    seeds = {};
    for k = 1:CC.NumObjects
        idx = order(k);
        if comp_vols(idx) < min_vol_mm3
            continue;
        end

        % Seed = deepest interior point of this component
        comp_mask = false(sz);
        comp_mask(CC.PixelIdxList{idx}) = true;
        Dm = bwdist(~comp_mask);
        [~, max_idx] = max(Dm(:));
        [si, sj, sk] = ind2sub(sz, max_idx);

        % Clamp away from borders
        si = max(4, min(sz(1)-3, si));
        sj = max(4, min(sz(2)-3, sj));
        sk = max(4, min(sz(3)-3, sk));

        seed = struct();
        seed.ijk = [si, sj, sk];
        seed.score = scores(idx);
        seed.comp_vol_mm3 = comp_vols(idx);
        seeds{end+1} = seed; %#ok<AGROW>
    end
end


% =========================================================================
%  FMM WEIGHT MAP (same as scaphoid buildWeights)
% =========================================================================
function W = build_fmm_weights(vol, seedMask, artifact_w, opts)
    fgVals = vol(seedMask);
    mu1 = median(fgVals);
    s1 = mad(fgVals, 1) + 50;

    bg = imdilate(seedMask, strel('sphere', 6)) & ~imdilate(seedMask, strel('sphere', 2));
    bgVals = vol(bg);
    mu0 = median(bgVals);
    s0 = mad(bgVals, 1) + 50;

    pBone = 1 ./ (1 + exp(-(vol - mu1) ./ s1));
    pTiss = 1 ./ (1 + exp(-(mu0 - vol) ./ s0));
    edgeW = 1 - mat2gray(gradientweight(vol));
    dataW = mat2gray(pBone ./ (pTiss + eps));

    alpha = 1.0; beta = 0.5; gamma = 1.0;
    base = alpha * dataW + beta * edgeW;
    W = base ./ (1 + gamma * artifact_w);
    W = max(W, eps);
end


% =========================================================================
%  ADAPTIVE FMM THRESHOLD (same as scaphoid segmentScaphoidFMM scoring)
% =========================================================================
function mask = adaptive_fmm_threshold(D, vol, G, softMed, specimen)
    Gsrc = mat2gray(imgradient3(vol));
    ths = linspace(0.14, 0.42, 9);

    HU_MIN = max(180, min(400, softMed + 220));

    best_score = -Inf;
    best_mask = false(size(vol));

    for t = ths
        t = min(max(t, eps), 0.999);
        B = D <= t;
        B = B & specimen;
        B = keep_largest_3d(B);
        if ~any(B(:)), continue; end

        P = bwperim(B, 26);
        if ~any(P(:)), continue; end
        s_edge = mean(Gsrc(P), 'omitnan');

        Rin = imerode(B, strel('sphere', 1));
        if ~any(Rin(:)), continue; end
        medHU = median(double(vol(Rin)), 'omitnan');
        if ~isfinite(medHU), medHU = -Inf; end

        penHU = max(0, HU_MIN - medHU) / HU_MIN;
        penVol = 1e-7 * double(nnz(B));

        s = s_edge - 0.5 * penHU - penVol;
        if s > best_score
            best_score = s;
            best_mask = B;
        end
    end

    mask = best_mask;
end


% =========================================================================
%  POST-PROCESSING (scaphoid approach: marker carve + boundary refinement)
% =========================================================================
function mask = postprocess_bone(mask, marker_mask, vol, G, softMed, core, spacing)
    if ~any(mask(:)), return; end

    % Fill holes, remove tiny fragments
    mask = imfill(mask, 'holes');
    mask = bwareaopen(mask, 500);

    % Scaphoid-style marker carve: remove markers at boundary, protect interior
    interior = imerode(mask, strel('sphere', 1));
    mask = (mask & ~imdilate(marker_mask, strel('sphere', 1))) | interior;

    mask = keep_largest_3d(mask);
    if ~any(mask(:)), return; end

    % Boundary-band cling (scaphoid lines 206-223)
    voxmm = mean(spacing);
    D_in = bwdist(~mask) * voxmm;
    deep_interior = D_in >= 1.0;
    band = mask & ~deep_interior;

    vals = vol(~marker_mask & vol > -300 & vol < 2000);
    if isempty(vals), vals = vol(isfinite(vol)); end
    band_core_thr = max(240, min(650, prctile(vals, 92)));
    core_seed = band & (vol > band_core_thr);
    HU_SUPPORT_MIN = max(60, min(180, softMed + 90));
    gThr = prctile(G(:), 80);
    support = band & ((vol > HU_SUPPORT_MIN) | (G > gThr));
    if any(core_seed(:))
        cling_band = imreconstruct(core_seed, support);
        mask = deep_interior | cling_band;
    end

    mask = keep_largest_3d(mask);
    if ~any(mask(:)), return; end
    mask = imclose(mask, strel('sphere', 1));
    mask = imfill(mask, 'holes');

    % Edge-backed perimeter prune (scaphoid lines 225-236)
    perim = bwperim(mask, 26);
    band1 = imdilate(perim, strel('sphere', 1));
    T_hu = max(160, min(340, softMed + 190));
    T_g = prctile(G(:), 70);
    kill = band1 & (vol < T_hu) & (G < T_g);
    if any(kill(:))
        mask(kill) = false;
        mask = keep_largest_3d(mask);
        mask = imclose(mask, strel('sphere', 1));
        mask = imfill(mask, 'holes');
    end

    % Final boundary carve (scaphoid lines 258-264)
    band = imdilate(bwperim(mask, 26), strel('sphere', 1));
    HU_CARVE_FLOOR = max(180, min(400, softMed + 220));
    kill = band & (vol < HU_CARVE_FLOOR);
    if any(kill(:))
        mask(kill) = false;
    end

    % Final cleanup
    mask = keep_largest_3d(mask);
    mask = imopen(mask, strel('sphere', 1));
    mask = keep_largest_3d(mask);
    mask = imclose(mask, strel('sphere', 1));
    mask = imfill(mask, 'holes');
end


% =========================================================================
%  SPECIMEN MASK (largest non-air component)
% =========================================================================
function specimen = build_specimen_mask(vol, spacing)
    non_air = vol > -500;
    rClose = max(1, round(0.6 / max(mean(spacing), eps)));
    non_air = imclose(non_air, strel('sphere', rClose));

    CC = bwconncomp(non_air, 26);
    if CC.NumObjects == 0
        specimen = non_air;
        return;
    end

    [~, iMax] = max(cellfun(@numel, CC.PixelIdxList));
    specimen = false(size(vol));
    specimen(CC.PixelIdxList{iMax}) = true;
end


% =========================================================================
%  MARKER AND ARTIFACT MAPS (same as scaphoid markerAndArtifactMaps)
% =========================================================================
function [marker_mask, artifact_w] = marker_and_artifact_maps(HU, marker_range, sigma_mm, spacing)
    lead = HU > 1200;
    flags = (HU >= marker_range(1) & HU <= marker_range(2)) & ...
            imdilate(lead, strel('sphere', 2));
    marker_mask = lead | flags;

    d_vox = bwdist(marker_mask);
    d_mm = d_vox * mean(spacing);
    artifact_w = exp(-(d_mm / sigma_mm).^2);
end


% =========================================================================
%  TAG ASSOCIATION
% =========================================================================
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
        fprintf('      Marker %d -> Bone %d: %.1f mm\n', t, idx, d);
        if isempty(bones{idx}.tag_id) || d < bones{idx}.tag_dist
            bones{idx}.tag_id = tags{t}.label;
            bones{idx}.tag_dist = d;
        end
    end
end


% =========================================================================
%  UTILITY
% =========================================================================
function mask = keep_largest_3d(mask)
    CC = bwconncomp(mask, 26);
    if CC.NumObjects <= 1, return; end
    [~, iMax] = max(cellfun(@numel, CC.PixelIdxList));
    mask = false(size(mask));
    mask(CC.PixelIdxList{iMax}) = true;
end
