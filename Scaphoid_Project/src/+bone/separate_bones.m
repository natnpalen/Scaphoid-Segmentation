function result = separate_bones(ds, opts)
% SEPARATE_BONES  Isolate individual bones from a multi-bone excised-in-air CT scan.
%
%   result = bone.separate_bones(ds, opts)
%
% Uses the scaphoid pipeline's seed-and-grow approach adapted for multiple
% bones.  Each bone is GROWN from a seed point using FMM with artifact-
% weighted speeds, constrained to a local region around its source component.
%
% Algorithm:
%   1. Detect markers (lead + dense flags + light flags) and build artifact field
%   2. Find seed points: one per bone (deepest interior of compact components)
%   3. Per-seed LOCAL FMM growth with adaptive threshold scoring
%   4. Post-processing: marker carve, shell sealing, boundary refine
%   5. Reject non-bone objects (negative mean HU)

vol = double(ds.HU);
spacing = ds.spacing;
voxel_vol = prod(spacing);
sz = size(vol);

% Hard metal mask: voxels that must NEVER be included in any bone
lead_metal = vol > 4000;

% ---- Stage 1: Markers and artifact field ----
fprintf('  [Separate] Stage 1: Marker detection & artifact field...\n');
[marker_mask, artifact_w] = marker_and_artifact_maps(vol, opts.ArtifactSigmaMM, spacing);
metal = vol > opts.TagHUMin;

fprintf('    Marker mask: %d voxels (%.0f mm^3)\n', sum(marker_mask(:)), sum(marker_mask(:))*voxel_vol);
fprintf('    Metal (HU>%d): %d voxels (%.0f mm^3)\n', opts.TagHUMin, sum(metal(:)), sum(metal(:))*voxel_vol);
fprintf('    Lead metal (HU>4000): %d voxels\n', nnz(lead_metal));
fprintf('    Artifact sigma: %.1f mm\n', opts.ArtifactSigmaMM);

% Identify individual marker assemblies
CC_metal = bwconncomp(metal, 26);
min_tag_vox = max(5, round(2.0 / voxel_vol));
real_tags = {};
for i = 1:CC_metal.NumObjects
    if numel(CC_metal.PixelIdxList{i}) >= min_tag_vox
        [rr, cc, ss] = ind2sub(sz, CC_metal.PixelIdxList{i});
        tag = struct();
        tag.label = numel(real_tags) + 1;
        tag.centroid_mm = [mean(rr) mean(cc) mean(ss)] .* spacing;
        tag.volume_mm3 = numel(CC_metal.PixelIdxList{i}) * voxel_vol;
        tag.voxel_idx = CC_metal.PixelIdxList{i};
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
    seed_hu = vol(seeds{si}.ijk(1), seeds{si}.ijk(2), seeds{si}.ijk(3));
    fprintf('      Seed %d: [%d %d %d] vox, HU=%.0f, component %.0f mm^3 (mean HU %.0f), score %.2f\n', ...
        si, seeds{si}.ijk, seed_hu, seeds{si}.comp_vol_mm3, seeds{si}.comp_mean_hu, seeds{si}.score);
end

if isempty(seeds)
    warning('No bone seeds found.');
    result = struct('bones', {{}}, 'specimen', false(sz), ...
        'marker_mask', marker_mask, 'artifact_weight', artifact_w, 'n_tags', numel(real_tags));
    return;
end

% ---- Stage 3: Grow each bone from its seed using FMM ----
fprintf('  [Separate] Stage 3: Growing bones from seeds (FMM)...\n');

% Global stats (used for thresholds)
softMed = median(vol(vol > -300), 'omitnan');
if ~isfinite(softMed), softMed = -300; end
fprintf('    Global softMed: %.0f HU\n', softMed);

bones = {};
all_bone_masks = false(sz);

for si = 1:numel(seeds)
    seed_ijk = seeds{si}.ijk;
    comp_mask = seeds{si}.comp_mask;
    fprintf('    Bone %d: seed [%d %d %d]...\n', si, seed_ijk);

    % === LOCAL CROP around source component + margin ===
    margin_mm = 10.0;
    margin_vox = ceil(margin_mm ./ spacing);
    [ri, ci, si_idx] = ind2sub(sz, find(comp_mask));
    r1 = max(1, min(ri) - margin_vox(1));  r2 = min(sz(1), max(ri) + margin_vox(1));
    c1 = max(1, min(ci) - margin_vox(2));  c2 = min(sz(2), max(ci) + margin_vox(2));
    s1 = max(1, min(si_idx) - margin_vox(3)); s2 = min(sz(3), max(si_idx) + margin_vox(3));
    fprintf('      Local ROI: [%d:%d, %d:%d, %d:%d] = %dx%dx%d\n', ...
        r1, r2, c1, c2, s1, s2, r2-r1+1, c2-c1+1, s2-s1+1);

    % Extract local volumes
    vol_L = vol(r1:r2, c1:c2, s1:s2);
    art_L = artifact_w(r1:r2, c1:c2, s1:s2);
    mk_L  = marker_mask(r1:r2, c1:c2, s1:s2);
    all_L = all_bone_masks(r1:r2, c1:c2, s1:s2);
    lead_L = lead_metal(r1:r2, c1:c2, s1:s2);

    % Local seed mask
    seed_local = [seed_ijk(1)-r1+1, seed_ijk(2)-c1+1, seed_ijk(3)-s1+1];
    seedMask_L = false(size(vol_L));
    seedMask_L(seed_local(1), seed_local(2), seed_local(3)) = true;

    % Learn seed stats from the SOURCE COMPONENT (not just one voxel)
    comp_L = comp_mask(r1:r2, c1:c2, s1:s2);
    comp_vals = vol_L(comp_L & vol_L > -200);
    if numel(comp_vals) >= 10
        mu1 = median(comp_vals);
        s1_stat = mad(comp_vals, 1) + 50;
    else
        mu1 = vol_L(seed_local(1), seed_local(2), seed_local(3));
        s1_stat = 50;
    end
    fprintf('      Seed stats: mu1=%.0f, s1=%.0f (from %d component voxels)\n', ...
        mu1, s1_stat, numel(comp_vals));

    % Build FMM weight map (scaphoid buildWeights approach — NO normalization)
    W_L = build_fmm_weights_local(vol_L, seedMask_L, art_L, mu1, s1_stat);

    % Hard-block lead metal and already-assigned bones in the weight map
    W_L(lead_L) = eps;
    W_L(all_L) = eps;

    fprintf('      W range: [%.4f, %.4f]\n', min(W_L(:)), max(W_L(:)));

    % Run FMM (pass weights directly, as scaphoid does — no normalization)
    th0 = min(max(0.01, mean(W_L(seedMask_L)) * 0.5), 0.99);
    fprintf('      FMM th0: %.4f\n', th0);

    try
        [~, D_L] = imsegfmm(W_L, seedMask_L, th0);
    catch ME
        fprintf('      FMM failed: %s — skipping\n', ME.message);
        continue;
    end

    fprintf('      FMM D range: [%.4f, %.4f]\n', min(D_L(:)), max(D_L(:)));

    % Non-air local specimen
    specimen_L = vol_L > -500;
    specimen_L = imclose(specimen_L, strel('sphere', 1));

    % Adaptive threshold sweep (scaphoid approach)
    G_L = imgradient3(vol_L);
    mask_bone_L = adaptive_fmm_threshold(D_L, vol_L, G_L, softMed, specimen_L);

    if ~any(mask_bone_L(:))
        fprintf('      -> empty after threshold sweep, skipped\n');
        continue;
    end
    fprintf('      After threshold: %d voxels (%.0f mm^3)\n', ...
        nnz(mask_bone_L), nnz(mask_bone_L)*voxel_vol);

    % Don't overlap with already-assigned bone voxels
    mask_bone_L = mask_bone_L & ~all_L;

    % Hard-remove lead metal from bone mask
    mask_bone_L = mask_bone_L & ~lead_L;

    % Post-processing (marker carve + boundary refine)
    vals_L = vol_L(~mk_L & vol_L > -300 & vol_L < 2000);
    if isempty(vals_L), vals_L = vol_L(isfinite(vol_L)); end
    core_thr_L = max(280, min(700, prctile(vals_L, 94)));
    core_L = vol_L > core_thr_L;
    mask_bone_L = postprocess_bone(mask_bone_L, mk_L, lead_L, vol_L, G_L, softMed, core_L, spacing);

    if ~any(mask_bone_L(:))
        fprintf('      -> empty after post-processing, skipped\n');
        continue;
    end

    % Shell sealing (ensures closed surface — ported from scaphoid sealOuterShell)
    mask_bone_L = seal_outer_shell(mask_bone_L, spacing);

    % Paste back to full volume
    mask_bone = false(sz);
    mask_bone(r1:r2, c1:c2, s1:s2) = mask_bone_L;

    % Final hard-remove of all marker material from bone mask
    mask_bone = mask_bone & ~marker_mask;

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
    [rr, cc, ss] = ind2sub(sz, find(mask_bone));
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

% ---- Stage 4: Reject non-bone objects (negative mean HU) ----
MIN_BONE_HU = 50;
n_before = numel(bones);
keep = true(1, numel(bones));
for bi = 1:numel(bones)
    if bones{bi}.mean_hu < MIN_BONE_HU
        fprintf('    Rejecting bone %d: mean HU %.0f < %d (not bone tissue)\n', ...
            bi, bones{bi}.mean_hu, MIN_BONE_HU);
        keep(bi) = false;
    end
end
bones = bones(keep);
if numel(bones) < n_before
    fprintf('    Rejected %d non-bone objects\n', n_before - numel(bones));
end

% ---- Stage 5: Tag association ----
fprintf('  [Separate] Stage 5: Tag association...\n');
bones = associate_tags(bones, real_tags, spacing);

% Sort by volume (largest first)
if ~isempty(bones)
    vols = cellfun(@(b) b.volume_mm3, bones);
    [~, order] = sort(vols, 'descend');
    bones = bones(order);
end

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

% Specimen mask for output
specimen = build_specimen_mask(vol, spacing);

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
    fprintf('    Initial components (after marker exclusion): %d\n', CC.NumObjects);
    if CC.NumObjects == 0
        seeds = {};
        return;
    end

    % Merge nearby fragments that were split by marker exclusion.
    % Marker exclusion creates ~2-voxel gaps in bones that touch markers,
    % splitting one bone into multiple fragments. Bridge these gaps by
    % checking pairwise distances and merging components within 5mm.
    MERGE_DIST_MM = 5.0;
    merge_dist_vox = MERGE_DIST_MM / mean(spacing);
    merged = merge_nearby_components(CC, sz, merge_dist_vox);
    fprintf('    After merging nearby fragments: %d components\n', merged.NumObjects);

    CC = merged;

    % Score each component
    scores = zeros(CC.NumObjects, 1);
    comp_vols = zeros(CC.NumObjects, 1);
    comp_mean_hus = zeros(CC.NumObjects, 1);

    for n = 1:CC.NumObjects
        M = false(sz);
        M(CC.PixelIdxList{n}) = true;
        V_vox = numel(CC.PixelIdxList{n});
        comp_vols(n) = V_vox * voxel_vol;

        hvals = vol(CC.PixelIdxList{n});
        hvals_tissue = hvals(hvals > -200);
        if ~isempty(hvals_tissue)
            comp_mean_hus(n) = mean(hvals_tissue);
        else
            comp_mean_hus(n) = mean(hvals);
        end

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

        fprintf('      Component %d: %.0f mm^3, mean HU %.0f, score %.2f\n', ...
            n, comp_vols(n), comp_mean_hus(n), scores(n));
    end

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
        [max_depth, max_idx] = max(Dm(:));
        [si, sj, sk] = ind2sub(sz, max_idx);

        % Clamp away from borders
        si = max(4, min(sz(1)-3, si));
        sj = max(4, min(sz(2)-3, sj));
        sk = max(4, min(sz(3)-3, sk));

        seed = struct();
        seed.ijk = [si, sj, sk];
        seed.score = scores(idx);
        seed.comp_vol_mm3 = comp_vols(idx);
        seed.comp_mean_hu = comp_mean_hus(idx);
        seed.comp_mask = comp_mask;
        seed.max_depth_vox = max_depth;
        seeds{end+1} = seed; %#ok<AGROW>
    end
end


% =========================================================================
%  FMM WEIGHT MAP (scaphoid buildWeights, using component-learned stats)
% =========================================================================
function W = build_fmm_weights_local(vol, seedMask, artifact_w, mu1, s1)
    % Background stats from shell around seed
    bg = imdilate(seedMask, strel('sphere', 6)) & ~imdilate(seedMask, strel('sphere', 2));
    bgVals = vol(bg);
    if isempty(bgVals)
        mu0 = -300;
        s0 = 100;
    else
        mu0 = median(bgVals);
        s0 = mad(bgVals, 1) + 50;
    end

    pBone = 1 ./ (1 + exp(-(vol - mu1) ./ s1));
    pTiss = 1 ./ (1 + exp(-(mu0 - vol) ./ s0));

    % Use gradientweight (same as scaphoid buildWeights)
    edgeW = 1 - mat2gray(gradientweight(vol));
    dataW = mat2gray(pBone ./ (pTiss + eps));

    alpha = 1.0; beta = 0.5; gamma = 1.0;
    base = alpha * dataW + beta * edgeW;
    W = base ./ (1 + gamma * artifact_w);
    W = max(W, eps);
end


% =========================================================================
%  ADAPTIVE FMM THRESHOLD (scaphoid segmentScaphoidFMM scoring)
% =========================================================================
function mask = adaptive_fmm_threshold(D, vol, G, softMed, specimen)
    % Use same boundary score image as scaphoid: 1 - gradientweight
    Gsrc = 1 - mat2gray(gradientweight(vol));

    ths = linspace(0.14, 0.42, 9);
    HU_MIN = max(180, min(400, softMed + 220));
    lambda = 0.5;

    best_score = -Inf;
    best_mask = false(size(vol));

    for ti = 1:numel(ths)
        t = min(max(ths(ti), eps), 0.999);
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

        s = s_edge - lambda * penHU - penVol;

        fprintf('        th=%.3f: %d vox, edge=%.3f, medHU=%.0f, penHU=%.3f, score=%.4f%s\n', ...
            t, nnz(B), s_edge, medHU, penHU, s, ternary(s > best_score, ' *', ''));

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
function mask = postprocess_bone(mask, marker_mask, lead_metal, vol, G, softMed, core, spacing)
    if ~any(mask(:)), return; end

    % Hard-remove lead metal (HU>4000) — never allowed in bone mask
    mask = mask & ~lead_metal;

    % Fill holes, remove tiny fragments
    mask = imfill(mask, 'holes');
    mask = bwareaopen(mask, 500);

    % Scaphoid-style marker carve: remove markers at boundary, protect interior
    % but never protect voxels overlapping with marker material
    interior = imerode(mask, strel('sphere', 1));
    interior = interior & ~marker_mask;
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
%  MARKER AND ARTIFACT MAPS
%  Detects the full marker assembly: lead letter (HU>1200), dense flags
%  (~700-1200 HU near lead), and light flags/housing (200-700 HU near lead).
% =========================================================================
function [marker_mask, artifact_w] = marker_and_artifact_maps(HU, sigma_mm, spacing)
    % Lead letter cores: HU > 1200 (typically 4000-7000 HU)
    lead = HU > 1200;

    % Dense flags/tabs: 700-1200 HU within 5 voxels of lead
    % Flags can be ~1mm from the letter; at 0.25mm spacing that's 4 voxels
    dense_flags = (HU >= 700 & HU <= 1200) & imdilate(lead, strel('sphere', 5));

    % Light flags/housing: 200-700 HU within 4 voxels of lead or dense flags
    lead_plus_dense = lead | dense_flags;
    light_flags = (HU >= 200 & HU <= 700) & imdilate(lead_plus_dense, strel('sphere', 4));

    marker_mask = lead | dense_flags | light_flags;

    fprintf('    Marker layers: lead=%d, dense_flags=%d, light_flags=%d, total=%d voxels\n', ...
        nnz(lead), nnz(dense_flags), nnz(light_flags), nnz(marker_mask));

    % Artifact weight field: Gaussian decay from marker boundary
    d_vox = bwdist(marker_mask);
    d_mm = d_vox * mean(spacing);
    artifact_w = exp(-(d_mm / sigma_mm).^2);
end


% =========================================================================
%  TAG ASSOCIATION (mask overlap, not centroid distance)
% =========================================================================
function bones = associate_tags(bones, tags, spacing)
    if isempty(tags) || isempty(bones), return; end

    fprintf('    Tag-bone associations:\n');
    vox_mm = mean(spacing);

    for t = 1:numel(tags)
        best_dist = Inf;
        best_bone = 0;

        for bi = 1:numel(bones)
            % Surface-to-surface distance (more robust than centroid for irregular shapes)
            D_from_bone = bwdist(bones{bi}.mask);
            tag_dists = D_from_bone(tags{t}.voxel_idx);
            min_dist_mm = min(tag_dists) * vox_mm;

            if min_dist_mm < best_dist
                best_dist = min_dist_mm;
                best_bone = bi;
            end
        end

        if best_bone > 0
            fprintf('      Marker %d -> Bone %d: %.1f mm (surface-to-surface)\n', ...
                t, best_bone, best_dist);
            if isempty(bones{best_bone}.tag_id) || best_dist < bones{best_bone}.tag_dist
                bones{best_bone}.tag_id = tags{t}.label;
                bones{best_bone}.tag_dist = best_dist;
            end
        end
    end
end


% =========================================================================
%  MERGE NEARBY COMPONENTS (bridges marker-exclusion gaps)
% =========================================================================
function CC_out = merge_nearby_components(CC, sz, merge_dist_vox)
    if CC.NumObjects <= 1
        CC_out = CC;
        return;
    end

    n = CC.NumObjects;
    parent = 1:n;

    % Check pairwise distances between components
    dists = cell(n, 1);
    for i = 1:n
        m = false(sz);
        m(CC.PixelIdxList{i}) = true;
        dists{i} = bwdist(m);
    end

    for i = 1:n
        for j = (i+1):n
            min_d = min(dists{i}(CC.PixelIdxList{j}));
            if min_d <= merge_dist_vox
                ri = i; while parent(ri) ~= ri, ri = parent(ri); end
                rj = j; while parent(rj) ~= rj, rj = parent(rj); end
                if ri ~= rj
                    parent(rj) = ri;
                end
            end
        end
    end

    % Resolve all roots
    for i = 1:n
        r = i;
        while parent(r) ~= r, r = parent(r); end
        parent(i) = r;
    end

    % Collect merged groups
    roots = unique(parent);
    CC_out = struct();
    CC_out.Connectivity = CC.Connectivity;
    CC_out.ImageSize = CC.ImageSize;
    CC_out.NumObjects = numel(roots);
    CC_out.PixelIdxList = cell(1, numel(roots));
    for k = 1:numel(roots)
        members = find(parent == roots(k));
        combined = [];
        for m = 1:numel(members)
            combined = [combined; CC.PixelIdxList{members(m)}]; %#ok<AGROW>
        end
        CC_out.PixelIdxList{k} = combined;
    end
end


% =========================================================================
%  SHELL SEALING (ported from scaphoid sealOuterShell)
% =========================================================================
function BW = seal_outer_shell(BW, spacing)
    if ~any(BW(:)), return; end

    rClose = max(1, round(1.5 / max(mean(spacing), eps)));
    SE = strel('sphere', rClose);
    BW = imclose(BW, SE);
    BW = imfill(BW, 'holes');

    BW = keep_largest_3d(BW);
    BW = imclearborder(BW, 26);
end


% =========================================================================
%  UTILITIES
% =========================================================================
function mask = keep_largest_3d(mask)
    CC = bwconncomp(mask, 26);
    if CC.NumObjects <= 1, return; end
    [~, iMax] = max(cellfun(@numel, CC.PixelIdxList));
    mask = false(size(mask));
    mask(CC.PixelIdxList{iMax}) = true;
end

function s = ternary(cond, a, b)
    if cond, s = a; else, s = b; end
end
