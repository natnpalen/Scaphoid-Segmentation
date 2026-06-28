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
[marker_mask, artifact_w, marker_core] = marker_and_artifact_maps(vol, opts.ArtifactSigmaMM, spacing);
metal = vol > opts.TagHUMin;

fprintf('    Marker mask: %d voxels (%.0f mm^3)\n', sum(marker_mask(:)), sum(marker_mask(:))*voxel_vol);
fprintf('    Metal (HU>%d): %d voxels (%.0f mm^3)\n', opts.TagHUMin, sum(metal(:)), sum(metal(:))*voxel_vol);
fprintf('    Lead metal (HU>4000): %d voxels\n', nnz(lead_metal));
fprintf('    Artifact sigma: %.1f mm\n', opts.ArtifactSigmaMM);

% Identify individual marker assemblies from actual lead (HU>3000),
% not the opts.TagHUMin threshold which is too low for assembly detection
CC_metal = bwconncomp(vol > 3000, 26);
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

    % === Scaphoid-style allow region: core -> allow -> imreconstruct ===
    G_L = imgradient3(vol_L);

    % Distance from lead in local ROI (mm) — used to suppress flag tabs
    d_lead_L = bwdist(lead_L) * mean(spacing);

    vals_L = vol_L(~mk_L & vol_L > -300 & vol_L < 2000);
    if isempty(vals_L), vals_L = vol_L(isfinite(vol_L)); end
    core_thr_L = max(280, min(700, prctile(vals_L, 94)));
    core_L = vol_L > core_thr_L;

    % Exclude near-lead voxels from core: flag tabs (700-1200 HU) sit
    % within ~3-5mm of lead and would otherwise seed the allow region,
    % connecting flags to bone through imreconstruct.
    LEAD_BUFFER_MM = 5.0;
    near_lead_L = d_lead_L < LEAD_BUFFER_MM;
    core_L = core_L & ~near_lead_L;

    % Allow = moderate HU OR high gradient (catches cancellous bone)
    HU_ALLOW_MIN = max(70, min(220, softMed + 110));
    gThr_L = prctile(G_L(:), 85);
    maskR_L = (vol_L > HU_ALLOW_MIN) | (G_L > gThr_L);

    % Also exclude near-lead from the reachable mask
    maskR_L = maskR_L & ~near_lead_L;

    % Flood-fill from core through allow — gives connected bone region
    if any(core_L(:))
        allow_L = imreconstruct(core_L, maskR_L);
    else
        allow_L = maskR_L;
    end
    % Exclude markers and already-assigned bones from allow
    allow_L = allow_L & ~mk_L & ~lead_L & ~all_L;
    fprintf('      Allow region: core=%d, allow=%d, lead_buffer=%d voxels\n', ...
        nnz(core_L), nnz(allow_L), nnz(near_lead_L));

    % Build FMM weight map
    W_L = build_fmm_weights_local(vol_L, seedMask_L, art_L, mu1, s1_stat);

    % Block everything outside the allow region (scaphoid line 94)
    W_L(~allow_L) = eps;

    % Robust percentile normalization (scaphoid lines 97-99)
    Lo = prctile(W_L(:), 1);
    Hi = prctile(W_L(:), 99);
    W_L = min(max((W_L - Lo) / max(eps, Hi - Lo), 0), 1);
    W_L = max(W_L, eps);

    fprintf('      W range: [%.4f, %.4f]\n', min(W_L(:)), max(W_L(:)));

    % Run FMM
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
    mask_bone_L = adaptive_fmm_threshold(D_L, vol_L, G_L, softMed, specimen_L);

    if ~any(mask_bone_L(:))
        fprintf('      -> empty after threshold sweep, skipped\n');
        continue;
    end
    fprintf('      After threshold: %d voxels (%.0f mm^3)\n', ...
        nnz(mask_bone_L), nnz(mask_bone_L)*voxel_vol);

    % Constrain to allow region (no marker/metal/assigned leakage)
    mask_bone_L = mask_bone_L & allow_L;

    % Shell sealing: close small gaps, fill interior
    mask_bone_L = seal_outer_shell(mask_bone_L, spacing);

    if ~any(mask_bone_L(:))
        fprintf('      -> empty after shell sealing, skipped\n');
        continue;
    end

    % Remove marker material AFTER sealing (imclose can re-bridge).
    flag_hu_thr_L = 300 + 150 * d_lead_L;
    flag_carve_L = (d_lead_L < 4.0) & (vol_L > flag_hu_thr_L);
    mask_bone_L = mask_bone_L & ~mk_L & ~lead_L & ~flag_carve_L;
    mask_bone_L = imfill(mask_bone_L, 'holes');
    mask_bone_L = keep_largest_3d(mask_bone_L);

    % === Boundary refinement (ported from scaphoid pipeline) ===
    mask_bone_L = refine_bone_boundary(mask_bone_L, vol_L, G_L, mk_L, spacing, softMed);

    if ~any(mask_bone_L(:))
        fprintf('      -> empty after marker carve, skipped\n');
        continue;
    end

    fprintf('      After seal+carve: %d voxels (%.0f mm^3)\n', ...
        nnz(mask_bone_L), nnz(mask_bone_L)*voxel_vol);

    % Paste back to full volume
    mask_bone = false(sz);
    mask_bone(r1:r2, c1:c2, s1:s2) = mask_bone_L;

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

        % Reject air pockets (mean HU < 50) before wasting FMM slots
        if comp_mean_hus(idx) < 50
            fprintf('      Skipping component %d: mean HU %.0f (air pocket)\n', idx, comp_mean_hus(idx));
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
    % Cap volume to exclude metal extremes from gradient/normalization.
    % Metal voxels (4000-7000 HU) dominate mat2gray, squashing bone weights.
    vol_cap = max(-500, min(vol, 2000));

    % Background stats from shell around seed
    bg = imdilate(seedMask, strel('sphere', 6)) & ~imdilate(seedMask, strel('sphere', 2));
    bgVals = vol_cap(bg);
    if isempty(bgVals)
        mu0 = -300;
        s0 = 100;
    else
        mu0 = median(bgVals);
        s0 = mad(bgVals, 1) + 50;
    end

    pBone = 1 ./ (1 + exp(-(vol_cap - mu1) ./ s1));
    pTiss = 1 ./ (1 + exp(-(mu0 - vol_cap) ./ s0));

    edgeW = 1 - mat2gray(gradientweight(vol_cap));
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
    % For excised-in-air: softMed ~280-370 reflects bone tissue, not soft
    % tissue. Using softMed+220 gives HU_MIN=400-500 which penalizes
    % cancellous bone (100-300 HU), creating hollow interiors. Use a low
    % threshold that only rejects air, not real bone tissue.
    HU_MIN = max(50, min(200, softMed * 0.4));
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
%  Controlled limited growth from lead cores to capture marker assemblies
%  (lead letters + flag tabs) WITHOUT leaking into bone tissue.
% =========================================================================
function [marker_mask, artifact_w, marker_core] = marker_and_artifact_maps(HU, sigma_mm, spacing)
    % Lead letter cores: HU > 3000 (actual lead is 4000-7000 HU)
    lead = HU > 3000;

    if ~any(lead(:))
        marker_mask = false(size(HU));
        marker_core = marker_mask;
        d_mm = bwdist(marker_mask) * mean(spacing);
        artifact_w = exp(-(d_mm / sigma_mm).^2);
        fprintf('    Marker detection: no lead found\n');
        return;
    end

    % Controlled iterative dilation from lead cores.
    % Each step only grows into voxels above FLAG_HU_MIN, capturing the
    % flag tabs (~700-1200 HU) attached to each lead letter. Growth is
    % limited to MAX_STEPS iterations so it cannot flood into bone even
    % when the marker physically contacts bone tissue.
    FLAG_HU_MIN = 400;
    MAX_STEPS = 8;
    mean_sp = mean(spacing);
    max_extent_mm = MAX_STEPS * mean_sp;

    SE = strel('sphere', 1);
    grown = lead;
    growable = HU > FLAG_HU_MIN;

    for step = 1:MAX_STEPS
        expanded = imdilate(grown, SE);
        new_voxels = expanded & growable & ~grown;
        if ~any(new_voxels(:)), break; end
        grown = grown | new_voxels;
    end

    % Per-assembly isolation: only keep grown regions connected to lead
    CC_grown = bwconncomp(grown, 26);
    marker_mask = false(size(HU));
    for i = 1:CC_grown.NumObjects
        if any(lead(CC_grown.PixelIdxList{i}))
            marker_mask(CC_grown.PixelIdxList{i}) = true;
        end
    end

    marker_core = marker_mask & (HU > 1000);

    fprintf('    Marker detection: lead=%d, grown=%d (max %.1fmm, %d steps, HU>%d), core=%d voxels\n', ...
        nnz(lead), nnz(marker_mask), max_extent_mm, MAX_STEPS, FLAG_HU_MIN, nnz(marker_core));

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

    % Compute distance transforms for each component
    dists = cell(n, 1);
    comp_vols = zeros(n, 1);
    for i = 1:n
        m = false(sz);
        m(CC.PixelIdxList{i}) = true;
        dists{i} = bwdist(m);
        comp_vols(i) = numel(CC.PixelIdxList{i});
    end

    % Print pairwise distances so we can debug merge decisions
    fprintf('    Pairwise component distances (merge threshold: %.1f vox):\n', merge_dist_vox);
    for i = 1:n
        for j = (i+1):n
            min_d = min(dists{i}(CC.PixelIdxList{j}));
            merged_str = '';
            if min_d <= merge_dist_vox
                merged_str = ' -> MERGE';
                ri = i; while parent(ri) ~= ri, ri = parent(ri); end
                rj = j; while parent(rj) ~= rj, rj = parent(rj); end
                if ri ~= rj
                    parent(rj) = ri;
                end
            end
            fprintf('      C%d (%d vox) <-> C%d (%d vox): %.1f vox%s\n', ...
                i, comp_vols(i), j, comp_vols(j), min_d, merged_str);
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
%  BOUNDARY REFINEMENT (ported from scaphoid cling-prune-carve chain)
%  Protects the deep interior, aggressively cleans tissue from the surface.
% =========================================================================
function mask = refine_bone_boundary(mask, vol, G, marker_mask, spacing, softMed)
    if ~any(mask(:)), return; end
    voxmm = mean(spacing);

    % --- Step 1: Boundary-band cling ---
    % Deep interior (>1mm from surface) is always kept. The outer band
    % is rebuilt: only voxels connected to high-HU cores AND passing
    % HU/gradient support survive. Tissue at the surface gets dropped.
    D_in = bwdist(~mask) * voxmm;
    deep_interior = D_in >= 1.0;
    band = mask & ~deep_interior;

    vals = double(vol(~marker_mask & vol > -300 & vol < 2000));
    if isempty(vals), vals = double(vol(isfinite(vol))); end
    core_thr = max(240, min(650, prctile(vals, 92)));
    core_seed = band & (vol > core_thr);

    HU_SUPPORT_MIN = max(60, min(180, softMed + 90));
    gThr = prctile(G(:), 80);
    support = band & ((vol > HU_SUPPORT_MIN) | (G > gThr));

    if any(core_seed(:))
        cling_band = imreconstruct(core_seed, support);
    else
        cling_band = support;
    end
    mask = deep_interior | cling_band;
    mask = keep_largest_3d(mask);
    mask = imclose(mask, strel('sphere', 1));
    mask = imfill(mask, 'holes');

    % --- Step 2: Edge-backed perimeter prune ---
    % Kill perimeter voxels with BOTH low HU AND low gradient.
    % Tissue has low HU + smooth gradients; cortical bone has high gradient.
    perim = bwperim(mask, 26);
    band1 = imdilate(perim, strel('sphere', 1));
    T_hu = max(160, min(340, softMed + 190));
    T_g = prctile(G(:), 70);
    kill = band1 & (double(vol) < T_hu) & (G < T_g);
    if any(kill(:))
        mask(kill) = false;
        mask = keep_largest_3d(mask);
        mask = imclose(mask, strel('sphere', 1));
        mask = imfill(mask, 'holes');
    end

    % --- Step 3: Conservative boundary carve ---
    % Kill surface voxels with low HU near air, not protected by dense core.
    perim = bwperim(mask, 26);
    band1 = imdilate(perim, strel('sphere', 1));
    outer1 = imdilate(mask, strel('sphere', 1)) & ~mask;
    protected_core = imdilate(vol > core_thr, strel('sphere', 2));
    T_hi = max(170, min(360, softMed + 210));
    T_lo = T_hi - 80;
    airNear = outer1 & imdilate(vol < -300, strel('sphere', 1));
    kill = band1 & (double(vol) < T_lo) & imdilate(airNear, strel('sphere', 1)) & ~protected_core;
    if any(kill(:))
        mask(kill) = false;
        mask = keep_largest_3d(mask);
        mask = imclose(mask, strel('sphere', 1));
        mask = imfill(mask, 'holes');
    end

    % --- Step 4: Final boundary carve ---
    band1 = imdilate(bwperim(mask, 26), strel('sphere', 1));
    HU_CARVE_FLOOR = max(180, min(400, softMed + 220));
    kill = band1 & (double(vol) < HU_CARVE_FLOOR);
    if any(kill(:))
        mask(kill) = false;
    end

    % --- Step 5: Final cleanup ---
    mask = keep_largest_3d(mask);
    mask = imopen(mask, strel('sphere', 1));
    mask = keep_largest_3d(mask);
    mask = imclose(mask, strel('sphere', 1));
    mask = imfill(mask, 'holes');
end


% =========================================================================
%  SHELL SEALING (ported from scaphoid sealOuterShell)
% =========================================================================
function BW = seal_outer_shell(BW, spacing)
    if ~any(BW(:)), return; end

    % Cap at 3 voxels — at 0.25mm spacing the mm-based formula gives 6,
    % which bridges bone to nearby markers and grabs non-bone tissue
    rClose = min(3, max(1, round(1.5 / max(mean(spacing), eps))));
    SE = strel('sphere', rClose);
    BW = imclose(BW, SE);
    BW = imfill(BW, 'holes');

    BW = keep_largest_3d(BW);
    % No imclearborder — for excised-in-air specimens the bone itself
    % often spans the full height of the local ROI, so clearing border
    % voxels destroys the entire mask.
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
