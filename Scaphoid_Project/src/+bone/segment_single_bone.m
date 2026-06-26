function [refined_mask, qc] = segment_single_bone(ds, bone_mask, opts)
% SEGMENT_SINGLE_BONE  Refine a single bone mask using the scaphoid pipeline approach.
%
%   [refined_mask, qc] = bone.segment_single_bone(ds, bone_mask, opts)
%
% Takes a coarse bone envelope (from separate_bones) and refines the
% boundary using the same adaptive FMM strategy proven in the scaphoid
% pipeline.  Operates on a tight crop around the bone for speed.
%
% Inputs
%   ds        : dataset struct from dicom.series_load
%   bone_mask : logical 3D mask (coarse envelope of one bone)
%   opts      : pipeline options struct
%
% Outputs
%   refined_mask : logical 3D mask (same size as ds.HU)
%   qc           : struct with .seed, .method, .volume_mm3

vol = double(ds.HU);
spacing = ds.spacing;
voxel_vol = prod(spacing);

% ---- Crop to bone ROI with margin ----
margin_vox = max(3, round(5.0 ./ spacing));
[r, c, s] = ind2sub(size(bone_mask), find(bone_mask));
r1 = max(1, min(r) - margin_vox(1));
r2 = min(size(vol,1), max(r) + margin_vox(1));
c1 = max(1, min(c) - margin_vox(2));
c2 = min(size(vol,2), max(c) + margin_vox(2));
s1 = max(1, min(s) - margin_vox(3));
s2 = min(size(vol,3), max(s) + margin_vox(3));

crop_vol  = vol(r1:r2, c1:c2, s1:s2);
crop_mask = bone_mask(r1:r2, c1:c2, s1:s2);

% ---- Marker detection within crop ----
lead = crop_vol > opts.TagHUMin;
if any(lead(:))
    marker_mask = lead;
    near_lead = imdilate(lead, strel('sphere', 2));
    flags = (crop_vol >= 200) & (crop_vol <= 700) & near_lead;
    marker_mask = marker_mask | flags;
    d_mm = aniso_distance_mm(marker_mask, spacing);
    artifact_w = exp(-(d_mm / opts.ArtifactSigmaMM).^2);
else
    marker_mask = false(size(crop_vol));
    artifact_w = zeros(size(crop_vol));
end
marker_dil = imdilate(marker_mask, strel('sphere', 2));

% ---- Compute gradient ----
[Gr, Gc, Gs] = gradient(crop_vol, spacing(1), spacing(2), spacing(3));
G = sqrt(Gr.^2 + Gc.^2 + Gs.^2);

% ---- Statistics from bone envelope ----
vals_mask = crop_mask & ~marker_dil & (crop_vol > -300) & (crop_vol < 2000);
vals = crop_vol(vals_mask);
if isempty(vals)
    vals = crop_vol(crop_mask & (crop_vol > -300));
end
if isempty(vals)
    refined_mask = bone_mask;
    qc = struct('seed', [], 'method', 'passthrough', 'volume_mm3', sum(bone_mask(:))*voxel_vol);
    return;
end

softMed = min(0, median(vals));

% ---- Seed: distance transform maximum within bone envelope ----
D = bwdist(~crop_mask);
[~, seed_idx] = max(D(:));
[si, sj, sk] = ind2sub(size(crop_mask), seed_idx);

% ---- Core detection ----
core_thr = max(280, min(700, prctile(vals, 94)));
core = crop_mask & ~marker_dil & (crop_vol > core_thr);

if sum(core(:)) < 100
    for p = [92 90 88 86 84]
        alt = max(220, min(650, prctile(vals, p)));
        c_alt = crop_mask & ~marker_dil & (crop_vol > alt);
        if sum(c_alt(:)) > 500
            core = c_alt; core_thr = alt;
            fprintf('      Core relaxed to p%d (thr=%.0f)\n', p, alt);
            break;
        end
    end
end

% ---- Adaptive constraint sweep ----
constraints = {
    'Loose',  110, 85;
    'Medium', 135, 89;
    'Strict', 160, 92;
};

specimen_G = G(crop_mask);
best_mask = [];
best_score = -Inf;
best_name = '';

for ci = 1:size(constraints, 1)
    name = constraints{ci, 1};
    hu_off = constraints{ci, 2};
    g_pct = constraints{ci, 3};

    hu_floor = max(70, min(220, softMed + hu_off));
    g_thr = prctile(specimen_G, g_pct);

    allow = crop_mask & ~marker_dil & ((crop_vol > hu_floor) | (G > g_thr));

    if any(core(:))
        cand = imreconstruct(core & allow, allow);
    else
        cand = allow;
    end

    % Remove tiny components
    CC = bwconncomp(cand, 26);
    for j = 1:CC.NumObjects
        if numel(CC.PixelIdxList{j}) < 500
            cand(CC.PixelIdxList{j}) = false;
        end
    end

    score = perimeter_score(cand, crop_vol);

    if score > best_score
        best_score = score;
        best_mask = cand;
        best_name = name;
    end
end

if isempty(best_mask) || ~any(best_mask(:))
    best_mask = crop_mask;
    best_name = 'envelope-fallback';
end

% ---- Morphological cleanup + fill ----
se = strel('sphere', 1);
best_mask = imclose(best_mask, se);
for z = 1:size(best_mask, 3)
    sl = best_mask(:,:,z);
    if any(sl(:))
        best_mask(:,:,z) = imfill(sl, 'holes');
    end
end
best_mask = imfill(best_mask, 'holes');
best_mask = best_mask & crop_mask;

% ---- Boundary refinement ----
best_mask = boundary_cling(best_mask, crop_vol, G, spacing, vals, softMed, core);
best_mask = edge_prune(best_mask, crop_vol, G, softMed);

% Final cleanup
best_mask = imopen(best_mask, se);
best_mask = imclose(best_mask, se);
best_mask = imfill(best_mask, 'holes');

% ---- Uncrop ----
refined_mask = false(size(vol));
refined_mask(r1:r2, c1:c2, s1:s2) = best_mask;

qc = struct();
qc.seed = [si + r1 - 1, sj + c1 - 1, sk + s1 - 1];
qc.method = best_name;
qc.volume_mm3 = sum(refined_mask(:)) * voxel_vol;
qc.core_thr = core_thr;
qc.softMed = softMed;
end


% =========================================================================
function score = perimeter_score(mask, vol)
    if ~any(mask(:))
        score = -Inf;
        return;
    end
    perim = mask & ~imerode(mask, strel('sphere', 1));
    if ~any(perim(:))
        score = -Inf;
        return;
    end
    score = prctile(vol(perim), 90);
end


function result = boundary_cling(bone_mask, vol, G, spacing, vals, softMed, core)
    D_in = bwdist(~bone_mask) .* mean(spacing);
    deep = D_in >= 1.0;
    band = bone_mask & ~deep;

    if ~any(band(:))
        result = bone_mask;
        return;
    end

    core_thr = max(240, min(650, prctile(vals, 92)));
    core_seed = band & (vol > core_thr);

    hu_support = max(60, min(180, softMed + 90));
    g_thr = prctile(G(:), 80);
    support = band & ((vol > hu_support) | (G > g_thr));

    if any(core_seed(:)) && any(support(:))
        cling = imreconstruct(core_seed & support, support);
    elseif any(support(:))
        cling = support;
    else
        cling = band;
    end

    result = deep | cling;
    se = strel('sphere', 1);
    result = imclose(result, se);
    result = imfill(result, 'holes');
end


function result = edge_prune(bone_mask, vol, G, softMed)
    perim = bone_mask & ~imerode(bone_mask, strel('sphere', 1));
    band1 = imdilate(perim, strel('sphere', 1));

    T_hu = max(160, min(340, softMed + 190));
    T_g = prctile(G(:), 70);

    kill = band1 & (vol < T_hu) & (G < T_g);
    if ~any(kill(:))
        result = bone_mask;
        return;
    end

    result = bone_mask;
    result(kill) = false;
    se = strel('sphere', 1);
    result = imclose(result, se);
    result = imfill(result, 'holes');
end


function d_mm = aniso_distance_mm(BW, spacing)
    iso = min(spacing);
    scale = spacing ./ iso;
    sz_new = round(size(BW) .* scale);
    BW_iso = imresize3(uint8(BW), sz_new, 'nearest') > 0;
    d_iso = bwdist(BW_iso) * iso;
    d_mm = imresize3(single(d_iso), size(BW), 'linear');
end
