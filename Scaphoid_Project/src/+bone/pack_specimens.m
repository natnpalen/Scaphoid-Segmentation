function pack_result = pack_specimens(bone_mask, cortical, cancellous, ds, stl_paths, shape_names, opts, bone_axis)
% PACK_SPECIMENS  Pack mechanical test specimens into cortical and cancellous regions.
%
%   pack_result = bone.pack_specimens(bone_mask, cortical, cancellous, ds, ...
%       stl_paths, shape_names, opts, bone_axis)
%
% Uses oriented bounding box (OBB) fitting instead of voxelization.
% Each STL specimen is represented by its exact rectangular prism dimensions
% (from the mesh extents), then tested at bone-aligned orientations against
% the actual region mask. This is faster and geometrically exact for
% prismatic machined specimens.
%
% Strategy:
%   1. Read each STL to get its bounding box dimensions (mm)
%   2. Generate bone-aligned orientations from the principal axis
%   3. For each region (cortical, cancellous):
%      a. Place one of each specimen type first (priority constraint)
%      b. Then greedily pack additional specimens to maximize count
%   4. No overlap between placements within a region

spacing = ds.spacing;
vol = double(ds.HU);
voxel_vol = prod(spacing);
n_shapes = numel(stl_paths);

cort_vol = sum(cortical(:)) * voxel_vol;
canc_vol = sum(cancellous(:)) * voxel_vol;
fprintf('      Cortical region: %.0f mm^3\n', cort_vol);
fprintf('      Cancellous region: %.0f mm^3\n', canc_vol);

% ---- Read STL bounding box dimensions ----
fprintf('      Reading %d specimen geometries...\n', n_shapes);
rotations = generate_bone_aligned_rotations(bone_axis, opts.PackingOrientations);
n_orient = size(rotations, 3);

templates = {};
for si = 1:n_shapes
    fprintf('        %s: ', shape_names{si});
    try
        TR = stlread(stl_paths{si});
        V = double(TR.Points);
        V = V - mean(V, 1);
        bbox_mm = max(V, [], 1) - min(V, [], 1);
        mesh_vol = prod(bbox_mm);
        fprintf('%.1f x %.1f x %.1f mm (%.0f mm^3 OBB)\n', ...
            bbox_mm(1), bbox_mm(2), bbox_mm(3), mesh_vol);
    catch ME
        fprintf('FAILED: %s\n', ME.message);
        continue;
    end

    for oi = 1:n_orient
        R = rotations(:,:,oi);
        corners = bbox_corners(bbox_mm);
        rotated_corners = (R * corners')';
        obb_extent = max(rotated_corners, [], 1) - min(rotated_corners, [], 1);

        obb_vox = ceil(obb_extent ./ spacing);
        if any(obb_vox < 1), continue; end

        tpl = struct();
        tpl.shape_idx = si;
        tpl.shape_name = shape_names{si};
        tpl.orientation = oi;
        tpl.rotation = R;
        tpl.bbox_mm = bbox_mm;
        tpl.obb_extent_mm = obb_extent;
        tpl.obb_vox = obb_vox;
        tpl.volume_mm3 = mesh_vol;
        templates{end+1} = tpl; %#ok<AGROW>
    end
end

fprintf('      %d total templates (%d shapes x %d orientations)\n', ...
    numel(templates), n_shapes, n_orient);

if isempty(templates)
    pack_result = empty_result();
    return;
end

% ---- Pack each region ----
fprintf('      --- Cortical packing ---\n');
cort_placements = pack_region(cortical, templates, vol, spacing, shape_names, n_shapes);

fprintf('      --- Cancellous packing ---\n');
canc_placements = pack_region(cancellous, templates, vol, spacing, shape_names, n_shapes);

% ---- Build result ----
pack_result = struct();
pack_result.cortical_placements = cort_placements;
pack_result.cancellous_placements = canc_placements;
pack_result.n_cortical = numel(cort_placements);
pack_result.n_cancellous = numel(canc_placements);
pack_result.n_total = numel(cort_placements) + numel(canc_placements);

pack_result.summary = struct();
for si = 1:n_shapes
    n_cort = sum(arrayfun(@(p) p.shape_idx == si, cort_placements));
    n_canc = sum(arrayfun(@(p) p.shape_idx == si, canc_placements));
    pack_result.summary.(shape_names{si}) = struct('cortical', n_cort, 'cancellous', n_canc);
    fprintf('      %s: %d cortical + %d cancellous = %d total\n', ...
        shape_names{si}, n_cort, n_canc, n_cort + n_canc);
end

fprintf('      TOTAL: %d specimens (%d cortical + %d cancellous)\n', ...
    pack_result.n_total, pack_result.n_cortical, pack_result.n_cancellous);
end


% =========================================================================
%  REGION PACKING (OBB-based)
% =========================================================================
function placements = pack_region(region, templates, vol, spacing, shape_names, n_shapes)

    voxel_vol = prod(spacing);
    region_vol = sum(region(:)) * voxel_vol;
    placements = struct('shape_name', {}, 'shape_idx', {}, 'orientation', {}, ...
        'position_vox', {}, 'obb_vox', {}, 'obb_extent_mm', {}, ...
        'volume_mm3', {}, 'mean_hu', {}, 'fit_fraction', {});

    if region_vol < 1.0
        fprintf('        Region too small (%.0f mm^3), skipping\n', region_vol);
        return;
    end

    available = region;
    vol_sz = size(available);

    % Precompute distance transform once (recomputed after each placement)
    D = bwdist(~available) .* mean(spacing);

    % Phase 1: one of each type (priority)
    for si = 1:n_shapes
        type_idx = find(cellfun(@(t) t.shape_idx == si, templates));
        if isempty(type_idx), continue; end

        [p, available, D] = try_place_best(available, D, templates(type_idx), vol, spacing);
        if ~isempty(p)
            placements(end+1) = p; %#ok<AGROW>
            fprintf('        [priority] Placed %s (%.1fx%.1fx%.1f mm) at [%d %d %d] fit=%.0f%%\n', ...
                p.shape_name, p.obb_extent_mm, p.position_vox, p.fit_fraction*100);
        else
            fprintf('        [priority] %s: does not fit\n', shape_names{si});
        end
    end

    % Phase 2: greedily pack more specimens (any type)
    max_additional = 50;
    for attempt = 1:max_additional
        if ~any(available(:)), break; end

        [p, available, D] = try_place_best(available, D, templates, vol, spacing);
        if isempty(p), break; end

        placements(end+1) = p; %#ok<AGROW>
        fprintf('        [greedy] Placed %s (%.1fx%.1fx%.1f mm) at [%d %d %d] fit=%.0f%%\n', ...
            p.shape_name, p.obb_extent_mm, p.position_vox, p.fit_fraction*100);
    end

    fprintf('        Total: %d specimens in %.0f mm^3 region\n', ...
        numel(placements), region_vol);
end


% =========================================================================
%  PLACEMENT SEARCH (OBB sliding window)
% =========================================================================
function [placement, available, D] = try_place_best(available, D, templates, vol, spacing)

    placement = [];
    best_score = -Inf;
    best_pos = [];
    best_tpl = [];
    best_fit_frac = 0;

    vol_sz = size(available);

    S = cumsum3(single(available));
    S_d = cumsum3(single(D));

    for ti = 1:numel(templates)
        tpl = templates{ti};
        bsz = tpl.obb_vox;

        if any(bsz > vol_sz), continue; end

        n_box_vox = prod(bsz);

        overlap = box_sum(S, vol_sz, bsz);

        fit_frac = overlap / max(1, n_box_vox);
        good_fit = fit_frac >= 0.90;
        if ~any(good_fit(:)), continue; end

        depth_sum = box_sum(S_d, vol_sz, bsz);
        avg_depth = depth_sum / max(1, n_box_vox);

        score_map = fit_frac + avg_depth;
        score_map(~good_fit) = -Inf;

        [local_best, linear_idx] = max(score_map(:));
        if local_best > best_score
            out_sz = vol_sz - bsz + 1;
            [pr, pc, ps] = ind2sub(out_sz, linear_idx);
            best_score = local_best;
            best_pos = [pr, pc, ps];
            best_tpl = tpl;
            best_fit_frac = fit_frac(linear_idx);
        end
    end

    if isempty(best_pos), return; end

    r1 = best_pos(1); c1 = best_pos(2); s1 = best_pos(3);
    bsz = best_tpl.obb_vox;
    r2 = r1 + bsz(1) - 1;
    c2 = c1 + bsz(2) - 1;
    s2 = s1 + bsz(3) - 1;

    box_mask = false(vol_sz);
    box_mask(r1:r2, c1:c2, s1:s2) = true;
    placed = box_mask & available;

    actual_fit = sum(placed(:)) / prod(bsz);
    if actual_fit < 0.85
        return;
    end

    placement = struct();
    placement.shape_name = best_tpl.shape_name;
    placement.shape_idx = best_tpl.shape_idx;
    placement.orientation = best_tpl.orientation;
    placement.position_vox = best_pos;
    placement.obb_vox = bsz;
    placement.obb_extent_mm = best_tpl.obb_extent_mm;
    placement.volume_mm3 = best_tpl.volume_mm3;
    placement.mean_hu = mean(vol(placed));
    placement.fit_fraction = actual_fit;

    available(box_mask) = false;
    D = bwdist(~available) .* mean(spacing);
end


% =========================================================================
%  3D INTEGRAL IMAGE (zero-padded cumulative sum)
% =========================================================================
function S = cumsum3(A)
    % Zero-pad so S(0,:,:) = 0, allowing clean subtraction in box_sum
    sz = size(A);
    S = zeros(sz + 1, 'like', A);
    S(2:end, 2:end, 2:end) = cumsum(cumsum(cumsum(A, 1), 2), 3);
end


% =========================================================================
%  BOX SUM via integral image (1-indexed input, S is 0-padded)
% =========================================================================
function result = box_sum(S, vol_sz, bsz)
    out_sz = vol_sz - bsz + 1;
    if any(out_sz < 1)
        result = zeros(0);
        return;
    end

    % In the padded S, S(i+1,j+1,k+1) = sum of A(1:i, 1:j, 1:k)
    % Box sum at position (r,c,s) with size bsz:
    %   sum of A(r:r+bsz-1, c:c+bsz-1, s:s+bsz-1)
    r1 = 1:out_sz(1);  r2 = r1 + bsz(1);
    c1 = 1:out_sz(2);  c2 = c1 + bsz(2);
    s1 = 1:out_sz(3);  s2 = s1 + bsz(3);

    result = S(r2, c2, s2) ...
           - S(r1, c2, s2) - S(r2, c1, s2) - S(r2, c2, s1) ...
           + S(r1, c1, s2) + S(r1, c2, s1) + S(r2, c1, s1) ...
           - S(r1, c1, s1);
end


% =========================================================================
%  BBOX CORNERS (8 corners of a bounding box centered at origin)
% =========================================================================
function C = bbox_corners(bbox_mm)
    hx = bbox_mm(1)/2; hy = bbox_mm(2)/2; hz = bbox_mm(3)/2;
    C = [ hx  hy  hz;  hx  hy -hz;  hx -hy  hz;  hx -hy -hz;
         -hx  hy  hz; -hx  hy -hz; -hx -hy  hz; -hx -hy -hz];
end


% =========================================================================
%  BONE-ALIGNED ROTATIONS
% =========================================================================
function R = generate_bone_aligned_rotations(bone_axis, n_orient)

    bone_axis = bone_axis(:) / norm(bone_axis);

    if abs(bone_axis(1)) < 0.9
        perp = cross(bone_axis, [1;0;0]);
    else
        perp = cross(bone_axis, [0;1;0]);
    end
    perp = perp / norm(perp);
    perp2 = cross(bone_axis, perp);
    perp2 = perp2 / norm(perp2);

    R_base = [perp, perp2, bone_axis]';

    R = zeros(3, 3, min(n_orient, 12));
    idx = 0;

    angles_about_axis = [0, 90, 180, 270];
    angles_perpendicular = [0, 90];

    for ai = 1:numel(angles_about_axis)
        for pi = 1:numel(angles_perpendicular)
            idx = idx + 1;
            if idx > n_orient, break; end

            theta = deg2rad(angles_about_axis(ai));
            phi = deg2rad(angles_perpendicular(pi));

            Rz = [cos(theta) -sin(theta) 0; sin(theta) cos(theta) 0; 0 0 1];
            Rx = [1 0 0; 0 cos(phi) -sin(phi); 0 sin(phi) cos(phi)];

            R(:,:,idx) = R_base * Rz * Rx;
        end
        if idx >= n_orient, break; end
    end

    R = R(:,:,1:idx);
end


% =========================================================================
%  UTILITIES
% =========================================================================
function r = empty_result()
    r = struct();
    r.cortical_placements = struct('shape_name', {}, 'shape_idx', {}, ...
        'orientation', {}, 'position_vox', {}, 'obb_vox', {}, ...
        'obb_extent_mm', {}, 'volume_mm3', {}, 'mean_hu', {}, ...
        'fit_fraction', {});
    r.cancellous_placements = r.cortical_placements;
    r.n_cortical = 0;
    r.n_cancellous = 0;
    r.n_total = 0;
    r.summary = struct();
end
