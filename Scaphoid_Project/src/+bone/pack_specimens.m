function pack_result = pack_specimens(bone_mask, cortical, cancellous, ds, stl_paths, shape_names, opts, bone_axis)
% PACK_SPECIMENS  Pack mechanical test specimens into cortical and cancellous regions.
%
%   pack_result = bone.pack_specimens(bone_mask, cortical, cancellous, ds, ...
%       stl_paths, shape_names, opts, bone_axis)
%
% Places all 4 specimen types (Bend, Compression, Punch, Shear) into both
% cortical and cancellous regions. Uses convolution-based fit scoring for
% fast placement, with orientations aligned to the bone's principal axis.
%
% Strategy:
%   1. Voxelize each STL specimen at bone-aligned orientations
%   2. For each region (cortical, cancellous):
%      a. Place one of each specimen type first (priority constraint)
%      b. Then greedily pack additional specimens to maximize count
%   3. No overlap between placements within a region
%
% Inputs
%   bone_mask    : logical 3D full bone mask
%   cortical     : logical 3D cortical region
%   cancellous   : logical 3D cancellous region
%   ds           : dataset struct from dicom.series_load
%   stl_paths    : cell array of STL file paths
%   shape_names  : cell array of shape names
%   opts         : pipeline options struct
%   bone_axis    : [3x1] principal axis direction from PCA
%
% Output
%   pack_result  : struct with .cortical_placements, .cancellous_placements,
%                  .summary

spacing = ds.spacing;
vol = double(ds.HU);
voxel_vol = prod(spacing);
n_shapes = numel(stl_paths);

cort_vol = sum(cortical(:)) * voxel_vol;
canc_vol = sum(cancellous(:)) * voxel_vol;
fprintf('      Cortical region: %.0f mm^3\n', cort_vol);
fprintf('      Cancellous region: %.0f mm^3\n', canc_vol);

% ---- Voxelize all shapes at bone-aligned orientations ----
fprintf('      Voxelizing %d specimen types...\n', n_shapes);
rotations = generate_bone_aligned_rotations(bone_axis, opts.PackingOrientations);
n_orient = size(rotations, 3);

templates = {};
for si = 1:n_shapes
    fprintf('        %s: ', shape_names{si});
    n_valid = 0;
    for oi = 1:n_orient
        try
            [shape_mask, ~] = bone.voxelize_stl(stl_paths{si}, spacing);
            if oi > 1
                shape_mask = rotate_mask_3d(shape_mask, rotations(:,:,oi));
            end
            shape_vol = sum(shape_mask(:)) * voxel_vol;
            if shape_vol < 0.1, continue; end

            tpl = struct();
            tpl.shape_idx = si;
            tpl.shape_name = shape_names{si};
            tpl.orientation = oi;
            tpl.mask = shape_mask;
            tpl.volume_mm3 = shape_vol;
            tpl.sz = size(shape_mask);
            templates{end+1} = tpl; %#ok<AGROW>
            n_valid = n_valid + 1;
        catch
            continue;
        end
    end
    fprintf('%d orientations (%.0f mm^3)\n', n_valid, ...
        ternary(n_valid > 0, templates{end}.volume_mm3, 0));
end

fprintf('      %d total templates\n', numel(templates));

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

% Summary by type
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
%  REGION PACKING
% =========================================================================
function placements = pack_region(region, templates, vol, spacing, shape_names, n_shapes)
% Greedy packing into a single region. Priority: one of each type first.

    voxel_vol = prod(spacing);
    region_vol = sum(region(:)) * voxel_vol;
    placements = struct('shape_name', {}, 'shape_idx', {}, 'orientation', {}, ...
        'position_vox', {}, 'volume_mm3', {}, 'mean_hu', {}, 'mask', {});

    if region_vol < 1.0
        fprintf('        Region too small (%.0f mm^3), skipping\n', region_vol);
        return;
    end

    available = region;
    types_placed = false(1, n_shapes);

    % Phase 1: one of each type (priority)
    for si = 1:n_shapes
        type_templates = cellfun(@(t) t.shape_idx == si, templates);
        type_idx = find(type_templates);
        if isempty(type_idx), continue; end

        [p, available] = try_place_best(available, templates(type_idx), vol, spacing);
        if ~isempty(p)
            placements(end+1) = p; %#ok<AGROW>
            types_placed(si) = true;
            fprintf('        [priority] Placed %s at [%d %d %d] (%.0f mm^3)\n', ...
                p.shape_name, p.position_vox, p.volume_mm3);
        else
            fprintf('        [priority] %s: does not fit\n', shape_names{si});
        end
    end

    % Phase 2: greedily pack more specimens (any type)
    max_additional = 50;
    for attempt = 1:max_additional
        if ~any(available(:)), break; end

        [p, available] = try_place_best(available, templates, vol, spacing);
        if isempty(p), break; end

        placements(end+1) = p; %#ok<AGROW>
        fprintf('        [greedy] Placed %s at [%d %d %d] (%.0f mm^3)\n', ...
            p.shape_name, p.position_vox, p.volume_mm3);
    end

    fprintf('        Total: %d specimens in %.0f mm^3 region\n', ...
        numel(placements), region_vol);
end


% =========================================================================
%  PLACEMENT SEARCH (convolution-based)
% =========================================================================
function [placement, available] = try_place_best(available, templates, vol, spacing)
% Find the best placement among all templates using convolution scoring.

    placement = [];
    best_score = -Inf;
    best_pos = [];
    best_tpl = [];
    best_placed_mask = [];

    vol_sz = size(available);

    for ti = 1:numel(templates)
        tpl = templates{ti};
        tsz = tpl.sz;

        % Template must fit within volume dimensions
        if any(tsz > vol_sz), continue; end

        % Convolution: count how many template voxels overlap available region
        % convn with 'valid' mode gives overlap at every valid position
        overlap_count = convn(single(available), flip_3d(single(tpl.mask)), 'valid');
        n_template_vox = sum(tpl.mask(:));

        % Fit fraction at each position
        fit_frac = overlap_count / max(1, n_template_vox);

        % Require >= 95% fit
        good_fit = fit_frac >= 0.95;
        if ~any(good_fit(:)), continue; end

        % Score by depth: prefer placements deep inside the region
        D = bwdist(~available) .* mean(spacing);
        % Average depth at each candidate position via convolution
        depth_sum = convn(single(D), flip_3d(single(tpl.mask)), 'valid');
        avg_depth = depth_sum / max(1, n_template_vox);

        % Combined score: fit fraction + average depth
        score_map = fit_frac + avg_depth;
        score_map(~good_fit) = -Inf;

        [local_best, linear_idx] = max(score_map(:));
        if local_best > best_score
            [pr, pc, ps] = ind2sub(size(score_map), linear_idx);
            best_score = local_best;
            best_pos = [pr, pc, ps];
            best_tpl = tpl;
        end
    end

    if isempty(best_pos), return; end

    % Build placed mask
    r1 = best_pos(1); c1 = best_pos(2); s1 = best_pos(3);
    tsz = best_tpl.sz;
    r2 = r1 + tsz(1) - 1;
    c2 = c1 + tsz(2) - 1;
    s2 = s1 + tsz(3) - 1;

    placed_mask = false(vol_sz);
    placed_mask(r1:r2, c1:c2, s1:s2) = best_tpl.mask;
    placed_mask = placed_mask & available;

    % Verify fit
    if sum(placed_mask(:)) < 0.90 * sum(best_tpl.mask(:))
        return;
    end

    placement = struct();
    placement.shape_name = best_tpl.shape_name;
    placement.shape_idx = best_tpl.shape_idx;
    placement.orientation = best_tpl.orientation;
    placement.position_vox = best_pos;
    placement.volume_mm3 = best_tpl.volume_mm3;
    placement.mean_hu = mean(vol(placed_mask));
    placement.mask = placed_mask;

    % Remove placed voxels from available
    available(placed_mask) = false;
end


function flipped = flip_3d(A)
    flipped = flip(flip(flip(A, 1), 2), 3);
end


% =========================================================================
%  BONE-ALIGNED ROTATIONS
% =========================================================================
function R = generate_bone_aligned_rotations(bone_axis, n_orient)
% Generate rotations that align the specimen's Z-axis with the bone's
% principal axis, plus 90-degree rotations about and perpendicular to it.

    bone_axis = bone_axis(:) / norm(bone_axis);

    % Build orthonormal frame: bone_axis = new Z
    if abs(bone_axis(1)) < 0.9
        perp = cross(bone_axis, [1;0;0]);
    else
        perp = cross(bone_axis, [0;1;0]);
    end
    perp = perp / norm(perp);
    perp2 = cross(bone_axis, perp);
    perp2 = perp2 / norm(perp2);

    % Base rotation: align specimen Z with bone axis
    R_base = [perp, perp2, bone_axis]';  % 3x3

    % Generate additional rotations by rotating about bone axis
    R = zeros(3, 3, min(n_orient, 12));
    idx = 0;

    angles_about_axis = [0, 90, 180, 270];  % degrees
    angles_perpendicular = [0, 90];          % tip specimen sideways

    for ai = 1:numel(angles_about_axis)
        for pi = 1:numel(angles_perpendicular)
            idx = idx + 1;
            if idx > n_orient, break; end

            theta = deg2rad(angles_about_axis(ai));
            phi = deg2rad(angles_perpendicular(pi));

            % Rotation about bone axis
            Rz = [cos(theta) -sin(theta) 0; sin(theta) cos(theta) 0; 0 0 1];
            % Rotation about perp axis (tip)
            Rx = [1 0 0; 0 cos(phi) -sin(phi); 0 sin(phi) cos(phi)];

            R(:,:,idx) = R_base * Rz * Rx;
        end
        if idx >= n_orient, break; end
    end

    R = R(:,:,1:idx);
end


% =========================================================================
%  MASK ROTATION
% =========================================================================
function rotated = rotate_mask_3d(mask, R)
    sz = size(mask);
    center = (sz + 1) / 2;

    [rr, cc, ss] = ind2sub(sz, find(mask));
    coords = [rr - center(1), cc - center(2), ss - center(3)];
    coords_rot = coords * R';

    new_coords = round(coords_rot);
    new_coords = new_coords - min(new_coords, [], 1) + 1;

    new_sz = max(new_coords, [], 1);
    rotated = false(new_sz);

    valid = all(new_coords >= 1, 2) & ...
            new_coords(:,1) <= new_sz(1) & ...
            new_coords(:,2) <= new_sz(2) & ...
            new_coords(:,3) <= new_sz(3);

    idx = sub2ind(new_sz, new_coords(valid,1), new_coords(valid,2), new_coords(valid,3));
    rotated(idx) = true;

    rotated = imclose(rotated, strel('sphere', 1));
end


% =========================================================================
%  UTILITIES
% =========================================================================
function r = empty_result()
    r = struct();
    r.cortical_placements = struct('shape_name', {}, 'shape_idx', {}, ...
        'orientation', {}, 'position_vox', {}, 'volume_mm3', {}, ...
        'mean_hu', {}, 'mask', {});
    r.cancellous_placements = r.cortical_placements;
    r.n_cortical = 0;
    r.n_cancellous = 0;
    r.n_total = 0;
    r.summary = struct();
end

function s = ternary(cond, a, b)
    if cond, s = a; else, s = b; end
end
