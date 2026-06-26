function placements = pack_specimens(region_mask, ds, stl_paths, shape_names, opts)
% PACK_SPECIMENS  Greedy mixed packing of mechanical test specimens into a bone region.
%
%   placements = bone.pack_specimens(region_mask, ds, stl_paths, shape_names, opts)
%
% All specimen shapes compete for the SAME volume (mixed packing, no overlaps).
% Uses distance-transform-guided placement: specimens are placed at the
% deepest available interior point first, maximizing packing density.
%
% Inputs
%   region_mask  : logical 3D mask (cortical or cancellous region)
%   ds           : dataset struct from dicom.series_load
%   stl_paths    : cell array of STL file paths
%   shape_names  : cell array of shape names (e.g. {'Bend','Compression','Punch','Shear'})
%   opts         : pipeline options struct with PackingOrientations, PackingMinDepthMM
%
% Output
%   placements   : struct array with fields .shape_name, .shape_idx,
%                  .position_vox, .orientation, .mask, .volume_mm3, .mean_hu

spacing = ds.spacing;
vol = double(ds.HU);
voxel_vol = prod(spacing);
n_shapes = numel(stl_paths);

region_vol = sum(region_mask(:)) * voxel_vol;
fprintf('      Region volume: %.0f mm^3\n', region_vol);

if region_vol < 1.0
    placements = struct([]);
    return;
end

% ---- Voxelize all shapes at all orientations ----
fprintf('      Voxelizing %d shapes...\n', n_shapes);
candidates = {};

% Standard orientations: identity + 90-degree rotations about each axis
n_orient = opts.PackingOrientations;
rotations = generate_rotations(n_orient);

for si = 1:n_shapes
    fprintf('        %s: ', shape_names{si});
    for oi = 1:size(rotations, 3)
        try
            [shape_mask, ~] = bone.voxelize_stl(stl_paths{si}, spacing);
            % Apply rotation
            if oi > 1
                shape_mask = rotate_mask_3d(shape_mask, rotations(:,:,oi));
            end
            shape_vol = sum(shape_mask(:)) * voxel_vol;
            if shape_vol < 0.1, continue; end

            cand = struct();
            cand.shape_idx = si;
            cand.shape_name = shape_names{si};
            cand.orientation = oi;
            cand.template = shape_mask;
            cand.volume_mm3 = shape_vol;
            cand.size_vox = size(shape_mask);
            candidates{end+1} = cand; %#ok<AGROW>
        catch
            continue;
        end
    end
    fprintf('%d orientations\n', sum(cellfun(@(c) c.shape_idx == si, candidates)));
end

fprintf('      %d candidate templates total\n', numel(candidates));

% ---- Greedy packing ----
available = region_mask;
placements = struct([]);
max_attempts = 200;

for attempt = 1:max_attempts
    if ~any(available(:)), break; end

    % Distance transform of available region
    D = bwdist(~available) .* mean(spacing);

    % Try each candidate at the deepest interior point
    best_score = -Inf;
    best_placement = [];

    for ci = 1:numel(candidates)
        cand = candidates{ci};
        sz = cand.size_vox;

        % Find the deepest point with enough clearance
        min_depth = opts.PackingMinDepthMM;
        deep_enough = D >= min_depth;
        if ~any(deep_enough(:)), continue; end

        % Find best placement position
        [score, pos] = find_best_position(available, cand.template, D, spacing);

        if score > best_score && ~isempty(pos)
            best_score = score;
            placed = struct();
            placed.shape_name = cand.shape_name;
            placed.shape_idx = cand.shape_idx;
            placed.orientation = cand.orientation;
            placed.position_vox = pos;
            placed.volume_mm3 = cand.volume_mm3;

            % Build placed mask
            sz = cand.size_vox;
            r1 = pos(1); r2 = pos(1) + sz(1) - 1;
            c1 = pos(2); c2 = pos(2) + sz(2) - 1;
            s1 = pos(3); s2 = pos(3) + sz(3) - 1;

            if r2 <= size(available,1) && c2 <= size(available,2) && s2 <= size(available,3)
                placed_mask = false(size(available));
                placed_mask(r1:r2, c1:c2, s1:s2) = cand.template;
                placed_mask = placed_mask & available;

                if sum(placed_mask(:)) >= 0.9 * sum(cand.template(:))
                    placed.mask = placed_mask;
                    placed.mean_hu = mean(vol(placed_mask));
                    best_placement = placed;
                end
            end
        end
    end

    if isempty(best_placement), break; end

    % Place the specimen
    if isempty(placements)
        placements = best_placement;
    else
        placements(end+1) = best_placement; %#ok<AGROW>
    end
    available(best_placement.mask) = false;

    fprintf('      Placed %s (#%d) at [%d %d %d], %.1f mm^3\n', ...
        best_placement.shape_name, numel(placements), ...
        best_placement.position_vox, best_placement.volume_mm3);
end

fprintf('      Total: %d specimens placed\n', numel(placements));
end


% =========================================================================
function [score, best_pos] = find_best_position(available, template, D, spacing)
    score = -Inf;
    best_pos = [];

    sz = size(template);
    vol_sz = size(available);

    % Stride: check every few voxels for speed
    stride = max(1, round(1.0 ./ spacing));

    % Get candidate positions from distance-transform peaks
    D_smooth = imgaussfilt3(D, 1);

    for r = 1:stride(1):(vol_sz(1) - sz(1) + 1)
        for c = 1:stride(2):(vol_sz(2) - sz(2) + 1)
            for s = 1:stride(3):(vol_sz(3) - sz(3) + 1)
                % Check center depth
                cr = r + floor(sz(1)/2);
                cc = c + floor(sz(2)/2);
                cs = s + floor(sz(3)/2);

                if cr > vol_sz(1) || cc > vol_sz(2) || cs > vol_sz(3)
                    continue;
                end

                center_depth = D_smooth(cr, cc, cs);
                if center_depth < 0.3, continue; end

                % Check if template fits
                region = available(r:r+sz(1)-1, c:c+sz(2)-1, s:s+sz(3)-1);
                overlap = template & region;
                fit_frac = sum(overlap(:)) / max(1, sum(template(:)));

                if fit_frac >= 0.95
                    s_val = center_depth * fit_frac;
                    if s_val > score
                        score = s_val;
                        best_pos = [r c s];
                    end
                end
            end
        end
    end
end


function R = generate_rotations(n)
    % Generate n distinct 90-degree rotation matrices
    % For n <= 6: identity + 90 deg about each axis (positive and negative)
    R = zeros(3, 3, min(n, 24));
    R(:,:,1) = eye(3);  % identity
    idx = 1;

    if n >= 2
        idx = idx + 1;
        R(:,:,idx) = [1 0 0; 0 0 -1; 0 1 0];  % 90 about X
    end
    if n >= 3
        idx = idx + 1;
        R(:,:,idx) = [0 0 1; 0 1 0; -1 0 0];   % 90 about Y
    end
    if n >= 4
        idx = idx + 1;
        R(:,:,idx) = [0 -1 0; 1 0 0; 0 0 1];   % 90 about Z
    end
    if n >= 5
        idx = idx + 1;
        R(:,:,idx) = [1 0 0; 0 -1 0; 0 0 -1];  % 180 about X
    end
    if n >= 6
        idx = idx + 1;
        R(:,:,idx) = [-1 0 0; 0 1 0; 0 0 -1];  % 180 about Y
    end

    R = R(:,:,1:idx);
end


function rotated = rotate_mask_3d(mask, R)
    % Rotate a 3D binary mask by rotation matrix R (90-degree rotations only)
    sz = size(mask);
    center = (sz + 1) / 2;

    [rr, cc, ss] = ind2sub(sz, find(mask));
    coords = [rr - center(1), cc - center(2), ss - center(3)];
    coords_rot = coords * R';

    new_coords = round(coords_rot);
    new_center = round((max(new_coords, [], 1) - min(new_coords, [], 1) + 1) / 2) + 1;
    new_coords = new_coords - min(new_coords, [], 1) + 1;

    new_sz = max(new_coords, [], 1);
    rotated = false(new_sz);

    valid = all(new_coords >= 1, 2) & ...
            new_coords(:,1) <= new_sz(1) & ...
            new_coords(:,2) <= new_sz(2) & ...
            new_coords(:,3) <= new_sz(3);

    idx = sub2ind(new_sz, new_coords(valid,1), new_coords(valid,2), new_coords(valid,3));
    rotated(idx) = true;

    % Fill gaps from rotation
    rotated = imclose(rotated, strel('sphere', 1));
end
