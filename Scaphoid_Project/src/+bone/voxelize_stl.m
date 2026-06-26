function [vox_mask, vox_size] = voxelize_stl(stl_file, spacing)
% VOXELIZE_STL  Convert an STL mesh to a binary voxel mask.
%
%   [vox_mask, vox_size] = bone.voxelize_stl(stl_file, spacing)
%
% Reads an STL file, scales it to the given voxel spacing, and returns
% a logical 3D mask where voxels inside the mesh are true.
%
% Inputs
%   stl_file : path to STL file
%   spacing  : [dr dc ds] voxel spacing in mm
%
% Outputs
%   vox_mask : logical 3D array (tight bounding box of shape)
%   vox_size : [R C S] size of the mask in voxels

TR = stlread(stl_file);
V = TR.Points;   % Nx3 in mm
F = TR.ConnectivityList;

% Center the shape at origin
V = V - mean(V, 1);

% Convert mm coordinates to voxel indices
V_vox = V ./ spacing;

% Shift so all coordinates are positive (1-based)
V_vox = V_vox - min(V_vox, [], 1) + 2;  % +2 for 1-based + 1-voxel margin

grid_size = ceil(max(V_vox, [], 1)) + 2;  % +2 for margin
vox_size = grid_size;

% Ray-casting voxelization (slice-by-slice in Z)
vox_mask = false(grid_size);

for z = 1:grid_size(3)
    z_plane = z;
    % Find triangles that cross this Z plane
    z_vals = V_vox(F, 3);  % [nFaces x 3]
    z_min_f = min(z_vals, [], 2);
    z_max_f = max(z_vals, [], 2);
    active = find(z_min_f <= z_plane & z_max_f >= z_plane);

    if isempty(active), continue; end

    % For each active triangle, find the intersection polygon with z=z_plane
    % then fill using even-odd rule
    slice_img = false(grid_size(1), grid_size(2));

    for fi = 1:numel(active)
        tri_idx = active(fi);
        v1 = V_vox(F(tri_idx, 1), :);
        v2 = V_vox(F(tri_idx, 2), :);
        v3 = V_vox(F(tri_idx, 3), :);

        % Find intersection of triangle with z=z_plane
        pts_2d = tri_z_intersection(v1, v2, v3, z_plane);
        if size(pts_2d, 1) < 2, continue; end

        % Scan-line fill between intersection points on each row
        y_range = [floor(min(pts_2d(:,1))), ceil(max(pts_2d(:,1)))];
        y_range(1) = max(1, y_range(1));
        y_range(2) = min(grid_size(1), y_range(2));

        for y = y_range(1):y_range(2)
            % Find x intersections of the line segments at this y
            x_hits = [];
            for ei = 1:size(pts_2d, 1)
                p1 = pts_2d(ei, :);
                p2 = pts_2d(mod(ei, size(pts_2d,1)) + 1, :);
                if (p1(1) <= y && p2(1) > y) || (p2(1) <= y && p1(1) > y)
                    t = (y - p1(1)) / (p2(1) - p1(1));
                    x_hit = p1(2) + t * (p2(2) - p1(2));
                    x_hits(end+1) = x_hit; %#ok<AGROW>
                end
            end
            if numel(x_hits) >= 2
                x_hits = sort(x_hits);
                for pi = 1:2:numel(x_hits)-1
                    x1 = max(1, round(x_hits(pi)));
                    x2 = min(grid_size(2), round(x_hits(pi+1)));
                    if x1 <= x2
                        slice_img(y, x1:x2) = ~slice_img(y, x1:x2);
                    end
                end
            end
        end
    end

    vox_mask(:,:,z) = slice_img;
end

% Use a simpler but more robust approach: surface voxelization + fill
% The ray-casting above can have parity issues, so we use MATLAB's
% built-in approach as validation
vox_mask2 = surface_voxelize(V_vox, F, grid_size);
if sum(vox_mask2(:)) > sum(vox_mask(:)) * 0.5
    vox_mask = vox_mask2;
end
end


function pts_2d = tri_z_intersection(v1, v2, v3, z)
    % Find intersection of triangle (v1,v2,v3) with plane z=z
    % Returns 2D points [y, x] of the intersection polygon
    edges = [v1; v2; v3; v1];
    pts_2d = [];
    for e = 1:3
        p1 = edges(e, :);
        p2 = edges(e+1, :);
        if (p1(3) <= z && p2(3) >= z) || (p2(3) <= z && p1(3) >= z)
            dz = p2(3) - p1(3);
            if abs(dz) < 1e-10
                pts_2d = [pts_2d; p1(1:2); p2(1:2)]; %#ok<AGROW>
            else
                t = (z - p1(3)) / dz;
                t = max(0, min(1, t));
                pt = p1 + t * (p2 - p1);
                pts_2d = [pts_2d; pt(1:2)]; %#ok<AGROW>
            end
        end
    end
end


function mask = surface_voxelize(V, F, grid_size)
    % Voxelize by marking surface voxels then filling interior
    mask = false(grid_size);

    % Mark all voxels that the surface passes through
    for fi = 1:size(F, 1)
        v1 = V(F(fi,1), :);
        v2 = V(F(fi,2), :);
        v3 = V(F(fi,3), :);

        % Sample triangle at fine resolution
        for u = 0:0.05:1
            for w = 0:0.05:(1-u)
                pt = v1 * (1-u-w) + v2 * u + v3 * w;
                idx = round(pt);
                if all(idx >= 1) && idx(1) <= grid_size(1) && ...
                   idx(2) <= grid_size(2) && idx(3) <= grid_size(3)
                    mask(idx(1), idx(2), idx(3)) = true;
                end
            end
        end
    end

    % Fill interior (the surface should form a closed shell)
    filled = imfill(mask, 'holes');
    mask = filled;
end
