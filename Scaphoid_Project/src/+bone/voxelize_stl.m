function [vox_mask, vox_size] = voxelize_stl(stl_file, spacing)
% VOXELIZE_STL  Convert an STL mesh to a binary voxel mask.
%
%   [vox_mask, vox_size] = bone.voxelize_stl(stl_file, spacing)
%
% Reads an STL file and returns a logical 3D mask where voxels inside
% the mesh surface are true.  Uses per-slice polygon filling.
%
% Inputs
%   stl_file : path to STL file
%   spacing  : [dr dc ds] voxel spacing in mm
%
% Outputs
%   vox_mask : logical 3D array
%   vox_size : [R C S] size in voxels

TR = stlread(stl_file);
V = double(TR.Points);
F = double(TR.ConnectivityList);

% Center at origin
V = V - mean(V, 1);

% Convert mm to voxel coordinates
V_vox = zeros(size(V));
V_vox(:,1) = V(:,1) / spacing(1);
V_vox(:,2) = V(:,2) / spacing(2);
V_vox(:,3) = V(:,3) / spacing(3);

% Shift so all coordinates are positive with margin
V_vox = V_vox - min(V_vox, [], 1) + 2;

grid_sz = ceil(max(V_vox, [], 1)) + 2;
vox_size = grid_sz;

% Per-slice polygon filling in the Z dimension
vox_mask = false(grid_sz);

z_min = floor(min(V_vox(:,3)));
z_max = ceil(max(V_vox(:,3)));

for z = max(1, z_min):min(grid_sz(3), z_max)
    % Find triangles that cross this Z plane
    z_vals = reshape(V_vox(F, 3), size(F));  % [nFaces x 3]
    f_min = min(z_vals, [], 2);
    f_max = max(z_vals, [], 2);
    active = find(f_min <= z & f_max >= z);

    if isempty(active), continue; end

    % Collect intersection segments with z-plane
    segments = [];
    for fi = 1:numel(active)
        tri = F(active(fi), :);
        v1 = V_vox(tri(1), :);
        v2 = V_vox(tri(2), :);
        v3 = V_vox(tri(3), :);

        pts = intersect_triangle_z(v1, v2, v3, z);
        if size(pts, 1) >= 2
            segments = [segments; pts(1,:) pts(2,:)]; %#ok<AGROW>
        end
    end

    if isempty(segments), continue; end

    % Fill using scanline: for each row, find x-intersections
    all_y = [segments(:,1); segments(:,3)];
    y_min_s = max(1, floor(min(all_y)));
    y_max_s = min(grid_sz(1), ceil(max(all_y)));

    for y = y_min_s:y_max_s
        x_hits = [];
        for si = 1:size(segments, 1)
            p1 = segments(si, 1:2);  % [y1 x1]
            p2 = segments(si, 3:4);  % [y2 x2]
            if (p1(1) <= y && p2(1) > y) || (p2(1) <= y && p1(1) > y)
                t = (y - p1(1)) / (p2(1) - p1(1));
                x_hit = p1(2) + t * (p2(2) - p1(2));
                x_hits(end+1) = x_hit; %#ok<AGROW>
            end
        end
        x_hits = sort(x_hits);
        % Fill between pairs (even-odd rule)
        for pi = 1:2:numel(x_hits)-1
            x1 = max(1, round(x_hits(pi)));
            x2 = min(grid_sz(2), round(x_hits(pi+1)));
            if x1 <= x2
                vox_mask(y, x1:x2, z) = true;
            end
        end
    end
end

% If scanline produced an empty result, fall back to surface sampling + fill
if ~any(vox_mask(:))
    vox_mask = surface_fill(V_vox, F, grid_sz);
end

fprintf('%.0f mm^3 ', sum(vox_mask(:)) * prod(spacing));
end


function pts = intersect_triangle_z(v1, v2, v3, z)
    % Find 2D intersection points [y, x] of triangle with plane z=const
    edges = {v1, v2; v2, v3; v3, v1};
    pts = zeros(0, 2);
    for e = 1:3
        p1 = edges{e, 1};
        p2 = edges{e, 2};
        if (p1(3) <= z && p2(3) >= z) || (p2(3) <= z && p1(3) >= z)
            dz = p2(3) - p1(3);
            if abs(dz) < 1e-10
                pts = [pts; p1(1:2); p2(1:2)]; %#ok<AGROW>
            else
                t = (z - p1(3)) / dz;
                t = max(0, min(1, t));
                pt = p1 + t * (p2 - p1);
                pts = [pts; pt(1:2)]; %#ok<AGROW>
            end
        end
    end
end


function mask = surface_fill(V, F, grid_sz)
    % Mark surface voxels by sampling triangles, then fill interior
    mask = false(grid_sz);
    for fi = 1:size(F, 1)
        v1 = V(F(fi,1), :);
        v2 = V(F(fi,2), :);
        v3 = V(F(fi,3), :);
        for u = 0:0.1:1
            for w = 0:0.1:(1-u)
                pt = v1*(1-u-w) + v2*u + v3*w;
                idx = round(pt);
                if all(idx >= 1) && idx(1) <= grid_sz(1) && ...
                   idx(2) <= grid_sz(2) && idx(3) <= grid_sz(3)
                    mask(idx(1), idx(2), idx(3)) = true;
                end
            end
        end
    end
    mask = imfill(mask, 'holes');
end
