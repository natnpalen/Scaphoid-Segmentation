function run_gyroid_generation(analysis_file, cfg)
% Fresh, minimal gyroid generator with parallel φ evaluation.
% Uses: infill.gyroid_phi_at_point, infill.gyroid_phi_trilinear, infill.eval_gyroid_layer

fprintf('--- Gyroid generation: loading analysis ---\n');
D = load(analysis_file);

if nargin < 2 || isempty(cfg)
    cfg = struct();
end

% ---------------- User parameters ----------------
w          = 0.40;      % mm, extrusion width
rho_eps    = 0.02;      % clamp for L=2.432*w/rho
rho_min    = 0.20;      % min printable density
rho_max    = 0.80;      % max printable density
% Display / UI scale for clarity (relative "infill %" not used directly for L)
ui_infill_min = 0.20;   % 20% for lowest HU in-mask
ui_infill_max = 0.80;   % 80% for highest HU in-mask

preview_step = 0.25;     % mm voxel for 3D preview
xy_step      = 0.25;    % mm sampling for layer contours
layer_height = 0.20;    % mm layer height for example slicing

do_preview_3d    = get_cfg_flag(cfg, 'gyroidPreview3d', true);
do_single_slice  = get_cfg_flag(cfg, 'gyroidSingleSlice', false); % build one mid-slice contour as a sanity check
do_zoned_preview = get_cfg_flag(cfg, 'gyroidZonedPreview3d', true);   % show 3D preview for K-means (zoned) gyroid
% -------------------------------------------------
% Optional: spatial phase shift to move the gyroid pattern in space (mm)
phase_mm = [1.5, 0.0, 0.0];  % tweak ±1–3 mm if a channel lines up with a thin region

origin  = D.ds.origin(:)';         % mm
spacing = D.ds.spacing(:)';        % mm
mask    = D.scaphoid_mask > 0;

% -------- Build density fields (robust) ----------
% Continuous: map gradient to [rho_min, rho_max], clamp inside mask
% (a) UI "infill %" for display (20–80% mapped from gradient 0..1)
infill_pct_cont = nan(size(D.gradient_map));           % store as fraction 0..1
infill_pct_cont(mask) = ui_infill_min + (ui_infill_max - ui_infill_min) * D.gradient_map(mask);

% (b) Actual printable rho used for gyroid
rho_cont = nan(size(D.gradient_map));
rho_cont(mask) = rho_min + (rho_max - rho_min) * ( (infill_pct_cont(mask) - ui_infill_min) / (ui_infill_max - ui_infill_min) );
rho_cont(mask) = max(rho_cont(mask), rho_min);  % guardrail

% ----- Zoned (optional): order K-means zones by median gradient (low->high) -----
k = double(D.k);
Zraw = D.zone_map_kmeans_smoothed;

% Compute median gradient per raw label
lab_stats = nan(1,k);
for i = 1:k
    lab_stats(i) = median(D.gradient_map(Zraw == i), 'omitnan');
end
[~, order] = sort(lab_stats, 'ascend');          % low gradient first
old2rank = zeros(1,k); old2rank(order) = 1:k;    % map old label -> rank 1..k

% Build rank_map where values are 1..k in sorted order
rank_map = zeros(size(Zraw), 'like', Zraw);
for i = 1:k
    rank_map(Zraw == i) = old2rank(i);
end

% Choose display % per (ordered) zone
zone_pct_vals = linspace(ui_infill_min, ui_infill_max, max(1, k));

% (a) Display "%", piecewise-constant per ordered zone
infill_pct_zoned = nan(size(rank_map));
rho_zoned        = nan(size(rank_map));
for r = 1:numel(zone_pct_vals)
    tgt_pct = zone_pct_vals(r);
    tgt_rho = rho_min + (rho_max - rho_min) * ((tgt_pct - ui_infill_min) / (ui_infill_max - ui_infill_min));
    sel = (rank_map == r);
    infill_pct_zoned(sel) = tgt_pct;
    rho_zoned(sel)        = tgt_rho;
end

% Fill any unlabeled in-mask holes conservatively (min % / min rho)
holes_pct = mask & isnan(infill_pct_zoned);
infill_pct_zoned(holes_pct) = ui_infill_min;

holes_rho = mask & isnan(rho_zoned);
rho_zoned(holes_rho) = rho_min;

rho_zoned(mask & (rho_zoned <= 0)) = rho_min;
rho_zoned(~mask) = NaN;


% (c) OPTIONAL feathering (mm) to soften sharp zone transitions before φ eval
% --- Feathering with NaN-safe mask handling and clamping ---
zones_feather_mm = 0.8;   % re-enable after optional test
if zones_feather_mm > 0
    sig = max(eps, zones_feather_mm ./ spacing); % [σx σy σz] in voxels

    R = rho_zoned;
    R(~mask) = 0;                          % zero outside to prevent NaN bleed
    Rb = imgaussfilt3(R, sig);
    Mb = imgaussfilt3(double(mask), sig);

    rho_zoned = Rb ./ max(Mb, eps);        % normalized masked blur
    rho_zoned(~mask) = NaN;

    % Clamp and sanitize inside the mask
    inb = mask;
    rho_zoned(inb) = min(max(rho_zoned(inb), rho_min), rho_max);
    bad = inb & ~isfinite(rho_zoned);
    rho_zoned(bad) = rho_min;

    fprintf('NaNs in rho_zoned (in-mask): %d\n', nnz(mask & isnan(rho_zoned)));
end


% -------- Diagnostics (quick) --------------------
stats = @(x) [min(x), median(x), max(x)];

rc = rho_cont(mask);       rz = rho_zoned(mask);
pc = infill_pct_cont(mask);pz = infill_pct_zoned(mask);

Lc = 2.432*w./max(rc, rho_eps); 
Lz = 2.432*w./max(rz, rho_eps);

fprintf('UI %% (cont) min/med/max: %.0f / %.0f / %.0f\n', 100*min(pc), 100*median(pc), 100*max(pc));
fprintf('rho_cont    min/med/max: %.3f / %.3f / %.3f\n', stats(rc));
fprintf('L_cont      min/med/max: %.2f / %.2f / %.2f mm\n', stats(Lc));

fprintf('UI %% (zone) min/med/max: %.0f / %.0f / %.0f\n', 100*min(pz), 100*median(pz), 100*max(pz));
fprintf('rho_zoned   min/med/max: %.3f / %.3f / %.3f\n', stats(rz));
fprintf('L_zoned     min/med/max: %.2f / %.2f / %.2f mm\n', stats(Lz));

% -------- Preview grid & mask resampling ----------
% Axis-aligned, grid-aligned resample (fast & robust)
sz = size(mask);
xv0 = origin(1):spacing(1):origin(1) + spacing(1)*(sz(1)-1);
yv0 = origin(2):spacing(2):origin(2) + spacing(2)*(sz(2)-1);
zv0 = origin(3):spacing(3):origin(3) + spacing(3)*(sz(3)-1);

world_max = [xv0(end), yv0(end), zv0(end)];
xv = origin(1):preview_step:world_max(1);
yv = origin(2):preview_step:world_max(2);
zv = origin(3):preview_step:world_max(3);
[X,Y,Z] = meshgrid(xv, yv, zv);

% Build explicit source grids in ndgrid/meshgrid convention:
[X0, Y0, Z0] = meshgrid(xv0, yv0, zv0);  % sizes: [numel(yv0) x numel(xv0) x numel(zv0)]
V = permute(double(mask), [2 1 3]);      % make V match [length(y) x length(x) x length(z)]
mask_preview = interp3(X0, Y0, Z0, V, X, Y, Z, 'nearest', 0) > 0.5;

% -------- Parallel 3D φ evaluation (continuous) ---
if do_preview_3d
    fprintf('Evaluating 3D φ (continuous) on preview grid...\n');

    if isempty(gcp('nocreate'))
        try, parpool; catch, end
    end

    Phi3D = nan(size(X));
    idx = find(mask_preview);                         % evaluate only inside mask
    Xq = X(idx); Yq = Y(idx); Zq = Z(idx);

    phi_vals = zeros(numel(idx),1);
    parfor i = 1:numel(idx)
        p = [Xq(i), Yq(i), Zq(i)] + phase_mm;
        phi_vals(i) = infill.gyroid_phi_trilinear(p, rho_cont, origin, spacing, w, rho_eps);
    end
    Phi3D(idx) = phi_vals;
    % ---- Crop to the tight bounding box of mask_preview ----
    [iy, ix, iz] = ind2sub(size(mask_preview), find(mask_preview));
    iy0 = min(iy); iy1 = max(iy);
    ix0 = min(ix); ix1 = max(ix);
    iz0 = min(iz); iz1 = max(iz);

    Xc   = X(  iy0:iy1, ix0:ix1, iz0:iz1 );
    Yc   = Y(  iy0:iy1, ix0:ix1, iz0:iz1 );
    Zc   = Z(  iy0:iy1, ix0:ix1, iz0:iz1 );
    PhiC = Phi3D(iy0:iy1, ix0:ix1, iz0:iz1);

    fv = isosurface(Xc, Yc, Zc, PhiC, 0);
    if ~isempty(fv.vertices)
        figure('Color','w','Name','Continuous Gyroid - 3D preview');
        p = patch(fv); p.FaceColor = [0.2, 0.6, 1.0]; p.EdgeColor = 'none';
        isonormals(Xc, Yc, Zc, PhiC, p);
        daspect([1 1 1]); view(3); camlight headlight; lighting gouraud; grid on;
        xlabel('X (mm)'); ylabel('Y (mm)'); zlabel('Z (mm)');
        title('Continuous Gyroid (φ=0 isosurface)');
        % Context: add the scaphoid shell as a translucent surface
        fv_mask = isosurface(X0, Y0, Z0, V, 0.5);
        if ~isempty(fv_mask.vertices)
            hold on;
            pShell = patch(fv_mask);
            set(pShell, 'FaceColor', [0.85 0.85 0.85], 'EdgeColor', 'none', 'FaceAlpha', 0.12);
        end

% Optional: limit the axes to the same cropped region used for the gyroid preview
xlim([min(Xc(:)) max(Xc(:))]);
ylim([min(Yc(:)) max(Yc(:))]);
zlim([min(Zc(:)) max(Zc(:))]);
        if ~isempty(fv_mask.vertices)
            hold on;
            pShell = patch(fv_mask);
            set(pShell, 'FaceColor', [0.85 0.85 0.85], 'EdgeColor', 'none', 'FaceAlpha', 0.12);
        end
    end
    % -------- Parallel 3D φ evaluation (zoned K-means) ---
if do_preview_3d && do_zoned_preview
    fprintf('Evaluating 3D φ (zoned) on preview grid...\n');

    % Evaluate only inside the same preview mask
    Phi3D_z = nan(size(X));
    idx = find(mask_preview);
    Xq = X(idx); Yq = Y(idx); Zq = Z(idx);

    phi_vals_z = zeros(numel(idx),1);
    parfor i = 1:numel(idx)
        p = [Xq(i), Yq(i), Zq(i)] + phase_mm;
        phi_vals_z(i) = infill.gyroid_phi_trilinear(p, rho_zoned, origin, spacing, w, rho_eps);
    end
    Phi3D_z(idx) = phi_vals_z;

    % ---- Crop to the tight bounding box of mask_preview ----
    [iy, ix, iz] = ind2sub(size(mask_preview), find(mask_preview));
    iy0 = min(iy); iy1 = max(iy);
    ix0 = min(ix); ix1 = max(ix);
    iz0 = min(iz); iz1 = max(iz);

    Xc   = X(  iy0:iy1, ix0:ix1, iz0:iz1 );
    Yc   = Y(  iy0:iy1, ix0:ix1, iz0:iz1 );
    Zc   = Z(  iy0:iy1, ix0:ix1, iz0:iz1 );
    PhiC = Phi3D_z(iy0:iy1, ix0:ix1, iz0:iz1);

    % Compute isosurface on the cropped grid
    fv = isosurface(Xc, Yc, Zc, PhiC, 0);
    if ~isempty(fv.vertices)
        figure('Color','w','Name','Zoned (K-means) Gyroid - 3D preview');
        p = patch(fv); p.FaceColor = [0.10, 0.75, 0.40]; p.EdgeColor = 'none';
        isonormals(Xc, Yc, Zc, PhiC, p);
        daspect([1 1 1]); view(3); camlight headlight; lighting gouraud; grid on;
        xlabel('X (mm)'); ylabel('Y (mm)'); zlabel('Z (mm)');
        title('Zoned (K-means) Gyroid (φ=0 isosurface)');
        % Context: add the scaphoid shell as a translucent surface
        fv_mask = isosurface(X0, Y0, Z0, V, 0.5);
        if ~isempty(fv_mask.vertices)
            hold on;
            pShell = patch(fv_mask);
            set(pShell, 'FaceColor', [0.85 0.85 0.85], 'EdgeColor', 'none', 'FaceAlpha', 0.12);
        end
        if ~isempty(fv_mask.vertices)
            hold on;
            pShell = patch(fv_mask);
            set(pShell, 'FaceColor', [0.85 0.85 0.85], 'EdgeColor', 'none', 'FaceAlpha', 0.12);
        end

    end
end

end

% -------- One layer contour (continuous) ----------
if do_single_slice
    z_mid = origin(3) + spacing(3)*(floor(sz(3)/2));
    fprintf('Evaluating one layer at Z=%.2f mm (continuous)...\n', z_mid);
    [C, xv2, yv2, Phi2] = infill.eval_gyroid_layer(z_mid, rho_cont, origin, spacing, w, rho_eps, xy_step);

    % Build a 2D mask on the SAME (xv2,yv2) grid at this Z to clamp the field
    [X2, Y2] = meshgrid(xv2, yv2);

    % Source axes for the original grid
    sz  = size(mask);
    xv0 = origin(1):spacing(1):origin(1) + spacing(1)*(sz(1)-1);
    yv0 = origin(2):spacing(2):origin(2) + spacing(2)*(sz(2)-1);

    % Pick the nearest voxel slice to z_mid
    kz = round( (z_mid - origin(3)) / spacing(3) ) + 1;
    kz = max(1, min(sz(3), kz));

    % Take that binary mask slice and resample it to (xv2,yv2)
    % Note: mask is [Nx x Ny x Nz]; for interp2 with (yv0,xv0) axes, transpose.
    mask_slice = double( mask(:,:,kz) )';
    mask2D = interp2( xv0, yv0, mask_slice, X2, Y2, 'nearest', 0 ) > 0.5;

    % Clamp: outside the mask slice, force Phi2 to NaN so contours cannot draw there
    Phi2(~mask2D) = NaN;

    % Plot
    % Plot base image (hidden outside mask via AlphaData)
    figure('Color','w','Name','Continuous Gyroid - layer');
    hImg = imagesc(xv2, yv2, Phi2); axis image xy; colormap(jet); hold on;
    set(hImg, 'AlphaData', double(~isnan(Phi2)));   % hide background heatmap

    % OPTIONAL: uncomment to visualize the mask boundary
    % contour(xv2, yv2, double(mask2D), [0.5 0.5], 'y--', 'LineWidth', 1);

    % Compute φ=0 contours (no draw), then keep only segments inside mask2D
    Cmat = contourc(xv2, yv2, Phi2, [0 0]);
    if isempty(Cmat), return; end  % nothing to draw on this slice

    col = 1;
    while col < size(Cmat,2)
        % header
        % lvl = Cmat(1,col);  % always 0 here
        n   = Cmat(2,col);

        % segment points
        seg = Cmat(:, col+1:col+n);  % [2 x n]

        % inside-mask filter
        inside = interp2(xv2, yv2, double(mask2D), seg(1,:), seg(2,:), 'nearest', 0) > 0.5;
        if nnz(inside) >= 0.9 * numel(inside)
            plot(seg(1,:), seg(2,:), 'c-', 'LineWidth', 1.2);
        end

        col = col + n + 1;
    end

title(sprintf('φ=0 contours @ Z=%.2f mm (continuous)', z_mid));
colorbar; xlabel('X (mm)'); ylabel('Y (mm)');

end

% -------- Optional: Zoned previews (safe clamp) ---
% (Enable if you want to inspect zoned behavior now)
do_zoned = get_cfg_flag(cfg, 'gyroidZonedSlice', false);
if do_zoned
    fprintf('Evaluating one layer (zoned)...\n');
    z_mid = origin(3) + spacing(3)*(floor(sz(3)/2));
    [Cz, xvz, yvz, Phiz] = infill.eval_gyroid_layer(z_mid, rho_zoned, origin, spacing, w, rho_eps, xy_step);
    figure('Color','w','Name','Zoned Gyroid - layer');
    imagesc(xvz, yvz, Phiz); axis image xy; colormap(jet); hold on;
    contour(xvz, yvz, Phiz, [0 0], 'k-', 'LineWidth', 1.2);
    title(sprintf('φ=0 contours @ Z=%.2f mm (zoned)', z_mid));
    colorbar; xlabel('X (mm)'); ylabel('Y (mm)');
end

fprintf('Gyroid generation done.\n');
end

function value = get_cfg_flag(cfg, fieldName, defaultValue)
    value = defaultValue;
    if ~isstruct(cfg) || ~isfield(cfg, 'features')
        return;
    end
    if isfield(cfg.features, fieldName)
        candidate = cfg.features.(fieldName);
        if islogical(candidate) && isscalar(candidate)
            value = candidate;
        else
            error('cfg.features.%s must be a logical scalar.', fieldName);
        end
    end
end
