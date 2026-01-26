function [mask, qc_data, ds] = run_segmentation(ds, opts)
% RUN_SEGMENTATION performs a robust, adaptive scaphoid segmentation
% with seed sanity checks, core/allow fallbacks, border-safe cleanup,
% and a relaxed retry path if the first pass finds air-like surfaces.

% --- Preprocessing Steps ---
if opts.AirRecalibrate
    [ds.HU, ds.calibration] = airRecalibrateIfNeeded(ds.HU, opts.AirTargetHU, opts.AirMaxOffsetHU);
    fprintf('[Cal] Air mode=%.1f → applied offset %+g HU (cap=%g)\n', ...
        ds.calibration.airMode, ds.calibration.offsetHU, opts.AirMaxOffsetHU);
else
    ds.calibration = struct('airMode', NaN, 'offsetHU', 0);
end

% Build specimen crop (adaptive margin if specimen touches image border)
[crop, HUc] = buildSpecimenCrop(ds.HU, ds.spacing);

% Markers / artifact weights
[markerMask_c, artifactW_c] = markerAndArtifactMaps(HUc, opts.MarkerRangeHU, opts.ArtifactSigmaMM, ds.spacing);
fprintf('[Markers] Range=[%g,%g] HU; sigma=%.1f mm (cropped)\n', opts.MarkerRangeHU(1), opts.MarkerRangeHU(2), opts.ArtifactSigmaMM);

ds.markerMask     = utils.paste_crop_volume(logical(markerMask_c), crop, size(ds.HU));
ds.artifactWeight = utils.paste_crop_volume(artifactW_c,        crop, size(ds.HU));

% Optional sheetness prior
S_c = [];
if opts.UseSheetnessPrior
    fprintf('[Sheetness] sigmas(mm)=%s, weight=%.2f (expensive multi-scale filter)\n', ...
        mat2str(opts.SheetnessSigmasMM), opts.SheetnessWeight);
    S_c = computeSheetness3D(HUc, ds.spacing, opts.SheetnessSigmasMM, ...
                             [opts.SheetnessAlpha, opts.SheetnessBeta, opts.SheetnessC], ...
                             opts.SheetnessBrightStructures, ...
                             'DownsampleFactor', opts.SheetnessDownsampleFactor, ...
                             'MaxSigmaMM', opts.SheetnessMaxSigmaMM, ...
                             'SigmaSpacingCap', opts.SheetnessSigmaSpacingCap);
end

% Seed (with quality guard + auto-reseat if HU too low)
[seedMask_c, seed_ijk_c, ~] = proposeScaphoidSeed(HUc, markerMask_c);
fprintf('[Seed] Proposed (cropped) at [i j k]=[%d %d %d]\n', seed_ijk_c);

[seedMask_c, seed_ijk_c, reseatNote] = reseatSeedIfNeeded(HUc, markerMask_c, seedMask_c, seed_ijk_c, ds.spacing);
if ~isempty(reseatNote), fprintf('[Seed] %s\n', reseatNote); end

% ---------- FIRST PASS: adaptive constraint sweep ----------
fprintf('[Segment] Running adaptive constraint sweep...\n');

constraint_sets = { ...
    struct('name', 'Loose',  'hu_offset', 110, 'grad_prctile', 85), ...
    struct('name', 'Medium', 'hu_offset', 135, 'grad_prctile', 89), ...
    struct('name', 'Strict', 'hu_offset', 160, 'grad_prctile', 92)  ...
};

% Stats for thresholds
softMed = median(double(HUc(HUc > -300)), 'omitnan'); if ~isfinite(softMed), softMed = -300; end
vals    = double(HUc(~markerMask_c & HUc > -300 & HUc < 2000)); if isempty(vals), vals = double(HUc(isfinite(HUc))); end
G       = imgradient3(HUc);

% Define core and allow, with fallback if core is empty
core_thr = max(280, min(700, prctile(vals, 94)));
core     = HUc > core_thr;
useCoreInAllow = any(core(:));
if ~useCoreInAllow
    % Relax core threshold progressively until we get a minimal core
    for p = [92 90 88 86 84]
        core_alt = HUc > max(220, min(650, prctile(vals, p)));
        if nnz(core_alt) > 2000
            core = core_alt; useCoreInAllow = true;
            fprintf('[Segment] Core empty at p94 — relaxed to p%d (nnz=%d)\n', p, nnz(core));
            break;
        end
    end
end

candidate_masks = {};
D_c = []; % keep one distance map for QC

for i = 1:length(constraint_sets)
    C = constraint_sets{i};
    HU_ALLOW_MIN = max(70,  min(220, softMed + C.hu_offset));
    gThr         = prctile(G(:), C.grad_prctile);

    maskR = (HUc > HU_ALLOW_MIN) | (G > gThr);
    if useCoreInAllow
        allow = imreconstruct(core, maskR) & utils.apply_crop(crop.specimen, crop);
    else
        allow = maskR & utils.apply_crop(crop.specimen, crop); % fallback: don't depend on core
    end

    Wc = buildWeights(HUc, seedMask_c, artifactW_c, opts.FMM_AlphaBetaGamma);
    if opts.UseSheetnessPrior && ~isempty(S_c)
        Wc = Wc .* (1 + opts.SheetnessWeight * S_c);
    end
    Wc(~allow) = eps;

    % Robust normalize
    Lo = prctile(Wc(:), 1); Hi = prctile(Wc(:), 99);
    Wc = min(max((Wc - Lo) / max(eps, (Hi - Lo)), 0), 1);
    Wc = max(Wc, eps);

    [mask_candidate, D_c_candidate] = segmentScaphoidFMM(Wc, seedMask_c, HUc, opts);
    mask_candidate = mask_candidate & utils.apply_crop(crop.specimen, crop);

    % Border-safe clearborder: only if small fraction touches border
    mask_candidate = clearborder_if_safe(mask_candidate, 0.05);  % 5% threshold

    mask_candidate = utils.keep_largest_component_3d(mask_candidate);

    candidate_masks{i} = mask_candidate;
    if i == 1, D_c = D_c_candidate; end

    fprintf('[Segment] Candidate %-6s | core=%d allow=%d cand=%d\n', ...
            constraint_sets{i}.name, nnz(core), nnz(allow), nnz(mask_candidate));
end

% Score candidates by 90th percentile of perimeter HU
scores = zeros(length(candidate_masks), 1);
for i = 1:length(candidate_masks)
    m = candidate_masks{i};
    if ~any(m(:)), scores(i) = -Inf; continue; end
    perim = bwperim(m, 26);
    surface_hu = HUc(perim);
    scores(i) = prctile(surface_hu, 90);
end

[best_score, best_idx] = max(scores);
mask_c = candidate_masks{best_idx};
fprintf('[Segment] Best result from ''%s'' constraints (Score=%.1f HU).\n', ...
    constraint_sets{best_idx}.name, best_score);
% Flag low-contrast scans (perimeter 90th percentile < ~350 HU)
lowContrast = best_score < 350;
fprintf('[Segment] Low-contrast mode: %d (perim90=%.1f HU)\n', lowContrast, best_score);

% ---------- RELAXED RETRY if surface looks like air/soft ----------
if (~any(mask_c(:))) || best_score < 150
    fprintf('[Segment] Surface HU looks too low (%.1f). Retrying with relaxed constraints...\n', best_score);
    constraint_sets_relaxed = { ...
        struct('name', 'UltraLooseA', 'hu_offset', 80, 'grad_prctile', 80), ...
        struct('name', 'UltraLooseB', 'hu_offset', 95, 'grad_prctile', 83), ...
        struct('name', 'UltraLooseC', 'hu_offset', 110, 'grad_prctile', 85) ...
    };

    candidate_masks2 = {};
    for i = 1:length(constraint_sets_relaxed)
        C = constraint_sets_relaxed{i};
        HU_ALLOW_MIN = max(50, min(200, softMed + C.hu_offset));
        gThr         = prctile(G(:), C.grad_prctile);
        maskR        = (HUc > HU_ALLOW_MIN) | (G > gThr);

        Wc = buildWeights(HUc, seedMask_c, artifactW_c, opts.FMM_AlphaBetaGamma);
        if opts.UseSheetnessPrior && ~isempty(S_c)
            Wc = Wc .* (1 + opts.SheetnessWeight * S_c);
        end
        % In the relaxed retry, don't rely on core; just gate by maskR
        Wc(~(maskR & utils.apply_crop(crop.specimen, crop))) = eps;

        Lo = prctile(Wc(:), 1); Hi = prctile(Wc(:), 99);
        Wc = min(max((Wc - Lo) / max(eps, (Hi - Lo)), 0), 1);
        Wc = max(Wc, eps);

        [m2, ~] = segmentScaphoidFMM(Wc, seedMask_c, HUc, opts);
        m2 = m2 & utils.apply_crop(crop.specimen, crop);
        m2 = clearborder_if_safe(m2, 0.05);
        m2 = utils.keep_largest_component_3d(m2);
        candidate_masks2{i} = m2;

        % score
        if any(m2(:))
            per = bwperim(m2, 26);
            scores2(i,1) = prctile(HUc(per), 90);
        else
            scores2(i,1) = -Inf;
        end
        fprintf('[Segment] Retry candidate %-11s | cand=%d | score=%.1f\n', ...
                constraint_sets_relaxed{i}.name, nnz(m2), scores2(i));
    end

    [score2_best, idx2] = max(scores2);
    if score2_best > best_score || ~any(mask_c(:))
        mask_c     = candidate_masks2{idx2};
        best_score = score2_best;
        fprintf('[Segment] Retry winner ''%s'' (Score=%.1f HU).\n', constraint_sets_relaxed{idx2}.name, best_score);
    end
end

% ---------- Last-ditch simple grow if still air-like ----------
if (~any(mask_c(:))) || best_score < 150
    fprintf('[Segment] Fallback: simple region-grow from seed.\n');
    T_lo = max(130, min(240, softMed + 140));
    seedDil = imdilate(seedMask_c, strel('sphere', 1));
    allowed = (HUc > T_lo) & ~imdilate(markerMask_c, strel('sphere', 2));
    mask_c  = imreconstruct(seedDil, allowed);
    mask_c  = utils.keep_largest_component_3d(mask_c);
    mask_c  = imclose(mask_c, strel('sphere', 1));
    mask_c  = imfill(mask_c, 'holes');
end

% --- Final Post-Processing on the Winning Mask ---
mask_c = postprocessMask(mask_c, markerMask_c);

% Optional shell sealing
if opts.UseShellSealing
    mask_c = sealOuterShell(mask_c, ds.spacing, opts);
end

% Boundary-band cling with interior protection (unchanged logic)
voxmm = mean(ds.spacing);
D_in  = bwdist(~mask_c) * voxmm;
deep_interior = D_in >= 1.0;
band = mask_c & ~deep_interior;

softMed = median(double(HUc(HUc > -300)), 'omitnan'); if ~isfinite(softMed), softMed = -300; end
vals = double(HUc(~markerMask_c & HUc > -300 & HUc < 2000)); if isempty(vals), vals = double(HUc(isfinite(HUc))); end
core_thr = max(240, min(650, prctile(vals, 92)));
core_seed = band & (HUc > core_thr);
HU_SUPPORT_MIN = max(60,  min(180, softMed + 90));
gThr = prctile(G(:), 80);
support = band & ( (HUc > HU_SUPPORT_MIN) | (G > gThr) );
cling_band = imreconstruct(core_seed, support);
mask_c = deep_interior | cling_band;
mask_c = utils.keep_largest_component_3d(mask_c);
mask_c = imclose(mask_c, strel('sphere',1));
mask_c = imfill(mask_c,'holes');

% Edge-backed perimeter prune (unchanged thresholds)
perim = bwperim(mask_c, 26);
band1 = imdilate(perim, strel('sphere',1));
T_hu  = max(160, min(340, softMed + 190));
T_g   = prctile(G(:), 70);
kill2 = band1 & (double(HUc) < T_hu) & (G < T_g);
if any(kill2(:))
   mask_c(kill2) = false;
   mask_c = utils.keep_largest_component_3d(mask_c);
   mask_c = imclose(mask_c, strel('sphere',1));
   mask_c = imfill(mask_c,'holes');
end

% Conservative boundary carve (unchanged)
perim  = bwperim(mask_c, 26);
band1  = imdilate(perim, strel('sphere', 1));
outer1 = imdilate(mask_c, strel('sphere', 1)) & ~mask_c;
protected_core = imdilate(core, strel('sphere', 2));
T_hi = max(170, min(360, softMed + 210));
T_lo = T_hi - 80;
airNear = outer1 & imdilate(HUc < -300, strel('sphere',1));
plateOK = true(size(HUc));
if exist('S_c','var') && ~isempty(S_c)
   plateOK = S_c < 0.15;
end
kill = band1 & (double(HUc) < T_lo) & imdilate(airNear, strel('sphere',1)) & ~protected_core & plateOK;
if any(kill(:))
   mask_c(kill) = false;
   mask_c = utils.keep_largest_component_3d(mask_c);
   mask_c = imclose(mask_c, strel('sphere', 1));
   mask_c = imfill(mask_c, 'holes');
end

% Final boundary carve
band = imdilate(bwperim(mask_c, 26), strel('sphere', 1));
HU_CARVE_FLOOR = max(180, min(400, softMed + 220));
kill = band & (double(HUc) < HU_CARVE_FLOOR);
if any(kill(:))
    mask_c(kill) = false;
end

% Consolidated final cleanup
mask_c = utils.keep_largest_component_3d(mask_c);
se_open = strel('sphere', 1);
mask_c = imopen(mask_c, se_open);
mask_c = utils.keep_largest_component_3d(mask_c);
se_close = strel('sphere', 1);
mask_c = imclose(mask_c, se_close);
mask_c = imfill(mask_c, 'holes');

% --- Uncrop and Finalize Outputs ---
mask = utils.paste_crop_mask(mask_c, crop, size(ds.HU));
seed_ijk_full = [crop.rRange(1), crop.cRange(1), crop.sRange(1)] + seed_ijk_c - 1;
if isempty(D_c), D_c = zeros(size(HUc),'like',HUc); end
D_full = utils.paste_crop_volume(D_c, crop, size(ds.HU));
seedMask_full = utils.paste_crop_mask(seedMask_c, crop, size(ds.HU));

qc_data = struct();
qc_data.seed_full = seed_ijk_full;
qc_data.seed_cropped = seed_ijk_c;
qc_data.dist_map_full = D_full;
qc_data.seed_mask_full = seedMask_full;
qc_data.crop = crop;

end



% =========================================================================
% =================== ALL LOCAL HELPER FUNCTIONS GO HERE ==================
% =========================================================================
% (Paste all the previous local helper functions from your old version of this file here)
% function [HU,cal] = airRecalibrateIfNeeded(...) ... end
% function [markerMask, A] = markerAndArtifactMaps(...) ... end
% etc...


% =========================================================================
% =================== LOCAL HELPER FUNCTIONS ==============================
% =========================================================================

% ----------------------- Air recalibration -------------------------------
function [HU,cal] = airRecalibrateIfNeeded(HU, targetHU, maxOffset)
% Estimate air mode from a 5-voxel border shell; adjust toward targetHU.
sz = size(HU); m = false(sz);
m(1:5,:,:)=1; m(end-4:end,:,:)=1; m(:,1:5,:)=1; m(:,end-4:end,:)=1; m(:,:,1:5)=1; m(:,:,end-4:end)=1;
air = HU(m);
air = air(isfinite(air));
edges = -2000:5:200;
if isempty(air), cal = struct('airMode',NaN,'offsetHU',0); return; end
h = histcounts(air, edges);
[~,i] = max(h); airMode = mean([edges(i), edges(i+1)]);
rawOffset = targetHU - airMode;
offset = sign(rawOffset)*min(abs(rawOffset), maxOffset);
if abs(offset) > 10
 HU = HU + offset;
else
 offset = 0;
end
cal = struct('airMode',airMode,'offsetHU',offset);
end

% ----------------------- Markers & artifacts -----------------------------
function [markerMask, A] = markerAndArtifactMaps(HU, rng, sigma_mm, spacing)
if nargin<2 || isempty(rng), rng = [200 700]; end
if nargin<3 || isempty(sigma_mm), sigma_mm = 3; end
lead  = HU > 1200;
flags = (HU>=rng(1) & HU<=rng(2)) & imdilate(lead, strel('sphere',2));  % only HU-flag near lead
markerMask = lead | flags;
Dvox = bwdist(markerMask);
d_mm = Dvox * mean(spacing);          % approximate voxel->mm
A = exp(-(d_mm/sigma_mm).^2);         % [0..1]
end

% ----------------------- Seed proposal -----------------------------------
function [seedMask, seed_ijk, coarseMask] = proposeScaphoidSeed(HU, markerMask)
sz = size(HU);
bw = HU > -300;
bw = bw & ~imdilate(markerMask, strel('sphere',2));
border = false(sz); border([1 end],:,:) = true; border(:,[1 end],:) = true; border(:,:,[1 end]) = true;
touchBorder = imdilate(border, strel('sphere',1));
bw = bw & ~touchBorder;
bw = bwareaopen(bw, 200);
CC = bwconncomp(bw, 26);
if CC.NumObjects==0
 safe = ~imdilate(markerMask, strel('sphere',2)) & ~touchBorder;
 HU2 = HU; HU2(~safe) = -Inf;
 [~,idx] = max(HU2(:)); [i,j,k] = ind2sub(sz, idx);
 seed_ijk = [i j k];
 seedMask = false(sz); seedMask(i,j,k) = true;
 coarseMask = false(sz);
 if isfinite(idx), coarseMask(idx) = true; end
 return;
end
scores = zeros(CC.NumObjects,1);
for n = 1:CC.NumObjects
 M = false(sz); M(CC.PixelIdxList{n}) = true;
 stats = regionprops3(M,'Volume','SurfaceArea','PrincipalAxisLength');
 V = double(stats.Volume(1));
 A = double(stats.SurfaceArea(1));
 pa = stats.PrincipalAxisLength(1,:);
 elong = max(pa)/max(1e-6,min(pa));
 sph = (pi^(1/3))*((6*max(V,eps))^(2/3))/max(A,eps);
 sph = max(0, min(1, sph));
 scores(n) = (0.6*sph + 0.4*(1/elong)) * log1p(V);
end
[~,kbest] = max(scores);
coarseMask = false(sz); coarseMask(CC.PixelIdxList{kbest}) = true;
Dm = bwdist(~coarseMask);
[~,idx] = max(Dm(:));
[i,j,k] = ind2sub(sz, idx);
i = max(4, min(sz(1)-3, i));
j = max(4, min(sz(2)-3, j));
k = max(4, min(sz(3)-3, k));
seed_ijk = [i j k];
seedMask = false(sz); seedMask(i,j,k) = true;
end
% ----------------------- Viewer (non-blocking) ---------------------------
function seed_ijk = showTriplanarAndMaybePick(HU, spacing, seed_ijk, overlays)
sz = size(HU); slAx = seed_ijk(3); slSag = seed_ijk(2); slCor = seed_ijk(1);
fig = figure('Name','Scaphoid QC (click axial to set seed)','Color','w');
ax1=subplot(2,2,1); title('Axial');
hAxial = imshow2D_img(ax1, squeeze(HU(:,:,slAx)), spacing(1:2));
hold(ax1,'on'); drawOverlays(ax1, overlays, 'axial', slAx);
sPlot = plot(ax1, seed_ijk(2), seed_ijk(1), 'r+', 'MarkerSize',12, 'LineWidth',2);
ax2=subplot(2,2,2); title('Sagittal');
imshow2D_img(ax2, squeeze(HU(:,slSag,:)), spacing([1 3]));
hold(ax2,'on'); drawOverlays(ax2, overlays, 'sagittal', slSag);
ax3=subplot(2,2,3); title('Coronal');
imshow2D_img(ax3, squeeze(HU(slCor,:,:))', spacing([2 3]));
hold(ax3,'on'); drawOverlays(ax3, overlays, 'coronal', slCor);
set(hAxial, 'HitTest','on', 'PickableParts','all', ...
 'ButtonDownFcn', @(h,evt)onClickAxial(evt));
uiwait(msgbox({'Click once on the AXIAL view to set the seed.', ...
            'Close this dialog when done.'}, 'Seed confirm','modal'));
 function onClickAxial(evt)
     cp = get(ax1,'CurrentPoint'); j = round(cp(1,1)); i = round(cp(1,2)); k = slAx;
     i = max(1,min(sz(1),i)); j = max(1,min(sz(2),j));
     seed_ijk = [i j k];
     set(sPlot, 'XData', j, 'YData', i);
 end
end
function showTriplanar(HU, spacing, seed_ijk, overlays)
figure('Name','Scaphoid QC (triplanar)','Color','w');
sz = size(HU);
slAx = seed_ijk(3); slSag = seed_ijk(2); slCor = seed_ijk(1);
ax1=subplot(2,2,1); title('Axial');
imshow2D_img(ax1, squeeze(HU(:,:,slAx)), spacing(1:2));
hold(ax1,'on'); drawOverlays(ax1, overlays, 'axial', slAx);
plot(ax1, seed_ijk(2), seed_ijk(1), 'r+', 'MarkerSize',12, 'LineWidth',2);
ax2=subplot(2,2,2); title('Sagittal');
imshow2D_img(ax2, squeeze(HU(:,slSag,:)), spacing([1 3]));
hold(ax2,'on'); drawOverlays(ax2, overlays, 'sagittal', slSag);
ax3=subplot(2,2,3); title('Coronal');
imshow2D_img(ax3, squeeze(HU(slCor,:,:))', spacing([2 3]));
hold(ax3,'on'); drawOverlays(ax3, overlays, 'coronal', slCor);
end
function hImg = imshow2D_img(ax, I, pix)
C=150; W=1000; clim=[C-W/2 C+W/2];
axes(ax); hImg = imagesc(I, clim);
axis(ax,'image','off'); colormap(ax, gray);
set(hImg,'HitTest','on','PickableParts','all');
end
function drawOverlays(ax, overlays, plane, sl)
if nargin<4, sl=1; end
hold(ax,'on');
if isfield(overlays,'coarseMask') && ~isempty(overlays.coarseMask)
 switch plane
     case 'axial',    M = overlays.coarseMask(:,:,sl);
     case 'sagittal', M = squeeze(overlays.coarseMask(:,sl,:));
     case 'coronal',  M = squeeze(overlays.coarseMask(sl,:,:))';
 end
 h=imshow(double(M)); set(h,'AlphaData',0.2);
end
if isfield(overlays,'markerMask') && ~isempty(overlays.markerMask)
 switch plane
     case 'axial',    M = overlays.markerMask(:,:,sl);
     case 'sagittal', M = squeeze(overlays.markerMask(:,sl,:));
     case 'coronal',  M = squeeze(overlays.markerMask(sl,:,:))';
 end
 B = bwperim(M); [r,c]=find(B); plot(ax, c, r, 'y.', 'MarkerSize',1);
end
if isfield(overlays,'artifactW') && ~isempty(overlays.artifactW)
 switch plane
     case 'axial',    A = overlays.artifactW(:,:,sl);
     case 'sagittal', A = squeeze(overlays.artifactW(:,sl,:));
     case 'coronal',  A = squeeze(overlays.artifactW(sl,:,:))';
 end
 A = mat2gray(A);
 h=imshow(A); set(h,'AlphaData',0.2);
end
end

% ----------------------- Segmentation core -------------------------------
function W = buildWeights(V, seedMask, A, abg)
alpha = abg(1); beta = abg(2); gamma = abg(3);
fgVals = V(seedMask); mu1=median(fgVals); s1=mad(fgVals,1)+50;
bg = imdilate(seedMask, strel('sphere',6)) & ~imdilate(seedMask, strel('sphere',2));
bgVals = V(bg); mu0=median(bgVals); s0=mad(bgVals,1)+50;
pBone = 1./(1+exp(-(V-mu1)./s1));
pTiss = 1./(1+exp(-(mu0-V)./s0));
edgeW = 1 - mat2gray(gradientweight(V));
dataW = mat2gray(pBone ./ (pTiss+eps));
base = alpha*dataW + beta*edgeW;
if gamma >= 0
 W = base ./ (1 + gamma*A);
else
 W = base .* (1 + (-gamma)*A);
end
W = max(W, eps);
end
function [BW, D] = segmentScaphoidFMM(W, seedMask, HU, opts)
th0 = min(max(0.01, mean(W(seedMask))*0.5), 0.99);
[~, D] = imsegfmm(W, seedMask, th0);
if opts.UseAdaptiveSweep
   tmin = max(eps, min(opts.FMMThresholdRange));
   tmax = min(0.999, max(opts.FMMThresholdRange));
   nT   = max(3, opts.FMMNumSteps);
   ths  = linspace(tmin, tmax, nT);
else
   ths  = 0.50;
end
switch lower(string(opts.BoundaryScoreImage))
   case "edge"
       Gsrc = 1 - mat2gray(gradientweight(HU));
   otherwise
       Gsrc = mat2gray(imgradient3(HU));
end
lambda     = opts.BoundaryScoreLambda;
needSingle = opts.BoundaryMustBeSingleComponent;
t_base = median(ths);
B_base = imsegfmm(W, seedMask, t_base);
if needSingle
   B_base = utils.keep_largest_component_3d(B_base);
end
if ~any(B_base(:))
   B_base = imdilate(seedMask, strel('sphere',1));
end
P0 = bwperim(B_base,26);
if any(P0(:))
   s_edge0 = mean(Gsrc(P0), 'omitnan');
else
   s_edge0 = -Inf;
end
Rin0 = imerode(B_base, strel('sphere',1));
hu_in0 = HU(Rin0);
if isempty(hu_in0) || all(~isfinite(hu_in0))
   medHU0 = -Inf;
else
   medHU0 = median(double(hu_in0), 'omitnan');
end
softMed = median(double(HU(HU > -300)), 'omitnan');
if ~isfinite(softMed), softMed = -300; end
HU_MIN = max(180, min(400, softMed + 220));
penHU0  = max(0, HU_MIN - medHU0) / HU_MIN;
penVol0 = 1e-7 * double(nnz(B_base));
best    = s_edge0 - lambda * penHU0 - penVol0;
BWbest  = B_base;
for t = ths
   t = min(max(t, eps), 0.999);
   B = imsegfmm(W, seedMask, t);
   if needSingle
       B = utils.keep_largest_component_3d(B);
   end
   if ~any(B(:)), continue; end
   P = bwperim(B,26);
   if ~any(P(:)), continue; end
   s_edge = mean(Gsrc(P), 'omitnan');
   Rin = imerode(B, strel('sphere',1));
   if ~any(Rin(:)), continue; end
   hu_in = HU(Rin);
   if isempty(hu_in) || all(~isfinite(hu_in))
       medHU = -Inf;
   else
       medHU = median(double(hu_in), 'omitnan');
   end
   penHU  = max(0, HU_MIN - medHU) / HU_MIN;
   penVol = 1e-7 * double(nnz(B));
   s      = s_edge - lambda * penHU - penVol;
   if s > best
       best  = s;
       BWbest = B;
   end
end
BW = BWbest;
end

function s = surfaceEdgeScore(B, G)
P = bwperim(B,26); vals = G(P); if isempty(vals), s=0; else, s=mean(vals); end
end

function BW = postprocessMask(BW, markerMask)
BW = imfill(BW,'holes');
BW = bwareaopen(BW, 500);
core = imerode(BW, strel('sphere',1));
BW = (BW & ~imdilate(markerMask, strel('sphere',1))) | core;
end

function BW = sealOuterShell(BW, spacing, opts)
rClose = max(1, round(opts.ShellCloseRadiusMM / max(mean(spacing),eps)));
SE = strel('sphere', rClose);
BW = imclose(BW, SE);
if opts.ShellFillHoles
 BW = imfill(BW, 'holes');
end
tmm = max(0, opts.ShellThicknessMM);
if tmm > 0
 D = bwdist(~BW) .* mean(spacing);
 BW = BW & (D <= tmm);
end
if opts.ShellKeepLargestOnly
 CC = bwconncomp(BW, 26);
 if CC.NumObjects >= 1
     [~,iMax] = max(cellfun(@numel, CC.PixelIdxList));
     Tmp = false(size(BW)); Tmp(CC.PixelIdxList{iMax})=true; BW = Tmp;
 end
end
BW = imclearborder(BW, 26);
end

function S = computeSheetness3D(HU, spacing, sigmasMM, frangiABC, brightStructures, varargin)
if nargin<5, brightStructures = true; end
if isempty(sigmasMM), S = zeros(size(HU),'like',HU); return; end
sigmasMM = sigmasMM(:).';
opts = struct('DownsampleFactor', 1, 'MaxSigmaMM', [], 'SigmaSpacingCap', []);
if ~isempty(varargin)
    for k = 1:2:numel(varargin)
        opts.(varargin{k}) = varargin{k+1};
    end
end
if ~isempty(opts.MaxSigmaMM)
    sigmasMM = sigmasMM(sigmasMM <= opts.MaxSigmaMM);
end
if ~isempty(opts.SigmaSpacingCap) && opts.SigmaSpacingCap > 0
    sigmasMM = sigmasMM(sigmasMM <= opts.SigmaSpacingCap * max(spacing));
end
if isempty(sigmasMM)
    S = zeros(size(HU),'like',HU);
    return;
end
V = double(HU);
if opts.DownsampleFactor > 1
    dsFactor = opts.DownsampleFactor;
    V_ds = imresize3(V, 1 / dsFactor, 'linear');
    spacing_ds = spacing * dsFactor;
    S_ds = computeSheetnessCore(V_ds, spacing_ds, sigmasMM, frangiABC, brightStructures);
    S = imresize3(S_ds, size(V), 'linear');
    return;
end
S = computeSheetnessCore(V, spacing, sigmasMM, frangiABC, brightStructures);
end
function S = computeSheetnessCore(V, spacing, sigmasMM, frangiABC, brightStructures)
[R,C,Sz] = size(V);
S_all = zeros(R,C,Sz,'like',V);
sx = spacing(1); sy = spacing(2); sz = spacing(3);
for s = sigmasMM
 sigx = s / max(sx,eps); sigy = s / max(sy,eps); sigz = s / max(sz,eps);
 kx = max(2, ceil(3*sigx)); ky = max(2, ceil(3*sigy)); kz = max(2, ceil(3*sigz));
 gx  = gauss1d(sigx, kx);   gy  = gauss1d(sigy, ky);   gz  = gauss1d(sigz, kz);
 g2x = gauss1d_second(sigx, kx); g2y = gauss1d_second(sigy, ky); g2z = gauss1d_second(sigz, kz);
 g1x = gauss1d_first(sigx, kx); g1y = gauss1d_first(sigy, ky); g1z = gauss1d_first(sigz, kz);
 Vx   = imfilter(V, gx', 'replicate','same');
 Vxx  = imfilter(V, g2x','replicate','same');
 Vx1  = imfilter(V, g1x','replicate','same');
 Vxy  = imfilter(Vx,  gy,  'replicate','same');
 Vx_y2 = imfilter(Vx,  g2y,'replicate','same');
 Vx_y1 = imfilter(Vx,  g1y,'replicate','same');
 Vxx_y = imfilter(Vxx, gy,  'replicate','same');
 Vx1_y = imfilter(Vx1, gy,  'replicate','same');
 Vx1_y1 = imfilter(Vx1, g1y,'replicate','same');
 Hxx = imfilter(Vxx_y, gz, 'replicate','same') * (s^2);
 Hyy = imfilter(Vx_y2, gz, 'replicate','same') * (s^2);
 Hzz = imfilter(Vxy,  g2z,'replicate','same') * (s^2);
 Hxy = imfilter(Vx1_y1, gz, 'replicate','same') * (s^2);
 Hxz = imfilter(Vx1_y, g1z,'replicate','same') * (s^2);
 Hyz = imfilter(Vx_y1, g1z,'replicate','same') * (s^2);
 [l1,l2,l3] = eigvals3sym(Hxx,Hyy,Hzz,Hxy,Hxz,Hyz);
 if brightStructures
     l2n = -l2; l3n = -l3;
 else
     l2n =  l2; l3n =  l3;
 end
 alpha = frangiABC(1); beta = frangiABC(2); C = frangiABC(3);
 Ra  = abs(l1) ./ (abs(l2)+eps);
 Rb  = abs(l2) ./ (abs(l3)+eps);
 Sst = sqrt(l1.^2 + l2.^2 + l3.^2);
 plate = exp(-(Ra.^2)/(2*alpha^2)) .* exp(-(Rb.^2)/(2*beta^2)) .* (1 - exp(-(Sst.^2)/(2*C^2)));
 plate(~isfinite(plate)) = 0;
 S_all = max(S_all, plate);
end
S = mat2gray(S_all);
 function g = gauss1d(sig, k)
     x = (-k:k);
     g = exp(-(x.^2)/(2*max(sig,eps)^2));
     g = g/sum(g);
 end
 function g = gauss1d_first(sig, k)
     x = (-k:k);
     g = -x .* exp(-(x.^2)/(2*max(sig,eps)^2));
     g = g / sum(abs(g)+eps);
 end
 function g = gauss1d_second(sig, k)
     x = (-k:k);
     s2 = max(sig,eps)^2;
     g = ((x.^2 - s2) .* exp(-(x.^2)/(2*s2))) / (s2^2);
     g = g - mean(g);
     g = g / (sum(abs(g))+eps);
 end
end

function [l1,l2,l3] = eigvals3sym(a,b,c,d,e,f)
sz = size(a);
T = a + b + c;
Q = (a.*b + a.*c + b.*c) - (d.^2 + e.^2 + f.^2);
R = a.*(b.*c - f.^2) - d.*(d.*c - e.*f) + e.*(d.*f - e.*b);
p = Q - (T.^2)/3;
q = (2*T.^3)/27 - (T.*Q)/3 + R;
phi = acos( max(-1,min(1, (-q./2) ./ sqrt(abs((p.^3)/27)+eps))) );
r = 2*sqrt(abs(p)/3);
lam1 = T/3 + r .* cos(phi/3);
lam2 = T/3 + r .* cos((phi+2*pi)/3);
lam3 = T/3 + r .* cos((phi+4*pi)/3);
[l1,l2,l3] = deal(lam1,lam2,lam3);
swap = abs(l1)>abs(l2); tmp=l1(swap); l1(swap)=l2(swap); l2(swap)=tmp;
swap = abs(l2)>abs(l3); tmp=l2(swap); l2(swap)=l3(swap); l3(swap)=tmp;
swap = abs(l1)>abs(l2); tmp=l1(swap); l1(swap)=l2(swap); l2(swap)=tmp;
end

function [crop, HUc] = buildSpecimenCrop(HU, spacing, varargin)
% Adaptive crop: if specimen touches the full image border, use a larger margin
nonAir = HU > -500;
rClose = max(1, round(0.6 / max(mean(spacing),eps)));
nonAir = imclose(nonAir, strel('sphere', rClose));
CC = bwconncomp(nonAir, 26);
if CC.NumObjects == 0
    crop = struct('rRange',1:size(HU,1), 'cRange',1:size(HU,2), 'sRange',1:size(HU,3), 'specimen', false(size(HU)));
    HUc  = HU;
    return;
end

[~,iMax] = max(cellfun(@numel, CC.PixelIdxList));
maskSpec = false(size(HU)); maskSpec(CC.PixelIdxList{iMax}) = true;
maskSpec = imerode(maskSpec, strel('sphere', 1));

[R,C,S] = size(HU);
props = regionprops3(maskSpec, 'BoundingBox');
bbox = props.BoundingBox(1, :); % [x y z width height depth]
rMin = max(1, floor(bbox(2) + 0.5));
cMin = max(1, floor(bbox(1) + 0.5));
sMin = max(1, floor(bbox(3) + 0.5));
rMax = min(R, ceil(bbox(2) + bbox(5) - 0.5));
cMax = min(C, ceil(bbox(1) + bbox(4) - 0.5));
sMax = min(S, ceil(bbox(3) + bbox(6) - 0.5));

% Detect if specimen touches the full image border
touches = any([rMin == 1, rMax == R, cMin == 1, cMax == C, sMin == 1, sMax == S]);
base_margin_mm = 2.5; edge_margin_mm = 8.0;
marg_mm = touches * edge_margin_mm + (~touches)*base_margin_mm;
marg    = max(3, round(marg_mm / max(mean(spacing), eps)));

r1 = max(1, rMin - marg); r2 = min(R, rMax + marg);
c1 = max(1, cMin - marg); c2 = min(C, cMax + marg);
s1 = max(1, sMin - marg); s2 = min(S, sMax + marg);

crop = struct('rRange', r1:r2, 'cRange', c1:c2, 'sRange', s1:s2, 'specimen', maskSpec);
HUc  = HU(crop.rRange, crop.cRange, crop.sRange);
end


function [bbox, Msub] = tightMaskBBox(M, marginVox)
if nargin<2, marginVox = 0; end
[R,C,S] = size(M);
[idxR, idxC, idxS] = ind2sub([R,C,S], find(M));
if isempty(idxR)
  bbox = struct('r', 1:R, 'c', 1:C, 's', 1:S);
  Msub = M;
  return;
end
r1 = max(1, min(idxR) - marginVox); r2 = min(R, max(idxR) + marginVox);
c1 = max(1, min(idxC) - marginVox); c2 = min(C, max(idxC) + marginVox);
s1 = max(1, min(idxS) - marginVox); s2 = min(S, max(idxS) + marginVox);
bbox = struct('r', r1:r2, 'c', c1:c2, 's', s1:s2);
Msub = M(bbox.r, bbox.c, bbox.s);
end

function BW = clearborder_if_safe(BW, fracThresh)
% Only clear components that touch the volume border if a small fraction
% of the mask touches the border (avoids wiping real anatomy pinned to crop)
if ~any(BW(:)), return; end
S = size(BW);
border = false(S);
border(1,:,:)   = true; border(end,:,:) = true;
border(:,1,:)   = true; border(:,end,:) = true;
border(:,:,1)   = true; border(:,:,end) = true;

onBorder = BW & border;
frac = nnz(onBorder) / max(1, nnz(BW));
if frac < fracThresh
    BW = imclearborder(BW, 26);
else
    % too much touches border — skip clearborder
end
end

function [seedMask, seed_ijk, note] = reseatSeedIfNeeded(HU, markerMask, seedMask, seed_ijk, spacing)
% Memory-safe reseat: avoids ndgrid, caps search window, and clamps
% voxel radius computed from spacing.

note = '';
if ~any(seedMask(:)), return; end

% If seed already looks like bone, keep it.
Hseed = double(HU(seed_ijk(1), seed_ijk(2), seed_ijk(3)));
softMed = median(double(HU(HU > -300)), 'omitnan'); if ~isfinite(softMed), softMed = -300; end
MIN_OK = max(200, min(450, softMed + 200));
if isfinite(Hseed) && Hseed >= MIN_OK
    return;
end

% Convert 10 mm to voxels, with sane clamps (avoid tiny spacing explosions)
R_mm = 10;
vox_mm = max(0.25, min(2.0, mean(spacing)));   % clamp to [0.25, 2.0] mm
rvox  = min(24, max(1, round(R_mm / vox_mm))); % cap radius to 24 vox

% Build small index ranges (no ndgrid, direct slicing)
R = size(HU,1); C = size(HU,2); S = size(HU,3);
r = max(1, seed_ijk(1)-rvox) : min(R, seed_ijk(1)+rvox);
c = max(1, seed_ijk(2)-rvox) : min(C, seed_ijk(2)+rvox);
s = max(1, seed_ijk(3)-rvox) : min(S, seed_ijk(3)+rvox);

% Hard cap the block size to stay tiny in RAM
if numel(r)*numel(c)*numel(s) > 5e6   % ~5M vox cap
    shrink = (5e6/(numel(r)*numel(c)*numel(s)))^(1/3);
    rvox2 = max(1, floor(rvox*shrink));
    r = max(1, seed_ijk(1)-rvox2) : min(R, seed_ijk(1)+rvox2);
    c = max(1, seed_ijk(2)-rvox2) : min(C, seed_ijk(2)+rvox2);
    s = max(1, seed_ijk(3)-rvox2) : min(S, seed_ijk(3)+rvox2);
end

% Slice directly (MATLAB expands combinations without explicit ndgrid)
subHU = HU(r, c, s);
subMK = ~markerMask(r, c, s) & isfinite(subHU);

if any(subMK(:))
    % Best bright voxel nearby
    subHU(~subMK) = -Inf;
    [~, idx] = max(subHU(:));
    [ir, ic, is] = ind3(idx, size(subHU));
    ii = r(ir); jj = c(ic); kk = s(is);

    note = sprintf('Seed reseated to brighter voxel (HU %.1f → %.1f).', Hseed, double(HU(ii,jj,kk)));
    seed_ijk = [ii jj kk];
    seedMask = false(size(HU)); seedMask(ii,jj,kk) = true;
else
    note = sprintf('Seed HU %.1f seems low, but no brighter candidate found nearby.', Hseed);
end
end

function [i,j,k] = ind3(idx, sz)
% tiny helper: linear -> 3D subs
i = mod(idx-1, sz(1)) + 1;
j = mod(floor((idx-1)/sz(1)), sz(2)) + 1;
k = floor((idx-1)/(sz(1)*sz(2))) + 1;
end
