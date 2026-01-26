function [mesh_outer, mesh_shell, outerMethod] = create_meshes(mask, ds, opts)
% CREATE_MESHES generates surface models from the final segmentation mask.
%
% Inputs:
%   mask (logical): The final, full-sized 3D logical mask of the scaphoid.
%   ds   (struct): The dataset struct from dicom.series_load.
%   opts (struct): The options struct for the pipeline.
%
% Outputs:
%   mesh_outer (struct): The primary, high-quality subvoxel mesh.
%   mesh_shell (struct): The secondary, binary shell mesh.

if ~any(mask(:))
    error('create_meshes:EmptyMask', ...
        ['Segmentation mask is empty. Upstream segmentation likely failed ', ...
         '(seed/thresholds/crop). Check run_segmentation logs for Score and core/allow stats.']);
end
% --------------------------- Mesh + per-vertex HU ------------------------
% Keep the current segmentation result as the "shell"
BW_shell = mask;
% Also make a solid outer envelope (fills internal voids)
BW_outer = solidifyMask3D(mask);
% Sanity: ensure BW_outer is not empty or all-ones
if ~any(BW_outer(:)) || all(BW_outer(:))
  warning('BW_outer degenerate (all 0 or all 1). Falling back to original mask for outer.');
  BW_outer = mask;
end
% Remove anything touching the volume border, then keep the largest interior component
BW_outer = imclearborder(BW_outer, 26);
BW_outer = utils.keep_largest_component_3d(BW_outer);
% Extra safety: fill any residual 3-D voids so the SDF has a single boundary
BW_outer = imfill(BW_outer, 'holes');
% Post-cleanup guard: if we removed everything, fall back to original mask
if ~any(BW_outer(:))
  warning('BW_outer empty after cleanup; falling back to original mask for outer.');
  BW_outer = mask;
end
% Shell mesh (unchanged: binary marching cubes) ---------------------------
[mesh_shell, perHU_shell] = meshing.buildMeshAndScalars(ds.HU, BW_shell, ds.spacing);
mesh_shell.HU = perHU_shell;
% Outer mesh (Phase A: sub-voxel SDF + Taubin + snap-to-gradient) --------
if opts.UseSubvoxelOuter
  outerMethod = 'subvoxel';
  try
      % 1) Signed distance in mm (positive inside)
      sdf_mm = signedDistanceMM(BW_outer, ds.spacing);
      % 2) Light Gaussian blur in mm (stabilizes 0-level)
      if opts.SDFSmoothMM > 0
          sig_vox = max(eps, [opts.SDFSmoothMM, opts.SDFSmoothMM, opts.SDFSmoothMM] ./ ds.spacing);
          sdf_mm = imgaussfilt3(double(sdf_mm), sig_vox);
      else
          sdf_mm = double(sdf_mm);
      end
      % 3) Sub-voxel isosurface at 0-level; map HU
      [mesh_outer, perHU_outer] = buildMeshFromFieldAndScalars(sdf_mm, 0.0, ds.spacing, ds.HU);
      mesh_outer.HU = perHU_outer;
      % 4) Taubin smoothing (shrinkage-free)
      mesh_outer = taubinSmooth(mesh_outer, opts.TaubinIters, opts.TaubinLambdaMu(1), opts.TaubinLambdaMu(2));
      % 5) Snap vertices to strongest HU gradient along normals (±band)
      mesh_outer = snapVerticesToGradient(mesh_outer, ds.HU, ds.spacing, ...
                                          sdf_mm, opts.SnapBandMM, opts.SnapStepMM, ...
                                          opts.SnapOutwardTolMM, opts.SnapUseLikelihood, ...
                                          opts.SnapInwardCapMM, opts.SnapSDFTauMM, ...
                                          opts.SnapUseParfor);
      % Recompute per-vertex HU after snapping
      F_HU = buildVolumeInterpolant(ds.HU, ds.spacing);
      mesh_outer.HU = sampleVolumeAtVertices(F_HU, mesh_outer.vertices);
  catch ME
      outerMethod = 'binary-fallback';
      warning('Sub-voxel outer mesh failed (%s); falling back to binary isosurface.', ME.message);
      [mesh_outer, perHU_outer] = meshing.buildMeshAndScalars(ds.HU, BW_outer, ds.spacing);
      mesh_outer.HU = perHU_outer;
  end
else
  outerMethod = 'binary-legacy';
  [mesh_outer, perHU_outer] = meshing.buildMeshAndScalars(ds.HU, BW_outer, ds.spacing);
  mesh_outer.HU = perHU_outer;
end
end

% --- All the meshing HELPER functions will go here ---
function BWsolid = solidifyMask3D(BW)
% Fill internal cavities so isosurface returns only the outer shell.
% Keep the object as-is, but add interior background cavities to it.
B0 = ~BW;
cavities = imclearborder(B0, 26);  % leaves ONLY interior background components
BWsolid = BW | cavities;           % fill cavities into the object
BWsolid = logical(BWsolid);
end

function sdf = signedDistanceMM(BW, spacing)
% signedDistanceMM: positive inside (foreground), negative outside, in mm.
% Uses exact Euclidean distance transform per voxel, then scaled to mm.
d_in  = bwdist(~BW);   % distance from inside voxels to boundary (in vox)
d_out = bwdist(BW);    % distance from outside to boundary (in vox)
% Convert voxel distances to mm with an isotropic approximation
voxel_mm = mean(spacing);
sdf = (d_in - d_out) * voxel_mm;
end

function [mesh, perVertexHU] = buildMeshFromFieldAndScalars(field, isoval, spacing, HUvolume)
% Toolbox-agnostic: force base MATLAB isosurface on a double array.
% field: 3-D double; isoval: scalar; spacing: [dr dc ds]; HUvolume: 3-D.
field = double(field);           % <-- ensure 'double matrix'
HUvolume = double(HUvolume);     % <-- ensure 'double matrix'
% Quick sanity: require a zero crossing for the isovalue
fmin = min(field(:)); fmax = max(field(:));
if ~(fmin <= isoval && isoval <= fmax)
  error('buildMeshFromFieldAndScalars:isosurfaceFailed', ...
        'Field has no %g-level crossing (min=%g, max=%g).', isoval, fmin, fmax);
end
% Base MATLAB isosurface
fv = isosurface(field, isoval);    % struct with .faces / .vertices
faces = fv.faces;
vertices = fv.vertices;
% Convert iso coords (x=cols,y=rows,z=slices) to mm with axis reorder
verts_vox = vertices;
verts_vox = vertices - 1;
verts_mm  = [verts_vox(:,2)*spacing(1), ...
           verts_vox(:,1)*spacing(2), ...
           verts_vox(:,3)*spacing(3)];
% Per-vertex HU via trilinear interpolation (in mm space)
F_HU = buildVolumeInterpolant(HUvolume, spacing);
perVertexHU = sampleVolumeAtVertices(F_HU, verts_mm);
mesh = struct('vertices', verts_mm, 'faces', faces);
end

function F = buildVolumeInterpolant(V, spacing)
% Build an interpolant using coordinate vectors (mm).
V = double(V);
[R,C,S] = size(V);
rv = (0:R-1)*spacing(1);
cv = (0:C-1)*spacing(2);
sv = (0:S-1)*spacing(3);
F = griddedInterpolant({rv, cv, sv}, V, 'linear', 'none');
end

function vals = sampleVolumeAtVertices(F_or_V, spacing_or_verts, verts_mm)
% Trilinear HU sampling at vertex positions (mm).
if isa(F_or_V, 'griddedInterpolant')
    F = F_or_V;
    verts = spacing_or_verts;
else
    F = buildVolumeInterpolant(F_or_V, spacing_or_verts);
    verts = verts_mm;
end
vals = F(verts(:,1), verts(:,2), verts(:,3));
end

function mesh = taubinSmooth(mesh, iters, lambda, mu)
% Shrinkage-free smoothing (Taubin) using a compatible adjacency method.
V = mesh.vertices; F = mesh.faces;
if isempty(V) || isempty(F), return; end
TR = triangulation(F, V);
nbrs = vertexAttachments(TR); % This is a compatible way to get neighbors
adj = cell(size(V,1),1);
for vi = 1:size(V,1)
  faces_i = nbrs{vi};
  if isempty(faces_i), adj{vi} = []; continue; end
  verts = unique(TR.ConnectivityList(faces_i(:),:));
  adj{vi} = setdiff(verts, vi);
end
for t = 1:max(0,iters)
  V = laplaceStep(V, adj, +lambda);
  V = laplaceStep(V, adj, +mu);
end
mesh.vertices = V;
  function Vnext = laplaceStep(Verts, Adj, step)
      Vnext = Verts;
      for k = 1:size(Verts,1)
          nb = Adj{k};
          if isempty(nb), continue; end
          delta = mean(Verts(nb,:),1) - Verts(k,:);
          Vnext(k,:) = Verts(k,:) + step * delta;
      end
  end
end

function mesh = snapVerticesToGradient(mesh, HU, spacing, sdf_mm, bandMM, stepMM, outwardTolMM, useLikelihood, inwardCapMM, sdfTauMM, useParfor)
% Parallel, interpolant-based snapping with A1–A4 guardrails.
% No nested functions; uses file-level subfunctions for parfor-compatibility.
% ---- Setup & normals -----------------------------------------------------
V  = mesh.vertices;
F  = mesh.faces;
outTol   = max(0, outwardTolMM);
inCap    = max(0, inwardCapMM);
halfBand = max(0, bandMM);
step     = max(eps, stepMM);
tau      = max(eps, sdfTauMM);
TR = triangulation(F, V);
FN = faceNormal(TR);
VN = zeros(size(V));
for i = 1:size(F,1)
  tri = F(i,:);
  VN(tri,:) = VN(tri,:) + FN(i,:);
end
VN = VN ./ max(eps, vecnorm(VN,2,2));
HU  = double(HU);
sdf = double(sdf_mm);
[R,C,S] = size(HU);
rv = (0:R-1)*spacing(1);   % row coords in mm
cv = (0:C-1)*spacing(2);   % col coords in mm
sv = (0:S-1)*spacing(3);   % slice coords in mm
% ---- Build gridded interpolants (B2) ------------------------------------
F_HU   = griddedInterpolant({rv,cv,sv}, HU,  'linear','none');
[Hu_r,Hu_c,Hu_s] = gradient(HU, spacing(1), spacing(2), spacing(3));
F_Hu_r = griddedInterpolant({rv,cv,sv}, Hu_r,'linear','none');
F_Hu_c = griddedInterpolant({rv,cv,sv}, Hu_c,'linear','none');
F_Hu_s = griddedInterpolant({rv,cv,sv}, Hu_s,'linear','none');
Gmag   = imgradient3(HU);
F_Gmag = griddedInterpolant({rv,cv,sv}, Gmag,'linear','none');
[gSr,gSc,gSs] = gradient(sdf, spacing(1), spacing(2), spacing(3));
F_sdf  = griddedInterpolant({rv,cv,sv}, sdf, 'linear','none');
F_gSr  = griddedInterpolant({rv,cv,sv}, gSr, 'linear','none');
F_gSc  = griddedInterpolant({rv,cv,sv}, gSc, 'linear','none');
F_gSs  = griddedInterpolant({rv,cv,sv}, gSs, 'linear','none');
% Wrap for workers
CF_HU   = parallel.pool.Constant(F_HU);
CF_Hu_r = parallel.pool.Constant(F_Hu_r);
CF_Hu_c = parallel.pool.Constant(F_Hu_c);
CF_Hu_s = parallel.pool.Constant(F_Hu_s);
CF_Gmag = parallel.pool.Constant(F_Gmag);
CF_sdf  = parallel.pool.Constant(F_sdf);
CF_gSr  = parallel.pool.Constant(F_gSr);
CF_gSc  = parallel.pool.Constant(F_gSc);
CF_gSs  = parallel.pool.Constant(F_gSs);
% Ensure VN points outward using grad(SDF) at vertices
FgSr = CF_gSr.Value; FgSc = CF_gSc.Value; FgSs = CF_gSs.Value;
gS_r = FgSr(V(:,1), V(:,2), V(:,3)); gS_r(isnan(gS_r)) = 0;
gS_c = FgSc(V(:,1), V(:,2), V(:,3)); gS_c(isnan(gS_c)) = 0;
gS_s = FgSs(V(:,1), V(:,2), V(:,3)); gS_s(isnan(gS_s)) = 0;
gS   = [gS_r, gS_c, gS_s];
flipMask = sum(VN.*gS, 2) > 0;       % grad(SDF) points inward
VN(flipMask,:) = -VN(flipMask,:);
% Precompute global fallback likelihood params (used if local rings are NaN)
defMuBone  = median(HU(HU>-300),'omitnan');  defMADBone = mad(HU(HU>-300),1) + 50;
defMuTiss  = median(HU(HU<-700),'omitnan');  defMADTiss = mad(HU(HU<-700),1) + 50;
% Outputs
N = size(V,1);
Vnew        = zeros(N,3);
suspectFlag = false(N,1);
retractMM   = zeros(N,1);
% Sampling steps along normal
numSteps = max(1, ceil(halfBand/step));
stepsAll = (-numSteps:numSteps) * step;
% Decide parallelism (B1)
useParfor = logical(useParfor);
hasDCT = false; try, hasDCT = license('test','Distrib_Computing_Toolbox'); catch, end
pp = []; if useParfor && hasDCT, pp = gcp('nocreate'); end
useParfor = useParfor && ~isempty(pp);
if useParfor
  parfor k = 1:N
      [Vnew(k,:), suspectFlag(k), retractMM(k)] = snap_one_vertex_worker( ...
          V(k,:), VN(k,:), stepsAll, ...
          CF_HU, CF_Hu_r, CF_Hu_c, CF_Hu_s, CF_Gmag, CF_sdf, CF_gSr, CF_gSc, CF_gSs, ...
          outTol, inCap, tau, useLikelihood, ...
          defMuBone, defMADBone, defMuTiss, defMADTiss);
  end
else
  for k = 1:N
      [Vnew(k,:), suspectFlag(k), retractMM(k)] = snap_one_vertex_worker( ...
          V(k,:), VN(k,:), stepsAll, ...
          CF_HU, CF_Hu_r, CF_Hu_c, CF_Hu_s, CF_Gmag, CF_sdf, CF_gSr, CF_gSc, CF_gSs, ...
          outTol, inCap, tau, useLikelihood, ...
          defMuBone, defMADBone, defMuTiss, defMADTiss);
  end
end
mesh.vertices  = Vnew;
mesh.snapFlags = struct('suspect', logical(suspectFlag), 'retractMM', retractMM);
end

function [pBest, isSuspect, movedInward] = snap_one_vertex_worker(p0, n, stepsAll, ...
  CF_HU, CF_Hu_r, CF_Hu_c, CF_Hu_s, CF_Gmag, CF_sdf, CF_gSr, CF_gSc, CF_gSs, ...
  outTol, inCap, tau, useLikelihood, defMuBone, defMADBone, defMuTiss, defMADTiss)
pBest = p0; isSuspect = false; movedInward = 0;
if ~all(isfinite(n)) || norm(n)<eps, return; end
% Local likelihood rings
if useLikelihood
  innerD = [0.4, 0.8];
  outerD = [0.4, 0.8, 1.2];
  Pi = p0 - innerD(:).*n;  Po = p0 + outerD(:).*n;
  hui = gi_eval_const(CF_HU,  Pi(:,1), Pi(:,2), Pi(:,3), NaN);
  huo = gi_eval_const(CF_HU,  Po(:,1), Po(:,2), Po(:,3), NaN);
  mu1 = defMuBone; s1 = defMADBone;
  mu0 = defMuTiss; s0 = defMADTiss;
  if any(isfinite(hui)), mu1 = median(hui(isfinite(hui))); s1 = mad(hui(isfinite(hui)),1) + 50; end
  if any(isfinite(huo)), mu0 = median(huo(isfinite(huo))); s0 = mad(huo(isfinite(huo)),1) + 50; end
  pBone_at = @(v) 1./(1+exp(-(v - mu1)./s1));
  pTiss_at = @(v) 1./(1+exp(-(mu0 - v)./s0));
else
  pBone_at = []; pTiss_at = [];
end
% Candidates
steps = stepsAll;
P = p0 + steps(:).*n;
% Constraints: SDF + inward cap
sdfVals = gi_eval_const(CF_sdf, P(:,1), P(:,2), P(:,3), +Inf);
ok_sdf   = sdfVals >= -outTol;
ok_incap = (steps >= -inCap);
% Directional gate (signed HU gradient along outward normal <= 0)
Hu_r_c = gi_eval_const(CF_Hu_r, P(:,1), P(:,2), P(:,3), NaN);
Hu_c_c = gi_eval_const(CF_Hu_c, P(:,1), P(:,2), P(:,3), NaN);
Hu_s_c = gi_eval_const(CF_Hu_s, P(:,1), P(:,2), P(:,3), NaN);
gpar   = Hu_r_c.*n(1) + Hu_c_c.*n(2) + Hu_s_c.*n(3);
ok_dir = gpar <= 0;
% Edge magnitude
gmag = gi_eval_const(CF_Gmag, P(:,1), P(:,2), P(:,3), -Inf);
% Likelihood + SDF proximity weighting + HU floor
HU_FLOOR = 260;  % hard cortical floor
if useLikelihood
  huvals = gi_eval_const(CF_HU, P(:,1), P(:,2), P(:,3), NaN);
  Rvals  = (pBone_at(huvals) ./ (pTiss_at(huvals) + eps));
  likeOK = Rvals >= 1.0;
else
  huvals = gi_eval_const(CF_HU, P(:,1), P(:,2), P(:,3), NaN);
  likeOK = true(size(P,1),1);
end
huOK  = huvals >= HU_FLOOR;            % absolute HU acceptance
sdfW  = exp(-abs(sdfVals)/tau);
score = abs(gmag) .* (useLikelihood * Rvals + ~useLikelihood) .* sdfW;
ok_pre = ok_sdf & ok_incap & ok_dir & likeOK & huOK & isfinite(score);
% Air-avoidance: reject candidates whose immediate outward sample is air
AIR_CUTOFF = -300;   % anything below this is basically air after calibration
hu_out = gi_eval_const(CF_HU, P(:,1) + 0.4*n(1), P(:,2) + 0.4*n(2), P(:,3) + 0.4*n(3), NaN);
airOK = ~(hu_out < AIR_CUTOFF);
ok = ok_pre & airOK;
% Origin fallback
gmag0 = gi_eval_const(CF_Gmag, p0(1), p0(2), p0(3), 0);
if useLikelihood
  hu0 = gi_eval_const(CF_HU, p0(1), p0(2), p0(3), NaN);
  s0  = abs(gmag0) * (pBone_at(hu0) ./ (pTiss_at(hu0) + eps));
else
  s0  = abs(gmag0);
end
sdf0 = gi_eval_const(CF_sdf, p0(1), p0(2), p0(3), 0);
s0   = s0 * exp(-abs(sdf0)/tau);
if any(ok)
  [sBest, iBest] = max([s0; score(ok)]);
  if iBest==1 || ~isfinite(sBest)
      pBest = p0;
  else
      Pok = P(ok,:);
      pBest = Pok(iBest-1,:);
  end
else
  pBest = p0;
end
% Repair pass (unchanged)
if useLikelihood
  huB = gi_eval_const(CF_HU, pBest(1), pBest(2), pBest(3), NaN);
  RB  = (pBone_at(huB) ./ (pTiss_at(huB) + eps));
else
  RB = 1;
end
Hu_rB = gi_eval_const(CF_Hu_r, pBest(1), pBest(2), pBest(3), 0);
Hu_cB = gi_eval_const(CF_Hu_c, pBest(1), pBest(2), pBest(3), 0);
Hu_sB = gi_eval_const(CF_Hu_s, pBest(1), pBest(2), pBest(3), 0);
gparB = Hu_rB*n(1) + Hu_cB*n(2) + Hu_sB*n(3);
if (RB < 1.0) || (gparB > 0)
  maxRepair = 0.2; rStep = min(0.1, maxRepair);
  moved = 0; pCurr = pBest;
  while moved < maxRepair
      pTry = pCurr - rStep*n;  % retract inward
      sdfTry = gi_eval_const(CF_sdf, pTry(1), pTry(2), pTry(3), -Inf);
      if ~isfinite(sdfTry) || sdfTry < 0, break; end
      if useLikelihood
          huT = gi_eval_const(CF_HU, pTry(1), pTry(2), pTry(3), NaN);
          RT  = (pBone_at(huT) ./ (pTiss_at(huT) + eps));
      else
          RT = 1;
      end
      Hu_rT = gi_eval_const(CF_Hu_r, pTry(1), pTry(2), pTry(3), 0);
      Hu_cT = gi_eval_const(CF_Hu_c, pTry(1), pTry(2), pTry(3), 0);
      Hu_sT = gi_eval_const(CF_Hu_s, pTry(1), pTry(2), pTry(3), 0);
      gparT = Hu_rT*n(1) + Hu_cT*n(2) + Hu_sT*n(3);
      pCurr = pTry; moved = moved + rStep;
      if (RT >= 1.0) && (gparT <= 0)
          pBest = pCurr; break;
      end
  end
  if useLikelihood
      huF = gi_eval_const(CF_HU, pBest(1), pBest(2), pBest(3), NaN);
      RF  = (pBone_at(huF) ./ (pTiss_at(huF) + eps));
  else
      RF = 1;
  end
  Hu_rF = gi_eval_const(CF_Hu_r, pBest(1), pBest(2), pBest(3), 0);
  Hu_cF = gi_eval_const(CF_Hu_c, pBest(1), pBest(2), pBest(3), 0);
  Hu_sF = gi_eval_const(CF_Hu_s, pBest(1), pBest(2), pBest(3), 0);
  gparF = Hu_rF*n(1) + Hu_cF*n(2) + Hu_sF*n(3);
  isSuspect   = (RF < 1.0) || (gparF > 0);
  movedInward = moved;
else
  isSuspect   = false;
  movedInward = 0;
end
end

function val = gi_eval_const(CF, rr, cc, ss, fillVal)
% Evaluate a griddedInterpolant stored in a parallel.pool.Constant.
F = CF.Value;
val = F(rr, cc, ss);
nanmask = isnan(val);
if any(nanmask), val(nanmask) = fillVal; end
end
