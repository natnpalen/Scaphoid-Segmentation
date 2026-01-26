function out = run_segmentation(dicomFolder, varargin)
% run_segmentation
%
% Inputs:
%   dicomFolder (char/string): folder containing a single DICOM CT series.
%
% Name-value options (common):
%   'TargetIsoMM'           : [] (no resample) or scalar (e.g., 0.5) for isotropic mm.
%   'ShowViewer'            : true/false to display a triplanar viewer for QC.
%   'InteractiveSeedConfirm': true/false allow a single mouse-click to relocate seed.
%   'AirRecalibrate'        : true/false (default true) shift air to target HU.
%   'AirTargetHU'           : -1000 (default) target HU for air calibration.
%   'AirMaxOffsetHU'        : 300 (cap on |offset| to avoid wild shifts).
%   'MarkerRangeHU'         : [200 700] default (lead collars / flags).
%   'ArtifactSigmaMM'       : 3 (falloff radius for artifact down-weighting).
%   'Smoothing'             : false (edge-preserving if true; conservative).
%   'FMM_AlphaBetaGamma'    : [0.6 0.4 1.0] weights for data, edge, artifact.
%
% Output struct (out):
%   .mesh         : struct with fields vertices (Nx3, mm), faces (Mx3), HU (Nx1)
%   .mask         : logical 3-D scaphoid mask
%   .ds           : dataset struct (HU, spacing, transforms, markerMask, artifactWeight, calibration)
%   .qc           : qc struct (confidenceVolume, vertexFlags, summary)
%   .seed         : [i j k] voxel index used to seed segmentation
%
% Requirements:
%   Image Processing Toolbox
%   Statistics & Machine Learning Toolbox
%   Medical Imaging Toolbox (Optional, for robust DICOM loading)
%   Parallel Computing Toolbox (Optional, for mesh snapping speedup)
% --------------------------- Options -------------------------------------
opts = struct( ...
 'TargetIsoMM', [], ...
 'ShowViewer', false, ...
 'InteractiveSeedConfirm', false, ...
 'AirRecalibrate', true, ...
 'AirTargetHU', -1000, ...
 'AirMaxOffsetHU', 300, ...
 'MarkerRangeHU', [200 700], ...
 'ArtifactSigmaMM', 3, ...
 'Smoothing', false, ...
 'FMM_AlphaBetaGamma', [0.8 0.2 1.0], ...
 'SaveOutputs', true, ...          % NEW: write files to a folder
 'OutputDir', '', ...               % NEW: leave empty to auto-create
 ... % --- NEW: sheetness prior ---
 'UseSheetnessPrior', true, ...
 'SheetnessSigmasMM', [0.5 0.8 1.2], ...
 'SheetnessWeight', 0.35, ...
 'SheetnessAlpha', 0.5, ...   % ratio term
 'SheetnessBeta',  0.5, ...   % structureness term
 'SheetnessC',     0.5, ...   % contrast normalizer (Frangi-style)
 'SheetnessBrightStructures', true, ...
 ... % --- NEW: adaptive FMM sweep ---
 'UseAdaptiveSweep', true, ...
 'FMMThresholdRange', [0.14 0.42], ...
 'FMMNumSteps', 9, ...
 'BoundaryScoreImage', 'gradHU', ... % 'gradHU' or 'edge'
 'BoundaryScoreLambda', 1e-5, ...
 'BoundaryMustBeSingleComponent', true, ...
 ... % --- NEW: outer shell sealing ---
 'UseShellSealing', false, ...
 'ShellCloseRadiusMM', 0.6, ...
 'ShellFillHoles', true, ...
 'ShellThicknessMM', 0.0, ...
 'ShellKeepLargestOnly', true, ...
... % --- Phase A: sub-voxel outer mesh + fairing + snapping ---
 'UseSubvoxelOuter', true, ...
 'SDFSmoothMM', 0.4, ...
 'TaubinIters', 12, ...
 'TaubinLambdaMu', [0.5 -0.53], ...
 'SnapBandMM', 2.5, ...
 'SnapStepMM', 0.25, ...
 'SnapOutwardTolMM', 0.1, ...
 'SnapUseLikelihood', true, ...
 'SnapInwardCapMM', 2.0, ...
 'SnapSDFTauMM', 0.25, ...
 'SnapUseParfor', true, ...
 'EmitBinaryShell', false, ...
 ... % --- Batch/output conveniences (used in next steps) ---
 'EmitMaskedHU',          true, ...
 'MaskedHUFormat',        'nii', ...
 'MaskedCropToBBox',      true, ...
 'MaskedBBoxMarginVox',   2, ...
 'MaskedFillValue',       NaN, ...
 'UseStableOutputDir', false, ...
 'WriteMaskedHU', true, ...
 'WriteNifti', true ...
);
opts = utils.parse_opts(opts, varargin{:});
% --------------------------- Load DICOM ----------------------------------
fprintf('[Load] Scanning DICOM: %s\n', string(dicomFolder));
ds = dicom.series_load(dicomFolder, ...
 'TargetIsoMM', opts.TargetIsoMM, 'Smoothing', opts.Smoothing);

[mask, qc_data, ds] = segment.run_segmentation(ds, opts);

[mesh_outer, mesh_shell, outerMethod] = meshing.create_meshes(mask, ds, opts);
% --------------------------- Masked HU (for downstream density binning) --
HU_masked_full = double(ds.HU);
outsideVal     = opts.MaskedFillValue;
if ~isfinite(outsideVal), outsideVal = NaN; end
HU_masked_full(~mask) = outsideVal;
% Optional: crop to tight bbox of the mask for compact storage
if opts.MaskedCropToBBox
  [bbox, bboxMask] = utils.tight_mask_bbox(mask, opts.MaskedBBoxMarginVox);
  HU_masked = HU_masked_full(bbox.r, bbox.c, bbox.s);
  masked_meta = struct('bbox', bbox, ...
                       'spacing', ds.spacing, ...
                       'outsideVal', outsideVal);
else
  HU_masked = HU_masked_full;
  bbox = struct('r', 1:size(mask,1), 'c', 1:size(mask,2), 's', 1:size(mask,3));
  masked_meta = struct('bbox', bbox, ...
                       'spacing', ds.spacing, ...
                       'outsideVal', outsideVal);
end

% Use OUTER as the primary preview/return value (preferred)
if opts.UseSubvoxelOuter
  mesh = mesh_outer;
else
  mesh = mesh_shell;
end

qc = buildQC(ds.HU, mask, qc_data.seed_mask_full, ds.artifactWeight, qc_data.dist_map_full, mesh);
if exist('outerMethod','var')
  qc.outerMethod = outerMethod;
else
  qc.outerMethod = 'unknown';
end

seed_out = qc_data.seed_full;

out = struct('mesh',         mesh, ...
           'mesh_shell',   mesh_shell, ...
           'mesh_outer',   mesh_outer, ...
           'mask',         mask, ...
           'ds',           ds, ...
           'qc',           qc, ...
           'seed',         seed_out);

fprintf('[Done] Shell: V=%d, F=%d | Outer: V=%d, F=%d | method=%s | meanConf=%.3f, lowConfFrac=%.3f\n', ...
  size(mesh_shell.vertices,1), size(mesh_shell.faces,1), ...
  size(mesh_outer.vertices,1), size(mesh_outer.faces,1), ...
  qc.outerMethod, ...
  qc.summary.meanConf, qc.summary.lowConfFrac);

% --------------------------- Save outputs to folder ----------------------
if opts.SaveOutputs
  seriesDir = string(dicomFolder);
  [parentDir, seriesName] = fileparts(seriesDir);
  baseOut   = fullfile(parentDir, "outputs", seriesName);
  if isempty(opts.OutputDir)
      tstamp = datestr(now,'yyyymmdd_HHMMSS');
      if isfield(opts,'UseStableOutputDir') && opts.UseStableOutputDir
          outDir = fullfile(baseOut, "latest");
      else
          outDir = fullfile(baseOut, tstamp);
      end
  else
      outDir = string(opts.OutputDir);
  end
  if ~exist(baseOut, 'dir'), mkdir(baseOut); end
  if ~exist(outDir,  'dir'), mkdir(outDir);  end
  out.outputDir = outDir; % keep path in results

  try
      save(fullfile(outDir, 'out.mat'), 'out', '-v7.3');
      save(fullfile(outDir, 'mask.mat'), 'mask', '-v7.3');
  catch ME
      warning('Save MAT failed: %s', ME.message);
  end

  if opts.WriteNifti
    fprintf('[Save] Writing NIfTI file...\n');
    try
        HU_for_nifti = HU_masked_full;
        HU_for_nifti(isnan(HU_for_nifti)) = -3000;
        HU_for_nifti = int16(HU_for_nifti);

        M = [ds.dir_row'   * ds.spacing(1), ...
             ds.dir_col'   * ds.spacing(2), ...
             ds.dir_slice' * ds.spacing(3), ...
             ds.origin'; ...
             0 0 0 1];

        seedFile = fullfile(tempdir,'scaphoid_seed.nii');
        if exist(seedFile,'file'), delete(seedFile); end
        niftiwrite(zeros(size(HU_for_nifti),'int16'), seedFile);
        info = niftiinfo(seedFile);
        delete(seedFile);

        info.Datatype = 'int16';
        info.PixelDimensions = ds.spacing;
        info.Transform = affine3d(M');

        niftiFilename = fullfile(outDir, 'scaphoid_masked_hu.nii.gz');
        niftiwrite(HU_for_nifti, niftiFilename, info, 'Compressed', true);
        save(fullfile(outDir,'scaphoid_masked_hu_affine_LPS.mat'), 'M');
        fprintf('[Save] NIfTI file and affine matrix written successfully.\n');
        % -------------------------------------------------------------------

    catch ME
        warning('Save NIfTI failed: %s', ME.message);
    end
  end

  try
      if isfield(opts,'EmitBinaryShell') && opts.EmitBinaryShell
          meshing.write_stl(fullfile(outDir, 'scaphoid_shell.stl'), out.mesh_shell);
      end
      meshing.write_stl(fullfile(outDir, 'scaphoid_outer.stl'), out.mesh_outer);
  catch ME
      warning('Save STL failed: %s', ME.message);
  end
  fprintf('[Save] Wrote outputs to %s\n', outDir);
end

try
 meshing.show_mesh_3d(mesh, mesh.HU);
catch
end
end

% ----------------------- QC ----------------------------------------------
function qc = buildQC(V, BW, seed, A, D, mesh)
pBone = mat2gray(V);
edge = mat2gray(imgradient3(V));
C = (0.5*pBone + 0.5*edge) ./ (1 + A);
dt = bwdist(A>0.4);
vIdx = round(mesh.vertices);
vIdx(:,1)=max(1,min(size(V,1), vIdx(:,1)));
vIdx(:,2)=max(1,min(size(V,2), vIdx(:,2)));
vIdx(:,3)=max(1,min(size(V,3), vIdx(:,3)));
lin = sub2ind(size(V), vIdx(:,1), vIdx(:,2), vIdx(:,3));
nearArt = dt(lin) < 3;
qc.confidenceVolume = C;
qc.distanceMap = D;
qc.vertexFlags.nearArtifact = nearArt;
if isfield(mesh, 'snapFlags')
  if isfield(mesh.snapFlags, 'suspect')
      qc.vertexFlags.snapSuspect = logical(mesh.snapFlags.suspect);
  end
  if isfield(mesh.snapFlags, 'retractMM')
      qc.vertexFlags.snapRetractMM = mesh.snapFlags.retractMM;
  end
end
qc.summary = struct('meanConf', mean(C(BW),'omitnan'), 'lowConfFrac', mean(C(BW)<0.3,'omitnan'));
end
