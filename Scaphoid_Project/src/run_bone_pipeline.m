function out = run_bone_pipeline(dicomFolder, stlFolder, varargin)
% RUN_BONE_PIPELINE  Full pipeline for multi-bone CT segmentation and specimen packing.
%
%   out = run_bone_pipeline(dicomFolder, stlFolder)
%   out = run_bone_pipeline(dicomFolder, stlFolder, 'Option', value, ...)
%
% Pipeline stages:
%   1. DICOM loading (reuses scaphoid pipeline's robust loader)
%   2. Bone separation (envelope detection for excised-in-air specimens)
%   3. Per-bone boundary refinement (adaptive FMM-style from scaphoid pipeline)
%   4. Cortical / cancellous segmentation (Otsu + depth-based)
%   5. Specimen packing (greedy mixed packing of STL shapes)
%   6. Visualization (3D + axial slices + histograms)
%   7. Output saving (MAT + STL + NIfTI)
%
% Inputs
%   dicomFolder : path to folder containing DICOM CT series
%   stlFolder   : path to folder containing specimen STL files
%                  (Bend.STL, Compression.STL, Punch.STL, Shear.STL)
%
% Name-value options
%   'TagHUMin'            : 1200 (HU threshold for lead tag detection)
%   'MinBoneVolMM3'       : 200  (minimum bone component volume)
%   'ClosingRadiusMM'     : 3.0  (morphological closing radius)
%   'ArtifactSigmaMM'     : 3.0  (Gaussian falloff for artifact weighting)
%   'RefineBones'         : true (run per-bone FMM refinement)
%   'PackingOrientations' : 6    (number of orientations per shape)
%   'PackingMinDepthMM'   : 0.5  (minimum depth for specimen placement)
%   'SaveOutputs'         : true
%   'OutputDir'           : ''   (auto-create if empty)
%   'ShowViewer'          : true (show 3D visualization)
%
% Output struct
%   .ds           : dataset from DICOM loading
%   .separation   : bone separation result
%   .segmentation : cell array of {cortical, cancellous, info} per bone
%   .packing      : cell array of packing results per bone
%   .outputDir    : path to saved outputs
%
% Example (156L-1 scan):
%   dicom = 'C:\Users\natha\OneDrive\Documents\Nathaniel\For Nick\New Bone Scans\156L-1\DICOMOBJ';
%   stls  = 'C:\Users\natha\OneDrive\Documents\Nathaniel\For Nick\Mechancial Specimens';
%   out   = run_bone_pipeline(dicom, stls);

% ---- Parse options ----
opts = struct( ...
    'TagHUMin',            1200, ...
    'MinBoneVolMM3',       200.0, ...
    'ClosingRadiusMM',     3.0, ...
    'ArtifactSigmaMM',     3.0, ...
    'RefineBones',         true, ...
    'PackingOrientations', 6, ...
    'PackingMinDepthMM',   0.5, ...
    'SaveOutputs',         true, ...
    'OutputDir',           '', ...
    'ShowViewer',          true, ...
    'TargetIsoMM',         [], ...
    'Smoothing',           false ...
);
opts = utils.parse_opts(opts, varargin{:});

fprintf('============================================================\n');
fprintf('  BONE SEGMENTATION PIPELINE\n');
fprintf('============================================================\n');
fprintf('  DICOM: %s\n', dicomFolder);
fprintf('  STL:   %s\n', stlFolder);
fprintf('============================================================\n\n');

% ==== Stage 1: DICOM Loading ====
fprintf('[Stage 1] Loading DICOM series...\n');
ds = dicom.series_load(dicomFolder, ...
    'TargetIsoMM', opts.TargetIsoMM, 'Smoothing', opts.Smoothing);
fprintf('  Volume: %dx%dx%d, spacing [%.3f %.3f %.3f] mm\n', ...
    ds.size(1), ds.size(2), ds.size(3), ds.spacing);
fprintf('  HU range: [%.0f, %.0f]\n\n', min(ds.HU(:)), max(ds.HU(:)));

% ==== Stage 2: Bone Separation ====
fprintf('[Stage 2] Separating bones...\n');
sep_result = bone.separate_bones(ds, opts);
n_bones = numel(sep_result.bones);
fprintf('  %d bones separated\n\n', n_bones);

if n_bones == 0
    warning('No bones found. Check DICOM data and thresholds.');
    out = struct('ds', ds, 'separation', sep_result, ...
        'segmentation', {{}}, 'packing', {{}});
    return;
end

% ==== Stage 3: Per-bone refinement (optional) ====
if opts.RefineBones
    fprintf('[Stage 3] Refining bone boundaries...\n');
    for bi = 1:n_bones
        fprintf('  Bone %d/%d (%.0f mm^3):\n', bi, n_bones, ...
            sep_result.bones{bi}.volume_mm3);
        [refined, qc] = bone.segment_single_bone(ds, sep_result.bones{bi}.mask, opts);
        sep_result.bones{bi}.mask = refined;
        sep_result.bones{bi}.volume_mm3 = qc.volume_mm3;
        sep_result.bones{bi}.qc = qc;
        fprintf('    Refined: %.0f mm^3 (method: %s)\n', qc.volume_mm3, qc.method);
    end
    fprintf('\n');
else
    fprintf('[Stage 3] Skipped (RefineBones = false)\n\n');
end

% ==== Stage 4: Cortical / Cancellous Segmentation ====
fprintf('[Stage 4] Cortical / cancellous segmentation...\n');
seg_results = cell(1, n_bones);
for bi = 1:n_bones
    fprintf('  Bone %d/%d:\n', bi, n_bones);
    [cortical, cancellous, info] = bone.cortical_cancellous(ds, sep_result.bones{bi}.mask, opts);
    seg_results{bi} = struct('cortical', cortical, 'cancellous', cancellous, 'info', info);
    fprintf('    Otsu threshold: %.0f HU\n', info.otsu_threshold);
    fprintf('    Cortical thickness: %.2f mm\n', info.cortical_thickness_mm);
    fprintf('    Cortical: %.0f mm^3 (%.0f%%)\n', info.cortical_volume_mm3, info.cortical_fraction*100);
    fprintf('    Cancellous: %.0f mm^3 (%.0f%%)\n\n', info.cancellous_volume_mm3, (1-info.cortical_fraction)*100);
end

% ==== Stage 5: Specimen Packing ====
fprintf('[Stage 5] Specimen packing...\n');

% Find STL files
shape_names = {'Bend', 'Compression', 'Punch', 'Shear'};
stl_paths = {};
stl_found = {};
for si = 1:numel(shape_names)
    candidates = {
        fullfile(stlFolder, [shape_names{si} '.STL']);
        fullfile(stlFolder, [shape_names{si} '.stl']);
        fullfile(stlFolder, [lower(shape_names{si}) '.stl']);
        fullfile(stlFolder, [lower(shape_names{si}) '.STL']);
    };
    found = false;
    for ci = 1:numel(candidates)
        if exist(candidates{ci}, 'file')
            stl_paths{end+1} = candidates{ci}; %#ok<AGROW>
            stl_found{end+1} = shape_names{si}; %#ok<AGROW>
            found = true;
            fprintf('  Found: %s\n', candidates{ci});
            break;
        end
    end
    if ~found
        fprintf('  Warning: %s.STL not found in %s\n', shape_names{si}, stlFolder);
    end
end

pack_results = cell(1, n_bones);
if ~isempty(stl_paths)
    for bi = 1:n_bones
        fprintf('\n  Bone %d/%d:\n', bi, n_bones);
        bone_packs = cell(1, 2);

        % Pack cortical region
        fprintf('    Cortical region:\n');
        bone_packs{1} = bone.pack_specimens(seg_results{bi}.cortical, ds, ...
            stl_paths, stl_found, opts);

        % Pack cancellous region
        fprintf('    Cancellous region:\n');
        bone_packs{2} = bone.pack_specimens(seg_results{bi}.cancellous, ds, ...
            stl_paths, stl_found, opts);

        pack_results{bi} = bone_packs;
    end
else
    fprintf('  No STL files found — skipping packing\n');
end
fprintf('\n');

% ==== Stage 6: Visualization ====
if opts.ShowViewer
    fprintf('[Stage 6] Generating visualizations...\n');
    bone.visualize_results(ds, sep_result, seg_results, pack_results, opts);
    fprintf('\n');
else
    fprintf('[Stage 6] Skipped (ShowViewer = false)\n\n');
end

% ==== Stage 7: Save Outputs ====
if opts.SaveOutputs
    fprintf('[Stage 7] Saving outputs...\n');

    if isempty(opts.OutputDir)
        [parentDir, seriesName] = fileparts(string(dicomFolder));
        baseOut = fullfile(parentDir, 'bone_pipeline_outputs', seriesName);
        tstamp = datestr(now, 'yyyymmdd_HHMMSS');
        outDir = fullfile(baseOut, tstamp);
    else
        outDir = opts.OutputDir;
    end
    if ~exist(outDir, 'dir'), mkdir(outDir); end
    opts.OutputDir = outDir;

    % Save main results
    try
        % Strip masks for compact saving (save separately)
        bones_compact = cell(size(sep_result.bones));
        for bi = 1:n_bones
            b = sep_result.bones{bi};
            bones_compact{bi} = rmfield(b, 'mask');
        end

        save(fullfile(outDir, 'pipeline_results.mat'), ...
            'sep_result', 'seg_results', 'pack_results', 'opts', '-v7.3');
        fprintf('  Saved pipeline_results.mat\n');
    catch ME
        warning('Save MAT failed: %s', ME.message);
    end

    % Per-bone masks as NIfTI
    try
        for bi = 1:n_bones
            bone_mask = sep_result.bones{bi}.mask;

            % Full bone mask
            write_mask_nifti(fullfile(outDir, sprintf('bone_%02d_mask.nii.gz', bi)), ...
                bone_mask, ds);

            % Cortical + cancellous
            write_mask_nifti(fullfile(outDir, sprintf('bone_%02d_cortical.nii.gz', bi)), ...
                seg_results{bi}.cortical, ds);
            write_mask_nifti(fullfile(outDir, sprintf('bone_%02d_cancellous.nii.gz', bi)), ...
                seg_results{bi}.cancellous, ds);

            % Masked HU (bone region only)
            HU_masked = int16(ds.HU);
            HU_masked(~bone_mask) = -3000;
            write_volume_nifti(fullfile(outDir, sprintf('bone_%02d_hu.nii.gz', bi)), ...
                HU_masked, ds);

            fprintf('  Saved bone_%02d NIfTI files\n', bi);
        end
    catch ME
        warning('Save NIfTI failed: %s', ME.message);
    end

    % Per-bone STL meshes
    try
        for bi = 1:n_bones
            mask_i = sep_result.bones{bi}.mask;
            if ~any(mask_i(:)), continue; end

            fv = isosurface(smooth3(double(mask_i), 'gaussian', 3), 0.5);
            if isempty(fv.vertices), continue; end
            fv.vertices(:,1) = fv.vertices(:,1) * ds.spacing(2);
            fv.vertices(:,2) = fv.vertices(:,2) * ds.spacing(1);
            fv.vertices(:,3) = fv.vertices(:,3) * ds.spacing(3);

            mesh = struct('vertices', fv.vertices, 'faces', fv.faces);
            meshing.write_stl(fullfile(outDir, sprintf('bone_%02d.stl', bi)), mesh);
            fprintf('  Saved bone_%02d.stl\n', bi);
        end
    catch ME
        warning('Save STL failed: %s', ME.message);
    end

    fprintf('  All outputs saved to: %s\n', outDir);
    out.outputDir = outDir;
end

% ==== Build output struct ====
out.ds = ds;
out.separation = sep_result;
out.segmentation = seg_results;
out.packing = pack_results;

% ==== Summary ====
fprintf('\n============================================================\n');
fprintf('  PIPELINE COMPLETE\n');
fprintf('============================================================\n');
total_vol = 0;
for bi = 1:n_bones
    b = sep_result.bones{bi};
    seg = seg_results{bi};
    total_vol = total_vol + b.volume_mm3;

    if ~isempty(b.tag_id)
        tag_str = sprintf('tag %d', b.tag_id);
    else
        tag_str = 'no tag';
    end

    n_specimens = 0;
    if bi <= numel(pack_results) && ~isempty(pack_results{bi})
        for ri = 1:numel(pack_results{bi})
            if ~isempty(pack_results{bi}{ri})
                n_specimens = n_specimens + numel(pack_results{bi}{ri});
            end
        end
    end

    fprintf('  Bone %d: %.0f mm^3 | cort %.0f%% | %s | %d specimens\n', ...
        bi, b.volume_mm3, seg.info.cortical_fraction*100, tag_str, n_specimens);
end
fprintf('  Total bone volume: %.0f mm^3\n', total_vol);
fprintf('============================================================\n');
end


% =========================================================================
%  Local helper functions
% =========================================================================

function write_mask_nifti(filename, mask, ds)
    data = int16(mask) * 1000;
    write_volume_nifti(filename, data, ds);
end


function write_volume_nifti(filename, data, ds)
    % Build affine matrix
    M = [ds.dir_row'   * ds.spacing(1), ...
         ds.dir_col'   * ds.spacing(2), ...
         ds.dir_slice' * ds.spacing(3), ...
         ds.origin'; ...
         0 0 0 1];

    % Create NIfTI using seed trick (same as scaphoid pipeline)
    seedFile = fullfile(tempdir, 'bone_pipeline_seed.nii');
    if exist(seedFile, 'file'), delete(seedFile); end
    niftiwrite(zeros(size(data), 'int16'), seedFile);
    info = niftiinfo(seedFile);
    delete(seedFile);

    info.Datatype = 'int16';
    info.PixelDimensions = ds.spacing;
    info.Transform = affine3d(M');

    niftiwrite(int16(data), filename, info, 'Compressed', true);
end
