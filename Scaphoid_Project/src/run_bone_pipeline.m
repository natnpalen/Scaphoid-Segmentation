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
    'MinBoneVolMM3',       500.0, ...
    'ClosingRadiusMM',     3.0, ...
    'ArtifactSigmaMM',     3.0, ...
    'MarkerRangeHU',       [200 700], ...
    'RefineBones',         false, ...
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
    fprintf('  Bone %d/%d (%.0f mm^3):\n', bi, n_bones, ...
        sep_result.bones{bi}.volume_mm3);
    [cort, canc, seg_info] = bone.cortical_cancellous(ds, sep_result.bones{bi}.mask, opts);
    seg_results{bi} = struct('cortical', cort, 'cancellous', canc, 'info', seg_info);
    fprintf('    %s | %d slabs | mean depth: %.2f mm | mean thr: %.0f HU\n', ...
        seg_info.method, seg_info.n_slabs, ...
        seg_info.mean_cortical_depth_mm, seg_info.mean_threshold_hu);
    fprintf('    Cortical: %.0f mm^3 (%.0f%%) | Cancellous: %.0f mm^3\n', ...
        seg_info.cortical_volume_mm3, seg_info.cortical_fraction*100, ...
        seg_info.cancellous_volume_mm3);
end
fprintf('\n');

% ==== Stage 5: Specimen Packing ====
fprintf('[Stage 5] Specimen packing...\n');

% Discover STL specimen files
stl_names = {'Bend', 'Compression', 'Punch', 'Shear'};
stl_paths = {};
stl_found = {};
for si = 1:numel(stl_names)
    candidates = {fullfile(stlFolder, [stl_names{si} '.STL']), ...
                  fullfile(stlFolder, [stl_names{si} '.stl'])};
    for ci = 1:numel(candidates)
        if exist(candidates{ci}, 'file')
            stl_paths{end+1} = candidates{ci}; %#ok<AGROW>
            stl_found{end+1} = stl_names{si}; %#ok<AGROW>
            break;
        end
    end
end

if isempty(stl_paths)
    fprintf('  No STL specimen files found in: %s\n', stlFolder);
    fprintf('  Expected: Bend.STL, Compression.STL, Punch.STL, Shear.STL\n');
    fprintf('[Stage 5] Skipped (no specimen files)\n\n');
    pack_results = cell(1, n_bones);
else
    fprintf('  Found %d specimen types: %s\n', numel(stl_found), strjoin(stl_found, ', '));
    pack_results = cell(1, n_bones);
    for bi = 1:n_bones
        fprintf('  Bone %d/%d (%.0f mm^3):\n', bi, n_bones, ...
            sep_result.bones{bi}.volume_mm3);

        bone_axis = [0; 0; 1];  % default
        if isfield(seg_results{bi}.info, 'bone_shape')
            % Recompute principal axis from bone mask for alignment
            bm = sep_result.bones{bi}.mask;
            [rr, cc, ss] = ind2sub(size(ds.HU), find(bm));
            coords = [rr(:)*ds.spacing(1), cc(:)*ds.spacing(2), ss(:)*ds.spacing(3)];
            coords = coords - mean(coords, 1);
            [V, ~] = eig(coords' * coords);
            bone_axis = V(:, 3);
        end

        pack_results{bi} = bone.pack_specimens( ...
            sep_result.bones{bi}.mask, ...
            seg_results{bi}.cortical, ...
            seg_results{bi}.cancellous, ...
            ds, stl_paths, stl_found, opts, bone_axis);
    end
    fprintf('\n');
end

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

    % Text summary file
    try
        summary_file = fullfile(outDir, 'pipeline_summary.txt');
        fid = fopen(summary_file, 'w');
        fprintf(fid, 'BONE SEGMENTATION PIPELINE SUMMARY\n');
        fprintf(fid, '===================================\n');
        fprintf(fid, 'Date: %s\n', datestr(now));
        fprintf(fid, 'DICOM: %s\n', dicomFolder);
        fprintf(fid, 'Volume: %dx%dx%d, spacing [%.3f %.3f %.3f] mm\n', ...
            ds.size(1), ds.size(2), ds.size(3), ds.spacing);
        fprintf(fid, 'HU range: [%.0f, %.0f]\n\n', min(ds.HU(:)), max(ds.HU(:)));
        fprintf(fid, 'Bones found: %d\n', n_bones);
        fprintf(fid, 'Markers found: %d\n\n', sep_result.n_tags);
        total_vol = 0;
        for bi = 1:n_bones
            b = sep_result.bones{bi};
            total_vol = total_vol + b.volume_mm3;
            if ~isempty(b.tag_id)
                tag_str = sprintf('tag %d (%.1f mm)', b.tag_id, b.tag_dist);
            else
                tag_str = 'no tag';
            end
            fprintf(fid, 'Bone %d: %.1f mm^3 | mean HU %.0f | dense %.0f%% | %s\n', ...
                bi, b.volume_mm3, b.mean_hu, b.dense_fraction*100, tag_str);
            fprintf(fid, '  Centroid: [%.1f %.1f %.1f] mm\n', b.centroid_mm);
            fprintf(fid, '  BBox: [%d %d %d] to [%d %d %d]\n', b.bbox);
        end
        fprintf(fid, '\nTotal bone volume: %.0f mm^3\n', total_vol);

        % Packing summary
        has_packing = ~isempty(pack_results) && isstruct(pack_results{1}) && ...
            isfield(pack_results{1}, 'n_total');
        if has_packing
            fprintf(fid, '\n--- SPECIMEN PACKING ---\n');
            for bi = 1:n_bones
                pr = pack_results{bi};
                fprintf(fid, 'Bone %d: %d cortical + %d cancellous = %d specimens\n', ...
                    bi, pr.n_cortical, pr.n_cancellous, pr.n_total);
                if isfield(pr, 'summary')
                    fnames = fieldnames(pr.summary);
                    for fi = 1:numel(fnames)
                        s = pr.summary.(fnames{fi});
                        fprintf(fid, '  %s: %d cortical, %d cancellous\n', ...
                            fnames{fi}, s.cortical, s.cancellous);
                    end
                end
            end
        end
        fclose(fid);
        fprintf('  Saved pipeline_summary.txt\n');
    catch ME
        warning('Save summary failed: %s', ME.message);
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
    total_vol = total_vol + b.volume_mm3;

    if ~isempty(b.tag_id)
        tag_str = sprintf('tag %d', b.tag_id);
    else
        tag_str = 'no tag';
    end

    fprintf('  Bone %d: %.0f mm^3 | mean HU %.0f | dense %.0f%% | %s\n', ...
        bi, b.volume_mm3, b.mean_hu, b.dense_fraction*100, tag_str);
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
