function out = run_bone_pipeline(dicomFolder, stlFolder, varargin)
% RUN_BONE_PIPELINE  Full pipeline for multi-bone CT segmentation and specimen packing.
%
%   out = run_bone_pipeline(dicomFolder, stlFolder)
%   out = run_bone_pipeline(dicomFolder, stlFolder, 'Option', value, ...)
%
% Pipeline stages:
%   1. DICOM loading
%   2. Bone separation (FMM envelope detection for excised-in-air specimens)
%   3. Cortical / cancellous segmentation (gradient-based)
%   4. Specimen packing (greedy mixed packing of STL shapes)
%   5. Visualization (3D overview)
%   6. Output saving (MAT + STL + NIfTI)
%
% Inputs
%   dicomFolder : path to folder containing DICOM CT series
%   stlFolder   : path to folder containing specimen STL files
%                  (Bend.STL, Compression.STL, Punch.STL, Shear.STL)
%
% Name-value options
%   'TagHUMin'            : 1200 (HU threshold for lead tag detection)
%   'MinBoneVolMM3'       : 500  (minimum bone component volume)
%   'ClosingRadiusMM'     : 3.0  (morphological closing radius)
%   'ArtifactSigmaMM'     : 3.0  (Gaussian falloff for artifact weighting)
%   'PackSpecimens'       : true (run specimen packing — slow)
%   'PackingOrientations' : 6    (number of orientations per shape)
%   'PackingMinDepthMM'   : 0.5  (minimum depth for specimen placement)
%   'SaveOutputs'         : true (export MAT, NIfTI, STL files)
%   'OutputDir'           : ''   (auto-create if empty)
%   'ShowViewer'          : true (show 3D visualization)

% ---- Ensure shared packages are on the path ----
thisDir = fileparts(mfilename('fullpath'));
sharedSrc = fullfile(fileparts(fileparts(thisDir)), 'Scaphoid_Project', 'src');
if exist(sharedSrc, 'dir') && ~contains(path, sharedSrc)
    addpath(sharedSrc);
end

% ---- Parse options ----
opts = struct( ...
    'TagHUMin',            1200, ...
    'MinBoneVolMM3',       500.0, ...
    'ClosingRadiusMM',     3.0, ...
    'ArtifactSigmaMM',     3.0, ...
    'MarkerRangeHU',       [200 700], ...
    'PackSpecimens',       true, ...
    'PackingOrientations', 6, ...
    'PackingMinDepthMM',   0.5, ...
    'SaveOutputs',         true, ...
    'OutputDir',           '', ...
    'ShowViewer',          true, ...
    'TargetIsoMM',         [], ...
    'Smoothing',           false ...
);
opts = utils.parse_opts(opts, varargin{:});

t_start = tic;

fprintf('\n');
fprintf('==========================================================\n');
fprintf('  BONE SEGMENTATION PIPELINE\n');
fprintf('==========================================================\n');
fprintf('  DICOM : %s\n', dicomFolder);
fprintf('  STL   : %s\n', stlFolder);
fprintf('==========================================================\n\n');

% ==== Stage 1: DICOM Loading ====
fprintf('[1/6] Loading DICOM series...\n');
t1 = tic;
ds = dicom.series_load(dicomFolder, ...
    'TargetIsoMM', opts.TargetIsoMM, 'Smoothing', opts.Smoothing);
fprintf('[1/6] Done (%.1fs)\n', toc(t1));
fprintf('       Volume %dx%dx%d  |  spacing [%.3f  %.3f  %.3f] mm  |  HU [%.0f, %.0f]\n\n', ...
    ds.size(1), ds.size(2), ds.size(3), ds.spacing, min(ds.HU(:)), max(ds.HU(:)));

% ==== Stage 2: Bone Separation ====
fprintf('[2/6] Separating bones...');
t2 = tic;
sep_result = bone.separate_bones(ds, opts);
n_bones = numel(sep_result.bones);
fprintf(' done (%.1fs)\n', toc(t2));
fprintf('       %d bones found, %d markers detected\n\n', n_bones, sep_result.n_tags);

if n_bones == 0
    warning('No bones found. Check DICOM data and thresholds.');
    out = struct('ds', ds, 'separation', sep_result, ...
        'segmentation', {{}}, 'packing', {{}});
    return;
end

% ==== Stage 3: Cortical / Cancellous Segmentation ====
fprintf('[3/6] Cortical/cancellous segmentation...');
t3 = tic;
seg_results = cell(1, n_bones);

bone_masks = cell(1, n_bones);
for bi = 1:n_bones
    bone_masks{bi} = sep_result.bones{bi}.mask;
end

use_parallel = ~isempty(ver('parallel')) && n_bones > 1;
if use_parallel
    parfor bi = 1:n_bones
        [cort, canc, seg_info] = bone.cortical_cancellous(ds, bone_masks{bi}, opts);
        seg_results{bi} = struct('cortical', cort, 'cancellous', canc, 'info', seg_info);
    end
else
    for bi = 1:n_bones
        [cort, canc, seg_info] = bone.cortical_cancellous(ds, bone_masks{bi}, opts);
        seg_results{bi} = struct('cortical', cort, 'cancellous', canc, 'info', seg_info);
    end
end
fprintf(' done (%.1fs)%s\n\n', toc(t3), ternary(use_parallel, ' [parallel]', ''));

% ==== Stage 4: Specimen Packing ====
pack_results = cell(1, n_bones);
if ~opts.PackSpecimens
    fprintf('[4/6] Specimen packing skipped (PackSpecimens=false)\n\n');
else
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
        fprintf('[4/6] Specimen packing skipped (no STL files found)\n\n');
    else
        fprintf('[4/6] Packing specimens (%s)...', strjoin(stl_found, ', '));
        t4 = tic;

        bone_axes = cell(1, n_bones);
        corticals = cell(1, n_bones);
        cancellouses = cell(1, n_bones);
        for bi = 1:n_bones
            bone_axes{bi} = [0; 0; 1];
            if isfield(seg_results{bi}.info, 'bone_shape')
                bm = bone_masks{bi};
                [rr, cc, ss] = ind2sub(size(ds.HU), find(bm));
                coords = [rr(:)*ds.spacing(1), cc(:)*ds.spacing(2), ss(:)*ds.spacing(3)];
                coords = coords - mean(coords, 1);
                [V, ~] = eig(coords' * coords);
                bone_axes{bi} = V(:, 3);
            end
            corticals{bi} = seg_results{bi}.cortical;
            cancellouses{bi} = seg_results{bi}.cancellous;
        end

        if use_parallel
            parfor bi = 1:n_bones
                pack_results{bi} = bone.pack_specimens( ...
                    bone_masks{bi}, corticals{bi}, cancellouses{bi}, ...
                    ds, stl_paths, stl_found, opts, bone_axes{bi});
            end
        else
            for bi = 1:n_bones
                pack_results{bi} = bone.pack_specimens( ...
                    bone_masks{bi}, corticals{bi}, cancellouses{bi}, ...
                    ds, stl_paths, stl_found, opts, bone_axes{bi});
            end
        end
        fprintf(' done (%.1fs)%s\n\n', toc(t4), ternary(use_parallel, ' [parallel]', ''));
    end
end

% ==== Stage 5: Visualization ====
if opts.ShowViewer
    fprintf('[5/6] Generating visualizations...');
    t5 = tic;
    bone.visualize_results(ds, sep_result, seg_results, pack_results, opts);
    fprintf(' done (%.1fs)\n\n', toc(t5));
else
    fprintf('[5/6] Visualization skipped\n\n');
end

% ==== Stage 6: Save Outputs ====
if opts.SaveOutputs
    fprintf('[6/6] Saving outputs...');
    t6 = tic;

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

    % --- MAT file ---
    try
        save(fullfile(outDir, 'pipeline_results.mat'), ...
            'sep_result', 'seg_results', 'pack_results', 'opts', '-v7.3');
    catch ME
        warning('Save MAT failed: %s', ME.message);
    end

    % --- Per-bone NIfTI + STL ---
    for bi = 1:n_bones
        bm = sep_result.bones{bi}.mask;

        % NIfTI masks
        try
            write_mask_nifti(fullfile(outDir, sprintf('bone_%02d_mask.nii.gz', bi)), bm, ds);
            write_mask_nifti(fullfile(outDir, sprintf('bone_%02d_cortical.nii.gz', bi)), ...
                seg_results{bi}.cortical, ds);
            write_mask_nifti(fullfile(outDir, sprintf('bone_%02d_cancellous.nii.gz', bi)), ...
                seg_results{bi}.cancellous, ds);

            HU_masked = int16(ds.HU);
            HU_masked(~bm) = -3000;
            write_volume_nifti(fullfile(outDir, sprintf('bone_%02d_hu.nii.gz', bi)), HU_masked, ds);
        catch ME
            warning('NIfTI save failed for bone %d: %s', bi, ME.message);
        end

        % Voxelized STL (fast, preserves exact mask geometry)
        try
            if any(bm(:))
                fv = isosurface(smooth3(double(bm), 'gaussian', 3), 0.5);
                if ~isempty(fv.vertices)
                    fv.vertices(:,1) = fv.vertices(:,1) * ds.spacing(2);
                    fv.vertices(:,2) = fv.vertices(:,2) * ds.spacing(1);
                    fv.vertices(:,3) = fv.vertices(:,3) * ds.spacing(3);
                    mesh = struct('vertices', fv.vertices, 'faces', fv.faces);
                    meshing.write_stl_binary(fullfile(outDir, sprintf('bone_%02d_voxelized.stl', bi)), mesh);
                end
            end
        catch ME
            warning('Voxelized STL save failed for bone %d: %s', bi, ME.message);
        end

        % Smooth anatomical STL (Laplacian-smoothed, decimated)
        try
            if any(bm(:))
                smooth_field = smooth3(double(bm), 'gaussian', 7, 1.5);
                fv_smooth = isosurface(smooth_field, 0.5);
                if ~isempty(fv_smooth.vertices) && size(fv_smooth.faces, 1) > 100
                    fv_smooth.vertices(:,1) = fv_smooth.vertices(:,1) * ds.spacing(2);
                    fv_smooth.vertices(:,2) = fv_smooth.vertices(:,2) * ds.spacing(1);
                    fv_smooth.vertices(:,3) = fv_smooth.vertices(:,3) * ds.spacing(3);

                    % Iterative Laplacian smoothing for anatomical fidelity
                    V_s = fv_smooth.vertices;
                    F_s = fv_smooth.faces;
                    n_smooth_iters = 15;
                    lambda = 0.5;
                    nv = size(V_s, 1);
                    adj = sparse(nv, nv);
                    for fi = 1:size(F_s, 1)
                        adj(F_s(fi,1), F_s(fi,2)) = 1; adj(F_s(fi,2), F_s(fi,1)) = 1;
                        adj(F_s(fi,2), F_s(fi,3)) = 1; adj(F_s(fi,3), F_s(fi,2)) = 1;
                        adj(F_s(fi,3), F_s(fi,1)) = 1; adj(F_s(fi,1), F_s(fi,3)) = 1;
                    end
                    valence = full(sum(adj, 2));
                    valence(valence == 0) = 1;
                    for iter = 1:n_smooth_iters
                        V_neighbor = adj * V_s;
                        V_avg = V_neighbor ./ valence;
                        V_s = V_s + lambda * (V_avg - V_s);
                    end

                    smooth_mesh = struct('vertices', V_s, 'faces', F_s);
                    meshing.write_stl_binary( ...
                        fullfile(outDir, sprintf('bone_%02d_smooth.stl', bi)), ...
                        smooth_mesh, 'Decimate', 0.3);
                end
            end
        catch ME
            warning('Smooth STL save failed for bone %d: %s', bi, ME.message);
        end
    end

    % --- Text summary ---
    try
        write_summary_file(fullfile(outDir, 'pipeline_summary.txt'), ...
            ds, sep_result, seg_results, pack_results, dicomFolder, n_bones);
    catch ME
        warning('Summary save failed: %s', ME.message);
    end

    fprintf(' done (%.1fs)\n', toc(t6));
    fprintf('       Output: %s\n\n', outDir);
    out.outputDir = outDir;
else
    fprintf('[6/6] Output saving skipped\n\n');
end

% ==== Build output struct ====
out.ds = ds;
out.separation = sep_result;
out.segmentation = seg_results;
out.packing = pack_results;

% ==== Final Summary Table ====
elapsed = toc(t_start);

fprintf('==========================================================\n');
fprintf('  PIPELINE COMPLETE  (%.1f s)\n', elapsed);
fprintf('==========================================================\n\n');

% --- Bone summary table ---
fprintf('  %-6s  %10s  %8s  %8s  %10s  %10s  %8s  %5s\n', ...
    'Bone', 'Volume', 'Mean HU', 'Shape', 'Cortical', 'Cancellous', 'Cort %%', 'Tag');
fprintf('  %-6s  %10s  %8s  %8s  %10s  %10s  %8s  %5s\n', ...
    '------', '----------', '--------', '--------', '----------', '----------', '--------', '-----');

total_vol = 0;
total_cort = 0;
total_canc = 0;
for bi = 1:n_bones
    b = sep_result.bones{bi};
    si = seg_results{bi}.info;
    total_vol = total_vol + b.volume_mm3;
    total_cort = total_cort + si.cortical_volume_mm3;
    total_canc = total_canc + si.cancellous_volume_mm3;

    if ~isempty(b.tag_id)
        tag_str = sprintf('%d', b.tag_id);
    else
        tag_str = '-';
    end

    fprintf('  %-6s  %8.0f mm3  %6.0f HU  %8s  %8.0f mm3  %8.0f mm3  %6.1f%%  %5s\n', ...
        sprintf('#%d', bi), b.volume_mm3, b.mean_hu, si.bone_shape, ...
        si.cortical_volume_mm3, si.cancellous_volume_mm3, ...
        si.cortical_fraction * 100, tag_str);
end

fprintf('  %-6s  %8.0f mm3  %8s  %8s  %8.0f mm3  %8.0f mm3  %6.1f%%\n', ...
    'TOTAL', total_vol, '', '', total_cort, total_canc, ...
    total_cort / max(1, total_cort + total_canc) * 100);

% --- Packing summary table ---
has_packing = ~isempty(pack_results) && ~isempty(pack_results{1}) && ...
    isstruct(pack_results{1}) && isfield(pack_results{1}, 'n_total');
if has_packing
    fprintf('\n  %-6s  %8s  %8s  %8s\n', 'Bone', 'Cortical', 'Cancel.', 'Total');
    fprintf('  %-6s  %8s  %8s  %8s\n', '------', '--------', '--------', '--------');

    grand_cort = 0; grand_canc = 0;
    for bi = 1:n_bones
        pr = pack_results{bi};
        grand_cort = grand_cort + pr.n_cortical;
        grand_canc = grand_canc + pr.n_cancellous;
        fprintf('  %-6s  %8d  %8d  %8d\n', sprintf('#%d', bi), ...
            pr.n_cortical, pr.n_cancellous, pr.n_total);
    end
    fprintf('  %-6s  %8d  %8d  %8d\n', 'TOTAL', ...
        grand_cort, grand_canc, grand_cort + grand_canc);

    % Per-type breakdown
    if isfield(pack_results{1}, 'summary')
        fprintf('\n  Specimens by type:\n');
        all_types = fieldnames(pack_results{1}.summary);
        for ti = 1:numel(all_types)
            type_total = 0;
            for bi = 1:n_bones
                if isfield(pack_results{bi}.summary, all_types{ti})
                    s = pack_results{bi}.summary.(all_types{ti});
                    type_total = type_total + s.cortical + s.cancellous;
                end
            end
            fprintf('    %-15s : %d\n', all_types{ti}, type_total);
        end
    end
end

fprintf('\n==========================================================\n\n');
end


% =========================================================================
%  Local helper functions
% =========================================================================

function write_mask_nifti(filename, mask, ds)
    data = int16(mask) * 1000;
    write_volume_nifti(filename, data, ds);
end


function write_volume_nifti(filename, data, ds)
    M = [ds.dir_row'   * ds.spacing(1), ...
         ds.dir_col'   * ds.spacing(2), ...
         ds.dir_slice' * ds.spacing(3), ...
         ds.origin'; ...
         0 0 0 1];

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


function write_summary_file(filepath, ds, sep_result, seg_results, pack_results, dicomFolder, n_bones)
    fid = fopen(filepath, 'w');
    cleanup = onCleanup(@() fclose(fid));

    fprintf(fid, 'BONE SEGMENTATION PIPELINE SUMMARY\n');
    fprintf(fid, '===================================\n');
    fprintf(fid, 'Date   : %s\n', datestr(now));
    fprintf(fid, 'DICOM  : %s\n', dicomFolder);
    fprintf(fid, 'Volume : %dx%dx%d, spacing [%.3f %.3f %.3f] mm\n', ...
        ds.size(1), ds.size(2), ds.size(3), ds.spacing);
    fprintf(fid, 'HU range: [%.0f, %.0f]\n\n', min(ds.HU(:)), max(ds.HU(:)));
    fprintf(fid, 'Bones   : %d\n', n_bones);
    fprintf(fid, 'Markers : %d\n\n', sep_result.n_tags);

    for bi = 1:n_bones
        b = sep_result.bones{bi};
        si = seg_results{bi}.info;
        if ~isempty(b.tag_id)
            tag_str = sprintf('tag %d (%.1f mm)', b.tag_id, b.tag_dist);
        else
            tag_str = 'no tag';
        end
        fprintf(fid, 'Bone %d: %.1f mm3 | HU %.0f | %s | %s\n', ...
            bi, b.volume_mm3, b.mean_hu, si.bone_shape, tag_str);
        fprintf(fid, '  Cortical: %.0f mm3 (%.1f%%) depth %.2f mm\n', ...
            si.cortical_volume_mm3, si.cortical_fraction*100, si.mean_cortical_depth_mm);
        fprintf(fid, '  Cancellous: %.0f mm3\n', si.cancellous_volume_mm3);
    end

    has_packing = ~isempty(pack_results) && ~isempty(pack_results{1}) && ...
        isstruct(pack_results{1}) && isfield(pack_results{1}, 'n_total');
    if has_packing
        fprintf(fid, '\nSPECIMEN PACKING\n');
        fprintf(fid, '================\n');
        for bi = 1:n_bones
            pr = pack_results{bi};
            fprintf(fid, 'Bone %d: %d cortical + %d cancellous = %d specimens\n', ...
                bi, pr.n_cortical, pr.n_cancellous, pr.n_total);
            if isfield(pr, 'summary')
                fnames = fieldnames(pr.summary);
                for fi = 1:numel(fnames)
                    s = pr.summary.(fnames{fi});
                    fprintf(fid, '  %-15s: %d cortical, %d cancellous\n', ...
                        fnames{fi}, s.cortical, s.cancellous);
                end
            end
        end
    end
end


function s = ternary(cond, a, b)
    if cond, s = a; else, s = b; end
end
