% full_workflow.m
%
% Master script to run the entire Scaphoid analysis, gyroid generation,
% and density-region mesh partition. Supports:
%   MODE = "single" : run one DICOM series folder
%   MODE = "batch"  : scan a top-level folder and process each subfolder
%
% This is the ONLY script you need to run.
clearvars
clc
close all
rehash toolboxcache

%% ===================== USER CONFIG =====================
% Canonical feature flags live under cfg.features.
% Toggle any stage or preview in one place:
%   cfg.features.runGyroid            -> Step 2 (gyroid generation)
%   cfg.features.runPartition         -> Step 3 (mesh partition)
%   cfg.features.showPopups           -> density analysis visualizations
%   cfg.features.gyroidPreview3d      -> 3D gyroid preview (continuous)
%   cfg.features.gyroidSingleSlice    -> single-slice gyroid preview
%   cfg.features.gyroidZonedPreview3d -> 3D zoned gyroid preview
%   cfg.features.gyroidZonedSlice     -> single-slice zoned preview
cfg = struct();

% Choose mode: "single" or "batch"
cfg.mode = "single";     % <- change to "batch" to process many

% Single-case input (used if cfg.mode == "single")
% UPDATE THESE PATHS to point to your DICOM data location
cfg.single.dicomFolder = fullfile(pwd, 'data', 'Scaphoid 1');

% Batch input (used if cfg.mode == "batch")
cfg.batch.topLevelFolder = fullfile(pwd, 'data');

% Common options
cfg.mesh.upsamplingFactor = 4;        % for run_mesh_partition

% Feature toggles (single source of truth)
cfg.features.runGyroid            = false;
cfg.features.runPartition         = true;
cfg.features.showPopups           = false;
cfg.features.gyroidPreview3d      = true;
cfg.features.gyroidSingleSlice    = false;
cfg.features.gyroidZonedPreview3d = true;
cfg.features.gyroidZonedSlice     = false;
% ========================================================


%% --- Setup: Add code to path (once) ---
projectRoot = fileparts(mfilename('fullpath'));
addpath(fullfile(projectRoot, 'src'));

% (Optional) use a short, stable temp base to avoid long paths
short_tmp_root = fullfile(tempdir, 'iso2mesh_tmp');
if ~exist(short_tmp_root,'dir'), mkdir(short_tmp_root); end
setenv('TMPDIR', short_tmp_root);
setenv('TEMP',   short_tmp_root);
setenv('TMP',    short_tmp_root);

t_all = tic;
switch lower(string(cfg.mode))
    case "single"
        fprintf('=====================================================\n');
        fprintf('                MODE: SINGLE CASE\n');
        fprintf('=====================================================\n');
        process_one_case(cfg.single.dicomFolder, cfg);

    case "batch"
        fprintf('=====================================================\n');
        fprintf('                MODE: BATCH PROCESS\n');
        fprintf('  Root: %s\n', cfg.batch.topLevelFolder);
        fprintf('=====================================================\n');

        caseFolders = list_candidate_cases(cfg.batch.topLevelFolder);
        if isempty(caseFolders)
            warning('No candidate case folders found under: %s', cfg.batch.topLevelFolder);
        end

        n = numel(caseFolders);
        fprintf('Found %d candidate case folder(s).\n\n', n);

        for idx = 1:n
            caseDir = caseFolders{idx};
            banner(sprintf('STARTING CASE %d/%d: %s', idx, n, caseDir));

            try
                process_one_case(caseDir, cfg);
                banner(sprintf('COMPLETED CASE %d/%d', idx, n));
            catch ME
                banner(sprintf('FAILED CASE %d/%d', idx, n));
                fprintf('Error: %s\n', ME.message);
                for s = 1:numel(ME.stack)
                    fprintf('  at %s (line %d)\n', ME.stack(s).name, ME.stack(s).line);
                end
            end

            % --- Memory-friendly cleanup between cases ---
            cleanup_between_cases();

        end

    otherwise
        error('Unknown MODE: %s. Use "single" or "batch".', cfg.mode);
end

fprintf('\n=====================================================\n');
fprintf('  ALL DONE. TOTAL ELAPSED: %.1f s\n', toc(t_all));
fprintf('=====================================================\n');


%% ======================= LOCAL FUNCTIONS =======================

function process_one_case(dicomFolder, cfg)
    % Validate input
    if ~isfolder(dicomFolder)
        error('DICOM folder not found: %s', dicomFolder);
    end

    % --- Step 1: Run Density Analysis ---
    fprintf('=====================================================\n');
    fprintf('      STARTING: DENSITY ANALYSIS\n');
    fprintf('  Input: %s\n', dicomFolder);
    fprintf('=====================================================\n');
    analysis_results_file = run_density_analysis(dicomFolder, cfg);

    % --- Step 2: Run Gyroid Generation (optional) ---
    if cfg.features.runGyroid
        fprintf('\n=====================================================\n');
        fprintf('      STARTING: GYROID TOOLPATH GENERATION\n');
        fprintf('=====================================================\n');
        run_gyroid_generation(analysis_results_file, cfg);
    else
        fprintf('\n[SKIP] Gyroid generation disabled by user.\n');
    end

    % --- Step 3: Partition into density-region meshes (optional) ---
    if cfg.features.runPartition
        fprintf('\n=====================================================\n');
        fprintf('      STARTING: DENSITY REGION MESH PARTITION\n');
        fprintf('=====================================================\n');
        run_mesh_partition(analysis_results_file, cfg);
    else
        fprintf('\n[SKIP] Mesh partition disabled by user.\n');
    end

    fprintf('\n=====================================================\n');
    fprintf('        WORKFLOW COMPLETED SUCCESSFULLY\n');
    fprintf('=====================================================\n');
end

function folders = list_candidate_cases(topLevel)
    % Return immediate subfolders of topLevel that look like DICOM series folders.
    % Heuristics:
    %  1) Must be a folder
    %  2) Contains at least one *.dcm / *.IMA / *.dicom file (at any depth 1 level down)
    %
    % Adjust heuristics if your data layout differs.
    folders = {};
    if ~isfolder(topLevel), return; end

    d = dir(topLevel);
    d = d([d.isdir]);
    names = setdiff({d.name}, {'.','..'});

    for i = 1:numel(names)
        thisDir = fullfile(topLevel, names{i});
        if is_candidate_dicom_folder(thisDir)
            folders{end+1} = thisDir; %#ok<AGROW>
        end
    end
end

function tf = is_candidate_dicom_folder(folderPath)
    % Check for presence of DICOM-ish files in this folder (non-recursive + one level down)
    patterns = {'*.dcm','*.DCM','*.IMA','*.ima','*.dicom','*.DICOM'};
    tf = false;

    % Check current folder
    for p = 1:numel(patterns)
        if ~isempty(dir(fullfile(folderPath, patterns{p})))
            tf = true; return;
        end
    end

    % Check one level down
    dd = dir(folderPath);
    dd = dd([dd.isdir]);
    names = setdiff({dd.name}, {'.','..'});
    for i = 1:numel(names)
        sub = fullfile(folderPath, names{i});
        for p = 1:numel(patterns)
            if ~isempty(dir(fullfile(sub, patterns{p})))
                tf = true; return;
            end
        end
    end
end

function cleanup_between_cases()
    % Close figures
    close all force

    % Clear any parallel pool (helps free RAM between heavy iso2mesh runs)
    pool = gcp('nocreate');
    if ~isempty(pool)
        try
            delete(pool);
        catch
        end
    end

    % Purge iteration-specific temp dirs created by run_mesh_partition
    baseTmp = fullfile(tempdir, 'iso2mesh_tmp');
    if isfolder(baseTmp)
        try
            d = dir(baseTmp);
            d = d([d.isdir]);
            names = setdiff({d.name},{'.','..'});
            for k = 1:numel(names)
                if startsWith(names{k}, 'w') % our per-iteration "wNNN" pattern
                    thisTmp = fullfile(baseTmp, names{k});
                    if isfolder(thisTmp)
                        % Best-effort removal (ignore errors if files are in use)
                        try
                            rmdir(thisTmp, 's');
                        catch
                        end
                    end
                end
            end
        catch
        end
    end

    % Purge Java temp refs and flush graphics
    drawnow;
end

function banner(msg)
    line = repmat('=',1,max(55, numel(msg)+10));
    fprintf('%s\n', line);
    fprintf('  %s\n', msg);
    fprintf('%s\n', line);
end
