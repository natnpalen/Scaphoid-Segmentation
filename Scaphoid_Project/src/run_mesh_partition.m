function run_mesh_partition(analysis_results_file, cfg)
%RUN_MESH_PARTITION.m Generate separate, closed meshes for k density regions (hi-res).

% ---------- Step 0: parse inputs ----------
if nargin < 2 || isempty(cfg)
    cfg = struct();
end
upsamplingFactor = get_cfg_number(cfg, 'mesh', 'upsamplingFactor', 4);

fprintf('=====================================================\n');
fprintf(' STARTING: DENSITY REGION MESH PARTITION\n');
fprintf('=====================================================\n');

% ---------- Step 1: load analysis + NIfTI HU ----------
fprintf('--- Step 1: Loading Density Analysis Data ---\n');
if ~isfile(analysis_results_file), error('File not found: %s', analysis_results_file); end
data = load(analysis_results_file);

analysisDir = fileparts(analysis_results_file);
outputDir   = fullfile(analysisDir, 'partitioned_meshes');
if ~exist(outputDir, 'dir'), mkdir(outputDir); fprintf('Created output directory: %s\n', outputDir); end
fprintf('Loaded data from: %s\n\n', analysis_results_file);

% Load masked HU NIfTI
nifti_path = fullfile(analysisDir, 'scaphoid_masked_hu.nii.gz');
if ~isfile(nifti_path)
    error('Could not find scaphoid_masked_hu.nii.gz in the analysis directory to load HU data.');
end
HU = niftiread(nifti_path);

% === Load analysis results (mask, ds, k, centroids, clamps) ===
S = load(fullfile(analysisDir, 'density_analysis_results.mat'), ...
    'scaphoid_mask', 'ds', 'k', 'centroids_sorted', 'hu_min', 'hu_max');

scaphoid_mask = logical(S.scaphoid_mask); %#ok<NASGU>
ds            = S.ds; %#ok<NASGU>
k             = S.k;
centroids     = S.centroids_sorted;
hu_min        = S.hu_min;
hu_max        = S.hu_max;

% --- Load the smooth scaphoid outer shell as FV_shell (used only for optional QA/clip) ---
FV_shell = [];

% Use a short temp root to avoid long paths during booleans
short_tmp_root = fullfile(tempdir, 'iso2mesh_tmp');
if ~exist(short_tmp_root,'dir'), mkdir(short_tmp_root); end

% Keep analysis-scoped subdir too (optional)
iso_tmp_root = fullfile(analysisDir, 'iso2mesh_tmp');
if ~exist(iso_tmp_root,'dir'), mkdir(iso_tmp_root); end

% Default the environment to the short temp root
setenv('TMPDIR', short_tmp_root);
setenv('TEMP',   short_tmp_root);
setenv('TMP',    short_tmp_root);

% Add iso2mesh binaries (if available)
surfboolean_path = which('surfboolean');
if ~isempty(surfboolean_path)
    iso2mesh_bin = fullfile(fileparts(surfboolean_path), 'bin', 'win64');
    if exist(iso2mesh_bin,'dir'), setenv('PATH', [iso2mesh_bin ';' getenv('PATH')]); end
end

try
    Tout = load(fullfile(analysisDir, 'out.mat'), 'out');
    if isfield(Tout,'out') && isfield(Tout.out,'mesh_outer') ...
            && isstruct(Tout.out.mesh_outer) ...
            && all(isfield(Tout.out.mesh_outer, {'vertices','faces'}))
        FV_shell = Tout.out.mesh_outer;
    end
catch, end

% Fallback: load shell STL via built-in stlread (if present)
if isempty(FV_shell)
    stl_path = fullfile(analysisDir, 'scaphoid_outer.stl');
    if exist(stl_path, 'file')
        try
            TR = stlread(stl_path);
            FV_shell = struct('vertices', double(TR.Points), 'faces', double(TR.ConnectivityList));
        catch
            warning('Could not read shell STL via stlread; leaving FV_shell empty.');
            FV_shell = [];
        end
    end
end

if ~isempty(FV_shell)
    FV_shell.vertices = double(FV_shell.vertices);
    FV_shell.faces    = double(FV_shell.faces);
end

% ---------- Step 2: upsample (NaN-safe) ----------
fprintf('--- Step 2: Upsampling Data by Factor of %d ---\n', upsamplingFactor);
fprintf('This will increase voxel count by %.0fx and may take some time...\n', upsamplingFactor^3);

[R0,C0,S0] = size(HU);
R_hires = (R0-1) * upsamplingFactor + 1;
C_hires = (C0-1) * upsamplingFactor + 1;
S_hires = (S0-1) * upsamplingFactor + 1;
vox_hires = double(R_hires) * double(C_hires) * double(S_hires);
bytes_est = vox_hires * (8 + 1);
if bytes_est > 12e9
    error(['Upsampling to [%d %d %d] would allocate ~%.1f GB; ' ...
        'reduce UpsamplingFactor or crop more tightly.'], ...
        R_hires, C_hires, S_hires, bytes_est/1e9);
end

tic;
lowres_data.HU = HU;
lowres_data.scaphoid_mask = data.scaphoid_mask;
lowres_data.ds = data.ds;
hires_data = partition.upsample_data(lowres_data, upsamplingFactor);
clear lowres_data HU;  % free low-res data now that upsampling is done
fprintf('Upsampling complete. (%.2f seconds)\n', toc);

n_in  = nnz(hires_data.mask);
n_fin = nnz(hires_data.mask & isfinite(hires_data.HU));
fprintf('Finite HU inside hi-res mask: %d / %d (%.2f%%)\n\n', n_fin, n_in, 100*n_fin/max(1,n_in));

% ---------- Step 3: project + cleanup + relabel ----------
fprintf('--- Step 3: Projecting Low-Res Density Classes to High-Res Grid ---\n');
D = density.density_defaults();

t_all_step3 = tic;
t_proj = tic;
labels_hires = partition.project_labels_hires( ...
    hires_data.HU, hires_data.mask, centroids, hu_min, hu_max);
fprintf('[TIMER] project_labels_hires: %.3f s\n', toc(t_proj));

t_reassign = tic;
labels_hires = density.reassign_islands_majority( ...
    labels_hires, hires_data.mask, k, D, hires_data.HU, hires_data.ds.spacing);
fprintf('[TIMER] reassign_islands_majority: %.3f s\n', toc(t_reassign));

t_relabel = tic;
[labels_final_hires, order_hires, means_hires] = density.relabel_by_hu( ...
    labels_hires, hires_data.HU, k, hires_data.mask);
fprintf('[TIMER] relabel_by_hu: %.3f s\n', toc(t_relabel));

k_eff = double(max(labels_final_hires(:)));
if k_eff < k
    fprintf('[Info] Effective label count is %d (requested k=%d). Adjusting.\n', k_eff, k);
    k = k_eff;
end

% diagnostics
onehot_sum = zeros(size(labels_final_hires), 'uint8');
for ii = 1:k, onehot_sum = onehot_sum + uint8(labels_final_hires == ii); end
gap = hires_data.mask & (onehot_sum == 0);
over = onehot_sum > 1;
fprintf('[Diag] Seam check voxels: gaps=%d, overlaps=%d\n', nnz(gap), nnz(over));

fprintf('[TIMER] Step 3 total: %.3f s\n\n', toc(t_all_step3));

% ---------- Step 4: Generate region meshes ----------
fprintf('--- Step 4: Generating %d Meshes from Labeled Volume ---\n', k);
tic;
meshes = partition.mesh_labeled_regions(labels_final_hires, hires_data.ds, k);
fprintf('Mesh generation complete. (%.2f seconds)\n\n', toc);

% ---------- Step 5: (Optional) Clip against shell ----------
if ~isempty(FV_shell)
    fprintf('--- Step 5: Clipping regions against outer shell (mesh boolean) ---\n');
    t_step5 = tic;
    [FV_shell.vertices, FV_shell.faces] = meshcheckrepair(FV_shell.vertices, FV_shell.faces, 'dup');
    [FV_shell.vertices, FV_shell.faces] = meshcheckrepair(FV_shell.vertices, FV_shell.faces, 'isolated');

    tri_counts = zeros(numel(meshes),1);
    for i = 1:numel(meshes)
        m = meshes{i};
        tri_counts(i) = (~isempty(m) && isfield(m,'faces')) * size(m.faces,1);
    end
    fprintf('[TIMER] Step 5 pre-clip: total faces=%d (per-part max=%d)\n', sum(tri_counts), max(tri_counts));

    meshes_out = meshes;              % preallocate
    clip_msgs  = cell(numel(meshes),1);

    % Parallel clip per part with UNIQUE temp dirs to avoid surfboolean collisions
    parfor i = 1:numel(meshes)
        m = meshes{i};
        if isempty(m) || ~isstruct(m) || ~isfield(m,'vertices') || isempty(m.vertices)
            meshes_out{i} = m;
            clip_msgs{i}  = sprintf('[TIMER]   clip part %d: skipped (empty)\n', i);
            continue;
        end

        % === Unique temp dir per iteration to avoid file collisions and long paths ===
        iter_tmp = fullfile(tempdir, 'iso2mesh_tmp', sprintf('w%03d', i));
        if ~exist(iter_tmp,'dir')
            mkdir(iter_tmp);
        end
        setenv('TMPDIR', iter_tmp);
        setenv('TEMP',   iter_tmp);
        setenv('TMP',    iter_tmp);

        t0 = tic;
        faces_in  = size(m.faces,1);
        clipped   = m;    % default if error
        ok        = true;
        errmsg    = '';

        try
            clipped = partition.clip_region_with_shell(m, FV_shell);
        catch ME
            ok     = false;
            errmsg = ME.message;
        end

        elapsed   = toc(t0);
        faces_out = size(clipped.faces,1);
        if ok
            clip_msgs{i} = sprintf('[TIMER]   clip part %d: %.3f s (faces in=%d -> out=%d)\n', ...
                                   i, elapsed, faces_in, faces_out);
        else
            clip_msgs{i} = sprintf('[TIMER]   clip part %d: FAILED in %.3f s (%s)\n', ...
                                   i, elapsed, errmsg);
        end
        meshes_out{i} = clipped;
    end

    for i = 1:numel(clip_msgs), fprintf('%s', clip_msgs{i}); end
    meshes = meshes_out;

    fprintf('[TIMER] Step 5 total (all parts): %.3f s\n', toc(t_step5));
end

% ---------- Step 6: Save STLs + stats (deterministic order) ----------
fprintf('--- Step 6: Saving %d STL files to output directory ---\n', numel(meshes));
decimateRatio = 1.0;
msgs = cell(numel(meshes),1);
savedMask = false(numel(meshes),1);

% Build per-label stats for naming/report
stats = io.label_stats(labels_final_hires, hires_data.HU, hires_data.ds, k, hires_data.mask);

% Precompute inverse order once (broadcast into parfor)
invOrder = zeros(1,k);
for r=1:k, invOrder(order_hires(r)) = r; end

t_step6 = tic;
parfor i = 1:numel(meshes)
    m = meshes{i};
    if isempty(m) || ~isstruct(m) || ~isfield(m,'vertices') || isempty(m.vertices)
        msgs{i} = sprintf(' - Skipping Region %d (no mesh generated).', i);
        continue;
    end

    rank   = invOrder(i);
    meanHU = stats(rank).meanHU;
    volmm3 = stats(rank).mm3;

    stl_filename = fullfile(outputDir, sprintf('scaphoid_region_%02d_HU%0.0f_V%0.0f.stl', rank, meanHU, volmm3));
    try
        t_stl = tic;
        meshing.write_stl_binary(char(stl_filename), m, 'Decimate', decimateRatio);
        msgs{i} = sprintf(' - Saved %s (%.3f s, faces=%d)', stl_filename, toc(t_stl), size(m.faces,1));
        savedMask(i) = true;
    catch ME
        msgs{i} = sprintf(' - FAILED to save Region %d: %s', i, ME.message);
    end
end
fprintf('[TIMER] Step 6 STLs total: %.3f s\n', toc(t_step6));
for i = 1:numel(msgs), fprintf('%s\n', msgs{i}); end
fprintf('Successfully saved %d mesh files.\n\n', nnz(savedMask));

% Save a CSV of stats for quick audit
try
    T = struct2table(stats);
    writetable(T, fullfile(outputDir, 'region_stats.csv'));
catch ME
    warning('Could not write region_stats.csv: %s', ME.message);
end

% ---- Build names for regions in ascending density order (rank = 1..k) ----
names = cell(1,k);
for r = 1:k
    names{r} = sprintf('Scaphoid_Region_%02d', r);
end

% ---- Export 3MF (preferred) ----
try
    out3mf = fullfile(outputDir, 'scaphoid_regions.3mf');
    t_3mf = tic;
    io.export_3mf(meshes, names, out3mf);
    fprintf('[TIMER] 3MF export: %.3f s\n', toc(t_3mf));
catch ME
    warning('3MF export failed: %s', ME.message);
end

fprintf('=====================================================\n');
fprintf(' MESH PARTITIONING COMPLETED SUCCESSFULLY\n');
fprintf('=====================================================\n');
end

function value = get_cfg_number(cfg, sectionName, fieldName, defaultValue)
    value = defaultValue;
    if ~isstruct(cfg) || ~isfield(cfg, sectionName)
        return;
    end
    section = cfg.(sectionName);
    if isfield(section, fieldName)
        candidate = section.(fieldName);
        if isnumeric(candidate) && isscalar(candidate) && candidate > 0
            value = candidate;
        else
            error('cfg.%s.%s must be a positive scalar.', sectionName, fieldName);
        end
    end
end
