function output_filename = run_density_analysis(dicomFolder, cfg)
% RUN_DENSITY_ANALYSIS (refactored)
if nargin < 2 || isempty(cfg)
    cfg = struct();
end
viz = get_cfg_flag(cfg, 'showPopups', false);

% --- Step 1: pipeline ---
fprintf('--- Step 1: Running Scaphoid Segmentation for:\n%s\n', dicomFolder);
out = run_segmentation(dicomFolder, 'WriteNifti', true);
fprintf('Pipeline complete.\n\n');

% --- Step 2: load HU/mask/ds ---
fprintf('--- Step 2: Loading Volumetric Data for Analysis ---\n');
[HU, scaphoid_mask, ds] = density.load_masked_hu(out.outputDir, out.mask, out.ds);
fprintf('Volumetric data loaded.\n\n');

D = density.density_defaults();
k = D.k;
sigma_mm = D.sigma_mm;

% --- Step 3: K-means (masked) → smooth → cleanup (coverage-preserving) ---
fprintf('--- Step 3: Performing K-Means Clustering Analysis ---\n');
% NOTE: capture centroids for projection later (NEW)
[labels_raw, centroids_sorted] = density.kmeans_label_masked(HU, scaphoid_mask, D.k, D.rngSeed);

labels_smooth = density.smooth_labels_masked(labels_raw, scaphoid_mask, D.k, D.neigh_mm, ds.spacing);

% NEW: coverage-preserving cleanup (reassign; boundary-aware)
zone_map_kmeans_smoothed = density.reassign_islands_majority( ...
    labels_smooth, scaphoid_mask, D.k, D, HU, ds.spacing);

% NEW: enforce 1..k == low→high HU at low-res
[zone_map_kmeans_smoothed, ~, ~] = density.relabel_by_hu( ...
    zone_map_kmeans_smoothed, HU, D.k, scaphoid_mask);

fprintf('K-Means clustering, mask-aware smoothing, and coverage-safe cleanup complete.\n\n');

% --- Step 4: Continuous normalization (masked) ---
fprintf('--- Step 4: Performing Continuous Normalization Analysis (masked) ---\n');
HU_blur  = density.masked_gaussian_blur(HU, scaphoid_mask, D.sigma_mm, ds.spacing);
[hu_min, hu_max] = density.percentile_clamp(HU_blur(scaphoid_mask), D.clamp_prct);
gradient_map = density.normalize_hu_to_density(HU_blur, scaphoid_mask, hu_min, hu_max);
fprintf('Continuous normalization (masked) complete.\n\n');

% --- Step 5: Visualizations (optional) ---
if viz
    fprintf('--- Step 5: Generating Visualizations (ShowPopups = true) ---\n');
    vis.show_labels_and_gradient(zone_map_kmeans_smoothed, gradient_map, ds, D.k, scaphoid_mask);
else
    fprintf('--- Step 5: Skipped visualizations (ShowPopups = false) ---\n');
end

% --- Step 6: Save results (NEW: save centroids) ---
fprintf('--- Step 6: Saving analysis results to file ---\n');
output_filename = fullfile(out.outputDir, 'density_analysis_results.mat');
save(output_filename, ...
    'gradient_map', ...
    'zone_map_kmeans_smoothed', ...
    'scaphoid_mask', ...
    'ds', ...
    'k', ...
    'hu_min','hu_max', ...
    'sigma_mm', ...
    'centroids_sorted');   % <-- NEW

fprintf('Results saved to:\n%s\n', output_filename);
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
