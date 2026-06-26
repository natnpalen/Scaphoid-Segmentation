function visualize_results(ds, sep_result, seg_results, pack_results, opts)
% VISUALIZE_RESULTS  3D and 2D diagnostic visualization for the bone pipeline.
%
%   bone.visualize_results(ds, sep_result, seg_results, pack_results, opts)
%
% Creates:
%   Figure 1: 3D overview — all bones color-coded with tags
%   Figure 2: Per-bone cortical/cancellous 3D views
%   Figure 3: Axial slice montage with overlays
%   Figure 4: HU histograms per bone
%   Figure 5: Specimen packing views (if any placed)

vol = double(ds.HU);
spacing = ds.spacing;
bones = sep_result.bones;
n_bones = numel(bones);

colors = lines(max(n_bones, 4));

% ========================================================================
%  Figure 1: 3D overview — all bones
% ========================================================================
fig1 = figure('Name', 'Bone Separation Overview', 'Color', 'w', ...
    'Position', [50 50 900 700]);

for bi = 1:n_bones
    mask_i = bones{bi}.mask;
    if ~any(mask_i(:)), continue; end

    try
        fv = isosurface(smooth3(double(mask_i), 'gaussian', 3), 0.5);
        if isempty(fv.vertices), continue; end
        % Scale vertices to mm
        fv.vertices(:,1) = fv.vertices(:,1) * spacing(2);
        fv.vertices(:,2) = fv.vertices(:,2) * spacing(1);
        fv.vertices(:,3) = fv.vertices(:,3) * spacing(3);

        p = patch(fv, 'FaceColor', colors(bi,:), 'EdgeColor', 'none', ...
            'FaceAlpha', 0.7);
        hold on;

        % Label
        cm = bones{bi}.centroid_mm;
        if ~isempty(bones{bi}.tag_id)
            lbl = sprintf('Bone %d (tag %d)\n%.0f mm^3', bi, bones{bi}.tag_id, bones{bi}.volume_mm3);
        else
            lbl = sprintf('Bone %d\n%.0f mm^3', bi, bones{bi}.volume_mm3);
        end
        text(cm(2), cm(1), cm(3), lbl, 'FontSize', 9, 'FontWeight', 'bold', ...
            'HorizontalAlignment', 'center', 'Color', colors(bi,:)*0.6);
    catch
        continue;
    end
end

axis equal vis3d off;
camlight headlight; lighting gouraud;
title(sprintf('Bone Separation: %d bones found', n_bones));
rotate3d on;

% ========================================================================
%  Figure 2: Per-bone cortical/cancellous
% ========================================================================
if ~isempty(seg_results)
    fig2 = figure('Name', 'Cortical / Cancellous Segmentation', 'Color', 'w', ...
        'Position', [100 100 1200 400]);

    n_cols = min(4, n_bones);
    for bi = 1:min(n_bones, n_cols)
        subplot(1, n_cols, bi);

        seg = seg_results{bi};
        cortical_mask = seg.cortical;
        cancellous_mask = seg.cancellous;

        % Show cortical in red, cancellous in blue
        try
            if any(cortical_mask(:))
                fv_c = isosurface(smooth3(double(cortical_mask), 'gaussian', 3), 0.5);
                if ~isempty(fv_c.vertices)
                    fv_c.vertices = fv_c.vertices .* spacing([2 1 3]);
                    patch(fv_c, 'FaceColor', [0.9 0.2 0.2], 'EdgeColor', 'none', ...
                        'FaceAlpha', 0.5);
                    hold on;
                end
            end
            if any(cancellous_mask(:))
                fv_t = isosurface(smooth3(double(cancellous_mask), 'gaussian', 3), 0.5);
                if ~isempty(fv_t.vertices)
                    fv_t.vertices = fv_t.vertices .* spacing([2 1 3]);
                    patch(fv_t, 'FaceColor', [0.2 0.2 0.9], 'EdgeColor', 'none', ...
                        'FaceAlpha', 0.5);
                end
            end
        catch
        end

        axis equal vis3d off;
        camlight headlight; lighting gouraud;
        title(sprintf('Bone %d\nCort %.0f%% | Canc %.0f%%', bi, ...
            seg.info.cortical_fraction*100, (1-seg.info.cortical_fraction)*100));
    end

    % Legend
    legend({'Cortical', 'Cancellous'}, 'Location', 'southoutside');
end

% ========================================================================
%  Figure 3: Axial slice montage
% ========================================================================
fig3 = figure('Name', 'Axial Slice Montage', 'Color', 'w', ...
    'Position', [150 150 1200 800]);

% Combined bone mask
all_bones = false(size(vol));
bone_labels = zeros(size(vol));
for bi = 1:n_bones
    all_bones = all_bones | bones{bi}.mask;
    bone_labels(bones{bi}.mask) = bi;
end

% Pick representative slices
[~, ~, ss] = ind2sub(size(all_bones), find(all_bones));
if ~isempty(ss)
    s_range = [min(ss), max(ss)];
    n_slices = min(16, s_range(2) - s_range(1) + 1);
    slice_idx = round(linspace(s_range(1), s_range(2), n_slices));
else
    slice_idx = round(linspace(1, size(vol,3), 16));
end

n_rows = ceil(sqrt(numel(slice_idx)));
n_cols_grid = ceil(numel(slice_idx) / n_rows);

for si = 1:numel(slice_idx)
    subplot(n_rows, n_cols_grid, si);

    z = slice_idx(si);
    slice_hu = vol(:,:,z);
    slice_labels = bone_labels(:,:,z);

    % Display HU as grayscale
    imagesc(slice_hu, [-500 1500]);
    colormap(gray);
    hold on;

    % Overlay bone contours
    for bi = 1:n_bones
        bone_slice = bones{bi}.mask(:,:,z);
        if any(bone_slice(:))
            contour(bone_slice, [0.5 0.5], 'Color', colors(bi,:), 'LineWidth', 1.5);
        end
    end

    % Overlay cortical/cancellous if available
    if ~isempty(seg_results)
        for bi = 1:min(n_bones, numel(seg_results))
            cort_slice = seg_results{bi}.cortical(:,:,z);
            canc_slice = seg_results{bi}.cancellous(:,:,z);
            if any(cort_slice(:))
                contour(cort_slice, [0.5 0.5], 'Color', [1 0.3 0.3], 'LineWidth', 0.8);
            end
        end
    end

    axis image off;
    title(sprintf('z=%d', z), 'FontSize', 8);
end
sgtitle('Axial Slices with Bone Contours');

% ========================================================================
%  Figure 4: HU histograms per bone
% ========================================================================
fig4 = figure('Name', 'HU Histograms', 'Color', 'w', ...
    'Position', [200 200 1000 400]);

for bi = 1:min(n_bones, 4)
    subplot(1, min(n_bones, 4), bi);

    hu_vals = vol(bones{bi}.mask);
    histogram(hu_vals, 100, 'FaceColor', colors(bi,:), 'EdgeColor', 'none');
    hold on;

    if ~isempty(seg_results) && bi <= numel(seg_results)
        otsu = seg_results{bi}.info.otsu_threshold;
        xline(otsu, 'r--', 'LineWidth', 2);
        text(otsu, 0, sprintf(' Otsu=%.0f', otsu), 'Color', 'r', ...
            'VerticalAlignment', 'bottom', 'FontSize', 9);
    end

    xlabel('HU');
    ylabel('Count');
    title(sprintf('Bone %d: %.0f mm^3', bi, bones{bi}.volume_mm3));
    grid on;
end
sgtitle('HU Distribution per Bone');

% ========================================================================
%  Figure 5: Specimen packing (if any placed)
% ========================================================================
if ~isempty(pack_results)
    has_specimens = false;
    for bi = 1:numel(pack_results)
        if ~isempty(pack_results{bi}) && ~isempty(fieldnames(pack_results{bi}))
            for ri = 1:numel(pack_results{bi})
                if ~isempty(pack_results{bi}{ri})
                    has_specimens = true;
                    break;
                end
            end
        end
        if has_specimens, break; end
    end

    if has_specimens
        fig5 = figure('Name', 'Specimen Packing', 'Color', 'w', ...
            'Position', [250 250 1000 700]);

        shape_colors = [0.2 0.8 0.2;   % Bend - green
                        0.8 0.8 0.2;   % Compression - yellow
                        0.8 0.2 0.8;   % Punch - magenta
                        0.2 0.8 0.8];  % Shear - cyan

        for bi = 1:min(n_bones, 4)
            if bi > numel(pack_results), continue; end

            subplot(1, min(n_bones, 4), bi);

            % Show bone outline
            mask_i = bones{bi}.mask;
            if any(mask_i(:))
                try
                    fv = isosurface(smooth3(double(mask_i), 'gaussian', 3), 0.5);
                    fv.vertices = fv.vertices .* spacing([2 1 3]);
                    patch(fv, 'FaceColor', [0.8 0.8 0.8], 'EdgeColor', 'none', ...
                        'FaceAlpha', 0.15);
                    hold on;
                catch
                end
            end

            % Show specimens
            bone_packs = pack_results{bi};
            for ri = 1:numel(bone_packs)
                if isempty(bone_packs{ri}), continue; end
                for pi = 1:numel(bone_packs{ri})
                    p_info = bone_packs{ri}(pi);
                    if ~isfield(p_info, 'mask'), continue; end
                    try
                        fv_p = isosurface(smooth3(double(p_info.mask), 'gaussian', 3), 0.5);
                        if isempty(fv_p.vertices), continue; end
                        fv_p.vertices = fv_p.vertices .* spacing([2 1 3]);
                        ci = min(p_info.shape_idx, size(shape_colors, 1));
                        patch(fv_p, 'FaceColor', shape_colors(ci,:), ...
                            'EdgeColor', 'none', 'FaceAlpha', 0.8);
                    catch
                    end
                end
            end

            axis equal vis3d off;
            camlight headlight; lighting gouraud;
            title(sprintf('Bone %d', bi));
        end
        sgtitle('Specimen Packing');
    end
end

% ---- Save figures ----
if isfield(opts, 'OutputDir') && ~isempty(opts.OutputDir)
    outDir = opts.OutputDir;
    if ~exist(outDir, 'dir'), mkdir(outDir); end

    try
        saveas(fig1, fullfile(outDir, 'bone_separation_3d.png'));
        saveas(fig3, fullfile(outDir, 'axial_slices.png'));
        saveas(fig4, fullfile(outDir, 'hu_histograms.png'));
        fprintf('  [Viz] Saved figures to %s\n', outDir);
    catch ME
        warning('Figure save failed: %s', ME.message);
    end
end
end
