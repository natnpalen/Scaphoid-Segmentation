function visualize_results(ds, sep_result, seg_results, pack_results, opts)
% VISUALIZE_RESULTS  3D diagnostic visualization for the bone pipeline.
%
%   bone.visualize_results(ds, sep_result, seg_results, pack_results, opts)
%
% Creates:
%   Figure 1: 3D overview — all bones color-coded with tags
%   Figure 2: Per-bone cortical/cancellous 3D views

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
        fv.vertices(:,1) = fv.vertices(:,1) * spacing(2);
        fv.vertices(:,2) = fv.vertices(:,2) * spacing(1);
        fv.vertices(:,3) = fv.vertices(:,3) * spacing(3);

        patch(fv, 'FaceColor', colors(bi,:), 'EdgeColor', 'none', ...
            'FaceAlpha', 0.7);
        hold on;

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

% Show marker locations as small red dots
if isfield(sep_result, 'marker_mask') && any(sep_result.marker_mask(:))
    mk = sep_result.marker_mask;
    try
        fv_mk = isosurface(smooth3(double(mk), 'gaussian', 3), 0.5);
        if ~isempty(fv_mk.vertices)
            fv_mk.vertices(:,1) = fv_mk.vertices(:,1) * spacing(2);
            fv_mk.vertices(:,2) = fv_mk.vertices(:,2) * spacing(1);
            fv_mk.vertices(:,3) = fv_mk.vertices(:,3) * spacing(3);
            patch(fv_mk, 'FaceColor', [1 0 0], 'EdgeColor', 'none', ...
                'FaceAlpha', 0.9);
        end
    catch
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
        'Position', [100 100 1200 500]);

    n_cols = min(4, n_bones);
    for bi = 1:min(n_bones, n_cols)
        subplot(1, n_cols, bi);

        seg = seg_results{bi};
        cortical_mask = seg.cortical;
        cancellous_mask = seg.cancellous;

        try
            if any(cortical_mask(:))
                fv_c = isosurface(smooth3(double(cortical_mask), 'gaussian', 3), 0.5);
                if ~isempty(fv_c.vertices)
                    fv_c.vertices = fv_c.vertices .* spacing([2 1 3]);
                    patch(fv_c, 'FaceColor', [0.9 0.2 0.2], 'EdgeColor', 'none', ...
                        'FaceAlpha', 0.5, 'DisplayName', 'Cortical');
                    hold on;
                end
            end
            if any(cancellous_mask(:))
                fv_t = isosurface(smooth3(double(cancellous_mask), 'gaussian', 3), 0.5);
                if ~isempty(fv_t.vertices)
                    fv_t.vertices = fv_t.vertices .* spacing([2 1 3]);
                    patch(fv_t, 'FaceColor', [0.2 0.2 0.9], 'EdgeColor', 'none', ...
                        'FaceAlpha', 0.5, 'DisplayName', 'Cancellous');
                end
            end
        catch
        end

        axis equal vis3d off;
        camlight headlight; lighting gouraud;
        title(sprintf('Bone %d\nCort %.0f%% (red) | Canc %.0f%% (blue)', bi, ...
            seg.info.cortical_fraction*100, (1-seg.info.cortical_fraction)*100));
        legend('Location', 'southoutside');
    end
end

% ========================================================================
%  Figure 3: Specimen packing (if any placed)
% ========================================================================
has_specimens = false;
if ~isempty(pack_results)
    for bi = 1:numel(pack_results)
        bp = pack_results{bi};
        if ~iscell(bp), continue; end
        for ri = 1:numel(bp)
            if isstruct(bp{ri}) && ~isempty(bp{ri})
                has_specimens = true;
                break;
            end
        end
        if has_specimens, break; end
    end
end

if has_specimens
    shape_colors = [0.2 0.8 0.2;   % Bend - green
                    0.8 0.8 0.2;   % Compression - yellow
                    0.8 0.2 0.8;   % Punch - magenta
                    0.2 0.8 0.8];  % Shear - cyan

    fig3 = figure('Name', 'Specimen Packing', 'Color', 'w', ...
        'Position', [250 250 1000 700]);

    for bi = 1:min(n_bones, 4)
        if bi > numel(pack_results), continue; end

        subplot(1, min(n_bones, 4), bi);

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

        bp = pack_results{bi};
        if iscell(bp)
            for ri = 1:numel(bp)
                if ~isstruct(bp{ri}), continue; end
                for pi = 1:numel(bp{ri})
                    p_info = bp{ri}(pi);
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
        end

        axis equal vis3d off;
        camlight headlight; lighting gouraud;
        title(sprintf('Bone %d', bi));
    end
    sgtitle('Specimen Packing');
end

% ---- Save figures ----
if isfield(opts, 'OutputDir') && ~isempty(opts.OutputDir)
    outDir = opts.OutputDir;
    if ~exist(outDir, 'dir'), mkdir(outDir); end

    try
        saveas(fig1, fullfile(outDir, 'bone_separation_3d.png'));
        if exist('fig2', 'var')
            saveas(fig2, fullfile(outDir, 'cortical_cancellous_3d.png'));
        end
        fprintf('  [Viz] Saved figures to %s\n', outDir);
    catch ME
        warning('Figure save failed: %s', ME.message);
    end
end
end
