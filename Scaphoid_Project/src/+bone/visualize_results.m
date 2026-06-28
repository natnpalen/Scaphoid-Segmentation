function visualize_results(ds, sep_result, seg_results, pack_results, opts)
% VISUALIZE_RESULTS  3D overview for the bone pipeline.
%
%   bone.visualize_results(ds, sep_result, seg_results, pack_results, opts)
%
% Creates one figure: 3D overview with color-coded bones and metal markers.

vol = double(ds.HU);
spacing = ds.spacing;
bones = sep_result.bones;
n_bones = numel(bones);

colors = [0.2 0.6 1.0;   % blue
          1.0 0.5 0.0;   % orange
          0.2 0.8 0.3;   % green
          0.9 0.2 0.9;   % magenta
          0.0 0.8 0.8;   % cyan
          1.0 0.8 0.2;   % yellow
          0.6 0.3 0.1;   % brown
          0.5 0.5 0.5];  % gray
if n_bones > size(colors, 1)
    colors = [colors; lines(n_bones - size(colors, 1))];
end

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
            lbl = sprintf('Bone %d (tag %d)\n%.0f mm^3, HU=%.0f', ...
                bi, bones{bi}.tag_id, bones{bi}.volume_mm3, bones{bi}.mean_hu);
        else
            lbl = sprintf('Bone %d\n%.0f mm^3, HU=%.0f', ...
                bi, bones{bi}.volume_mm3, bones{bi}.mean_hu);
        end
        text(cm(2), cm(1), cm(3), lbl, 'FontSize', 9, 'FontWeight', 'bold', ...
            'HorizontalAlignment', 'center', 'Color', colors(bi,:)*0.6);
    catch
        continue;
    end
end

% Markers intentionally NOT shown — bones only

axis equal vis3d off;
camlight headlight; lighting gouraud;
title(sprintf('Bone Separation: %d bones found', n_bones));
rotate3d on;

% ---- Cortical / Cancellous 3D view ----
has_seg = ~isempty(seg_results) && numel(seg_results) >= n_bones && ...
    isstruct(seg_results{1}) && isfield(seg_results{1}, 'info') && ...
    isfield(seg_results{1}.info, 'cortical_volume_mm3') && ...
    seg_results{1}.info.cortical_volume_mm3 > 0;

if has_seg
    fig2 = figure('Name', 'Cortical / Cancellous Segmentation', 'Color', 'w', ...
        'Position', [100 100 900 700]);

    for bi = 1:n_bones
        cort = seg_results{bi}.cortical;
        canc = seg_results{bi}.cancellous;

        % Cortical shell — opaque, bone-tinted
        if any(cort(:))
            try
                fv_c = isosurface(smooth3(double(cort), 'gaussian', 3), 0.5);
                if ~isempty(fv_c.vertices)
                    fv_c.vertices(:,1) = fv_c.vertices(:,1) * spacing(2);
                    fv_c.vertices(:,2) = fv_c.vertices(:,2) * spacing(1);
                    fv_c.vertices(:,3) = fv_c.vertices(:,3) * spacing(3);
                    patch(fv_c, 'FaceColor', colors(bi,:), 'EdgeColor', 'none', ...
                        'FaceAlpha', 0.3);
                    hold on;
                end
            catch, end
        end

        % Cancellous interior — darker shade, semi-transparent
        if any(canc(:))
            try
                fv_n = isosurface(smooth3(double(canc), 'gaussian', 3), 0.5);
                if ~isempty(fv_n.vertices)
                    fv_n.vertices(:,1) = fv_n.vertices(:,1) * spacing(2);
                    fv_n.vertices(:,2) = fv_n.vertices(:,2) * spacing(1);
                    fv_n.vertices(:,3) = fv_n.vertices(:,3) * spacing(3);
                    patch(fv_n, 'FaceColor', colors(bi,:)*0.5, 'EdgeColor', 'none', ...
                        'FaceAlpha', 0.6);
                    hold on;
                end
            catch, end
        end

        cm = bones{bi}.centroid_mm;
        si = seg_results{bi}.info;
        lbl = sprintf('Bone %d\nCort %.0f%% (depth %.1fmm)', ...
            bi, si.cortical_fraction*100, si.mean_cortical_depth_mm);
        text(cm(2), cm(1), cm(3), lbl, 'FontSize', 9, 'FontWeight', 'bold', ...
            'HorizontalAlignment', 'center', 'Color', colors(bi,:)*0.6);
    end

    axis equal vis3d off;
    camlight headlight; lighting gouraud;
    title('Cortical (translucent) / Cancellous (solid) Segmentation');
    rotate3d on;
end

% ---- Save figures ----
if isfield(opts, 'OutputDir') && ~isempty(opts.OutputDir)
    outDir = opts.OutputDir;
    if ~exist(outDir, 'dir'), mkdir(outDir); end

    try
        saveas(fig1, fullfile(outDir, 'bone_separation_3d.png'));
        fprintf('  [Viz] Saved bone_separation_3d.png\n');
    catch ME
        warning('Figure save failed: %s', ME.message);
    end

    if has_seg
        try
            saveas(fig2, fullfile(outDir, 'cortical_cancellous_3d.png'));
            fprintf('  [Viz] Saved cortical_cancellous_3d.png\n');
        catch ME
            warning('Figure save failed: %s', ME.message);
        end
    end
end
end
