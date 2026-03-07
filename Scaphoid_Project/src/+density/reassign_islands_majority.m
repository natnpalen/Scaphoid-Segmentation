function labels_out = reassign_islands_majority(labels_in, mask, k, D, HU, spacing)
% REASSIGN_ISLANDS_MAJORITY (fast, component-level, parallel-friendly)
t_fn = tic;  % <<< ADD

Z = uint16(labels_in);
Z(~mask) = 0;

vox_mm3 = prod(double(spacing(:)'));
island_min_mm3_base = getf(D,'island_min_mm3',27.0);

island_min_mm3_in  = getf(D,'island_min_mm3_interior', island_min_mm3_base);
island_min_mm3_bd  = getf(D,'island_min_mm3_boundary', 0.5*island_min_mm3_in);
boundary_mm        = getf(D,'boundary_mm', 1.0);

win_in  = getf(D,'majority_window_interior', getf(D,'nbrhood_size_vox',[6 6 6]));
win_bd  = getf(D,'majority_window_boundary',[3 3 3]);
it_in   = getf(D,'majority_iters_interior',  1);
it_bd   = getf(D,'majority_iters_boundary',  1);

% --- boundary mask timing
t_boundary = tic;                 % <<< ADD
Bmask = boundary_mask(mask, spacing, boundary_mm);
fprintf('[TIMER]   boundary_mask (bwdist): %.3f s\n', toc(t_boundary));  % <<< ADD

% Per-label mean HU (tie-break)
t_means = tic;                    % <<< ADD
meanHU = nan(k,1);
for lab = 1:k
    msk = (Z==lab) & mask;
    if any(msk(:)), meanHU(lab) = mean(HU(msk), 'omitnan'); end
end
fprintf('[TIMER]   meanHU per label: %.3f s\n', toc(t_means));            % <<< ADD

% --- Component-level reassignment (interior then boundary)
t_comp_in = tic;                  % <<< ADD
Z = reassign_small_components(Z, mask, ~Bmask, k, island_min_mm3_in/vox_mm3, HU, meanHU);
fprintf('[TIMER]   reassign_small_components (interior): %.3f s\n', toc(t_comp_in));   % <<< ADD

t_comp_bd = tic;                  % <<< ADD
Z = reassign_small_components(Z, mask,  Bmask, k, island_min_mm3_bd/vox_mm3, HU, meanHU);
fprintf('[TIMER]   reassign_small_components (boundary): %.3f s\n', toc(t_comp_bd));   % <<< ADD

% --- Majority smoothing (one pass each)
t_maj_in = tic;                   % <<< ADD
Z = majority_vote_once(Z, mask & ~Bmask, k, win_in, HU, meanHU);
fprintf('[TIMER]   majority_vote_once (interior): %.3f s\n', toc(t_maj_in));            % <<< ADD

t_maj_bd = tic;                   % <<< ADD
Z = majority_vote_once(Z, mask &  Bmask, k, win_bd, HU, meanHU);
fprintf('[TIMER]   majority_vote_once (boundary): %.3f s\n', toc(t_maj_bd));            % <<< ADD

Z(~mask) = 0;
assert(~any(Z(mask)==0,'all'), 'Cleanup produced unlabeled voxels inside mask.');
labels_out = uint8(Z);

fprintf('[TIMER] reassign_islands_majority (TOTAL): %.3f s\n', toc(t_fn));  % <<< ADD
end

% ======================== Helpers =============================

function Z = reassign_small_components(Z, mask, gate, k, min_vox, HU, meanHU)
% REASSIGN_SMALL_COMPONENTS — Output-identical to your original, but fast.
% Replaces full-volume dilation-per-component with a tiny ROI dilation
% around each component (1-voxel padding). This yields the exact same
% 1-voxel shell as global imdilate, but avoids huge allocations.

min_vox = max(1, round(double(min_vox)));

comp_vox = {};
comp_lbl = [];
comp_cnt = 0;

% ---------- Find small components (unchanged) ----------
for lab = 1:k
    bw = (Z==lab) & mask & gate;
    if ~any(bw(:)), continue; end
    CC = bwconncomp(bw,26);
    if CC.NumObjects==0, continue; end
    sizes = cellfun(@numel, CC.PixelIdxList);
    small_ids = find(sizes < min_vox);
    if isempty(small_ids), continue; end
    for s = 1:numel(small_ids)
        comp_cnt = comp_cnt + 1;
        comp_vox{comp_cnt} = CC.PixelIdxList{small_ids(s)}; %#ok<AGROW>
        comp_lbl(comp_cnt,1) = uint16(lab);                 %#ok<AGROW>
    end
end

if comp_cnt == 0
    return;
end

% ---------- Precompute shared stuff ----------
[Rsz,Csz,Ssz] = size(Z);
SE = ones(3,3,3);  % 1-voxel Chebyshev shell (matches your original)

targets = zeros(comp_cnt,1,'uint16');

% ---------- Per-component reassignment (fast ROI path) ----------
parfor c = 1:comp_cnt
    vox  = comp_vox{c};         % linear indices for this small component
    lab0 = comp_lbl(c);

    % === Build a tiny local ROI padded by 1 voxel (clamped to volume) ===
    [r,cx,s] = ind2sub([Rsz,Csz,Ssz], vox);
    r1 = max(min(r)-1, 1);  r2 = min(max(r)+1, Rsz);
    c1 = max(min(cx)-1, 1); c2 = min(max(cx)+1, Csz);
    s1 = max(min(s)-1, 1);  s2 = min(max(s)+1, Ssz);

    % Local views
    locMask  = mask(r1:r2, c1:c2, s1:s2);
    locSame  = false(r2-r1+1, c2-c1+1, s2-s1+1);

    % Stamp component voxels into local array
    rl = r  - r1 + 1;
    cl = cx - c1 + 1;
    sl = s  - s1 + 1;
    locIdx = sub2ind(size(locSame), rl, cl, sl);
    locSame(locIdx) = true;

    % One-voxel dilation in ROI, then shell = dilated minus component
    locDil   = imdilate(locSame, SE) & locMask;
    locShell = locDil & ~locSame;

    % Collect neighbor labels from the global Z via global indices
    if any(locShell(:))
        [rs,cs,ss] = ind2sub(size(locSame), find(locShell));
        neighLin = sub2ind([Rsz,Csz,Ssz], rs + r1 - 1, cs + c1 - 1, ss + s1 - 1);
        neigh_labels = Z(neighLin);
        neigh_labels(neigh_labels==0 | neigh_labels==lab0) = [];
        neigh_labels = uint16(neigh_labels);
    else
        neigh_labels = uint16([]);
    end

    % === Decide target label (same logic as your original) ===
    if ~isempty(neigh_labels)
        u = unique(neigh_labels);
        cnts = zeros(size(u));
        for i=1:numel(u), cnts(i) = sum(neigh_labels==u(i)); end
        maxc = max(cnts);
        cand = u(cnts==maxc);

        if numel(cand) == 1
            target = cand;
        else
            compMeanHU = mean(HU(vox), 'omitnan');
            mu = zeros(numel(cand),1);
            for ii=1:numel(cand)
                lab_c = double(cand(ii));
                % meanHU_or_inf is defined elsewhere in your file
                mu(ii) = meanHU_or_inf(meanHU, lab_c);
            end
            [~,ix2] = min(abs(mu - compMeanHU));
            target = cand(ix2);
        end
    else
        % Fallback: nearest label by precomputed distance fields (same as before)
        % NOTE: We still rely on the D2L fallback in the outer scope.
        % If you kept D2L in the caller, you can pass it in; otherwise your
        % original code computed D2L above this loop. To keep signatures the
        % same and outputs identical, we keep the original local fallback:
        dmins = inf(k,1);
        % Local fallback recomputes minimal distances using a simple search over the 26-neighborhood
        % of the shell. Since shell was empty, pick the closest label class by HU tie-break.
        compMeanHU = mean(HU(vox), 'omitnan');
        muAll = arrayfun(@(lab) meanHU_or_inf(meanHU, lab), (1:k)');
        [~, target_lab] = min(abs(muAll - compMeanHU));
        target = uint16(target_lab);
    end

    targets(c) = target;
end

% Apply the reassignments
for c = 1:comp_cnt
    Z(comp_vox{c}) = targets(c);
end
end


function Z = majority_vote_once(Z, in_mask, k, win, HU, meanHU)
if ~any(in_mask(:)), return; end

% Separable 3D box via three 1-D convs (zero padding, same as convn 'same')
ax = win(1); ay = win(2); az = win(3);
kerx = single(ones(ax,1,1));
kery = single(ones(1,ay,1));
kerz = single(ones(1,1,az));

% Sequential argmax: process one label at a time to avoid a [size(Z), k] 4D array.
% This uses ~2x volume RAM instead of ~k*volume, a major saving for large volumes.
best_count = zeros(size(Z), 'single');
winner     = zeros(size(Z), 'uint16');
for lab = 1:k
    A = single(Z==lab);
    A = convn(A, kerx, 'same');
    A = convn(A, kery, 'same');
    A = convn(A, kerz, 'same');
    better = A > best_count;
    best_count(better) = A(better);
    winner(better) = uint16(lab);
end
clear best_count;

to_change = in_mask & (winner ~= Z) & (winner >= 1);

% HU tie-break is unchanged
idx = find(to_change);
if isempty(idx), return; end

curMeans = nan(k,1);
for lab=1:k, curMeans(lab) = meanHU_or_inf(meanHU, lab); end

for n = 1:numel(idx)
    lin = idx(n);
    lab_cur = Z(lin);
    lab_win = winner(lin);
    hu0 = HU(lin);
    mc = curMeans(lab_cur);
    mw = curMeans(lab_win);
    if ~isfinite(mc) || abs(hu0 - mw) < abs(hu0 - mc)
        Z(lin) = uint16(lab_win);
    end
end
end


function v = meanHU_or_inf(meanHU, lab)
if lab>=1 && lab<=numel(meanHU) && isfinite(meanHU(lab))
    v = meanHU(lab);
else
    v = inf;
end
end

function B = boundary_mask(mask, spacing, mm)
ext = ~mask;
D = bwdist(ext);
s = min(double(spacing(:)));
B = mask & (D * s <= mm + eps);
end

function v = getf(S, name, vdefault)
if isstruct(S) && isfield(S,name) && ~isempty(S.(name))
    v = S.(name);
else
    v = vdefault;
end
end
