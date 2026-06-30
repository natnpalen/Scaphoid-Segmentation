% ----------------------- DICOM robust loader -----------------------------
function ds = series_load(folder, varargin)
o = struct('TargetIsoMM',[],'Smoothing',false);
o = utils.parse_opts(o, varargin{:});
Vraw = []; infosCell = {};
% 1) Try dicomCollection (version-agnostic: no NumImages dependency)
try
 dc = dicomCollection(folder, 'IncludeSubfolders', true);
 fprintf('[Load] dicomCollection: %d series found\n', height(dc));
 if height(dc) > 0
     % Show what was found
     if any(strcmpi(dc.Properties.VariableNames,'Modality'))
         for ri = 1:height(dc)
             nImages = 0;
             if any(strcmpi(dc.Properties.VariableNames,'NumImages'))
                 nImages = dc.NumImages(ri);
             end
             fprintf('       Series %d: Modality=%s, %d images\n', ...
                 ri, string(dc.Modality(ri)), nImages);
         end
         isCT = strcmpi(dc.Modality,'CT');
         if any(isCT)
             dc2 = dc(isCT,:);
         else
             fprintf('[Load] No CT modality found, using all series\n');
             dc2 = dc;
         end
     else
         fprintf('[Load] No Modality column in dicomCollection\n');
         dc2 = dc;
     end
     % Pick first series — pass the table row directly (version-safe)
     idx = 1;
     try
         [Vraw,~,meta] = dicomreadVolume(dc2(idx,:));
         fprintf('[Load] dicomreadVolume(collection) OK: size %s\n', mat2str(size(Vraw)));
     catch ME_inner
         fprintf('[Load] dicomreadVolume(collection) failed: %s\n', ME_inner.message);
         try
             [Vraw,~,meta] = dicomreadVolume(folder);
             fprintf('[Load] dicomreadVolume(folder) fallback OK: size %s\n', mat2str(size(Vraw)));
         catch ME_inner2
             fprintf('[Load] dicomreadVolume(folder) fallback also failed: %s\n', ME_inner2.message);
         end
     end
     if ~isempty(Vraw)
         infosCell = normMeta(meta);
         fprintf('[Load] Meta cells: %d (volume slices: %d)\n', numel(infosCell), size(Vraw, ndims(Vraw)));
     end
 end
catch ME1
 fprintf('[Load] dicomCollection failed: %s\n', ME1.message);
end
% 2) Direct bulk read on the folder (works great for Slicer exports)
if isempty(Vraw)
 fprintf('[Load] Method 1 produced no volume, trying direct dicomreadVolume(folder)...\n');
 try
     [Vraw,~,meta] = dicomreadVolume(folder);
     infosCell = normMeta(meta);
     fprintf('[Load] dicomreadVolume(folder) OK: size %s, meta cells: %d\n', ...
         mat2str(size(Vraw)), numel(infosCell));
 catch ME2
     fprintf('[Load] dicomreadVolume(folder) failed: %s\n', ME2.message);
     % Show folder contents to help diagnose
     d = dir(folder);
     d = d(~[d.isdir]);
     if isempty(d)
         fprintf('[Load]   Folder is empty or contains only subdirectories\n');
         dsub = dir(fullfile(folder, '**', '*'));
         dsub = dsub(~[dsub.isdir]);
         fprintf('[Load]   Recursive search: %d files found\n', numel(dsub));
         if numel(dsub) > 0
             fprintf('[Load]   First file: %s (%.0f bytes)\n', ...
                 fullfile(dsub(1).folder, dsub(1).name), dsub(1).bytes);
         end
     else
         exts = {};
         for di = 1:numel(d)
             [~,~,ext] = fileparts(d(di).name);
             if ~ismember(ext, exts), exts{end+1} = ext; end %#ok<AGROW>
         end
         fprintf('[Load]   Folder has %d files, extensions: %s\n', numel(d), strjoin(exts, ', '));
         % Try reading the first file to see if it's valid DICOM
         try
             testFile = fullfile(d(1).folder, d(1).name);
             testInfo = dicominfo(testFile);
             fprintf('[Load]   First file IS valid DICOM: %s\n', testFile);
             fprintf('[Load]   Modality=%s, Rows=%d, Cols=%d\n', ...
                 getField(testInfo, 'Modality', '?'), ...
                 getField(testInfo, 'Rows', 0), ...
                 getField(testInfo, 'Columns', 0));
             if isfield(testInfo, 'SeriesInstanceUID')
                 fprintf('[Load]   SeriesInstanceUID=%s\n', testInfo.SeriesInstanceUID);
             end
         catch
             fprintf('[Load]   First file is NOT valid DICOM\n');
         end
     end
 end
end
% 3) Legacy fallback: enumerate files manually
if isempty(Vraw)
 fprintf('[Load] Methods 1-2 failed. Using legacy manual file enumeration...\n');
 files = listDicomFiles(folder);
 fprintf('[Load] Legacy: found %d valid DICOM files\n', numel(files));
 if isempty(files), error('No readable DICOM files found under: %s', folder); end
 [Vraw, infosCell] = tryDicomReadVolume(files, folder);
 if ~isempty(Vraw)
     fprintf('[Load] Legacy dicomreadVolume(files) OK: size %s\n', mat2str(size(Vraw)));
 else
     fprintf('[Load] Legacy dicomreadVolume(files) failed, using manualReadStack...\n');
     [Vraw, infosCell] = manualReadStack(files);
     fprintf('[Load] Manual stack read OK: size %s\n', mat2str(size(Vraw)));
 end
end
% --- Normalize shape: [R C 1 S] -> [R C S]
Vraw = squeezeDicomVolume(Vraw);
% If meta is empty, try to grab a single header from any .dcm file
if isempty(infosCell)
 info1 = tryGrabOneHeader(folder);
 if ~isempty(info1)
     infosCell = {info1};
     fprintf('[Load] Meta missing; using header from: %s\n', info1.Filename);
 else
     fprintf('[Load] Meta and header missing; will use safe defaults.\n');
     infosCell = {};
 end
end
% Convert to HU with per-slice slope/intercept
HU = applyRescaleSlopeIntercept(double(Vraw), infosCell);
% Orientation/ordering, spacing, origin & axes  (now tolerant to empty meta)
[rowSpacing,colSpacing,sliceSpacing,origin,dir_row,dir_col,dir_slice,HU,infosCell] = ...
 orientAndOrder(HU, infosCell);
% Optional very light edge-aware smoothing
if o.Smoothing
 G = imgradient3(HU);
 edgeMask = G > prctile(G(:), 95);
 HU(~edgeMask) = imdiffusefilt(HU(~edgeMask), 'ConductionMethod','quadratic', ...
     'NumberOfIterations', 2, 'GradientThreshold', 25);
end
% Optional isotropic resampling
if ~isempty(o.TargetIsoMM)
 HU = squeezeDicomVolume(HU);  % <-- add this line
 [HU,rowSpacing,colSpacing,sliceSpacing] = resampleIsotropicHU( ...
     HU, rowSpacing, colSpacing, sliceSpacing, o.TargetIsoMM);
end
% Build voxel<->world closures (LPS)
[R,C,S] = size(HU);
voxelToWorld = @(i_row,i_col,i_slice) voxelToWorld_impl( ...
 i_row,i_col,i_slice, origin, dir_row,dir_col,dir_slice, rowSpacing,colSpacing,sliceSpacing);
worldToVoxel = @(xyz) worldToVoxel_impl( ...
 xyz, origin, dir_row,dir_col,dir_slice, rowSpacing,colSpacing,sliceSpacing);
% Pack
ds = struct();
ds.HU          = HU;
ds.spacing     = [rowSpacing colSpacing sliceSpacing];
ds.origin      = origin(:).';
ds.dir_row     = dir_row(:).';
ds.dir_col     = dir_col(:).';
ds.dir_slice   = dir_slice(:).';
ds.size        = [R C S];
ds.infos       = infosCell;
ds.voxelToWorld= voxelToWorld;
ds.worldToVoxel= worldToVoxel;
ds.M_voxToLPS  = [dir_col*colSpacing, dir_row*rowSpacing, dir_slice*sliceSpacing, origin(:); 0 0 0 1];
fprintf('[Load] dicomreadVolume OK: size %dx%dx%d\n', R,C,S);
end
function filesList = listDicomFiles(folder)
if ~isfolder(folder), error('Folder not found: %s', folder); end
listing = dir(fullfile(folder,'**','*'));
files = listing(~[listing.isdir]);
paths = fullfile({files.folder},{files.name});
keep = false(size(paths));
for i=1:numel(paths)
 p = paths{i};
 try
     d = dir(p);
     if d.bytes < 512, continue; end
     dicominfo(p);
     keep(i) = true;
 catch
 end
end
filesList = paths(keep);
end
function [V, infosCell] = tryDicomReadVolume(files, folder)
V = []; infosCell = {};
try
 [V,~,meta] = dicomreadVolume(files);
 infosCell  = normMeta(meta);
 if isempty(infosCell) || size(V,3) ~= numel(infosCell)
     [V2,~,meta2] = dicomreadVolume(folder);
     info2 = normMeta(meta2);
     if ~isempty(info2) && size(V2,3)==numel(info2)
         V = V2; infosCell = info2;
     end
 end
catch
end
end
function infosCell = normMeta(meta)
if isempty(meta), infosCell={}; return; end
if iscell(meta), infosCell=meta;
elseif isstruct(meta), infosCell = num2cell(meta);
else, infosCell = {};
end
end
function [V, infosCell] = manualReadStack(filesList)
meta = struct('fname',[],'SeriesInstanceUID',[],'SOPInstanceUID',[], ...
        'InstanceNumber',[],'IPP',[],'IOP',[]);
M = repmat(meta,0,1);
for i=1:numel(filesList)
 try
     info = dicominfo(filesList{i});
     rec.fname = filesList{i};
     rec.SeriesInstanceUID = getField(info,'SeriesInstanceUID','');
     rec.SOPInstanceUID    = getField(info,'SOPInstanceUID','');
     rec.InstanceNumber    = getField(info,'InstanceNumber',i);
     rec.IOP               = getField(info,'ImageOrientationPatient',[1 0 0 0 1 0]);
     if isfield(info,'ImagePositionPatient')
         rec.IPP = double(info.ImagePositionPatient(:)).';
     else
         rec.IPP = [NaN NaN NaN];
     end
     M(end+1,1)=rec; %#ok<AGROW>
 catch
 end
end
if isempty(M), error('No readable DICOM headers.'); end
if ~all(cellfun(@isempty,{M.SeriesInstanceUID}))
 u = unique({M.SeriesInstanceUID});
 counts = cellfun(@(x)sum(strcmp({M.SeriesInstanceUID},x)), u);
 [~,iMax] = max(counts);
 M = M(strcmp({M.SeriesInstanceUID}, u{iMax}));
end
IOP0 = M(1).IOP;
drow = IOP0(1:3); drow = drow(:)/norm(drow+eps);
dcol = IOP0(4:6); dcol = dcol(:)/norm(dcol+eps);
nrm  = cross(drow,dcol); nrm  = nrm / norm(nrm+eps);
hasIPP = all(isfinite(M(1).IPP));
if hasIPP
 projs = cellfun(@(p)dot(p,nrm), {M.IPP});
 [~,ord] = sort(projs,'ascend');
else
 [~,ord] = sort([M.InstanceNumber], 'ascend');
end
M = M(ord);
infosCell = cell(numel(M),1);
info0 = dicominfo(M(1).fname);
R = double(getField(info0,'Rows',[]));
C = double(getField(info0,'Columns',[]));
S = numel(M);
V = zeros(R,C,S,'double');
for k=1:S
 infosCell{k} = dicominfo(M(k).fname);
 V(:,:,k) = double(dicomread(infosCell{k}));
end
end
function HU = applyRescaleSlopeIntercept(V, infosCell)
% Robustly apply DICOM RescaleSlope/Intercept.
% Handles: empty meta, single meta for whole stack, per-slice meta, or length mismatch.
S = size(V,3);
defaultSlope = 1;
defaultIntercept = 0;
% Normalize infosCell into a cell array of length >=1 (or empty)
if isempty(infosCell)
 infos = {};
elseif iscell(infosCell)
 infos = infosCell;
elseif isstruct(infosCell)
 % Some MATLAB releases return a struct or struct array
 if numel(infosCell)==1
     infos = {infosCell};
 else
     infos = num2cell(infosCell);
 end
else
 infos = {};
end
% Helper to get slope/intercept from one info struct
 function [s, b] = slopeInterceptOne(info)
     if isempty(info)
         s = defaultSlope; b = defaultIntercept; return;
     end
     if isfield(info,'RescaleSlope'),     s = double(info.RescaleSlope);     else, s = defaultSlope;     end
     if isfield(info,'RescaleIntercept'), b = double(info.RescaleIntercept); else, b = defaultIntercept; end
     if ~isfinite(s), s = defaultSlope; end
     if ~isfinite(b), b = defaultIntercept; end
 end
% Case A: no metadata at all → assume identity
if isempty(infos)
 HU = V;  % slope=1, intercept=0
 return
end
% Case B: one header applies to all slices
if numel(infos)==1
 [s,b] = slopeInterceptOne(infos{1});
 HU = s .* V + b;
 return
end
% Case C: per-slice headers (or at least multiple); tolerate length mismatch
slopes     = ones(S,1)*defaultSlope;
intercepts = zeros(S,1);
n = min(S, numel(infos));
for k = 1:n
 [s,b] = slopeInterceptOne(infos{k});
 slopes(k) = s;
 intercepts(k) = b;
end
% If meta shorter than S, broadcast the last known values to the rest
if n < S
 slopes(n+1:S)     = slopes(n);
 intercepts(n+1:S) = intercepts(n);
end
% Apply per-slice scaling
HU = zeros(size(V),'like',double(1));
for k = 1:S
 HU(:,:,k) = slopes(k).*double(V(:,:,k)) + intercepts(k);
end
end
function [dr,dc,ds,origin,dir_row,dir_col,dir_slice,HU,infosCell] = orientAndOrder(HU, infosCell)
% Tolerant spacing/orientation: only reorders when meta matches slice count.
% ---- Defaults
dr = 1; dc = 1; ds = 1;
origin = [0;0;0];
dir_row = [1;0;0]; dir_col = [0;1;0]; dir_slice = [0;0;1];
Svol = size(HU,3);
Smeta = numel(infosCell);
if Smeta==0
 % No meta at all: keep defaults; nothing to reorder
 return
end
% Use the first header for spacing/orientation defaults
info0 = infosCell{1};
% Pixel spacing
if isfield(info0,'PixelSpacing') && numel(info0.PixelSpacing)>=2
 dr = double(info0.PixelSpacing(1));
 dc = double(info0.PixelSpacing(2));
end
% Orientation
if isfield(info0,'ImageOrientationPatient') && numel(info0.ImageOrientationPatient)>=6
 IOP = double(info0.ImageOrientationPatient(:));
 dir_row = IOP(1:3); dir_row = dir_row / max(norm(dir_row),eps);
 dir_col = IOP(4:6); dir_col = dir_col / max(norm(dir_col),eps);
 dir_slice = cross(dir_row,dir_col); dir_slice = dir_slice / max(norm(dir_slice),eps);
end
% Initial slice spacing from header (fallbacks)
if isfield(info0,'SpacingBetweenSlices') && isfinite(info0.SpacingBetweenSlices)
 ds = double(info0.SpacingBetweenSlices);
elseif isfield(info0,'SliceThickness') && isfinite(info0.SliceThickness)
 ds = double(info0.SliceThickness);
end
% --- Only attempt IPP-based reorder/spacing if meta covers ALL slices
if Smeta == Svol
 [IPP_all, maskIPP] = collectIPP(infosCell);
 nIPP = sum(maskIPP);
 % Refine ds from IPP if we have at least two valid positions
 if nIPP >= 2
     proj = IPP_all(maskIPP,:)*dir_slice;
     dproj = abs(diff(sort(proj,'ascend')));
     if ~isempty(dproj)
         ds = median(dproj(~isnan(dproj)));
     end
 end
 % Reorder volume only if EVERY slice has IPP
 if nIPP == Svol
     [~,ord] = sort((IPP_all*dir_slice), 'ascend');
     HU = HU(:,:,ord);
     infosCell = infosCell(ord);
     origin = IPP_all(1,:).';
     return
 end
 % If not all IPPs present, keep original order; set origin from first valid IPP
 if nIPP >= 1
     firstIdx = find(maskIPP,1,'first');
     origin = IPP_all(firstIdx,:).';
 end
else
 % Metadata does not match slice count → do not reorder (avoids collapsing to 1 slice)
 % Try to set origin from first header if it has IPP
 if isfield(info0,'ImagePositionPatient') && numel(info0.ImagePositionPatient)==3
     origin = double(info0.ImagePositionPatient(:));
 end
end
end
function [IPP_all, mask] = collectIPP(infosCell)
S = numel(infosCell);
IPP_all = NaN(S,3); mask = false(S,1);
for k=1:S
 if isfield(infosCell{k},'ImagePositionPatient')
     v = double(infosCell{k}.ImagePositionPatient(:)).';
     if numel(v)==3 && all(isfinite(v))
         IPP_all(k,:) = v; mask(k)=true;
     end
 end
end
end
function [HU_iso, dr,dc,ds] = resampleIsotropicHU(HU, dr,dc,ds, iso_mm)
% Defensive isotropic resampling to iso_mm (mm). Accepts HU as [R C S] or [R C 1 S].
% Force to 3-D
HU = squeeze(HU);
if ndims(HU) ~= 3
 error('resampleIsotropicHU: expected 3-D volume after squeeze, got ndims=%d', ndims(HU));
end
[R,C,S] = size(HU);
if any([R C S] < 1)
 error('resampleIsotropicHU: invalid input size [%d %d %d].', R,C,S);
end
% Build interpolant on the native grid
F = griddedInterpolant({1:R, 1:C, 1:S}, double(HU), 'linear', 'nearest');
% Target grid extents in mm
Rmm = (R-1)*dr;  Cmm = (C-1)*dc;  Smm = (S-1)*ds;
% New sizes
newR = max(1, round(Rmm/iso_mm)+1);
newC = max(1, round(Cmm/iso_mm)+1);
newS = max(1, round(Smm/iso_mm)+1);
% Query points in source index units
rq = ((0:newR-1)*iso_mm)/dr + 1;
cq = ((0:newC-1)*iso_mm)/dc + 1;
sq = ((0:newS-1)*iso_mm)/ds + 1;
[RR,CC,SS] = ndgrid(rq, cq, sq);
HU_iso = F(RR, CC, SS);
% Update spacings
dr = iso_mm; dc = iso_mm; ds = iso_mm;
end
function val = getField(s, field, def)
if isfield(s,field), val = s.(field); else, val = def; end
end
function [x_mm,y_mm,z_mm] = voxelToWorld_impl(i_row,i_col,i_slice, origin, dir_row,dir_col,dir_slice, dr,dc,ds)
i_row = double(i_row); i_col = double(i_col); i_slice = double(i_slice);
d_row = (i_row-1).*dr; d_col = (i_col-1).*dc; d_sli = (i_slice-1).*ds;
dx = d_col.*dir_col(1) + d_row.*dir_row(1) + d_sli.*dir_slice(1);
dy = d_col.*dir_col(2) + d_row.*dir_row(2) + d_sli.*dir_slice(2);
dz = d_col.*dir_col(3) + d_row.*dir_row(3) + d_sli.*dir_slice(3);
x_mm = origin(1)+dx; y_mm = origin(2)+dy; z_mm = origin(3)+dz;
end
function ijk = worldToVoxel_impl(xyz, origin, dir_row,dir_col,dir_slice, dr,dc,ds)
d = bsxfun(@minus, xyz, origin(:).');
i_col   = (d*dir_col)/dc + 1;
i_row   = (d*dir_row)/dr + 1;
i_slice = (d*dir_slice)/ds + 1;
ijk = [i_row, i_col, i_slice];
end

function V = squeezeDicomVolume(V)
% Convert [R C 1 S] or [R C S 1] to [R C S]
if ndims(V)==4
 if size(V,3)==1, V = squeeze(V); end
 if size(V,4)==1, V = squeeze(V); end
end
end

function info1 = tryGrabOneHeader(folder)
info1 = [];
d = dir(fullfile(folder,'**','*.dcm'));
if isempty(d), d = dir(fullfile(folder,'**','*')); end
d = d(~[d.isdir]);
for k = 1:min(numel(d), 200)  % don't scan forever
 f = fullfile(d(k).folder, d(k).name);
 try
     info1 = dicominfo(f);
     info1.Filename = f;
     return
 catch
 end
end
end