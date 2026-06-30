function write_stl_binary(filename, mesh, varargin)
% WRITE_STL_BINARY  Fast, correct binary STL writer (little-endian).
%   write_stl_binary(file, mesh)
%   write_stl_binary(file, mesh, 'Decimate', 1.0)   % 0<r<=1; 1=no decimation
%
% mesh.vertices: Nx3 (double/single) in mm
% mesh.faces   : Mx3 (1-based indices)

% ---- Parse args ----
p = inputParser;
addParameter(p, 'Decimate', 1.0, @(x) isnumeric(x) && isscalar(x) && x>0 && x<=1);
parse(p, varargin{:});
r = p.Results.Decimate;

% ---- Validate mesh ----
if ~isstruct(mesh) || ~isfield(mesh,'vertices') || ~isfield(mesh,'faces')
    error('Mesh must have fields .vertices and .faces');
end
V = double(mesh.vertices);
F = double(mesh.faces);
if isempty(V) || isempty(F)
    error('Mesh is empty.');
end

% ---- Optional decimation ----
if r < 1.0
    try
        [F, V] = reducepatch(F, V, r);  %#ok<ASGLU>
    catch
        warning('reducepatch not available; skipping decimation.');
    end
end

% ---- Triangles & normals (vectorized) ----
tri  = round(F);                 % ensure integer indices
nTri = size(tri,1);
v1   = V(tri(:,1),:);
v2   = V(tri(:,2),:);
v3   = V(tri(:,3),:);
n    = cross(v2 - v1, v3 - v1, 2);
nn   = sqrt(sum(n.^2,2)); nn(nn==0) = 1;
n    = n ./ nn;

% ---- Open file (binary, little-endian) ----
filename_char = char(filename);
[fid, msg] = fopen(filename_char, 'w', 'ieee-le');
if fid < 0, error('Cannot open %s: %s', filename_char, msg); end
cleanup = onCleanup(@() fclose(fid));

% ---- 80-byte header ----
hdr = zeros(80,1,'uint8');
namebytes = uint8(filename_char);
hdr(1:min(80,numel(namebytes))) = namebytes(1:min(80,numel(namebytes)));
fwrite(fid, hdr, 'uint8');

% ---- Triangle count ----
fwrite(fid, uint32(nTri), 'uint32');

% ---- Batched records: [12 singles][uint16 attr] per triangle ----
B = 200000;

for s = 1:B:nTri
    e = min(s+B-1, nTri);
    nb = e - s + 1;

    % Pack 12 floats per tri: [nx ny nz x1 y1 z1 x2 y2 z2 x3 y3 z3]
    block = single([ n(s:e,:)  v1(s:e,:)  v2(s:e,:)  v3(s:e,:) ]);
    % Convert float data to bytes: 48 bytes per triangle
    floatBytes = typecast(reshape(block.', [], 1), 'uint8');
    floatMat = reshape(floatBytes, 48, nb);  % 48 x nb
    % Append 2-byte attribute (zero) after each triangle's floats
    recMat = [floatMat; zeros(2, nb, 'uint8')];  % 50 x nb
    fwrite(fid, recMat(:), 'uint8');
end
end
