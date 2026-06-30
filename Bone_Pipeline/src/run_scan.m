% RUN_SCAN  Quick-start script for the bone segmentation pipeline.
%
% Edit SCAN_NAME below to switch between scans.
% Add new scans by adding entries to the switch block.
%
% Usage: just hit Run (F5) in MATLAB, or type:  run_scan

clear all; close all; clc; %#ok<CLALL>

% ---- Which scan to run ----
SCAN_NAME = '156L-1';

% ---- Specimen STL folder (shared across all scans) ----
stlFolder = 'C:\Users\natha\OneDrive\Documents\Nathaniel\For Nick\Mechancial Specimens';

% ---- Scan-specific DICOM paths ----
baseDir = 'C:\Users\natha\OneDrive\Documents\Nathaniel\For Nick\New Bone Scans';

switch SCAN_NAME
    case '156L-1'
        dicomFolder = fullfile(baseDir, '156L-1', 'DICOMOBJ');
    case '156L-2'
        dicomFolder = fullfile(baseDir, '156L-2', 'DICOMOBJ');
    otherwise
        error('Unknown scan "%s". Add it to the switch block in run_scan.m.', SCAN_NAME);
end

% ---- Add this pipeline to the path ----
thisDir = fileparts(mfilename('fullpath'));
addpath(thisDir);

% ---- Run ----
out = run_bone_pipeline(dicomFolder, stlFolder, ...
    'SaveOutputs', true);
