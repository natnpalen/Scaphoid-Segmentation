% RUN_SCAN  Quick-start script for the bone segmentation pipeline.
%
% Set the DICOM and STL folder paths below, then hit Run (F5).

clear all; close all; clc; %#ok<CLALL>

% ---- Set your paths here ----
dicomFolder = 'C:\Users\natha\OneDrive\Documents\Nathaniel\For Nick\New Bone Scans\156L-1\DICOMOBJ';
stlFolder   = 'C:\Users\natha\OneDrive\Documents\Nathaniel\For Nick\Mechancial Specimens';

% ---- Add this pipeline to the path ----
thisDir = fileparts(mfilename('fullpath'));
addpath(thisDir);

% ---- Run ----
out = run_bone_pipeline(dicomFolder, stlFolder, ...
    'PackSpecimens', false, ...
    'SaveOutputs',   true);
