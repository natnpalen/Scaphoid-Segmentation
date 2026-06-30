% RUN_156L2  Quick-start script for the 156L-2 bone scan.
%
% Usage: just hit Run (F5) in MATLAB, or type:  run_156L2
%
% Paths are set to the 156L-2 scan and mechanical specimen STL files.
% Edit below if your files are in a different location.

clear all; close all; clc; %#ok<CLALL>

% Add this pipeline's src to the path
thisDir = fileparts(mfilename('fullpath'));
addpath(thisDir);

dicomFolder = 'C:\Users\natha\OneDrive\Documents\Nathaniel\For Nick\New Bone Scans\156L-2\DICOMOBJ';
stlFolder   = 'C:\Users\natha\OneDrive\Documents\Nathaniel\For Nick\Mechancial Specimens';

out = run_bone_pipeline(dicomFolder, stlFolder);
