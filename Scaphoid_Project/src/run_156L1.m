% RUN_156L1  Quick-start script for the 156L-1 bone scan.
%
% Usage: just hit Run (F5) in MATLAB, or type:  run_156L1
%
% Paths are set to the 156L-1 scan and mechanical specimen STL files.
% Edit below if your files are in a different location.

dicomFolder = 'C:\Users\natha\OneDrive\Documents\Nathaniel\For Nick\New Bone Scans\156L-1\DICOMOBJ';
stlFolder   = 'C:\Users\natha\OneDrive\Documents\Nathaniel\For Nick\Mechancial Specimens';

out = run_bone_pipeline(dicomFolder, stlFolder);
