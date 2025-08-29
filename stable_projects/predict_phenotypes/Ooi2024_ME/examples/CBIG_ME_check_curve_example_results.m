function CBIG_ME_check_curve_example_results(sav_file)

% CBIG_ME_check_curve_example_results(sav_file)
% This function checks if the generated example results are identical to
% the reference files.
%
% Input:
%   - sav_file
%     The path saving of the sav file with results for comparison
%
% Written by Leon Ooi and CBIG under MIT license: https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md

% get directories
example_dir = fullfile(getenv('CBIG_CODE_DIR'), 'stable_projects', 'predict_phenotypes', ...
    'Ooi2024_ME', 'examples');
ref_dir = fullfile(example_dir, 'ref_results');
addpath(example_dir);


% run python script to read resultscompare prediction accuracies of KRR
command = ['python ' fullfile(example_dir, 'CBIG_ME_check_example_savfile.py'), ...
    ' ', fullfile(ref_dir,'predacc_behav59_results.sav'), ' ', ...
    sav_file];
[status, cmdout] = system(command);

assert(status == 0, cmdout);

display('Ooi2024 example run successfully!')
rmpath(example_dir);

end