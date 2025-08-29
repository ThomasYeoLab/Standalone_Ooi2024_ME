function CBIG_ME_example_wrapper(output_dir)

% CBIG_ME_example_wrapper(out_dir)
%
% This function fits the theoretical and logarithm models for
% the NxT combinations of the KRR results from the cognition factor score 
% in the HCP 
%
% Input:
%  - out_dir
%    A path where the results of the example will be saved.
%
% Written by Leon Ooi and CBIG under MIT license: https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md

% define directories and set up dependencies
project_code_dir = fullfile(getenv('CBIG_CODE_DIR'), 'stable_projects', ...
    'predict_phenotypes', 'Ooi2024_ME', 'curve_fitting');
parent_dir = fullfile(getenv('CBIG_CODE_DIR'), 'stable_projects', 'predict_phenotypes', ...
    'Ooi2024_ME', 'examples');
input_dir = fullfile(parent_dir, 'example_data');
addpath(project_code_dir);

% fit curves using python
if ~exist(fullfile(input_dir,'HCP','output','full','curve_fit'))
    mkdir(fullfile(input_dir,'HCP','output','full','curve_fit'))
end
command = ['python ' fullfile(project_code_dir, 'CBIG_ME_fit_all.py'), ...
    ' HCP 59 full predacc ' input_dir];
[status, cmdout] = system(command);

% move curve fit the output directory
if ~exist(output_dir)
    mkdir(output_dir)
end
movefile(fullfile(parent_dir, 'example_data','HCP','output','full','curve_fit'), output_dir);

rmpath(project_code_dir);
end
