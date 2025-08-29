function CBIG_ME_check_KRR_example_results(out_dir)

% CBIG_ICCW_check_example_results(out_dir)
% This function checks if the generated example results are identical to
% the reference files.
%
% Input:
%   - out_dir
%     The output directory path saving results of example scripts
%
% Written by Leon Ooi and CBIG under MIT license: https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md

% get directories
ref_dir = fullfile(getenv('CBIG_CODE_DIR'), 'stable_projects', 'predict_phenotypes', ...
    'Ooi2024_ME', 'examples', 'ref_results');

% compare prediction accuracies of KRR
ref_acc = load(fullfile(ref_dir, 'final_result_2cog.mat'));
test_acc = load(fullfile(out_dir, 'final_result_2cog.mat'));

diff_acc = max(max(abs(ref_acc.optimal_acc - test_acc.optimal_acc)));
assert(diff_acc< 1e-5, sprintf('Difference in acc of KRR of: %f', diff_acc));

display('Ooi2024_ME KRR example run successfully!')

end