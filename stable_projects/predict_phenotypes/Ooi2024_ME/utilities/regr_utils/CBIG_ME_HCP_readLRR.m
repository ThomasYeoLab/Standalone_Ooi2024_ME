function CBIG_ME_HCP_readLRR(results_basedir, vers, output_vers, metric)

% function CBIG_ME_HCP_readLRR(results_basedir, vers, output_vers, metric)
% This function collates the results for LRR in the HCP.
%
% Inputs:
%     -results_basedir:
%      Directory in which all results are saved.
%      E.g. '/home/leon_ooi/storage/optimal_prediction/replication/HCP/output/full'
%
%     -vers:
%      The manner in which the first T frames of data used to generate the
%      FC was calculated. For LRR, we only run "full".
%
%     -output_vers:
%      Whether to read the full prediction workflow results or the split
%      half prediction results. For LRR, we only run "output".
%
%     -metric:
%      The accuracy metric to collate. Can be "corr" or "COD".
%
% Output:
%      Creates a directory called "images" in output directory. Creates a
%      mat file of collated results with dimensions 
%      #minutes x #sample sizes x #behaviors
%
% Written by Leon Ooi and CBIG under MIT license: https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md

%% Set function settings and paths
% Paths
results_dir = fullfile(results_basedir, output_vers, vers);
img_dir = fullfile(results_dir, 'images');
% create output directory to save collated results
if ~exist(img_dir); mkdir(img_dir); end
% Regression split details
num_behav = 61;
num_seeds = 50;
% Different scan durations
if strcmp(vers, 'only_censored')
    min = [2:2:40];
else
    min = [2:2:58];
end
% Different sample sizes and folds
if strcmp(output_vers, 'output_splithalf')
    subs = [792 350 300 250 200 150];
    num_folds = 2;
else
    subs = [792 600 500 400 300 200];
    num_folds = 10;
end

%% Start results collation
% Collate regression accuracies
fprintf('Collating regression results \n');
acc_landscape = zeros(length(min),length(subs),num_behav);
% loop over sample sizes
for s = 1:length(subs)
curr_subs = subs(s);
% loop over scan duration
for m = 1:length(min)
    curr_min = min(m);
    min_name = strcat(num2str(curr_min), 'min');
    acc_mean = zeros(num_seeds,num_behav);
    % loop over seeds
    for seed = 1:num_seeds
        fi_tmp = zeros(num_folds,num_behav);
        seed_name = strcat('seed_', num2str(seed));
        % read results based on whether it was the full sample size
        % or subsample (first sample should be full sample size)
        if s ~= 1 % assumes first entry is the full sample size)
            % loop over folds
            for fold = 1:num_folds
                fold_name = strcat('fold_', num2str(fold));
                load(fullfile(results_dir, seed_name, strcat('LRR_', min_name), ...
                    strcat(num2str(curr_subs), '_subjects'), fold_name, ...
                    'results', 'optimal_acc', strcat('LRR_', min_name)))
                fi_tmp(fold, :) = optimal_statistics{1}.(metric);
            end
        else
            load(fullfile(results_dir, seed_name, strcat('LRR_', min_name), ...
                strcat(num2str(curr_subs), '_subjects'), 'results', ...
                'results', 'optimal_acc', strcat('LRR_', min_name)))
            % loop over folds
            for fold = 1:num_folds
                fi_tmp(fold, :) = optimal_statistics{fold}.(metric);
            end
        end
        
        acc_mean(seed,:) = mean(fi_tmp, 1);
    end
    % average and save into matrix
    acc_landscape(m,s,:) = mean(acc_mean, 1);
end
end
save(fullfile(img_dir, strcat('acc_LRR_',metric,'_landscape.mat')), 'acc_landscape');

end