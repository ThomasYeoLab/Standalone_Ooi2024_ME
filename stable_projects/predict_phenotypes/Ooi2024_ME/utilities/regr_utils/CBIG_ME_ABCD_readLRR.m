function CBIG_ME_ABCD_readLRR(results_basedir, vers, output_vers, metric)

% function CBIG_ME_ABCD_readLRR(results_basedir, vers, output_vers, metric)
%
% This function collates the results for LRR in the ABCD.
%
% Inputs:
%     -results_basedir:
%      Directory in which all results are saved.
%      E.g. '/home/leon_ooi/storage/optimal_prediction/replication/ABCD/output/full'
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
%      #minutes x #sample sizes x #phenotypes
%
% Written by Leon Ooi and CBIG under MIT license: https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md

%% Set function settings and paths
% Paths
results_dir = fullfile(results_basedir, output_vers, vers); 
img_dir = fullfile(results_dir, 'images');
if ~exist(img_dir); mkdir(img_dir); end
% Regression split details
num_behav = 39;
% Different scan durations
if strcmp(vers, 'only_censored')
    min = [2:2:14 15];
else
    min = [2:2:20];
end
% Different sample sizes and folds
if strcmp(output_vers, 'output_splithalf')
    subs = [2565 1000 800 600 400 200];
    % compute matching folds: manually check redundant folds (optional) 
    % i.e. computing ICC for folds 1 and 252 is same as 252 and 1
    load(fullfile(results_dir, 'no_relative_5_fold_sub_list.mat'));
    for n = 1:length(sub_fold)
        curr_idxs = sub_fold(n).fold_index;
        for m = 1:length(sub_fold)
            search_idxs = sub_fold(m).fold_index;
            if ~any((search_idxs  + curr_idxs) ~= 1)
                matching_fold(n,1) = n;
                matching_fold(n,2) = m;
            end
        end
    end
    num_folds = 252;
else
    subs = [2565 1600 1400 1200 1000 800 600 400 200];
    num_folds = 120;
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
    acc_mean = zeros(num_folds,num_behav);
    % read results based on whether it was the full sample size
    % or subsample (first sample should be full sample size)
    if s ~= 1
        % loop over folds
        for fold = 1:num_folds
            fold_name = strcat('fold_', num2str(fold));
            load(fullfile(results_dir, strcat('LRR_', min_name), ...
                strcat(num2str(curr_subs), '_subjects'), fold_name, ...
                'results', 'optimal_acc', strcat('LRR_' , min_name, '.mat')))
            acc_mean(fold, :) = optimal_statistics{1}.(metric);
        end
    else
        load(fullfile(results_dir, strcat('LRR_', min_name), ...
                strcat(num2str(curr_subs), '_subjects'), 'results', ...
                'results', 'optimal_acc', strcat('LRR_' , min_name, '.mat')))
        % loop over folds
        for fold = 1:num_folds
            acc_mean(fold,:) = optimal_statistics{fold}.(metric);
        end
    end
    % average and save into matrix
    acc_landscape(m,s,:) = mean(acc_mean, 1);
end
end
save(fullfile(img_dir, strcat('acc_LRR_',metric,'_landscape.mat')), 'acc_landscape');

end