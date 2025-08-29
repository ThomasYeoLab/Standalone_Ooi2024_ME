function CBIG_ME_ABCD_readKRR(results_basedir, vers, output_vers, metric)

% function CBIG_ME_ABCD_readKRR(results_basedir, vers, output_vers, metric)
%
% This function collates the results for KRR in the ABCD, and also the univariate
% and multivariate reliability.
%
% Inputs:
%     -results_basedir:
%      Directory in which all results are saved.
%      E.g. '/home/leon_ooi/storage/optimal_prediction/replication/ABCD/output/full'
%
%     -vers:
%      The manner in which the first T frames of data used to generate the
%      FC was calculated. Can be one of the following:
%      "full", "uncensored_only", "no_censoring", "random"
%
%     -output_vers:
%      Whether to read the full prediction workflow results or the split
%      half prediction results. Can be either "output" for former or
%      "output_splithalf" for latter.
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
% Results to collate (set to 0 if you want to skip)
% if regression = 1, prediction accuracies are collated
% if regression_ind = 1, prediction accuracies are collated and saved for each fold
% if tstats_icc = 1, univariate reliability is collated
% if haufe_icc = 1, multivariate KRR reliability is collated
% if edge_corr = 1, edgewise correlation with the cognition factor score is collated
regression = 1;
regression_ind = 0;
haufe_icc = 0;
tstats_icc = 0;
edge_corr = 0;

% Paths
results_dir = fullfile(results_basedir, output_vers, vers);
img_dir = fullfile(results_dir, 'images');
% create output directory to save collated results
if ~exist(img_dir); mkdir(img_dir); end
% Regression split details
num_behav = 39;
% Different scan durations
if strcmp(vers, 'uncensored_only')
    min = [2:2:14 15];
else
    if contains(vers, 'MID')
        min = [1:1:11];
    elseif contains(vers, 'NBACK')
        min = [1:1:11];
    elseif contains(vers, 'SST')
        min = [1:1:12];
    else
        min = [2:2:20];
    end
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
    if contains(vers, 'MID') || contains(vers, 'NBACK') || contains(vers, 'SST')
        subs = [2262 1400 1200 1000 800 600 400 200];
    else
        subs = [2565 1600 1400 1200 1000 800 600 400 200];
    end
    num_folds = 120;
end

%% Start results collation
% collate regression accuracies
if regression
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
            if s ~= 1 || strcmp(vers, 'random') 
                % loop over folds
                for fold = 1:num_folds
                    fold_name = strcat('fold_', num2str(fold));
                    load(fullfile(results_dir, strcat('KRR_', min_name), ...
                        strcat(num2str(curr_subs), '_subjects'), fold_name, ...
                        strcat('final_result_KRR_' , min_name, '.mat')))
                    acc_mean(fold, :) = optimal_stats.(metric);
                end
            else
                % all folds are saved in single file for maximum sample size
                load(fullfile(results_dir, strcat('KRR_', min_name), ...
                        strcat(num2str(curr_subs), '_subjects'), 'results', ...
                        strcat('final_result_KRR_' , min_name, '.mat')))
                acc_mean = optimal_stats.(metric);
            end
            % average and save into matrix
            acc_landscape(m,s,:) = mean(acc_mean, 1);
        end
    end
    save(fullfile(img_dir, strcat('acc_KRR_',metric,'_landscape.mat')), 'acc_landscape');
end

% collate regression accuracies (save each fold)
if regression_ind
    fprintf('Collating regression results (save individual folds) \n');
    acc_landscape = zeros(length(min),length(subs),num_folds,num_behav);
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
            if s ~= 1 || strcmp(vers, 'random') 
                % loop over folds
                for fold = 1:num_folds
                    fold_name = strcat('fold_', num2str(fold));
                    load(fullfile(results_dir, strcat('KRR_', min_name), ...
                        strcat(num2str(curr_subs), '_subjects'), fold_name, ...
                        strcat('final_result_KRR_' , min_name, '.mat')))
                    acc_mean(fold, :) = optimal_stats.(metric);
                end
            else
                % all folds are saved in single file for maximum sample size
                load(fullfile(results_dir, strcat('KRR_', min_name), ...
                        strcat(num2str(curr_subs), '_subjects'), 'results', ...
                        strcat('final_result_KRR_' , min_name, '.mat')))
                acc_mean = optimal_stats.(metric);
            end
            acc_landscape(m,s,:,:) = acc_mean;
        end
    end
    save(fullfile(img_dir, strcat('acc_KRR_indiv_',metric,'_landscape.mat')), 'acc_landscape');
end

% collate univariate reliability
if tstats_icc
    fprintf('Collating reliability results (univariate icc) \n');
    tstats_icc_landscape = zeros(length(min),length(subs),num_behav);
    % loop over phenotypes
    for b = 1:num_behav
        % loop over sample sizes
        for s = 1:length(subs)
            curr_subs = subs(s);
            % loop over scan duration
            for m = 1:length(min)
                curr_min = min(m);
                min_name = strcat(num2str(curr_min), 'min');
                load(fullfile(results_dir, 'interpretation', min_name, ...
                    strcat(num2str(curr_subs), '_subjects'), ...
                    strcat('KRR_tstats_mat_behav',num2str(b),'.mat')));
                % only need to go through half the folds since ICC between 
                % folds 1 and 252 is same as 252 and 1 
                for mf = 1:(num_folds/2)
                    % Calculate ICC
                    tstats_tmp(mf) = CBIG_ICC_1to1([squeeze(tstats_mat_mean(mf,:))' ...
                            squeeze(tstats_mat_mean(num_folds - (mf-1),:))']);
                    % Calculate split half replicability
                    % binarize into < 0.05 and include sign
                    f1_p = squeeze(p_mat_mean(mf,:))' < 0.05;
                    f1_t_pos = squeeze(tstats_mat_mean(mf,:))' > 0;
                    f1_t_neg = -(squeeze(tstats_mat_mean(mf,:))' < 0);
                    f1_t = f1_t_pos + f1_t_neg;
                    f1_p_w_sign = f1_p .* f1_t;
                    % binarize into < 0.05 and include sign
                    f2_p = squeeze(p_mat_mean(num_folds - (mf-1),:))' < 0.05;
                    f2_t_pos = squeeze(tstats_mat_mean(num_folds - (mf-1),:))' > 0;
                    f2_t_neg = -(squeeze(tstats_mat_mean(num_folds - (mf-1),:))' < 0);
                    f2_t = f2_t_pos + f2_t_neg;
                    f2_p_w_sign = f2_p .* f2_t;
                    % find common significant edges
                    sum_replicated = sum((f1_p_w_sign .* f2_p_w_sign) == 1);
                    p_frac_tmp(mf) = ((sum_replicated / sum(f1_p)) + ...
                        (sum_replicated / sum(f2_p))) / 2;
                end
                % average and save into matrix
                tstats_icc_landscape(m,s,b) = mean(tstats_tmp);
                tstats_landscape(m,s,:,b) = tstats_tmp;
                p_frac_landscape(m,s,b) = mean(p_frac_tmp);
            end
        end
    end
    save(fullfile(img_dir, 'tstats_icc_landscape.mat'), 'tstats_icc_landscape');
    save(fullfile(img_dir, 'tstats_icc_indiv_landscape.mat'), 'tstats_landscape');
    save(fullfile(img_dir, 'p_frac_landscape.mat'), 'p_frac_landscape');
end

% collate Haufe ICC
if haufe_icc
    fprintf('Collating reliability results (multivariate icc) \n');
    fi_icc_landscape = zeros(length(min),length(subs),num_behav);
    % loop over phenotypes
    for b = 1:num_behav
        % loop over sample sizes
        for s = 1:length(subs)
            curr_subs = subs(s);
            % loop over scan duration
            for m = 1:length(min)
                curr_min = min(m);
                min_name = strcat(num2str(curr_min), 'min');
                load(fullfile(results_dir, 'interpretation', min_name, ...
                    strcat(num2str(curr_subs), '_subjects'), ...
                    strcat('KRR_cov_mat_behav',num2str(b),'.mat'))); 
                % only need to go through half the folds since ICC between 
                % folds 1 and 252 is same as 252 and 1 
                for mf = 1:(num_folds/2)
                    fi_tmp(mf) = CBIG_ICC_1to1([squeeze(cov_mat_mean(mf,:))' ...
                            squeeze(cov_mat_mean(num_folds - (mf-1),:))']);
                end
                % average and save into matrix
                fi_icc_landscape(m,s,b) = mean(fi_tmp);
            end
        end
    end
    save(fullfile(img_dir, 'fi_icc_KRR_landscape.mat'), 'fi_icc_landscape');
end

% find edgewise correlation to cognition factor score
if edge_corr
    fprintf('Find edgewise correlation to cognition factor score \n');
    b = 37;
    % for this, specify HCP replication directory inside results_basedir
    % e.g. '/mnt/nas/CSC7/Yeolab/Users/leon_ooi/optimal_prediction/replication/HCP'
    FC_mat = load(fullfile(results_basedir, 'input', 'FC', 'full', '20min_FC.mat'));
    behav_var = load(fullfile(results_basedir, 'output', 'full', 'subject2565_variables_to_predict.mat');
    for n = 1:size(FC_mat.feat_mat,3)
        FC_edges(:,n) = CBIG_FC_mat2vector(FC_mat.feat_mat(:,:,n));
    end
    % calculate correlation over each edge
    y_vals = behav_var.y(:,b);
    na_vals = find(sum(isnan(FC_edges)) > 0);
    FC_edges(:,na_vals) = [];
    y_vals(na_vals) = [];
    for e = 1:size(FC_edges,1)
        rho_k(e) = corr(y_vals,FC_edges(e,:)');
    end
    save(fullfile(img_dir, 'rho_k.mat'), 'rho_k');
end
    
end