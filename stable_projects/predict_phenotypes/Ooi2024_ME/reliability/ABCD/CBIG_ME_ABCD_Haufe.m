function CBIG_ME_ABCD_Haufe(input_dir,results_dir,vers,num_mins,num_subs,reg,b)

% function CBIG_ME_ABCD_Haufe(input_dir,results_dir,vers,num_mins,num_subs,reg,b)
% 
% This function calculated the Haufe-inverted feature importance for each model. This function
% is specifically for KRR models.
%
% Input:
% - input_dir
%   The directory in which the brain imaging features are saved.
%
% - results_dir 
%   The directory in which the regression results are results are saved.

% - vers
%   The type of analysis that was carried out. Can be either "full" or "random".

% - num_mins 
%   A string specifying the scan time that is being analysed. E.g. "2_mins"
%
% - num_subs 
%   A string specifying the sample size that is being analysed. E.g. "200_subjects"
%
% - reg
%   The regression type. In this analysis, we only run "KRR".
%
% - b
%   The index of the phenotype to calculate Haufe-inverted feature weights for.
%
% Output: 
% - cov_mat_mean
%   A mat file is saved with a matrix of #folds x #features x #behaviors.
% 
% Written by Leon Ooi and CBIG under MIT license: https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md

%% set up directories
save_dir = fullfile(results_dir, 'interpretation', num_mins, num_subs);
sub_txt = fullfile(input_dir, 'subject_list_2565.txt'); 
if ~exist(save_dir)
    mkdir(save_dir);
end

% set up adjustable parameters
total_folds = 252;
model = strcat(reg,'_',num_mins);


% load fold
for f = 1:total_folds
        fprintf('Calculating for %s, %s, fold %i / %i \n', ...
            num_mins, num_subs, f, total_folds)
        fold_name = strcat('fold_', num2str(f));

        % load FC depending on whether analysing original run results
        % or randomized run results
        if strcmp(vers,'random')
            fprintf('Regenerating random FC... \n')
            % read subject list
            file = fopen(sub_txt);
            data = textscan(file,'%s');
            fclose(file);
            subj_list = data{:};
            % read all FC from different permutations
            num_perms = 24;
            % randomly sample and combine across seeds
            feat.feat_mat = zeros(419,419,length(subj_list));
            for subj_n = 1:length(subj_list)
                rng(f*subj_n,'twister')
                perm_idx = randsample(num_perms,1);
                sub_split = strsplit(subj_list{subj_n}, '_');
                sub_formatted = strcat(sub_split{1}, sub_split{2});
                sub_corrmat = load(fullfile(input_dir,'FC', strcat('full_perm', num2str(perm_idx)), ...
                    strcat(num_mins,'s'), strcat(sub_formatted,'_', num_mins, '_FC.mat')));
                feat.feat_mat(:,:,subj_n) = sub_corrmat.corr_mat;         
            end
        else
            input_mat = fullfile(input_dir, 'FC', vers, strcat(num_mins,'_FC.mat'));
            feat = load(input_mat);
        end

        % take lower triangle
        lower_tri = logical(tril(ones(419,419),-1));
        feat_mat = zeros(sum(lower_tri, 'all'), size(feat.feat_mat,3));
        for sub = 1:size(feat.feat_mat,3)
            sub_mat =  feat.feat_mat(:,:,sub);
            feat_mat(:,sub) = sub_mat(lower_tri);
        end 

        % load subfold and results
        if strcmp(num_subs, '2565_subjects')
            % load sub_fold
            load(fullfile(results_dir, 'no_relative_5_fold_sub_list.mat'));
            % calculate feature importance for each fold
            % find train fold idx
            train = ~sub_fold(f).fold_index;
            % load features and normalize
            feat_train = feat_mat(:,train);
            feat_train_norm = (feat_train - mean(feat_train,1)) ./ std(feat_train, [], 1);
            % load predictions
            if strcmp(vers,'random')
                results = load(fullfile(results_dir, model, num_subs, fold_name, ...
                    strcat('final_result_', model, '.mat'))); 
            else
                results = load(fullfile(results_dir, model, num_subs, 'results', ...
                    strcat('final_result_', model, '.mat'))); 
            end
            if strcmp(vers,'random')
                y_pred = results.y_pred_train{1};
            else
                y_pred = results.y_pred_train{f};
            end
        
            % remove nan before calculating covariance 
            nan_sub = any(isnan(feat_train_norm));
            feat_train_norm(:,nan_sub) = [];
            y_pred(nan_sub,:) = [];

            % compute covariance
            cov_mat_mean(f,:) = bsxfun(@minus,feat_train_norm,mean(feat_train_norm,2)) * ...
                bsxfun(@minus,y_pred(:,b),mean(y_pred(:,b))) / (size(feat_train_norm,2));
        else
            % load sub_fold
            load(fullfile(results_dir, strcat(num_subs, '_10_fold_sub_list.mat')));
            % calculate feature importance for each fold
            % find train fold idx
            train = ~subsampled_subfold.fold_index{f};
            % load features and normalize
            feat_subset = feat_mat(:,subsampled_subfold.subfold_pos{f});
            feat_train = feat_subset(:,train);
            feat_train_norm = (feat_train - mean(feat_train,1)) ./ std(feat_train, [], 1);
            % load results
            results_fold = load(fullfile(results_dir, model, num_subs, fold_name, ...
                strcat('final_result_', model, '.mat'))); 
            y_pred = results_fold.y_pred_train{1};
              
            % remove nan before calculating covariance 
            nan_sub = any(isnan(feat_train_norm));
            feat_train_norm(:,nan_sub) = [];
            y_pred(nan_sub,:) = [];

            % compute covariance
            cov_mat_mean(f,:) = bsxfun(@minus,feat_train_norm,mean(feat_train_norm,2)) * ...
                bsxfun(@minus,y_pred(:,b),mean(y_pred(:,b))) / (size(feat_train_norm,2));
        end
    end

    % save results
    cov_mat_mean = single(cov_mat_mean);
    save(fullfile(save_dir, strcat(reg,'_cov_mat_behav',num2str(b),'.mat')), ...
        'cov_mat_mean', '-v7.3');
end