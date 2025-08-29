function CBIG_ME_HCP_Haufe(input_dir,results_dir,vers,num_mins,num_subs,reg,b)

% function CBIG_ME_HCP_Haufe(input_dir, results_dir, feature)
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
%   A mat file is saved with a matrix of #seeds x #features x #behaviors.
%
% Written by Leon Ooi and CBIG under MIT license: https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md

%% set up directories
save_dir = fullfile(results_dir, 'interpretation', num_mins, num_subs);
sub_txt = fullfile(input_dir, 'subject_list_792.txt'); 
if ~exist(save_dir)
    mkdir(save_dir);
end

% set up adjustable parameters
total_seeds = 50;
model = strcat(reg,'_',num_mins);

% load seed
for s = 1:total_seeds
	fprintf('Calculating for %s, %s, seed %i / %i \n', ...
	    num_mins, num_subs, s, total_seeds)
	seed_name = strcat('seed_', num2str(s));
	model_dir = fullfile(results_dir, seed_name);

	% load FC depending on whether analysing original run results
        % or randomized run results
	if strcmp(vers,'random')
	    fprintf('Regenerating random FC... \n')
	    % read subject list
	    f = fopen(sub_txt);
	    data = textscan(f,'%f');
	    fclose(f);
	    subj_list = data{:};
	    % read all FC from different permutations
	    num_perms = 24;
	    feat_mats = zeros(num_perms,419,419,length(subj_list));
	    idx = 1;
	    for permno = 1:num_perms
		FC_mat = fullfile(input_dir, 'FC', strcat('full_perm',num2str(permno)), strcat(num_mins,'_FC.mat'));
		fc_perm = load(FC_mat);
		feat_mats(idx,:,:,:) = fc_perm.feat_mat;
		idx = idx + 1;
	    end
	    % randomly sample and combine across seeds
	    feat.feat_mat = zeros(419,419,length(subj_list));
	    for subj_n = 1:length(subj_list)
		rng(s*subj_n,'twister')
		perm_idx = randsample(num_perms,1);
		feat.feat_mat(:,:,subj_n) = feat_mats(perm_idx,:,:,subj_n);
	    end
	    % clear feat mats to save memory
	    clear feat_mats
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
        if strcmp(num_subs, '792_subjects')
            % load sub_fold
            load(fullfile(model_dir, 'no_relative_2_fold_sub_list.mat'));
            
            % pre allocate space for cov_mat
            cov_mat = zeros(size(sub_fold,1), size(feat_mat,1));
            
            % calculate feature importance for each fold
            for i = 1:size(sub_fold,1)
                fold_name = strcat('fold_', num2str(i));
                % calculate feature importance for each fold
                % find train fold idx
                train = ~sub_fold(i).fold_index;
                % load features and normalize
                results = load(fullfile(model_dir, model, num_subs, 'results', ...
                    strcat('final_result_', model, '.mat')));
                % load predictions
                y_pred = results.y_pred_train{i};
                % get training set
                feat_train = feat_mat(:,train);
                % remove 0 FC sub
                zeroFC_sub = find(sum(feat_train == 0));
                feat_train(:,zeroFC_sub) = [];
                y_pred(zeroFC_sub,:) = [];
                % normalize
                feat_train_norm = (feat_train - mean(feat_train,1)) ./ std(feat_train, [], 1);

                % remove nan before calculating covariance
                nan_sub = any(isnan(feat_train_norm));
                feat_train_norm(:,nan_sub) = [];
                y_pred(nan_sub,:) = [];
                nan_sub = any(isnan(y_pred),2);
                feat_train_norm(:,nan_sub) = [];
                y_pred(nan_sub,:) = [];
          
                % compute covariance
                cov_mat(i,:) = bsxfun(@minus,feat_train_norm,mean(feat_train_norm,2)) * ...
                    bsxfun(@minus,y_pred(:,b),mean(y_pred(:,b))) / (size(feat_train_norm,2));
            end
        else
            % load sub_fold
            load(fullfile(model_dir, strcat(num_subs, '_10_fold_sub_list.mat')));
            % calculate feature importance for each fold
            % find train fold idx
            for fold = 1:size(subsampled_subfold.fold_index, 2)
                fold_name = strcat('fold_', num2str(fold));
                % find train fold idx
                train = ~subsampled_subfold.fold_index{fold};
                % load results
                results_fold = load(fullfile(model_dir, model, num_subs, fold_name, ...
                strcat('final_result_', model, '.mat')));

                % get subset of features
                feat_subset = feat_mat(:,subsampled_subfold.subfold_pos{fold});
                % get training set
                feat_train = feat_subset(:,train);
                % load predictions
                y_pred = results_fold.y_pred_train{1};
                % remove 0 FC sub
                zeroFC_sub = find(sum(feat_train == 0));
                feat_train(:,zeroFC_sub) = [];
                y_pred(zeroFC_sub,:) = [];
                % normalize
                feat_train_norm = (feat_train - mean(feat_train,1)) ./ std(feat_train, [], 1);
                % remove nan before calculating covariance
                nan_sub = any(isnan(feat_train_norm));
                feat_train_norm(:,nan_sub) = [];
                y_pred(nan_sub,:) = [];
                nan_sub = any(isnan(y_pred),2);
                feat_train_norm(:,nan_sub) = [];
                y_pred(nan_sub,:) = [];

                % compute covariance
                cov_mat(fold,:) = bsxfun(@minus,feat_train_norm,mean(feat_train_norm,2)) * ...
                    bsxfun(@minus,y_pred(:,b),mean(y_pred(:,b))) / (size(feat_train_norm,2));
            end
        end
        % don't take average over outerfolds
        cov_mat_mean(s,:,:) = cov_mat;
    end

    if isa(cov_mat_mean,'double')
        cov_mat_mean = single(cov_mat_mean);
    end

    % save results
    save(fullfile(save_dir, strcat(reg,'_cov_mat_behav',num2str(b),'.mat')), ...
        'cov_mat_mean', '-v7.3');
end