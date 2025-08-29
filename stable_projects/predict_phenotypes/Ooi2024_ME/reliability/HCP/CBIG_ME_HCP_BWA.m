function CBIG_ME_HCP_BWA(input_dir,results_dir,vers,num_mins,num_subs,b)

% function CBIG_ME_ABCD_BWA(input_dir,results_dir,vers,num_mins,num_subs,b)
% 
% This function calculated the univariate t-statistics for each phenotype.
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
if ~exist(save_dir)
    mkdir(save_dir);
end

% set up adjustable parameters
total_seeds = 50;
reg = "KRR";
model = strcat(reg,'_',num_mins);
sub_txt = fullfile(input_dir, 'subject_list_792.txt');

% process for each seed
for s = 1:total_seeds
	fprintf('Calculating for %s, %s, seed %i / %i \n', ...
	    num_mins, num_subs, s, total_seeds)
	seed_name = strcat('seed_', num2str(s));
	model_dir = fullfile(results_dir, seed_name);

	% load FC
	if strcmp(vers,'random')
	    fprintf('Regenerating random FC... \n')
	    % read subject list
	    file = fopen(sub_txt);
	    data = textscan(file,'%f');
	    fclose(file);
	    subj_list = data{:};
	    % read all FC from different permutations
	    num_perms = 24;
	    feat_mats = zeros(num_perms,419,419,length(subj_list));
	    idx = 1;
	    for permno = 1:num_perms
		FC_mat = fullfile(input_dir, 'FC', ...
                    strcat('full_perm',num2str(permno)), strcat(num_mins,'_FC.mat'));
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

            % calculate feature importance for each fold
            for f = 1:size(sub_fold,1)
                fold_name = strcat('fold_', num2str(f));
                % find train fold idx
                train = ~sub_fold(f).fold_index;
                % load features
                feat_train = feat_mat(:,train);
                % load y
                y_fold = load(fullfile(model_dir, model, num_subs, 'results', ...
                    'y', fold_name, strcat('y_regress_', model, '.mat'))); 
                y_all = y_fold.y_resid;
                y_train = y_all(train,:);
                % remove 0 FC sub and normalize
                zeroFC_sub = find(sum(feat_train == 0));
                feat_train(:,zeroFC_sub) = [];
                y_train(zeroFC_sub,:) = [];
                feat_train_norm = (feat_train - mean(feat_train,1)) ./ std(feat_train, [], 1);

                % remove nan before calculating tstats 
                [feat_train_norm, y_train, nan_sub] = ...
                    remove_nan(feat_train_norm, y_train);

                % compute tstats
                [r_all, p_all] = cellfun(@(f) corr(f', y_train(:,b)), ...
                    mat2cell(feat_train_norm,ones(size(feat_train_norm,1),1), ...
                    [size(feat_train_norm,2)]));
                tstats_mat(f,:) = r_all.*sqrt((size(feat_train_norm,2)-2)./(1-r_all.^2));
                p_mat(f,:) = p_all;
            end
        else
            % load sub_fold
            load(fullfile(model_dir, strcat(num_subs, '_10_fold_sub_list.mat')));
            
            for f = 1:size(subsampled_subfold.fold_index, 2)
                fold_name = strcat('fold_', num2str(f));
                % find train fold idx
                train = ~subsampled_subfold.fold_index{f};
                % get subset of features
                feat_subset = feat_mat(:,subsampled_subfold.subfold_pos{f});
                % load features and normalize
                feat_train = feat_subset(:,train);
                feat_train_norm = (feat_train - mean(feat_train,1)) ./ std(feat_train, [], 1);
                % load y
                y_fold = load(fullfile(model_dir, model, num_subs, ...
                    fold_name, 'y', 'fold_1', strcat('y_regress_', model, '.mat'))); 
                y_all = y_fold.y_resid;
                y_train = y_all(train,:);
                % remove 0 FC sub and normalize
                zeroFC_sub = find(sum(feat_train == 0));
                feat_train(:,zeroFC_sub) = [];
                y_train(zeroFC_sub,:) = [];
                feat_train_norm = (feat_train - mean(feat_train,1)) ./ std(feat_train, [], 1);

                % remove nan before calculating tstats 
                [feat_train_norm, y_train, nan_sub] = ...
                    remove_nan(feat_train_norm, y_train);

                % compute tstats
                [r_all, p_all] = cellfun(@(f) corr(f', y_train(:,b)), ...
                    mat2cell(feat_train_norm,ones(size(feat_train_norm,1),1), ...
                    [size(feat_train_norm,2)]));
                tstats_mat(f,:) = r_all.*sqrt((size(feat_train_norm,2)-2)./(1-r_all.^2));
                p_mat(f,:) = p_all;
            end
        end
        
        % don't take average over outerfolds
        tstats_mat_mean(s,:,:) = tstats_mat;
        p_mat_mean(s,:,:) = p_mat;
    end
    % convert to single
    tstats_mat_mean = convert_2_single(tstats_mat_mean);
    p_mat_mean = convert_2_single(p_mat_mean);
    % save results
    save(fullfile(save_dir, strcat(reg,'_tstats_mat_behav',num2str(b),'.mat')), ...
        'tstats_mat_mean', 'p_mat_mean', '-v7.3');
end

function mat = convert_2_single(mat)
% Check if input matrix is double - converts to single if it is
%
% Input:
% - mat
%   A 2D matrix with either double or single values.

if isa(mat,'double')
    mat = single(mat);
end

end

function [feat, y, nan_sub] = remove_nan(feat, y)
% Remove nan entries from features and y
%
% Input:
% - feat
%   A 2D matrix of #features x #subjects
% - y
%   A 2D matrix of #subjects x #targets.

nan_sub = any(isnan(feat));
feat(:,nan_sub) = [];
y(nan_sub,:) = [];

nan_sub = any(isnan(y),2);
feat(:,nan_sub) = [];
y(nan_sub,:) = [];

end