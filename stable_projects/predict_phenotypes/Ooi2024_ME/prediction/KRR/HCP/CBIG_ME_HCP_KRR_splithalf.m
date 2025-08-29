function CBIG_ME_HCP_KRR_splithalf(outerFolds, innerFolds, seed, mins, vers, ...
    basedir, subtxt, scorecsv, restrictedcsv, predvar, covtxt, ymat, covmat)

% function CBIG_ME_HCP_KRR_splithalf(outerFolds, innerFolds, seed, mins, vers, ...
%    basedir, subtxt, scorecsv, restrictedcsv, predvar, covtxt, ymat, covmat)
% 
% This function prepares the input parameters for a specified cross-validated split 
% of single-kernel regression.
% 
% Inputs:
%   - outerFolds
%     Number of outer-loop cross validation folds.
%
%   - innerFolds
%     Number of inner-loop cross validation folds.
%
%   - seed
%     Random seed for split to be generated.
%
%   - mins
%     An integer indicating which FC to run regression for.
%
%   - vers
%     The manner in which FC was calculated (See FC generation scripts). Can be "full", "no_censoring"
%     or "uncensored_only".
%
%   - basedir
%     The full path of output directory where `KRR_(outstem)` will be created.
%
%   - subtxt
%     Full path of the subject ID list. Each line in this list corresponds
%     to one subject ID. This list should be in the same order as the feature
%     matrix.
%
%   - scorecsv
%     Full path of csv file containing behavioral data and gender for all subjects. 
%
%   - restrictedcsv
%     Full path of csv file containing restricted data (Family ID and age) for all subjects. 
%  
%   - predvar
%     Full path to a text file with all behavioral (or demographic)
%     measures (measures to be predicted using kernel ridge regression). 
%     Each line corresponds to one behavioral name. The behavioral names
%     should exist as a header in "scorecsv".
% 
%   - covtxt
%     Full path to a text file stating all covariate names. Each line
%     corresponds to one covariate name. The covariate names should exist
%     as header in the "restrictedcsv".
%
%   - ymat
%     Name of the output mat file containing behavioural variables for all 
%     subjects in the regression analysis.
%
%   - covmat
%     Name of the output mat file containing covariates for all subjects in the
%     regression analysis.
% 
% Written by Leon Ooi and CBIG under MIT license: https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md

%% set up directories
indir = fullfile(basedir, 'input');
outdir = fullfile(basedir,'output_splithalf', vers);

% set up adjustable parameters
seed_name = strcat('seed_', num2str(seed));
min_str = strcat(num2str(mins),'min');
num_outer_folds = outerFolds;
sub_txt = fullfile(indir, subtxt);
score_csv = {scorecsv}; 
restricted_csv = {restrictedcsv};
cov_csv = {scorecsv restrictedcsv}; 
pred_var_txt =  predvar; 
cov_txt = covtxt;
y_mat = ymat; 
cov_mat = covmat;
lambda_set = [ 0 0.00001 0.0001 0.001 0.004 0.007 0.01 0.04 0.07 0.1 0.4 0.7 1 1.5 2 2.5 3 3.5 4 5 10 15 20];

% all other params
outstem = strcat('KRR_', min_str);
param.outstem = outstem;
param.num_inner_folds = innerFolds;
param.with_bias = 1;
param.ker_param.type = 'corr'; 
param.ker_param.scale = NaN;
param.lambda_set = lambda_set;
param.threshold_set = [];
param.cov_X = [];
param.metric = 'predictive_COD';

%% get subfold
fprintf('[1.1] Generate subfold... \n')

% generate folds
if ~exist(fullfile(outdir, seed_name, 'no_relative_2_fold_sub_list.mat'))
    sub_fold = CBIG_cross_validation_data_split(sub_txt, restricted_csv{1}, ...
     'Subject', 'Family_ID', num_outer_folds, seed, fullfile(outdir, seed_name), ','); 
else
    fprintf('Using existing sub_fold file \n')
    fold_temp = load(fullfile(outdir, seed_name, 'no_relative_2_fold_sub_list.mat'));
    sub_fold = fold_temp.sub_fold;
end

%% generate y matrix
fprintf('[2] Generate y matrix... \n')

if ~exist(fullfile(outdir, y_mat))
    % get names of tasks to predict
    fid = fopen(pred_var_txt,'r'); % variable names text file
    score_list = textscan(fid,'%s');
    score_names = score_list{1};
    fclose(fid);
    num_scores = size(score_names,1);
    % generate y
    score_types = cell(1,num_scores); % define score types
    score_types(:) = {'continuous'};
    y = CBIG_read_y_from_csv(score_csv, 'Subject', score_names, score_types,...
        sub_txt, fullfile(outdir, y_mat), ',');
else
    fprintf('Using existing y file \n')
    y_temp = load(fullfile(outdir,y_mat));
    y = y_temp.y;
end

%% generate covariate matrix
fprintf('[3] Generate covariate matrix... \n')

if ~exist(fullfile(outdir,cov_mat))
    % generate covariates
    fid = fopen(cov_txt,'r'); % covariate names text file
    cov_list = textscan(fid,'%s');
    cov_names = cov_list{1};
    fclose(fid);
    num_cov = size(cov_names,1);
    cov_types = {'continuous', 'categorical'}; % define covariate types
    cov = CBIG_generate_covariates_from_csv(cov_csv, 'Subject', cov_names, cov_types, ...
         sub_txt, 'none', 'none', fullfile(outdir,cov_mat), ',');
else
    fprintf('Using existing covariate file \n')
    cov_temp = load(fullfile(outdir, cov_mat));
    cov = cov_temp.covariates;
end

%% generate features
fprintf('[4] Collate feature matrices... \n')
% special sampling for random_full
if strcmp(vers,'random')
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
        FC_mat = fullfile(indir, 'FC', strcat('full_perm',num2str(permno)), strcat(min_str,'_FC.mat'));
        fc_perm = load(FC_mat);
        feat_mats(idx,:,:,:) = fc_perm.feat_mat;
        idx = idx + 1;
    end
    % randomly sample and combine across seeds
    feat_mat = zeros(419,419,length(subj_list));
    for subj_n = 1:length(subj_list)
        rng(seed*subj_n,'twister')
        perm_idx = randsample(num_perms,1);
        feat_mat(:,:,subj_n) = feat_mats(perm_idx,:,:,subj_n);            
    end
    % clear feat mats to save memory
    clear feat_mats
else
    FC_mat = fullfile(indir, 'FC', vers, strcat(min_str,'_FC.mat'));
    if ~exist(FC_mat)
        % read subject list
        f = fopen(sub_txt);
        data = textscan(f,'%f');
        fclose(f);
        subj_list = data{:};
        % pre allocate space
        feat_mat = zeros(419,419,length(subj_list));
        % read files
        for s = 1:length(subj_list)
            sub_corrmat = load(fullfile(indir,'FC', vers, strcat(num2str(mins),'mins'), ...
                strcat(num2str(subj_list(s)),'_', min_str, '_FC.mat')));
            feat_mat(:,:,s) = sub_corrmat.corr_mat;
        end
        save(FC_mat, 'feat_mat', '-v7.3')
    else
        fprintf('%s exists! Not regenerated.\n', strcat(min_str, '_FC.mat'))
        load(FC_mat)
    end
end

%% KRR workflow
fprintf('[5.1] Run KRR workflow (all subjects)...\n')
% set y variable of subjects with 0-FC to NaN so they are removed
zeroFC_subs = find(squeeze(sum(sum(feat_mat == 0, 2), 1)));
y(zeroFC_subs,:) = NaN;
param.sub_fold  = sub_fold;
param.y = y;
param.covariates = cov;
param.feature_mat = feat_mat;
param.outdir = fullfile(outdir, seed_name, outstem, ...
    strcat(num2str(length(param.y)),'_subjects'), 'results');
CBIG_KRR_workflow_LITE(param);

% run for each subsample
sub_samples = [50 100 150 200 250 300 350];
for j = 1:length(sub_samples)
    ss = sub_samples(j);
    fprintf('[5.%i] Generate subsample (%i)... \n',j+1, ss)
    subfold_size = floor(ss / (outerFolds - 1));
    extras = mod(ss, (outerFolds - 1));
    for k = 1:length(sub_fold)
        fprintf('\t Run KRR workflow (%i subjects, fold %i)... \n', ss, k)
        fold_name = strcat('fold_', num2str(k));
        % grab y, covariates, features
        fold_idx = [];
        fold_pos = [];
        extra_count = 0;
        for n = 1:length(sub_fold)
            if n == k
                chosen_subs = find(sub_fold(n).fold_index == 1);
                fold_pos = [fold_pos; chosen_subs ];
                fold_idx = [fold_idx; ones(length(chosen_subs),1)];
            else
                idx = find(sub_fold(n).fold_index == 1);
                rng(seed,'twister')
                if extra_count < extras
                    if subfold_size+1 >= length(idx)
                        fprintf('\t\t WARNING: %s has fewer subjects that required, using all subjects \n', fold_name)
                        chosen_subs = idx;
                    else
                        chosen_subs = datasample(idx, subfold_size+1, 'Replace',false);
                    end
                    extra_count = extra_count + 1;
                else
                    if subfold_size >= length(idx)
                        fprintf('\t\t WARNING: %s has fewer subjects that required, using all subjects \n', fold_name)
                        chosen_subs = idx;
                    else
                        chosen_subs = datasample(idx, subfold_size, 'Replace',false);
                    end
                end
                fold_pos = [fold_pos; chosen_subs];
                fold_idx = [fold_idx; zeros(length(chosen_subs),1)];
            end
        end
        new_subfold.fold_index = fold_idx;
        % record subfold indices for saving later
        subsampled_subfold.fold_index{k} = new_subfold.fold_index;
        subsampled_subfold.subfold_pos{k} = fold_pos;
        param.sub_fold  = new_subfold;
        param.y = y(fold_pos, :);
        % bug if only 2 types of variables left, add random noise
        for col = 1:size(param.y, 2)
            if numel(unique(param.y(:,col)))==2
                fprintf('\t WARNING: var %i is now binary, adding small noise. \n', col)
                rng(seed,'twister')
                param.y(:, col) = param.y(:, col) + rand(length(fold_pos),1) * 1e-15;
            end
        end
        param.covariates = cov(fold_pos, :);
        param.feature_mat = feat_mat(:,:,fold_pos);
        param.outdir = fullfile(outdir, seed_name, outstem, ...
            strcat(num2str(ss),'_subjects'), fold_name);
        CBIG_KRR_workflow_LITE(param);
    end
    % save subsampled indices
    save(fullfile(outdir, seed_name, strcat(strcat(num2str(ss),'_subjects'), ...
    '_10_fold_sub_list.mat')), 'subsampled_subfold')
end

end



