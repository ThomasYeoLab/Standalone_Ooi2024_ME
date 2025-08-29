function CBIG_ME_ABCD_KRR_randomFC_splithalf(num_sites, innerFolds, mins, vers, ...
    basedir, subtxt, subcsv, predvar, covtxt, ymat, covmat)

% Written by Leon Ooi and CBIG under MIT license: https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md

%% set up directories
addpath(fullfile(getenv('CBIG_CODE_DIR'),'stable_projects', 'predict_phenotypes', ...
    'Ooi2024_ME', 'utilities','regr_utils'))
indir = fullfile(basedir,'input');
outdir = fullfile(basedir,'output_splithalf', vers);

% set up adjustable parameters
min_str = strcat(num2str(mins),'min');
sub_txt = fullfile(indir, subtxt);
sub_csv= { fullfile(indir, subcsv) };
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
if ~exist(fullfile(outdir, 'no_relative_5_fold_sub_list.mat'))
    sub_fold = CBIG_ME_ABCD_LpOCV_split( sub_txt, sub_csv{1}, ...
        'subjectkey', 'site_group', num_sites, outdir, ',' ); 
else
    fprintf('Using existing sub_fold file \n')
    fold_temp = load(fullfile(outdir, 'no_relative_5_fold_sub_list.mat'));
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
    y = CBIG_read_y_from_csv(sub_csv, 'subjectkey', score_names, score_types,...
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
    cov_types = {'categorical', 'continuous'}; % define covariate types
    cov = CBIG_generate_covariates_from_csv(sub_csv, 'subjectkey', cov_names, cov_types, ...
         sub_txt, 'none', 'none', fullfile(outdir,cov_mat), ',');
else
    fprintf('Using existing covariate file \n')
    cov_temp = load(fullfile(outdir, cov_mat));
    cov = cov_temp.covariates;
end

%% KRR workflow
fprintf('[4.1] Run KRR workflow (all subjects)...\n')
for k = 1:length(sub_fold)
    fold_name = strcat('fold_', num2str(k));
    param.sub_fold = sub_fold(k);
    param.y = y;
    param.covariates = cov;
    param.outdir = fullfile(outdir, outstem, ...
    strcat(num2str(length(param.y)),'_subjects'), fold_name);
        
    % only run if results file does not exist
    if ~exist(fullfile(param.outdir,strcat('final_result_',param.outstem,'.mat')))
        fprintf('Collate feature matrices... \n')
        % read subject list
        f = fopen(sub_txt);
        data = textscan(f,'%s');
        fclose(f);
        subj_list = data{:};
        % read all FC from different permutations
        num_perms = 24;
        % randomly sample and combine across seeds
        feat_mat = zeros(419,419,length(subj_list));
        for subj_n = 1:length(subj_list)
            rng(k*subj_n,'twister')
            perm_idx = randsample(num_perms,1);
            sub_split = strsplit(subj_list{subj_n}, '_');
            sub_formatted = strcat(sub_split{1}, sub_split{2});
            sub_corrmat = load(fullfile(indir,'FC', strcat('full_perm', num2str(perm_idx)), ...
                strcat(num2str(mins),'mins'), strcat(sub_formatted,'_', min_str, '_FC.mat')));
            feat_mat(:,:,subj_n) = sub_corrmat.corr_mat;         
        end
        param.feature_mat = feat_mat;
        % run KRR workflow
        CBIG_KRR_workflow_LITE(param);
    else
        fprintf('Results exist, skipping... \n')
    end
end

% run for each subsample
sub_samples = [200 400 600 800 1000];
sites = CBIG_read_y_from_csv(sub_csv, 'subjectkey', {'site_group'}, ...
                {'categorical'}, sub_txt, fullfile(outdir, 'sites.mat'), ',');
trainFolds = 5;
for j = 1:length(sub_samples)
    ss = sub_samples(j);
    fprintf('[5.%i] Generate subsample (%i)... \n',j+1, ss)
    for k = 1:length(sub_fold)
        fprintf('\t Run KRR workflow (%i subjects, fold %i)... \n', ss, k)
        subfold_size = floor(ss / trainFolds);
        extras = mod(ss, trainFolds);
        fold_name = strcat('fold_', num2str(k));
        % grab y, covariates, features
        fold_idx = [];
        fold_pos = [];

        % get test subjects
        chosen_subs = find(sub_fold(k).fold_index == 1);
        fold_pos = [fold_pos; chosen_subs ];
        fold_idx = [fold_idx; ones(length(chosen_subs),1)];

        % get train subjects
        extra_count = 0;
        idx = find(sub_fold(k).fold_index == 0);
        train_sites = sites(idx);
        train_sites_unique = unique(sites(idx));
        assert(length(train_sites_unique) == trainFolds, ...
            "ERROR sites not equal to train folds")
        for i = 1:trainFolds
            % draw from each site cluster
            rng(1,'twister')
            site_idx = idx(train_sites == train_sites_unique(i));
            if extra_count < extras
                if subfold_size+1 >= length(site_idx)
                    fprintf('\t\t WARNING: %s has fewer subjects that required, using all subjects \n', fold_name)
                    chosen_subs = site_idx;
                else
                    chosen_subs = datasample(site_idx, subfold_size+1, 'Replace',false);
                end
                extra_count = extra_count + 1;
            else
                if subfold_size >= length(site_idx)
                    fprintf('\t\t WARNING: %s has fewer subjects that required, using all subjects \n', fold_name)
                    chosen_subs = site_idx;
                else
                    chosen_subs = datasample(site_idx, subfold_size, 'Replace',false);
                end
            end
            fold_pos = [fold_pos; chosen_subs];
            fold_idx = [fold_idx; zeros(length(chosen_subs),1)];
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
                rng(1,'twister')
                param.y(:, col) = param.y(:, col) + rand(length(fold_pos),1) * 1e-15;
            end
        end
        param.covariates = cov(fold_pos, :);
        param.outdir = fullfile(outdir, outstem, ...
            strcat(num2str(ss),'_subjects'), fold_name);
        
        if ~exist(fullfile(param.outdir,strcat('final_result_',param.outstem,'.mat')))
            fprintf('Collate feature matrices... \n')
            % read subject list
            f = fopen(sub_txt);
            data = textscan(f,'%s');
            fclose(f);
            subj_list = data{:};
            % read all FC from different permutations
            num_perms = 24;
            % randomly sample and combine across seeds
            feat_mat = zeros(419,419,length(subj_list));
            for subj_n = 1:length(subj_list)
                rng(k*subj_n,'twister')
                perm_idx = randsample(num_perms,1);
                sub_split = strsplit(subj_list{subj_n}, '_');
                sub_formatted = strcat(sub_split{1}, sub_split{2});
                sub_corrmat = load(fullfile(indir,'FC', strcat('full_perm', num2str(perm_idx)), ...
                    strcat(num2str(mins),'mins'), strcat(sub_formatted,'_', min_str, '_FC.mat')));
                feat_mat(:,:,subj_n) = sub_corrmat.corr_mat;
            end
            param.feature_mat = feat_mat(:,:,fold_pos);
            % run KRR workflow
            CBIG_KRR_workflow_LITE(param);
        else
            fprintf('Results exist, skipping... \n')
        end
    end
    % save subsampled indices
    save(fullfile(outdir, strcat(strcat(num2str(ss),'_subjects'), ...
    '_10_fold_sub_list.mat')), 'subsampled_subfold')
end

rmpath(fullfile(getenv('CBIG_CODE_DIR'),'stable_projects', 'predict_phenotypes', ...
    'Ooi2024_ME', 'utilities','regr_utils'))
end



