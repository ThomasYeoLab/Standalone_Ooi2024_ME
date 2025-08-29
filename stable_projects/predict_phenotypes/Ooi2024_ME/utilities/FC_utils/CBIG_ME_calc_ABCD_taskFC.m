function CBIG_ME_calc_ABCD_taskFC(subj_txt, mins, vers, perm_order, batch, task)

% function CBIG_ME_calc_ABCD_FC(subj_txt, mins, vers, perm_order, batch, task)
%
% This function calls CBIG_ME_ComputeFC to generate a FC matrix for each
% participant in the HCP dataset.
%
% Inputs:
%     -subj_txt:
%      A text file containing the participant IDs for participants used in
%      the analysis.
%
%     -mins:
%      A scalar. The scan duration (in minutes) that is used to calculate
%      the FC.
%
%     -vers:
%      The manner in which to calculate the the first t frames of data used
%      to generate the FC. Can be one of the following:
%      1) "full": Volumes will be grabbed in chronological order. Censoring
%      occurs after frames are grabbed.
%      2) "uncensored_only": Volumes will be grabbed in chronological order.
%      Censoring occurs before frames are grabbed.
%      3) "no_censoring": Volumes will be grabbed in chronological order.
%      Censoring files are ignored.
%
%     -perm_order: 
%      A number between 1 and (#runs)!. Since there are 4 runs in the ABCD
%      this should be between 1-24.
%                  
%     -batch:
%      A scalar which indicates which batch of 1000 subject to process.
%      This is to allow multiple jobs to be submitted to speed up
%      processing.
%
%     - task:
%      A string containing the task to be processed. Can be "MID", "NBACK"
%      or "SST".
%
% Output:
%     -FC_save_dir:
%      A directory in which the FC from each "mins" minutes are saved. One
%      mat file is saved per participant.
%
% Written by Leon Ooi and CBIG under MIT license: https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md

%% Set up directories and necessary files
% Project code directory
project_dir = fullfile(getenv('CBIG_CODE_DIR'), 'stable_projects', ...
    'predict_phenotypes', 'Ooi2024_ME');
addpath(fullfile(project_dir, 'utilities','FC_utils'));

% Path to parcellation details
schaefer_parc_dir = fullfile(getenv('CBIG_CODE_DIR'), 'stable_projects', ...
    'brain_parcellation', 'Schaefer2018_LocalGlobal', 'Parcellations');
ROI_list = { fullfile(schaefer_parc_dir, 'FreeSurfer5.3', 'fsaverage6', 'label', ...
    'lh.Schaefer2018_400Parcels_17Networks_order.annot') ...
    fullfile(schaefer_parc_dir, 'FreeSurfer5.3', 'fsaverage6', 'label', ...
    'rh.Schaefer2018_400Parcels_17Networks_order.annot') };

% Path to data (MODIFY THIS TO WHERE YOUR DATA IS STORED)
ABCD_dir = fullfile(getenv('ABCD_data_dir'), 'process', 'y0', ...
    'task_GSR_mf_FD0.3_DVARS50');

% Path to output (MODIFY THIS TO YOUR OWN OUTPUT PATH)
input_dir = fullfile(getenv('HOME'), 'storage', 'optimal_prediction', 'replication', ...
    'ABCD', 'input');
FC_save_dir = fullfile(input_dir, 'FC');
% create folder to save FC in if it doesn't exist
if ~exist(FC_save_dir)
    mkdir(FC_save_dir);
end
% create sub directory to save FC
min_output_dir = fullfile(FC_save_dir, strcat(vers,'_',task), ...
    strcat(num2str(mins),'mins'));
if ~exist(min_output_dir)
    mkdir(min_output_dir)
end

%% Generate FC matrices
% read subject list
f = fopen(subj_txt);
data = textscan(f,'%s');
fclose(f);
final_list = data{:};

% process in batches of 1000 to speed up job
start_s = (batch-1)*1000 + 1;
end_s = batch*1000;
if end_s < length(final_list)
	final_list = final_list(start_s:end_s);
else
	final_list = final_list(start_s:end);
end

% calculate how many frames correspond to mins
first_t_frames = ceil((mins*60)/0.8); 
fprintf('Generating FC matrices for first %i mins / %i frames \n', ....
    mins, first_t_frames)

% calculate FC for each subject in list
for n = 1:length(final_list)
    % check available files
    sub_tmp = strsplit(final_list{n}, '_'); % remove underscore in subj ID
    sub = strcat(sub_tmp{1}, sub_tmp{2});
    sub_outfile = strcat(sub,'_', num2str(mins),'min_FC.mat');
    output_file = fullfile(min_output_dir,sub_outfile);
    % create temporary directory to combine intermediate output
    sub_tmp_dir = fullfile(min_output_dir,sub);
    if ~exist(sub_tmp_dir)
        mkdir(sub_tmp_dir)
    end
    indiv_path = fullfile(ABCD_dir, sub);
    %bold_indiv_path = fullfile(getenv('ABCD_task_dir'), sub);
    %sc_indiv_path = fullfile(ABCD_dir, '..', 'task_GSR', sub); 
    
    % collate paths to runs and discard files into a cell   
    run_order_perm = perms([1 2]);
    run_order = run_order_perm(perm_order,:);
    
    % only run if FC doesn't exist already
    if ~exist(output_file)
        fprintf('\t Subj %i / %i :%s - generating FC \n', n, length(final_list), sub)
        clear subj_lh_bold subj_rh_bold subj_vol subj_cens
        subj_lh_bold = [];
        subj_rh_bold = [];
        subj_vol = [];
        subj_cens = [];
        
        for i = 1:2 % corresponds to the 2 runs
            run = run_order(i);
            if strcmp(task, "MID")
                run = num2str(run + 100);
            elseif strcmp(task, "NBACK")
                run = num2str(run + 200);
            elseif strcmp(task, "SST")
                run = num2str(run + 300);
            end

            lh_file = fullfile(indiv_path, 'surf', strcat('lh.', sub, '_bld', ...
                run,'_task_mc_skip_residc_interp_FDRMS0.3_DVARS50_bp_0.009_0.08_fs6_sm6.nii.gz'));
            rh_file = fullfile(indiv_path, 'surf', strcat('rh.', sub, '_bld', ...
                run,'_task_mc_skip_residc_interp_FDRMS0.3_DVARS50_bp_0.009_0.08_fs6_sm6.nii.gz'));
            bold_file = fullfile(indiv_path, 'bold', run, ...
                strcat(sub, '_bld', run,'_task_mc_skip_residc_interp_FDRMS0.3_DVARS50_bp_0.009_0.08.nii.gz'));
            cens_txt = fullfile(indiv_path, 'qc', ...
                strcat(sub, '_bld',run,'_FDRMS0.3_DVARS50_motion_outliers.txt'));
            if ~exist(cens_txt) | ~exist(lh_file) | ~exist(rh_file) | ~exist(bold_file)
                continue
            end
            subj_lh_bold = [ subj_lh_bold lh_file ' '];
            subj_rh_bold = [ subj_rh_bold rh_file ' '];
            subj_vol = [ subj_vol bold_file ' '];
            subj_cens = [ subj_cens cens_txt ' '];
        end
        subj_lh_bold = { subj_lh_bold };
        subj_rh_bold = { subj_rh_bold };
        subj_vol = { subj_vol };
        subj_cens = { subj_cens };
        
        % skip subject if no runs exist
        if isempty(subj_lh_bold{:})
            fprintf('\t \t No usable runs... skipping processing \n')
            continue
        end
        
        % compute FC matrix based on version settings
        regression_mask = 'NONE';
        subctx_file = fullfile(indiv_path, 'FC_metrics', 'ROIs', ...
            strcat(sub, '.subcortex.19aseg.func.nii.gz'));
        input_vers = vers;
        if strcmp(vers, 'no_censoring')
            subj_cens = 'NONE';
            input_vers = 'full';
        end
        
        % calculate for all combinations of lh and rh
        % lh2lh
        lh2lh_output = fullfile(sub_tmp_dir, 'lh2lh_tmp.mat');
        CBIG_ME_ComputeFC(lh2lh_output, subj_lh_bold, subj_lh_bold, ...
            subj_cens, ROI_list{1}, ROI_list{1}, regression_mask, regression_mask, 1, ...
            0, first_t_frames, input_vers)
        % lh2rh
        lh2rh_output = fullfile(sub_tmp_dir, 'lh2rh_tmp.mat');
        CBIG_ME_ComputeFC(lh2rh_output, subj_lh_bold, subj_rh_bold, ...
            subj_cens, ROI_list{1}, ROI_list{2}, regression_mask, regression_mask, 1, ...
            0, first_t_frames, input_vers)
        % rh2rh
        rh2rh_output = fullfile(sub_tmp_dir, 'rh2rh_tmp.mat');
        CBIG_ME_ComputeFC(rh2rh_output, subj_rh_bold, subj_rh_bold, ...
            subj_cens, ROI_list{2}, ROI_list{2}, regression_mask, regression_mask, 1, ...
            0, first_t_frames, input_vers)
        %lh2sc
        lh2sc_output = fullfile(sub_tmp_dir, 'lh2sc_tmp.mat');
        CBIG_ME_ComputeFC(lh2sc_output, subj_lh_bold, subj_vol, ...
            subj_cens, ROI_list{1}, subctx_file, regression_mask, regression_mask, 1, ...
            0, first_t_frames, input_vers)
        %rh2sc
        rh2sc_output = fullfile(sub_tmp_dir, 'rh2sc_tmp.mat');
        CBIG_ME_ComputeFC(rh2sc_output, subj_rh_bold, subj_vol, ...
            subj_cens, ROI_list{2}, subctx_file, regression_mask, regression_mask, 1, ...
            0, first_t_frames, input_vers)
        %sc2sc
        sc2sc_output = fullfile(sub_tmp_dir, 'sc2sc_tmp.mat');
        CBIG_ME_ComputeFC(sc2sc_output, subj_vol, subj_vol, ...
            subj_cens, subctx_file, subctx_file, regression_mask, regression_mask, 1, ...
            0, first_t_frames, input_vers)
        
        % combine each section
        lh2lh = load(lh2lh_output);
        lh2rh = load(lh2rh_output);
        rh2rh = load(rh2rh_output);
        lh2subcort = load(lh2sc_output);
        rh2subcort = load(rh2sc_output);
        subcort2subcort = load(sc2sc_output);
    
        lh2all = [lh2lh.corr_mat lh2rh.corr_mat lh2subcort.corr_mat];
        rh2all = [lh2rh.corr_mat' rh2rh.corr_mat rh2subcort.corr_mat];
        subcort2all = [lh2subcort.corr_mat' rh2subcort.corr_mat' subcort2subcort.corr_mat];
        corr_mat = [lh2all; rh2all; subcort2all];
        
        save(output_file, 'corr_mat', '-v7.3');
        % remove output files
        clear lh2lh lh2rh rh2rh lh2subcort rh2subcort subcort2subcort
        clear lh2all rh2all subcort2all corr_mat
        rmdir(sub_tmp_dir, 's')
    else
        fprintf('\t Subj %i / %i :%s - exists! \n', n, length(final_list), sub)
        rmdir(sub_tmp_dir, 's')
    end
end

rmpath(fullfile(project_dir, 'utilities','FC_utils'));
end
