function CBIG_ME_calc_HCP_FC(subj_txt, mins, vers, perm_order, res)

% CBIG_ME_calc_HCP_FC(subj_txt, mins, vers, perm_order)
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
%      A number between 1 and (#runs)!. Since there are 4 runs in the HCP
%      this should be between 1-24. Can also input a number greater than 24 
%      to use alphabetical order (i.e. RUNXXX-LR or RUNXXX-RL).
%
%     -res: 
%      An integer representing parcellation resolution. Can be either 400
%      or 1000.
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
if res == 400
    ROI_file = fullfile(project_dir, 'utilities', 'FC_utils', 'CBIG_ME_HCP_labels.txt');
elseif res == 1000
    ROI_file = fullfile(project_dir, 'utilities', 'FC_utils', 'CBIG_ME_HCP_labels_res1000.txt');
else
    error("Resolution not recognised!");
end

% Path to data (MODIFY THIS TO WHERE YOUR DATA IS STORED)
HCP_dir = fullfile(getenv('HCP_data_dir'), 'S1200', 'individuals');

% Path to output (MODIFY THIS TO YOUR OWN OUTPUT PATH)
input_dir = fullfile(getenv('HOME'), 'storage', 'optimal_prediction', 'replication', ...
    'HCP', 'input');
FC_save_dir = fullfile(input_dir, 'FC');
% create folder to save FC in if it doesn't exist
if ~exist(FC_save_dir)
    mkdir(FC_save_dir);
end
% create sub directory to save FC
if res == 400
    if contains(vers, 'day')
        min_output_dir = fullfile(FC_save_dir, vers, ...
            strcat(num2str(mins),'mins'));
    else
        min_output_dir = fullfile(FC_save_dir, strcat(vers,'_perm',num2str(perm_order)), ...
            strcat(num2str(mins),'mins'));
    end
elseif res == 1000
    min_output_dir = fullfile(FC_save_dir, strcat(vers,'_1000parcels'), ...
        strcat(num2str(mins),'mins'));
else
    error("Resolution not recognised!");
end
if ~exist(min_output_dir)
    mkdir(min_output_dir)
end

%% Generate FC matrices
% read subject list
f = fopen(subj_txt);
data = textscan(f,'%f');
fclose(f);
final_list = data{:};

% calculate how many frames correspond to mins
first_t_frames = ceil((mins*60)/0.72);
fprintf('Generating FC matrices for first %i mins / %i frames \n', ....
    mins, first_t_frames)

% calculate FC for each subject in list
for n = 1:length(final_list)
    sub = num2str(final_list(n));
    indiv_path = fullfile(HCP_dir, sub, 'MNINonLinear', 'Results');
    % check available files
    avail_files = dir(strcat(indiv_path,'/rfMRI*'));
    sub_outfile = strcat(sub,'_', num2str(mins),'min_FC.mat');
    output_file = fullfile(min_output_dir,sub_outfile);
    
    % check ordering of data from TOD study
    tod_data = readtable(fullfile(getenv('CBIG_CODE_DIR'),'stable_projects', 'preprocessing', ...
        'Orban2020_tod', 'data_release', 'HCP_S1200_physio_data_summary_2020_02_11.csv'));
    % permute run order: all have at least 4 runs
    if contains(vers, 'day1')
        vers = 'full';
        run_order_perm = perms([1 2]);
    elseif contains(vers, 'day2')
        vers = 'full';
        run_order_perm = perms([3 4]);
    else
        run_order_perm = perms([1 2 3 4]);
    end
        
    % only run if FC doesn't exist already
    if ~exist(output_file)
        fprintf('\t Subj %i / %i :%s - generating FC \n', n, length(final_list), sub)
        clear subj_bold subj_cens
        subj_bold = [];
        subj_cens = [];
        
        % save available bold and censoring files
        if perm_order < 25
            run_order = run_order_perm(perm_order,:);
            for i = 1:length(run_order)
                if run_order(i) < 3
                    ses_num = 1;
                    run_num = run_order(i);
                else
                    ses_num = 2;
                    run_num = run_order(i) - 2;
                end
                % run and session number according to TOD paper
                run_select = tod_data.subject_id == final_list(n) & tod_data.run == run_num ...
                    & tod_data.session == ses_num;
                r_label = tod_data.run_label(run_select);
                r_label_split = strsplit(r_label{:}, '_');
                ph_enc_dir = tod_data.phase_encoding_dir(run_select);
                run = strcat('rfMRI_', r_label_split{1}, '_', ph_enc_dir{:});
                bold_file = fullfile(indiv_path, run, 'postprocessing', 'MSM_reg_wbsgrayordinatecortex', ...
                    strcat(run, '_Atlas_MSMAll_hp2000_clean_regress.dtseries.nii'));
                cens_txt = fullfile(indiv_path, run, 'postprocessing', 'MSM_reg_wbsgrayordinatecortex', ...
                    'scripts', strcat(run, '_FD0.2_DV75_censoring.txt'));
                if ~exist(cens_txt) | ~exist(bold_file)
                    continue
                end
                subj_bold = [ subj_bold bold_file ' '];
                subj_cens = [ subj_cens cens_txt ' '];
            end
        else
            % collate paths to runs and discard files into a cell
            for i = 1:length(avail_files)
                run = avail_files(i).name;
                bold_file = fullfile(indiv_path, run, 'postprocessing', 'MSM_reg_wbsgrayordinatecortex', ...
                    strcat(run, '_Atlas_MSMAll_hp2000_clean_regress.dtseries.nii'));
                cens_txt = fullfile(indiv_path, run, 'postprocessing', 'MSM_reg_wbsgrayordinatecortex', ...
                    'scripts', strcat(run, '_FD0.2_DV75_censoring.txt'));
                % skip if file does not exist
                if ~exist(cens_txt) | ~exist(bold_file)
                    continue
                end
                subj_bold = [ subj_bold bold_file ' '];
                subj_cens = [ subj_cens cens_txt ' '];
             end
        end
        
        subj_bold = { subj_bold };
        subj_cens = { subj_cens };
        
        % skip subject if no runs exist
        if isempty(subj_bold{:})
            fprintf('\t \t No usable runs... skipping processing \n')
            continue
        end
            
        % compute FC matrix based on version settings
        regression_mask = 'NONE';
        if contains(vers, 'no_censoring')
            % do not use censoring
            CBIG_ME_ComputeFC(output_file, subj_bold, subj_bold, ...
                'NONE', ROI_file, ROI_file, regression_mask, regression_mask, 1, ...
                0, first_t_frames, 'full')
        else
            % either full or uncensored_only
            CBIG_ME_ComputeFC(output_file, subj_bold, subj_bold, ...
            subj_cens, ROI_file, ROI_file, regression_mask, regression_mask, 1, ...
            0, first_t_frames, vers)
        end
    else
        fprintf('\t Subj %i / %i :%s - exists! \n', n, length(final_list), sub)
    end
end

rmpath(fullfile(project_dir, 'utilities','FC_utils'));
end
