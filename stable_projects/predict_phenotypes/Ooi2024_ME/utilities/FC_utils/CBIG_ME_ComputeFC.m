function CBIG_ME_ComputeFC(output_file, subj_text_list1, subj_text_list2, ...
    discard_frames_list, ROIs1, ROIs2, regression_mask1, regression_mask2, ...
    all_comb_bool, avg_sub_bool, first_t_frames, vers)

% This function is adapted from CBIG_ComputeROIs2ROIsCorrelationMatrix.
% The function computes the ROIs to ROIs correlation matrix using a
% specified method.
%
% The method will specify how many volumes are used to generate the FC
% matrix and the manner in which how the volumes are counted (see below).
%
% Inputs:
%     -output_file:
%      Name of the output, it should end with '.mat' (e.g.
%      'lh2rh_corrmat.mat')
%
%     -subj_text_list1, subj_text_list2: 
%      Text files in which each line represents one subject's bold run
%      (see original function for full description of the forms
%      subj_text_list can take)
%
%     -discard_frames_list: 
%      Text file which has a structure like subj_text_list1 and subj_text_list2, 
%      i.e. each line will correspond to one subject but will point to the 
%      location of the frame index output of motion scrubing. The motion 
%      scrubing output should be a text file with a binary column, where 1 
%      means keep, 0 means throw.
%      Use discard_frames_list = 'NONE' for no motion scrubbing.
%
%     -ROIs1, ROIs2:
%      Files specifying the demarcations of regions of interest. 
%      (see original function for full description of the forms
%      ROIs can take) ute the correlation of a single
%      surface ROI to a volume ROI
%   
%     -regression_mask1, regression_mask2:
%      Files specifying signals to be regressed from the timecourses.
%      (see original function for full description of the forms
%      regression_mask can take)
%
%     -all_comb_bool:
%      all_comb_bool = 1 if we want to compute all possible combination of
%      ROIs1 and ROIs2, that means if we have have N ROIs and M ROIs, our
%      output matrix will be M-by-N-by number of subjects
%
%     -all_sub_bool:
%      avg_sub_bool = 1 if we want our output matrix to be averaged across
%      all subjects
%
%     -first_t_vols:
%      A scalar. Number of volumes than should be used to calculate the FC.
%
%     -vers:
%      The manner in which to calculate the first_t_vols. Can be one of the
%      following options.
%      "full"           : Volumes will be grabbed in chronological order.
%                         Censoring occurs after frames are grabbed.
%      "uncensored_only": Volumes will be grabbed in chronological order. 
%                         Censoring occurs before frames are grabbed.
%
% Output:
%     -mat file
%      The output will be a mat file with the calculated FC, saved
%      according to the output_file name.
%
% Written by CBIG under MIT license: https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md

if(ischar(all_comb_bool))
    all_comb_bool = str2double(all_comb_bool);
end

if(ischar(avg_sub_bool))
    avg_sub_bool = str2double(avg_sub_bool);
end

%% read in subject_lists
subj_list_1 = read_sub_list(subj_text_list1);
subj_list_2 = read_sub_list(subj_text_list2);

if(length(subj_list_1) ~= length(subj_list_2))
    error('both lists should contain the same number of subjects');
end

%% read in discarded frames file name
if strcmp(discard_frames_list, 'NONE')
    frame = 0;
else
    frame = 1;
    discard_list = read_sub_list(discard_frames_list);
    if (length(discard_list) ~= length(subj_list_1))
        error('number of subjects in discard list should be the same as that of subjects');
    end
end

%% read in ROIs
ROIs1_cell = read_ROI_list(ROIs1);
ROIs2_cell = read_ROI_list(ROIs2);

%% read in regression list
[regress_cell1, regress1] = read_regress_list(regression_mask1);
[regress_cell2, regress2] = read_regress_list(regression_mask2);

%% space allocation
if avg_sub_bool == 1
    if all_comb_bool == 1
        corr_mat = zeros(length(ROIs1_cell), length(ROIs2_cell));
    else
        if length(ROIs1_cell) ~= length(ROIs2_cell)
            error('ROIs1_cell should have the same length as ROIs2_cell when all_comb_bool = 0');
        end
        corr_mat = zeros(length(ROIs1_cell), 1);
    end % end if all_comb_bool = 1
else
    if all_comb_bool == 1
        corr_mat = zeros(length(ROIs1_cell), length(ROIs2_cell), length(subj_list_1));
    else
        if length(ROIs1_cell) ~= length(ROIs2_cell)
            error('ROIs1_cell should have the same length as ROIs2_cell when all_comb_bool = 0');
        end
        corr_mat = zeros(length(ROIs1_cell), length(subj_list_1));
    end
end % end if avg_sub_bool = 1

%% Compute correlation
for i = 1:length(subj_list_1) % loop through each subject
    disp(num2str(i));
    
    S1 = textscan(subj_list_1{i}, '%s');
    S1 = S1{1}; % a cell of size (#runs x 1) for subject i in the first list
    
    S2 = textscan(subj_list_2{i}, '%s');
    S2 = S2{1}; % a cell of size (#runs x 1) for subject i in the second list

    if frame
        discard = textscan(discard_list{i}, '%s');
        discard = discard{1}; % a cell of size (#runs x 1) for subject i in the discard list
    end
    
    %%% MODIFICATION FROM ORIGNAL FUNCTION HERE
    % define flags that are used in modified function
    frame_count = 0;
    used_frames = 0;
    break_flag = 0;
    % run ordering
    run_order = 1:length(S1);
    %%% END OF MODIFICATION
    
    %%% MODIFICATION FROM ORIGNAL FUNCTION HERE
    % allow run order to be different from list
    for ro = 1:length(S1)
        j = run_order(ro);
    %%% END OF MODIFICATION

        % retrieve the binary frame index
        if frame
            discard_file = discard{j};
            fid = fopen(discard_file, 'r');
            frame_index = fscanf(fid,'%d');
        end
        
        input = S1{j};
        if (isempty(strfind(input, '.dtseries.nii'))) % input is a nifti file: .nii.gz
            input_series = MRIread(input);
            % time_course1 will look like nframes x nvertices for e.g. 236 x 10242
            time_course1 = single(transpose(reshape(input_series.vol, ...
                size(input_series.vol, 1) * size(input_series.vol, 2) * size(input_series.vol, 3), ...
                size(input_series.vol, 4))));
        else % input is a cifti file: .dtseries.nii
            input_series = ft_read_cifti(input);
            time_course1 = single(transpose(input_series.dtseries));
        end
        input = S2{j};
        if (isempty(strfind(input, '.dtseries.nii'))) % input is a nifti file: .nii.gz
            input_series = MRIread(input);
            % time_course1 will look like nframes x nvertices for e.g. 236 x 10242
            time_course2 = single(transpose(reshape(input_series.vol, ...
                size(input_series.vol, 1) * size(input_series.vol, 2) * size(input_series.vol, 3), ...
                size(input_series.vol, 4))));
        else % input is a cifti file: .dtseries.nii
            input_series = ft_read_cifti(input);
            time_course2 = single(transpose(input_series.dtseries));
        end
        
        %%% MODIFICATION FROM ORIGNAL FUNCTION HERE
        % Use only t frames to compute FC
        % Time series will be shortened if length of timeseries is less than
        % first_t_frames
        
        % Full version - grab volumes to be used in
        % chronological order before censoring
        if contains(vers, "full")
            % if run is more than volumes needed then cut time series
            if size(time_course1,1) > (first_t_frames - frame_count)
                rem_frames = first_t_frames - frame_count;
                % FC cannot be calculated with 1 frame, disregard this run
                if rem_frames == 1
                    break
                end
                % cut timeseries
                time_course1 = time_course1(1:rem_frames,:);
                time_course2 = time_course2(1:rem_frames,:);
                % cut discard_frames_list
                if frame
                    frame_index = frame_index(1:rem_frames,:);
                end
                % signal that no more runs are needed
                break_flag = 1;
                
            % otherwise use the whole run
            else
                frame_count = frame_count + size(time_course1,1);
            end
        end
        %%% END OF MODIFICATION
        
        if frame
            time_course1(frame_index==0,:) = [];
            time_course2(frame_index==0,:) = [];
        end
        
        %%% MODIFICATION FROM ORIGNAL FUNCTION HERE
        % Use only t frames to compute FC
        % Time series will be shortened if length of timeseries is less than
        % first_t_frames
        
        % Uncensored only version - grab volumes to be used in
        % chronological order after censoring
        if contains(vers, "uncensored_only")
            % if run is more than volumes needed then cut time series
            if size(time_course1,1) > (first_t_frames - frame_count)
                rem_frames = first_t_frames - frame_count;
                % FC cannot be calculated with 1 frame, disregard this run
                if rem_frames == 1
                    break
                end
                % cut timeseries
                time_course1 = time_course1(1:rem_frames,:);
                time_course2 = time_course2(1:rem_frames,:);
                run_weight = rem_frames / first_t_frames;
                % signal that no more runs are needed
                break_flag = 1;
                
            % otherwise use the whole run
            else
                run_weight = size(time_course1,1) / first_t_frames;
                frame_count = frame_count + size(time_course1,1);
            end
        end
        %%% END OF MODIFICATION

        % create time_courses based on ROIs
        t_series1 = zeros(size(time_course1, 1), length(ROIs1_cell));
        for k = 1:length(ROIs1_cell)
            t_series1(:,k) = CBIG_nanmean(time_course1(:, ROIs1_cell{k}), 2);
        end

        t_series2 = zeros(size(time_course2, 1), length(ROIs2_cell));       
        for k = 1: length(ROIs2_cell)
            t_series2(:,k) = CBIG_nanmean(time_course2(:, ROIs2_cell{k}), 2);
        end

        % regression
        if(regress1)
            regress_signal = zeros(size(time_course1, 1), length(regress_cell1));
            for k = 1:length(regress_cell1)
               regress_signal(:, k) = CBIG_nanmean(time_course1(:, regress_cell1{k} == 1), 2); 
            end

            % faster than using glmfit in which we need to loop through
            % all voxels
            X = [ones(size(time_course1, 1), 1) regress_signal];
            pseudo_inverse = pinv(X);
            b = pseudo_inverse*t_series1;
            t_series1 = t_series1 - X*b;
        end

        if(regress2)
            regress_signal = zeros(size(time_course2, 1), length(regress_cell2));
            for k = 1:length(regress_cell2)
                regress_signal(:, k) = mean(time_course2(:, regress_cell2{k} == 1), 2);
            end

            % faster than using glmfit in which we need to loop through
            % all voxels
            X = [ones(size(time_course2, 1), 1) regress_signal];
            pseudo_inverse = pinv(X);
            b = pseudo_inverse*t_series2;
            t_series2 = t_series2 - X*b;
        end

        % normalize series (size of series now is nframes x nvertices)
        t_series1 = bsxfun(@minus, t_series1, mean(t_series1, 1));
        t_series1 = bsxfun(@times, t_series1, 1./sqrt(sum(t_series1.^2, 1)));

        t_series2 = bsxfun(@minus, t_series2, mean(t_series2, 1));
        t_series2 = bsxfun(@times, t_series2, 1./sqrt(sum(t_series2.^2, 1)));

        % compute correlation
        if all_comb_bool == 1
            subj_corr_mat = t_series1' * t_series2;
        else
            subj_corr_mat = transpose(sum(t_series1 .* t_series2, 1));
        end
        
        %%% MODIFICATION FROM ORIGNAL FUNCTION HERE
        % Uncensored only version - run weight for averaging FC calculated
        % based off uncesored frames (calculated earlier)
        
        if contains(vers, "uncensored_only")
            % Fisher r-to-z transform
            if ro == 1
                subj_z_mat = CBIG_StableAtanh(subj_corr_mat);
            else
                subj_z_mat = subj_z_mat + (CBIG_StableAtanh(subj_corr_mat) * run_weight);
            end
            
        % Full version - run weight for averaging FC calculated based off
        % final number of uncesored frames
        elseif contains(vers, "full")
            used_frames = used_frames + size(time_course1,1);
            run_weight = size(time_course1,1) / used_frames;
            % Fisher r-to-z transform
            if ro == 1
                subj_z_mat = CBIG_StableAtanh(subj_corr_mat);
            else
                subj_z_mat = subj_z_mat * (1 -run_weight) + ...
                    (CBIG_StableAtanh(subj_corr_mat) * run_weight);
            end
        end
        
        if break_flag == 1
            break
        end
        %%% END OF MODIFICATION
    end % inner for loop for each run j of subject i
    
    %%% MODIFICATION FROM ORIGNAL FUNCTION HERE
    % run weight was used to average runs already, so this step is not
    % necessary (commented out)
    
    % subj_z_mat = subj_z_mat/length(S1); % average across number of runs
    
    %%% END OF MODIFICATION
    
    if avg_sub_bool == 1
        corr_mat = corr_mat + subj_z_mat;
    else
        if all_comb_bool == 1
            corr_mat(:, :, i) = tanh(subj_z_mat);
        else
            corr_mat(:, i) = tanh(subj_z_mat);
        end
    end
end % outermost for loop
disp(['isnan: ' num2str(sum(isnan(corr_mat(:)))) ' out of ' num2str(numel(corr_mat))]);

if avg_sub_bool == 1
    corr_mat = corr_mat/length(subj_list_1);
    corr_mat = tanh(corr_mat);
end

%% write out results
if(~isempty(strfind(output_file, '.mat')))
    save(output_file, 'corr_mat', '-v7.3');
end
end

%% sub-function to read subject lists
function subj_list = read_sub_list(subject_text_list)
% this function will output a 1xN cell where N is the number of
% subjects in the text_list, each subject will be represented by one
% line in the text file
% NOTE: multiple runs of the same subject will still stay on the same
% line
% Each cell will contain the location of the subject, for e.g.
% '<full_path>/subject1_run1_bold.nii.gz <full_path>/subject1_run2_bold.nii.gz'

% skip if input is already a cell
if iscell(subject_text_list)
    subj_list = subject_text_list;
else
    fid = fopen(subject_text_list, 'r');
    i = 0;
    while(1);
        tmp = fgetl(fid);
        if(tmp == -1)
            break
        else
            i = i + 1;
            subj_list{i} = tmp;
        end
    end
    fclose(fid);
end

end

%% sub-function to read ROI lists
function ROI_cell = read_ROI_list(ROI_list)
% ROI_list can be a .nii.gz/.mgz/.mgh/.dlabel.nii file contains a parcellation.

% this is for an arbitrary nii.gz file
if(~isempty(strfind(ROI_list, '.nii.gz')) || ~isempty(strfind(ROI_list, '.mgz')) || ~isempty(strfind(ROI_list, '.mgh')))
    ROI_vol = MRIread(ROI_list);
    
    regions = unique(ROI_vol.vol(ROI_vol.vol ~= 0));
    for i = 1:length(regions)
        ROI_cell{i} = find(ROI_vol.vol == regions(i));
    end
    
    % this is for an arbitrary .dlabel.nii file
elseif (~isempty(strfind(ROI_list, '.dlabel.nii')))
    ROI_vol = ft_read_cifti(ROI_list, 'mapname','array');
    regions = unique(ROI_vol.dlabel(ROI_vol.dlabel ~= 0));
    for i = 1:length(regions)
        ROI_cell{i} = find(ROI_vol.dlabel == regions(i));
    end
    
    % it can also be a single .label file
elseif (~isempty(strfind(ROI_list, '.label'))) % input ROI as a single .label file
    tmp = read_label([], ROI_list);
    ROI_cell{1} = tmp(:,1) + 1;
    
elseif (~isempty(strfind(ROI_list, '.annot'))) % input ROI as a single .annot file
    vertex_label = CBIG_read_annotation(ROI_list);
    regions = unique(vertex_label(vertex_label ~= 1));   % exclude medial wall
    for i = 1:length(regions)
        ROI_cell{i} = find(vertex_label == regions(i));
    end
    
else % input ROIs is a list of its locations: .nii.gz, dlabel.nii or .label
    fid = fopen(ROI_list, 'r');
    i = 0;
    while(1);
        tmp = fgetl(fid);
        if(tmp == -1)
            break
        else
            if(~isempty(strfind(tmp, '.nii.gz')) || ~isempty(strfind(tmp, '.mgz')) || ~isempty(strfind(tmp, '.mgh')))
                ROI_vol = MRIread(tmp);
                regions = unique(ROI_vol.vol(ROI_vol.vol ~= 0));
                for n = 1:length(regions)
                    i = i + 1;
                    ROI_cell{i} = find(ROI_vol.vol == regions(n)); % each cell contains a list of vertex's indices
                end
            elseif(~isempty(strfind(tmp, '.dlabel.nii')))
                ROI_vol = ft_read_cifti(tmp, 'mapname','array');
                regions = unique(ROI_vol.dlabel(ROI_vol.dlabel ~= 0));
                for n = 1:length(regions)
                    i = i + 1;
                    ROI_cell{i} = find(ROI_vol.dlabel == regions(n));
                end
            elseif(~isempty(strfind(tmp, '.label')))
                i = i + 1;
                tmp = read_label([], tmp);
                ROI_cell{i} = tmp(:, 1) + 1;
            end
        end
    end % end while
    fclose(fid);
end

end

%% sub-function to read regression lists
function [regress_cell, isRegress] = read_regress_list(regression_mask)
if(strcmp(regression_mask, 'NONE'))
    isRegress = 0;
    regress_cell = [];
else
    isRegress = 1;
    %regression_cell is a existing variable
    if(exist('regression_cell', 'var') == 1)
        regress_cell = regression_mask;
        %regression_cell is a existing .mat file contains a variable regress_cell
    elseif (exist(regression_mask, 'file') == 2)
        load(regression_mask);
    else
        error('regression_cell should be a variable or a .mat file which contains a variable regress_cell');
    end
    
end

end
