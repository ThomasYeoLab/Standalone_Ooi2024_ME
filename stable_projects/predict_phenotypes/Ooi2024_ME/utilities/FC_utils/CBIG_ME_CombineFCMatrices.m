function CBIG_ME_CombineFCMatrices(indir, sub_list)

% function CBIG_ME_CombineFCMatrices(indir, sub_list)
%
% This function combines each individual's FC matrix into a 3D matrix of
% #ROIx#ROIx#subjects to make it easier to enter into regression algorithms.
% Only works for HCP, saving the files for ABCD were too huge.
%
% Inputs:
%     -indir:
%      The directory in which FC matrices were saved.
%      E.g. '/home/leon_ooi/storage/optimal_prediction/replication/HCP/input'
%
%     -sub_list:
%      A string cotaining the name of the text file with all subject IDs.
%      E.g. 'subject_list_792.txt'
%
%
% Output:
%      A mat file is saved for each run order permutation and each scan duration.
%
% Written by Leon Ooi and CBIG under MIT license: https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md

% collate for each permutation of run order
for perm = 1:24
    % collate for each scan duration interval 
    for mins = 2:2:58
        % version settings
        vers = strcat('full_perm', num2str(perm));
        min_str = strcat(num2str(mins),'min');
        sub_txt = fullfile(indir, sub_list);
        % path to individual FCs - assuming that this was not changed from the 
        % FC generation script
        FC_mat = fullfile(indir, 'FC', vers, strcat(min_str,'_FC.mat'));
        if ~exist(FC_mat)
            % read subject list
            f = fopen(sub_txt);
            data = textscan(f,'%f');
            fclose(f);
            subj_list = data{:};
            % pre allocate space
            feat_mat = zeros(419,419,length(subj_list));

            % read files and save mat file
            for s = 1:length(subj_list)
                sub_corrmat = load(fullfile(indir,'FC', vers, strcat(num2str(mins),'mins'), ...
                    strcat(num2str(subj_list(s)),'_', min_str, '_FC.mat')));
                feat_mat(:,:,s) = sub_corrmat.corr_mat;
            end
            save(FC_mat, 'feat_mat', '-v7.3')
        else
            fprintf('FC exists not regenerating \n')
        end
    end
end