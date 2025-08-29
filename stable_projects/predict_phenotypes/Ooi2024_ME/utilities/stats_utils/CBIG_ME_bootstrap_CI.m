function CBIG_ME_bootstrap_CI(K_path, start_rand, output_path)
% function CBIG_ME_bootstrap_CI(K, start_rand)
% 
% This function takes in a set of K1 and K2 values (estimated by fitting
% the theoretical model) and an index to perform bootstrapping 10 times.
% In the paper, the bootstrapping was used to compute a 95% confidence
% interval (CI) of the most cost-effective scan time (Figure 4C). 
%
% The bootstrapping process first randomly samples (with replacement) the
% datasets. Then within each sampled datasets, it randomly samples (with
% replacement) the phenotypes. So, one dataset might appear multiple times
% in a bootstrapping sample. Also, one phenotype might appear multiple times
% in a sampled dataset. In the paper, we performed bootstrapping 1000
% times, meaming that this script needs to be 100 times with different start_ind
% (i.e. starting indices)
% 
% After generating the results for 1000 bootstrapping samples, load and
% combine them into a single vector. The lower and upper bound of the
% 95% CI are defined as the 2.5th percentile and 97.5th percentile of that
% vector. (Note that this script does not handle this part of the calculation)
%
% IMPORTANT: In the paper, the CI was computed with the cognitive factor
% scores REMOVED for all datasets
% 
% Inputs:
%   - K_path
%     the path to a csv file with a N-by-2 matrix. N refers to the number
%     of pheotypes. 2 refers to 2 columns: 1 column of K1 and 1 column of
%     K2. K1 and K2 are estimated by fitting the theoretical model to the
%     empirical data. More specifically, we take all the phenotypes under 
%     'whole-brain, original' from 'stable_projects/predict_phenotypes/
%     Ooi2024_ME/results/CBIG_ME_TheoreticalModel_Params.xlsx' and remove
%     ALL the cognitive factor scores. Use 'K_bootstrapping.csv' to
%     replicate the results.
%
%   - start_rand
%     a number (in char format) used to set the random seed for
%     bootstrapping. 
%
%   - output_path
%     the path used to save the output (i.e. the most cost-effective scan time)
%
% Example:
%   CBIG_ME_bootstrap_CI('/home/shaoshi.z/CBIG_ME_K_for_bootstrapping.csv', '1');
% Written by Shaoshi Zhang and CBIG under MIT license: https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md

% load in K1 and K2 given the path of K values
K = csvread(K_path);

% t records the most cost-effective scan time for each bootstarpping sample
t = [];

% Line 62 generates a random seed.
% Due to the nature of bootstrapping, there is a chance that the sampled
% phenotypes cannot reach the specified target accuracy. In that case, we'll 
% draw another sample, the process repeats until we have 10 valid samples. 
% Consequently, because of the possibility of drawing a failed sample, this 
% batch of random seeds needs to be sufficiently far apart from the next batch 
% of random seeds to avoid any overlapping random seed. Thus, a factor of 12345 
% is used to make sure all batches (of 10) having a unique random seed.
i = str2double(start_rand)*12345;

% use a while loop to gather 10 valid samples
while length(t) < 10
    
    % total_budget_all records the total budget across 12 conditions (2 acc x 3 scan cost x 2 overhead cost)
    total_budget_all = [];
    
    % search range for the most CE scan time. NOTE: change the upper bound from 120 to 400
    % for the subcortical case (Figure 6A)
    for T = 10:5:120
        [total_budget, err] = get_total_budget(i, K, T);
        total_budget_all = [total_budget_all total_budget];
        
        % there are cases where there's no way for the sampled phenotypes
        % to reach the accuracy target. If that happens, a new sample is
        % needed
        if err
            break
        end        
    end
    
    % if there is no error!
    if ~err
        % get the minimum budget for each of the 12 conditions
        for j = 1:12
            minimum_total_budget = min(total_budget_all(j, :));
            % normalize the budget to the minimum budget and covert it to a
            % percentage
            total_budget_all(j, :) = (total_budget_all(j, :)/minimum_total_budget - 1)*100;
        end
        
        % compute the mean minimum budget across 12 conditions and fetch
        % the most CE scan time
        mean_minimum_total_budget = mean(total_budget_all);
        [~, ind] = min(mean_minimum_total_budget);
        t_cost = 10:5:120;
        t = [t; t_cost(ind)];
        i = i + 1;
    else
        i = i + 1;
    end
end

% save out the results (10 samples in total)
dlmwrite([output_path '/' start_rand '.csv'], t, 'delimiter', ',', 'precision', 15);
end


function [B, err] = get_total_budget(random_seed, K, T)
% B is the total budget
% K refers to the K values (K1 and K2). We don't need K0 because we only
% consider the normalized prediction accuracy

% number of phenotypes included for each dataset. The phenotype order follows 
% 'stable_projects/predict_phenotypes/Ooi2024_ME/results/CBIG_ME_TheoreticalModel_Params.xlsx'
num_phenotypes = [16; 18; 14; 7; 7; 6; 15; 18; 17];

% K_dataset is a cell array that groups the phenotypes by datasets into
% separate cells, which makes the dataset-level random sampling easier
K_dataset = {};

% since the K are arranged sequentially, we use a partition variable to
% indicate the starting and ending position
partition = cumsum(num_phenotypes);
for i = 1:length(num_phenotypes)
    if i == 1
        starting = 1;
    else
        starting = partition(i-1)+1;
    end
    ending = partition(i);
    K_dataset{i} = K(starting:ending, :);
end

rng(random_seed, 'twister')
B = [];

for i = 1:length(num_phenotypes)
   % dataset-level random sampling
   k = randi([1, length(num_phenotypes)], 1);
   KK = K_dataset{k};
   
   starting = 1; 
   ending = size(KK, 1);
   l = size(KK, 1);
   % within-dataset phenotype-level random sampling
   r = randi([starting, ending], l, 1);
   K_bootstrap = KK(r, :);
   
   % loop through the 12 conditions
   % 3 target accuracies (r)
   for r = [0.8, 0.9, 0.95]
       % 2 (o)verhead cost per participant
       for o = [500, 1000]
           % 2 (s)canning cost per minute
           for s = [500/60, 1000/60]
               
               % find the minimum sample size (N) that can achieve the target acc 
               [n, err] = find_N(r, T, K_bootstrap);
               
               % if the target acc can never be reached, return with error
               if err == 1
                   return
               end
               
               % total budget = overhead cost * # participants + scanning
               % cost * overhead cost * # participants
               b = o*n + s*n*T;
               
               % append the list
               B = [B; b];
           end
       end
   end    
end

end


function [n, err] = find_N(r, T, K)
% find the minimum sample size required to achieve the target accuracy (r)
% given a fixed scan time and the K values of phenotypes

% the process is like a binary search
n_lo = 1;
n_hi = 20000;

flag = 1;
err = 0;
while flag
    if n_lo == n_hi
        err = 1;
        return
    end
    n = ceil((n_lo + n_hi)/2);
    acc = get_acc(n, T, K);
    if acc == r
        flag = 0;
    elseif acc > r
        acc = get_acc(n-1, T, K);
        if acc < r
            flag = 0;
        elseif acc == r
            flag = 0;
            n = n - 1;
        else
            n_hi = n;
        end
    elseif acc < r
        acc = get_acc(n+1, T, K);
        if acc > r
            flag = 0;
            n = n + 1;
        elseif acc == r
            flag = 0;
            n = n + 1;
        else
            n_lo = n;
        end
    end
end

end


function acc = get_acc(n, T, K)
% compute the averaged accuracy across phenotypes using the theoretical
% model
acc = [];
for i = 1:size(K, 1)
   k = K(i, :);
   
   % 0.9 is to correct for the fact that only 90% of the total sample size goes
   % into model training
   acc = [acc; sqrt(1/(1+k(1)/floor(n*0.9)+k(2)/(floor(n*0.9)*T)))];
end
acc = mean(acc);

end
