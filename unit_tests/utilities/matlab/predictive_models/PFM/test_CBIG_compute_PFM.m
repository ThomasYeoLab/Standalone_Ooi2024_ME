classdef test_CBIG_compute_PFM < matlab.unittest.TestCase

    methods (Test)
        function test_CBIG_compute_PFM_general_example(testCase)
            % create output folder
            CBIG_CODE_DIR = getenv('CBIG_CODE_DIR');
            parent_dir = fullfile(CBIG_CODE_DIR, 'unit_tests', 'utilities', 'matlab',... 
                          'predictive_models', 'PFM');
            input_dir = fullfile(parent_dir, 'input', 'general');
            outdir = fullfile(parent_dir, 'output', 'general');
            ref_output = fullfile(parent_dir, 'ref_output', 'general');
            if exist(outdir,'dir')
                rmdir(outdir, 's')
            end
            mkdir(outdir)
            
            feature_file = fullfile(input_dir, 'feature_file.mat');
            y_pred_train_file = fullfile(input_dir, 'y_pred_train.mat');
           
            % compare the results or replace the reference output
            CBIG_CODE_DIR = getenv('CBIG_CODE_DIR');
            load(fullfile(CBIG_CODE_DIR, 'unit_tests','replace_unittest_flag'));
            if replace_unittest_flag
                % replace PFM results
                copyfile(fullfile(outdir,'PFM_score.mat'), ...
                    fullfile(ref_output,'PFM_score.mat'));
            else
                PFM_score = CBIG_compute_PFM_general(feature_file, y_pred_train_file);
                save(fullfile(outdir, 'PFM_score.mat'), 'PFM_score');

                
                % compare results
                test = load(fullfile(outdir, 'PFM_score.mat'));
                ref = load(fullfile(ref_output, 'PFM_score.mat'));
                assert(max(abs((test.PFM_score(:) - ref.PFM_score(:)))) < 1e-6, ...
                    'PFM different');
            end
            
            % remove the output directory
            rmdir(outdir, 's')
        end
        
        function test_CBIG_compute_PFM_singleKRR_example(testCase)
            % create output folder
            CBIG_CODE_DIR = getenv('CBIG_CODE_DIR');
            parent_dir = fullfile(CBIG_CODE_DIR, 'unit_tests', 'utilities', 'matlab',... 
                          'predictive_models', 'PFM');
            input_dir = fullfile(parent_dir, 'input', 'singlekrr');
            outdir = fullfile(parent_dir, 'output', 'singlekrr');
            ref_output = fullfile(parent_dir, 'ref_output', 'singlekrr');
            if exist(outdir,'dir')
                rmdir(outdir, 's')
            end
            mkdir(outdir)
            
            feature_file = fullfile(input_dir, 'feature_file.mat');
            singleKRR_dir = input_dir;
            sub_fold_file = fullfile(input_dir, 'subfold_file.mat');
            score_ind = [1, 2];
            outstem = '2cog';
           
            % compare the results or replace the reference output
            CBIG_CODE_DIR = getenv('CBIG_CODE_DIR');
            load(fullfile(CBIG_CODE_DIR, 'unit_tests','replace_unittest_flag'));
            if replace_unittest_flag
                % replace PFM results
                copyfile(fullfile(outdir,'PFM_score1_all_folds.mat'), ...
                    fullfile(ref_output,'PFM_score1_all_folds.mat'));
                copyfile(fullfile(outdir,'PFM_score2_all_folds.mat'), ...
                    fullfile(ref_output,'PFM_score2_all_folds.mat'));
            else
                CBIG_compute_singleKRR_PFM(feature_file, singleKRR_dir, sub_fold_file, ...
                    score_ind(1), outstem, outdir)
                CBIG_compute_singleKRR_PFM(feature_file, singleKRR_dir, sub_fold_file, ...
                    score_ind(2), outstem, outdir)
                
                % compare results
                for i = 1:length(score_ind)
                    test = load(fullfile(outdir, ['PFM_score' num2str(i) '_all_folds.mat']));
                    ref = load(fullfile(ref_output, ['PFM_score' num2str(i) '_all_folds.mat']));
                    assert(max(abs((test.PFM_all_folds(:) - ref.PFM_all_folds(:)))) < 1e-6, ...
                        'PFM different');
                end
            end
            
            % remove the output directory
            rmdir(outdir, 's')
        end
        
        function test_CBIG_compute_PFM_LRR_fitrlinear_example(testCase)
            % create output folder
            CBIG_CODE_DIR = getenv('CBIG_CODE_DIR');
            parent_dir = fullfile(CBIG_CODE_DIR, 'unit_tests', 'utilities', 'matlab',... 
                          'predictive_models', 'PFM');
            input_dir = fullfile(parent_dir, 'input', 'lrr');
            outdir = fullfile(parent_dir, 'output', 'lrr');
            ref_output = fullfile(parent_dir, 'ref_output', 'lrr');
            if exist(outdir,'dir')
                rmdir(outdir, 's')
            end
            mkdir(outdir)
            
            feature_file = fullfile(input_dir, 'RSFC.mat');
            LRR_dir = input_dir;
            sub_fold_file = fullfile(input_dir, 'no_relative_5_fold_sub_list.mat');
            score_ind = [1];
            outstem = '1cog';
           
            % compare the results or replace the reference output
            CBIG_CODE_DIR = getenv('CBIG_CODE_DIR');
            load(fullfile(CBIG_CODE_DIR, 'unit_tests','replace_unittest_flag'));
            if replace_unittest_flag
                % replace PFM results
                copyfile(fullfile(outdir,'PFM_score1_all_folds.mat'), ...
                    fullfile(ref_output,'PFM_score1_all_folds.mat'));
            else
                CBIG_compute_LRR_fitrlinear_PFM(feature_file, LRR_dir, sub_fold_file, ...
                    score_ind(1), outstem, outdir)
                
                % compare results
                for i = 1:length(score_ind)
                    test = load(fullfile(outdir, ['PFM_score' num2str(i) '_all_folds.mat']));
                    ref = load(fullfile(ref_output, ['PFM_score' num2str(i) '_all_folds.mat']));
                    assert(max(abs((test.PFM_all_folds(:) - ref.PFM_all_folds(:)))) < 1e-6, ...
                        'PFM different');
                end
            end
            
            % remove the output directory
            rmdir(outdir, 's')
        end
        
        function test_CBIG_compute_PFM_multiKRR_example(testCase)
            % create output folder
            CBIG_CODE_DIR = getenv('CBIG_CODE_DIR');
            parent_dir = fullfile(CBIG_CODE_DIR, 'unit_tests', 'utilities', 'matlab',... 
                          'predictive_models', 'PFM');
            input_dir = fullfile(parent_dir, 'input', 'multikrr');
            outdir = fullfile(parent_dir, 'output', 'multikrr');
            ref_output = fullfile(parent_dir, 'ref_output', 'multikrr');
            if exist(outdir,'dir')
                rmdir(outdir, 's')
            end
            mkdir(outdir)
            
            feature_file = fullfile(input_dir, 'feature_file.mat');
            multiKRR_dir = input_dir;
            sub_fold_file = fullfile(input_dir, 'no_relative_2_fold_sub_list.mat');
            score_ind = [1, 2];
            outstem = 'Cognitive';
           
            % compare the results or replace the reference output
            CBIG_CODE_DIR = getenv('CBIG_CODE_DIR');
            load(fullfile(CBIG_CODE_DIR, 'unit_tests','replace_unittest_flag'));
            if replace_unittest_flag
                % replace PFM results
                copyfile(fullfile(outdir,'PFM_score1_all_folds.mat'), ...
                    fullfile(ref_output,'PFM_score1_all_folds.mat'));
                copyfile(fullfile(outdir,'PFM_score2_all_folds.mat'), ...
                    fullfile(ref_output,'PFM_score2_all_folds.mat'));
            else
                CBIG_compute_multiKRR_PFM(feature_file, multiKRR_dir, sub_fold_file, ...
                    score_ind(1), outstem, outdir)
                CBIG_compute_multiKRR_PFM(feature_file, multiKRR_dir, sub_fold_file, ...
                    score_ind(2), outstem, outdir)
                
                % compare results
                for i = 1:length(score_ind)
                    test = load(fullfile(outdir, ['PFM_score' num2str(i) '_all_folds.mat']));
                    ref = load(fullfile(ref_output, ['PFM_score' num2str(i) '_all_folds.mat']));
                    assert(max(abs((test.PFM_all_folds(:) - ref.PFM_all_folds(:)))) < 1e-6, ...
                        'PFM different');
                end
            end
            
            % remove the output directory
            rmdir(outdir, 's')
        end
    end
end