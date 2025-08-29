classdef test_CBIG_LRR_frac_workflow < matlab.unittest.TestCase
% Written by Ruby Kong and CBIG under MIT license: https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md
    
    
    methods (Test)
        function test_LRR_frac_with_2_measures( TestCase )
        % Test LRR_fracridge prediction workflow with 2 measures
            parent_dir = fullfile(getenv('CBIG_CODE_DIR'), 'unit_tests', 'utilities', 'matlab', ...
            'predictive_models', 'LRR_fracridge');
            input_dir = fullfile(parent_dir, 'input', 'test_basic');
            load(fullfile(input_dir, 'no_relative_5_fold_sub_list.mat'))
            load(fullfile(input_dir, 'y.mat'))
            load(fullfile(input_dir, 'covariates.mat'))
            load(fullfile(input_dir, 'RSFC.mat'))
            
            
            params.sub_fold = sub_fold;
            params.feature_mat = corr_mat;
            params.y = y;
            params.covariates = covariates;
            params.num_innerfolds = length(params.sub_fold);
            params.outdir = fullfile(parent_dir, 'output', 'test_basic');
            params.outstem = '2cog';
            params.lambda = [0.05:0.05:0.5];
            
            if(exist(params.outdir, 'dir'))
                rmdir(params.outdir, 's')
            end
            mkdir(params.outdir)
            
            
            % get replace_unittest_flag
            CBIG_CODE_DIR = getenv('CBIG_CODE_DIR');
            load(fullfile(CBIG_CODE_DIR, 'unit_tests','replace_unittest_flag'));
            
            if replace_unittest_flag
                % replace reference result   
                ref_outdir = fullfile(parent_dir, 'ref_output', 'test_basic');
                mkdir(fullfile(ref_outdir, 'results', 'optimal_acc'))
                CBIG_LRR_frac_workflow( params );
                source = fullfile(params.outdir, 'results', 'optimal_acc', '2cog.mat');
                destination = fullfile(ref_outdir, 'results', 'optimal_acc', '2cog.mat');
                copyfile(source,destination)                
            else
            
                CBIG_LRR_frac_workflow( params );

                ref_dir = fullfile(parent_dir, 'ref_output', 'test_basic');
                ref = load(fullfile(ref_dir, 'results', 'optimal_acc', '2cog.mat'));
                test = load(fullfile(params.outdir, 'results', 'optimal_acc', '2cog.mat'));
                fields = fieldnames(ref);

                for i = 1:length(fields)
                    if isequal(fields{i}, 'y_predict')
                        for j = 1:length(ref.y_predict)
                            curr_ref = ref.y_predict{j};
                            curr_test = test.y_predict{j};

                            assert(isequal(size(curr_test),size(curr_ref)), ...
                                sprintf('%d -th array of field y_predict is of wrong size.', j));
                            assert(max(abs((curr_test(:) - curr_ref(:)))) < 1e-10, ...
                                sprintf('%d -th array of field y_predict is different from reference result.', j));
                        end
                    elseif isequal(fields{i}, 'optimal_statistics')
                        for j = 1:length(ref.optimal_statistics)
                            stats_ref = ref.optimal_statistics{j};
                            stats_test = test.optimal_statistics{j};
                            stats_names = fieldnames(stats_ref);
                            for k = 1:length(stats_names)
                                assert(max(abs(stats_ref.(stats_names{k}) - stats_test.(stats_names{k}))) < 1e-6, ...
                                    'optimal stasts are different from reference result.');
                            end
                        end
                    else
                        curr_ref = getfield(ref, fields{i});
                        curr_test = getfield(test, fields{i});

                        assert(isequal(size(curr_test),size(curr_ref)), ...
                            sprintf('field %s is of wrong size.', fields{i}));
                        assert(max(abs((curr_test(:) - curr_ref(:)))) < 1e-10, ...
                            sprintf('field %s is different from reference result.', fields{i}));
                    end
                end
            end
            rmdir(fullfile(parent_dir, 'output'), 's')
        end
        
    end
    
    
end