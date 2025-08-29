classdef test_CBIG_cov_matrix < matlab.unittest.TestCase
% Written by Leon Ooi and CBIG under MIT license: http://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md
    
    methods (Test)
        function testBasic(testCase)
            % set directories
            CBIG_CODE_DIR = getenv('CBIG_CODE_DIR');
            ref_dir = fullfile(CBIG_CODE_DIR,'unit_tests', 'utilities', ...
                'matlab', 'stats', 'CBIG_cov_matrix', 'ref_output');
            % get replace_unittest_flag
            load(fullfile(CBIG_CODE_DIR, 'unit_tests','replace_unittest_flag'));
            
            % get the current output using artificial data
            x = [[0.41,0.63,0.52,0.11];[0.03,0.29,0.35,0.45];[0.89,0.08,0.83,0.78]; ...
                [0.54,0.66,0.26,0.23];[0.18,0.07,0.30,0.66]];
            y = [[0.55,0.46,0.51];[0.39,0.04,0.89];[0.37,0.28,0.58];[0.09,0.37,0.91];[0.37,0.18,0.48]];
            covmat_test = CBIG_cov_matrix(x,y);
            
            if replace_unittest_flag
                covmat_ref = covmat_test;
                save(fullfile(ref_dir, 'ref_result.mat'),'covmat_ref')
            else
                % load result
                load(fullfile(ref_dir, 'ref_result.mat'))
                
                % compare the current output with expected output
                results_diff = sum(sum(abs(covmat_ref - (covmat_test))));
                assert(results_diff < 1e-6, sprintf('Results differ by: %f',results_diff))
            end
        end
        
    end
    
end
