classdef test_CBIG_ICC_1to1 < matlab.unittest.TestCase
% Written by Leon Ooi and CBIG under MIT license: http://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md
    
    methods (Test)
        
        function testBasic(testCase)
            % set directories
            CBIG_CODE_DIR = getenv('CBIG_CODE_DIR');
            ref_dir = fullfile(CBIG_CODE_DIR,'unit_tests', 'utilities', ...
                'matlab', 'stats', 'CBIG_ICC_1to1', 'ref_output');
            % get replace_unittest_flag
            load(fullfile(CBIG_CODE_DIR, 'unit_tests','replace_unittest_flag'));
            
            % get the current output using artificial data
            v1 = [0.45,0.34,0.45,0.89,0.09,0.61];
            v2 = [0.18,0.89,0.61,0.72,0.34,0.07];
            ICC_test = CBIG_ICC_1to1([v1',v2']);
            
            if replace_unittest_flag
                ICC_ref = ICC_test;
                save(fullfile(ref_dir, 'ref_result.mat'),'ICC_ref')
            else
                % load result
                load(fullfile(ref_dir, 'ref_result.mat'))
                
                % compare the current output with expected output
                results_diff = abs(ICC_ref - ICC_test);
                assert(results_diff < 1e-6, sprintf('Results differ by: %f',results_diff))
            end
        end
        
    end
    
end
