classdef test_CBIG_FC_vector2mat < matlab.unittest.TestCase
% Written by Leon Ooi and CBIG under MIT license: https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md
    
    methods (Test)
        function test_vector2mat(TestCase)
            CBIG_CODE_DIR = getenv('CBIG_CODE_DIR');
            ref_dir = fullfile(CBIG_CODE_DIR, 'unit_tests', 'utilities', 'matlab', ...
                'FC', 'CBIG_FC_vector2mat');
            replace_unit_test = load(fullfile(CBIG_CODE_DIR, 'unit_tests', 'replace_unittest_flag'));
            
            % load input
            load(fullfile(ref_dir, 'input', 'example.mat'))
            
            % run function
            test_mat = CBIG_FC_vector2mat(example_lt);
            
            % replace if flag is 1
            if replace_unit_test
                disp("Replacing unit test for CBIG_FC_vector2mat")
                % save new reference results
                example_mat = test_mat;
                save(fullfile(ref_dir, 'example.mat'), 'example_mat', 'example_lt');
            end
            
            % compare with reference result
            assert(isequal(size(example_mat),size(test_mat)),...
                'Output correlation matrix is of wrong size.');
            assert(isequal(example_mat,test_mat), ...
                sprintf('Output correlation matrix are not equal.'));            
        end
    end

end