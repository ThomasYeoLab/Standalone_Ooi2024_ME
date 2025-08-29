classdef test_CBIG_FC_mat2vector < matlab.unittest.TestCase
% Written by Leon Ooi and CBIG under MIT license: https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md
    
    methods (Test)
        function test_vector2mat(TestCase)
            CBIG_CODE_DIR = getenv('CBIG_CODE_DIR');
            ref_dir = fullfile(CBIG_CODE_DIR, 'unit_tests', 'utilities', 'matlab', ...
                'FC', 'CBIG_FC_mat2vector');
            replace_unit_test = load(fullfile(CBIG_CODE_DIR, 'unit_tests', 'replace_unittest_flag'));
            
            % load input
            load(fullfile(ref_dir, 'input', 'example.mat'))
            
            % run function
            test_lt = CBIG_FC_mat2vector(example_mat);
            
            % replace if flag is 1
            if replace_unit_test
                disp("Replacing unit test for CBIG_FC_mat2vector")
                % save new reference results
                example_lt = test_lt;
                save(fullfile(ref_dir, 'example.mat'), 'example_mat', 'example_lt');
            end
            
            % compare with reference result
            assert(isequal(test_lt, example_lt), ...
                sprintf('Output vectors are not equal.'));            
        end
    end

end