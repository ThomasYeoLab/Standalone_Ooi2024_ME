classdef test_CBIG_ReorderParcelIndex < matlab.unittest.TestCase
% Written by XUE Aihuiping and CBIG under MIT license: https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md

    methods (Test)
        
        function ReorderParcelIndex(TestCase)
            CBIG_CODE_DIR = getenv('CBIG_CODE_DIR');
            ref_dir = fullfile(CBIG_CODE_DIR, 'unit_tests', 'utilities', 'matlab', ...
                'FC', 'CBIG_ReorderParcelIndex');
            replace_unit_test = load(fullfile(CBIG_CODE_DIR, 'unit_tests', 'replace_unittest_flag'));
            
            % Input files are Schaefer2018 400 Parcels, Yeo17 networks and Kong17 networks
            input_file_dir = fullfile(CBIG_CODE_DIR, 'stable_projects', 'brain_parcellation', ...
                'Schaefer2018_LocalGlobal', 'Parcellations', 'FreeSurfer5.3', 'fsaverage6', 'label');
            lh_old_annot = fullfile(input_file_dir, 'lh.Schaefer2018_400Parcels_17Networks_order.annot');
            rh_old_annot = fullfile(input_file_dir, 'rh.Schaefer2018_400Parcels_17Networks_order.annot');
            lh_new_annot = fullfile(input_file_dir, 'lh.Schaefer2018_400Parcels_Kong2022_17Networks_order.annot');
            rh_new_annot = fullfile(input_file_dir, 'rh.Schaefer2018_400Parcels_Kong2022_17Networks_order.annot');
            
            % Run CBIG_ReorderParcelIndex
            [index, parcel_names] = CBIG_ReorderParcelIndex(lh_old_annot, rh_old_annot, lh_new_annot, rh_new_annot);
            
            % Load reference results
            ref_output_dir = fullfile(ref_dir, 'ref_output');
            ref_output_file = fullfile(ref_output_dir, 'Schaefer400_yeo17_to_kong17.mat');
            ref_result = load(ref_output_file);
            
            % Replace unit test results when flag is 1
            if replace_unit_test
                disp('Replacing unit test for test_CBIG_ReorderParcelIndex...')
                save(ref_output_file, 'index', 'parcel_names');
                ref_result = load(ref_output_file);
            end
            
            % Compare with reference results
            assert(isequal(index, ref_result.index), 'Output index matrix is different.');
            assert(isequal(parcel_names, ref_result.parcel_names), 'Output parcel name table is different.');
        end
    end
end
