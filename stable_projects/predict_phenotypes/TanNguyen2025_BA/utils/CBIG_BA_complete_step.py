'''
Written by Trevor Tan and CBIG under MIT license:
https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md

This script contains function(s) relevant to proceeding to
the next step in analysis pipeline.

Expected output(s):
1. f'{out_dir}/complete_{append}.txt'

Example (in script intending to generate flag to signify
         completion of analysis step):
    from utils.CBIG_BA_complete_step import generate_complete_flag
    generate_complete_flag(out_dir, append='train')
'''

import os

def generate_complete_flag(out_dir, append=None):
    '''
    This function should be used at the end of every analysis step.
    It will generate an empty .txt file to represent the successful
    completion of said step.
    Another .sh function `check_completed_step` in
    ${TANNGUYEN2025_BA_DIR}/replication/scripts/CBIG_BA_3a_ad_classification.sh,
    or ${TANNGUYEN2025_BA_DIR}/replication/scripts/CBIG_BA_3b_mci_progression.sh,
    will check if these .txt files are generated before moving onto the
    next step of the analysis pipeline.
    '''

    flag = os.path.join(out_dir, 'complete_{}'.format(append))
    with open(flag, 'w') as file:
        pass
