#!/bin/bash
# Please make sure your VNC resolution is 1920x1080 for figure utilities unit tests to pass
#
# Written by Nanbo Sun,Yang Qing and CBIG under MIT license: https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md

##########################
# Set parameters and paths
##########################

# output folder
out_dir=$1
mkdir -p $out_dir

# unit test version, 'light' or 'intermediate' or 'comprehensive'
if [ -z "$2" ]; then
    version="intermediate"
else
    version="$2"
fi

dt=`date +%Y%m%d`
# log file
LF=${out_dir}/${dt}_CBIG_UnitTests_Version_${version}.log

# verbose log file
LF_verbose=${out_dir}/${dt}_CBIG_UnitTests_Version_${version}_verbose.log

if [ -z "${out_dir}" ]; then
    echo "Error: please specify the output directory."
    exit 1
fi
> $LF_verbose
if [ ! -w $LF_verbose ]; then
    echo "Error: $LF_verbose is not writeable. Please change the output directory."
    exit 1
fi
> $LF
if [ ! -w $LF ]; then
    echo "Error: $LF is not writeable. Please change the output directory."
    exit 1
fi

###########################################
# Print start info to log file and terminal
###########################################

echo "**************************************************************************" | tee -a $LF
echo | tee -a $LF
echo "Running CBIG unit tests!" | tee -a $LF
echo "Unit Test Version: ${version}" | tee -a $LF
echo -n "Start Date: " | tee -a $LF
date | tee -a $LF

#####################
# Run CBIG unit tests
#####################

> ${out_dir}/error_testcase.log

# run light unit test
if [ "$version" = "light" ]; then
    matlab -nodesktop -nosplash -r "runtests('$CBIG_CODE_DIR/unit_tests', 'Recursively', true); exit"\
      > $LF_verbose
    # extract error message from the verbose log file
    line_num=`grep -n "Failure Summary" $LF_verbose | cut -d ":" -f 1`
    if [ -n "${line_num}" ]; then
        line_num_error=$(($line_num+4))
        total_lines=`wc -l $LF_verbose | cut -d " " -f 1`
        sed -n "$line_num_error,$total_lines p" < $LF_verbose > ${out_dir}/error_testcase.log
    fi

# run comprehensive unit test
elif [ "$version" = "comprehensive" ]; then
    matlab -nodesktop -nosplash -r "runtests('$CBIG_CODE_DIR/unit_tests', 'Recursively', true); exit" > $LF_verbose
    # extract error message from the verbose log file
    line_num=`grep -n "Failure Summary" $LF_verbose | cut -d ":" -f 1`
    if [ -n "${line_num}" ]; then
        line_num_error=$(($line_num+4))
        total_lines=`wc -l $LF_verbose | cut -d " " -f 1`
        sed -n "$line_num_error,$total_lines p" < $LF_verbose > ${out_dir}/error_testcase.log
    fi
    matlab -nodesktop -nosplash -r "runtests('$CBIG_CODE_DIR/stable_projects', 'Recursively', true); exit" \
      > ${out_dir}/tmp.log
    cat ${out_dir}/tmp.log >> $LF_verbose
    # extract error message from the verbose log file
    line_num=`grep -n "Failure Summary" ${out_dir}/tmp.log | cut -d ":" -f 1`
    if [ -n "${line_num}" ]; then
        line_num_error=$(($line_num+4))
        total_lines=`wc -l ${out_dir}/tmp.log | cut -d " " -f 1`
        sed -n "$line_num_error,$total_lines p" < ${out_dir}/tmp.log >> ${out_dir}/error_testcase.log
    fi
    # Run python unit tests
    sh $CBIG_CODE_DIR/unit_tests/CBIG_Python_Projects_UnitTests.sh $CBIG_CODE_DIR/stable_projects ${out_dir}/python_tmp.log
    cat ${out_dir}/python_tmp.log >> $LF_verbose

# run intermediate unit test
elif [ "$version" = "intermediate" ]; then
    matlab -nodesktop -nosplash -r "runtests('$CBIG_CODE_DIR/unit_tests', 'Recursively', true); exit" > $LF_verbose
    # extract error message from the verbose log file
    line_num=`grep -n "Failure Summary" $LF_verbose | cut -d ":" -f 1`
    if [ -n "${line_num}" ]; then
        line_num_error=$(($line_num+4))
        total_lines=`wc -l $LF_verbose | cut -d " " -f 1`
        sed -n "$line_num_error,$total_lines p" < $LF_verbose > ${out_dir}/error_testcase.log
    fi
    # run the selected stable project unit tests
    SP_tests=`cat $CBIG_CODE_DIR/unit_tests/CBIG_intermediate_unit_test_list`
    for SP_test in ${SP_tests}
    do
        matlab -nodesktop -nosplash -r "addpath(genpath('$CBIG_CODE_DIR/stable_projects')); runtests('$SP_test'); exit"\
          > ${out_dir}/tmp.log
        cat ${out_dir}/tmp.log >> $LF_verbose
        # extract error message from the verbose log file
        line_num=`grep -n "Failure Summary" ${out_dir}/tmp.log | cut -d ":" -f 1`
        if [ -n "${line_num}" ]; then
            line_num_error=$(($line_num+4))
            total_lines=`wc -l ${out_dir}/tmp.log | cut -d " " -f 1`
            sed -n "$line_num_error,$total_lines p" < ${out_dir}/tmp.log >> ${out_dir}/error_testcase.log
        fi
    done

# error message for unrecognized unit test set
else
    echo "Error: version should be [light], [intermediate] or [comprehensive]! Please specify."
    exit 1
fi

#########################################
# Print env info to log file and terminal
#########################################

echo -n "End Date: " | tee -a $LF
date | tee -a $LF
echo | tee -a $LF
echo "**************************************************************************" | tee -a $LF
echo | tee -a $LF

# print OS related info
echo "Test Environment Settings" | tee -a $LF
echo "OS:" | tee -a $LF
lsb_release -a | tee -a $LF

# print default software packages used
echo | tee -a $LF
echo -e "MATLAB:\t $CBIG_MATLAB_DIR" | tee -a $LF
echo -e "FREESURFER:\t $FREESURFER_HOME" | tee -a $LF
echo -e "FSL:\t $CBIG_FSLDIR" | tee -a $LF
echo -e "ANTS:\t $CBIG_ANTS_DIR" | tee -a $LF
echo -e "AFNI:\t $CBIG_AFNI_DIR" | tee -a $LF

# print last commit
echo | tee -a $LF
echo -n "Last commit: " | tee -a $LF
cd $CBIG_CODE_DIR
git log -1 --oneline | tee -a $LF
echo | tee -a $LF
echo "**************************************************************************" | tee -a $LF
echo | tee -a $LF

#############################################
# Print test summary to log file and terminal
#############################################

# calculate number of included functions
ut_ex_num=`cat $LF_verbose | grep '^Done' | sed 's/Done\ //g' | grep '^test_' | wc -l`
# calculate number of included stable projects
sp_num=`cat $LF_verbose | grep '^Done' | sed 's/Done\ //g' | grep -v '^test_' | wc -l`
# calculate number of python stable projects
py_num=`cat $LF_verbose | grep '^Python unit test done' | wc -l`

# calculate number of failed functions
ut_ex_err_num=`cat ${out_dir}/error_testcase.log | grep -e 'test' | sed 's/^\     //g' | grep '^test_' \
  | sort | cut -d "/" -f 1 | uniq | wc -l`
# calculate number of failed stable projects
sp_err_num=`cat ${out_dir}/error_testcase.log | grep -e 'test' | sed 's/^\     //g' | grep -v '^test_' \
  | sort | cut -d "/" -f 1 | uniq | wc -l`
# calculate number of failed python stable projects
py_err_num=`cat $LF_verbose | grep '^Some python tests failed' | wc -l`

# print summary report to log file and terminal
echo "Test Summary:" | tee -a $LF
echo -e "[$sp_num] matlab stable projects tested\t\t\t[$sp_err_num] FAILED" | tee -a $LF
echo -e "[$py_num] python stable projects tested\t\t\t[$py_err_num] FAILED" | tee -a $LF
echo -e "[$ut_ex_num] utilities&external_packages functions tested\t[$ut_ex_err_num] FAILED" \
  | tee -a $LF
echo | tee -a $LF

##########################################
# Print cleaned error messages to log file
##########################################

# unit test passed
if [ "${ut_ex_err_num}" -eq 0 ] && [ "${sp_err_num}" -eq 0 ] && [ "${py_err_num}" -eq 0 ]; then
    echo "Unit Tests [PASSED]!" | tee -a $LF
    echo "You can check $LF for more details."

# unit test failed
else
    echo "Unit Tests [FAILED]!" | tee -a $LF
    echo | tee -a $LF
    echo "**************************************************************************" | tee -a $LF
    echo "Please check $LF for error details."
    echo | tee -a $LF
    echo "Error details are as follows:" >> $LF
    echo | tee -a $LF
    if [ "${ut_ex_err_num}" -gt 0 ] || [ "${sp_err_num}" -gt 0 ]; then
        bounds=`grep -n '^=\+\{80\}' $LF_verbose | cut -d':' -f1`
        echo $bounds | xargs -n2 sh -c "sed -n \"\$0,\$1 p\" $LF_verbose >> $LF; echo -e \"\n\" >> $LF"
    fi
    if [ "${py_err_num}" -gt 0 ]; then
        awk '/FAILED \(/ {exit} /^=+/{print; p=0} /ERROR|FAIL/{p=1} p' ${out_dir}/python_tmp.log >> $LF
    fi
fi

###########################################
# list functions included in this unit test
###########################################

echo | tee -a $LF
echo "# **************************************************************************" >> $LF
echo "# " >> $LF
echo "# The matlab stable projects included are listed here:" >> $LF
echo "# " >> $LF
cat $LF_verbose | grep '^Done' | sed 's/Done\ //g' | grep -v '^test_' | sed 's/_unit_test//g' | sort | \
  awk '{print "# "$0}' >> $LF
echo "# The python stable projects included are listed here:" >> $LF
echo "# " >> $LF
cat "$LF_verbose" | grep "Python unit test done for" | sed 's/.*Python unit test done for /# /' >> "$LF"
echo "# " >> $LF
echo "# The utilities or exernal_packages functions included are listed here:" >> $LF
echo "# " >> $LF
cat $LF_verbose | grep '^Done' | sed 's/Done\ //g' | grep '^test_' | sed 's/test_//g' | sort | \
  awk '{print "# "$0}' >> $LF

###########################
# Remove intermediate files
###########################

rm -f ${out_dir}/error_testcase.log
rm -f ${out_dir}/tmp.log



