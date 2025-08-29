#!/bin/bash
#
# This script will loop through folders and find python unit test files (starting with "test_") and run unit test for
# these projects.
#
# Written by XUE Aihuiping and CBIG under MIT license: https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md

if [ -z "$1" ]; then
    test_dir="$CBIG_CODE_DIR/stable_projects"
else
    test_dir="$1"
fi

if [ ! -z "$2" ]; then
    log_file="$2"
    # Remove log file if exist
    if [ -e "${log_file}" ]; then
        rm ${log_file}
    fi
    # Redirect output to log file
    exec > "$log_file" 2>&1
fi

# Get .yml file number under given folder
get_yml_num() {
    local find_dir="$1"
    env_file=(`find -L ${find_dir} -maxdepth 1 -type f -name "*.yml"`)
    num_files=${#env_file[@]}
    return $num_files
}

# Find all test directories
test_dirs=$(find ${test_dir} -type f -name "test_*.py" | xargs -I{} dirname {} | sort -u)

# Loop to run unit tests
for test_dir in $test_dirs; do
    # Skip if the directory is not under unit_tests folder
    if [[ "$test_dir" != *"unit_tests"* ]]; then
        continue
    fi

    # Extract project dir and name
    proj_dir="${test_dir%/unit_tests*}"
    proj_name="${proj_dir##*/}"
    export PYTHONPATH="$proj_dir"
    #cd $proj_dir

    echo "Run unit test for ${proj_name}..."

    # If .yml file exists in the test folder, use the .yml file in the test folder
    # Should not include more than one .yml file in test folder.
    get_yml_num ${test_dir}
    yml_num=$?
    if [ ${yml_num} -gt 1 ]; then
        echo "[ERROR] Found multiple .yml files under ${test_dir}. "
        continue
    elif [ ${yml_num} -eq 0 ]; then
        get_yml_num ${proj_dir} # Search the project root dir
        if [ ! $? -eq 1 ]; then
            get_yml_num "${proj_dir}/replication/config" # Search the replication config
            if [ $? -eq 1 ]; then
                env_file=`find ${proj_dir}/replication/config -maxdepth 1 -type f -name "*.yml"`
            else
                echo "[ERROR] Cannot find proper .yml file."
                continue
            fi
        else
            env_file=`find ${proj_dir} -maxdepth 1 -type f -name "*.yml"`
        fi
    else 
        env_file=`find ${test_dir} -maxdepth 1 -type f -name "*.yml"`
    fi

    # Read the env name in the yml file
    while IFS= read -r line; do
        if [[ $line == name:* ]]; then
            env_name="${line#name: }"
            env_name=$(echo "$env_name" | xargs)
        fi
    done < "$env_file"

    # Check current conda env list
    env_list=$(conda env list)
    env_list=$(echo "$env_list" | awk '{print $1}' | tail -n +4)

    found=0
    for env in $env_list; do
        if [ "$env" = "$env_name" ]; then
            found=1
            break
        fi
    done

    # Install required env if not found
    if [ $found -eq 0 ]; then
        echo "$env_name does not exist. Installing $env_name..."
        conda env create -f $env_file
    fi

    echo "Current env: $env_name"
    source CBIG_init_conda
    conda activate $env_name
    result=`conda compare $env_file`
    if [[ "$result" != *Success* ]]; then
        echo "$env_name is different from $env_file."
        conda env update --file $env_file
    fi

    # Check whether pytest is included
    packages=$(conda list)
    if echo "$packages" | grep -q "^pytest "; then
        # Run pytest if the environment includes pytest
        echo "Running pytest..."
        pytest ${test_dir}
    else
        # Run unittest if the environment does not includes pytest
        echo "Running unittest..."
        python -m unittest discover -s ${test_dir}
    fi
    if grep -q "FAILED" "$log_file"; then
        echo "Some python tests failed for ${proj_name}."
    fi
    echo "Python unit test done for ${proj_name}"
    conda deactivate
done


