#!/bin/sh
# Written by Chen Zhang and CBIG under MIT license: https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md

#!/bin/bash

# --- Configuration ---
default_sleep_interval="30s" # Default interval if none provided

# Check if an argument (job pattern) was provided
if [ -z "$1" ]; then
    echo "Usage: $0 <job_pattern>"
    exit 1
fi

job_pattern="$1"
me=$(whoami)

# Set sleep interval: use default or override with $2 if provided
sleep_interval="$default_sleep_interval"
if [ -n "$2" ]; then
    sleep_interval="$2"
    echo "Using provided sleep interval: $sleep_interval"
else
    echo "Using default sleep interval: $sleep_interval"
fi

# Function to get the count of matching jobs
get_job_count() {
    # Use qstat, filter by user, grep for the pattern, count lines
    # Added -w to grep for whole word matching, might be useful depending on pattern
    # Use || true to prevent script exit if grep finds nothing (exit code 1)
    qstat -u "$me" | grep -- "$job_pattern" | wc -l || true
}

# Initial check
status=$(get_job_count)
echo "Initial job count for pattern \"$job_pattern\": $status"

# Loop while the job count is not zero
while [ "$status" -ne 0 ]; do
    # # Print status on two lines
    # echo "Waiting for $status job(s) matching \"$job_pattern\" to finish..."
    # echo "(Checking again in $sleep_interval)"
    # Wait
    sleep "$sleep_interval"
    # Check status again
    status=$(get_job_count)
done

echo "All jobs matching \"$job_pattern\" have finished."

exit 0