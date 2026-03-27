#!/bin/zsh
# get slurm job id and ideally "rank" as task id to print
sleep 10 && echo "Hello World"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_PROCID: $SLURM_PROCID"
echo "SLURM_NTASKS: $SLURM_NTASKS"
echo "SLURM_NNODES: $SLURM_NNODES"
echo "SLURM_NODEID: $SLURM_NODEID"
# print all node hostnames allocated to this job
scontrol show hostnames $SLURM_JOB_NODELIST
# also print current node hostname
echo "Hostname: $(hostname)"
python -c "import ssl; print(ssl.OPENSSL_VERSION)"
#OpenSSL 1.1.1u  30 May 2023

# if procid is 0
if [ "$SLURM_PROCID" -eq 0 ]; then
    echo "This is the master process."
    hostnames=$(scontrol show hostnames $SLURM_JOB_NODELIST)
    # Find common prefix
    common_prefix=$(echo $hostnames | tr ' ' '\n' | awk 'NR==1 {prefix=$0; next} {while (index($0, prefix) != 1) prefix=substr(prefix, 1, length(prefix)-1)} END {print prefix}')
    # Print prefix + "[" + list of unique suffixes + "]"
    suffixes=()
    for hn in $hostnames; do
        suffix=${hn#$common_prefix}
        suffixes+=($suffix)
    done
    unique_suffixes=($(echo "${suffixes[@]}" | tr ' ' '\n' | sort -u))
    # Calculate wait time and run time
    start_time=$SLURM_JOB_START_TIME
    current_time=$(date +%s)
    run_time=$((current_time - start_time))
    # Get submit time from sacct
    submit_time_str=$(sacct -j $SLURM_JOB_ID --format=Submit --noheader --parsable2 | head -1)
    if [ -n "$submit_time_str" ] && [ "$submit_time_str" != "None" ]; then
        submit_time=$(date -d "$submit_time_str" +%s)
        if [ "$submit_time" -gt 0 ] 2>/dev/null; then
            wait_time=$((start_time - submit_time))
        fi
    fi
    # print current time as day.month.year_hour:minute:second
    echo "[$(date +"%d.%m.%Y %H:%M:%S" -d $submit_time_str)][$(printf "%3s" $wait_time)s][$(printf "%3s" $run_time)s]:\t${SLURM_JOB_NAME}\t${SLURM_JOB_ID}\t${common_prefix}[$(IFS=,; echo "${unique_suffixes[*]}")]\t\t" >> hello_world_timings.log
fi