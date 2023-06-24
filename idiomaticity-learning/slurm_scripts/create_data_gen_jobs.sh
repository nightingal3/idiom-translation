#!/bin/bash
set -e 
output_folder="./slurm_job_scripts/"
mkdir -p $output_folder

base_script='generate_synchronous_data.py'

for context in true false
do
    if [ $context = true ]; then 
        sbatch_script="$base_script --add_context_beta"
    else
        sbatch_script=$base_script
    fi
    for data_size in 100k 1M 10M
    do
        for idiom_size in 10 100 1k 10k 100k 1M
        do
            if [[ "$data_size" = "$idiom_size" ]]; then
                break
            else
                echo "${sbatch_script} -n $data_size -i $idiom_size" >> ${output_folder}/create_data.txt
            fi
        done
    done
done
