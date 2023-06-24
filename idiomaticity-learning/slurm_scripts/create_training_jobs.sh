#!/bin/bash
set -e 
output_folder="./slurm_job_scripts/"
filename="train_models_w_context.tsv"
mkdir -p $output_folder

sep="\t"
transformer_sizes=(s m l)
contexts=(true)
# make header
echo -e "TaskID${sep}CorpusSize${sep}IdiomSize${sep}TransformerSize${sep}Seed${sep}UseContext${sep}Command" > ${output_folder}/${filename}

base_script='python train_transformer.py'
taskid=1
for seed in 42 43 44 45 46
do 
    for transformer_size in ${transformer_sizes[@]}
    do
        for context in ${contexts[@]}
        do
            if [ $context = true ]; then 
                sbatch_script="$base_script --use_context_beta"
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
                        echo -e "${taskid}${sep}${data_size}${sep}${idiom_size}${sep}${transformer_size}${sep}${seed}${sep}${context}${sep}${sbatch_script} -c $data_size -i $idiom_size -t $transformer_size --seed $seed" >> ${output_folder}/${filename}
                        taskid=$((taskid+1))
                    fi
                done
            done
        done
    done
done