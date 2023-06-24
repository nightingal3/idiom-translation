#!/bin/bash
#SBATCH --array=225-226%1
#SBATCH --time=20-00:00:00
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=20G
#SBATCH --exclude=tir-0-32,tir-0-36,tir-1-32,tir-1-36,tir-1-28
#SBATCH --mail-user=emmy@cmu.edu
#SBATCH --mail-type=END
#SBATCH --job-name=ctx_idiom_learning

config=./slurm_job_scripts/train_models_w_context.tsv

corpus_size=$(awk -v TaskID=$SLURM_ARRAY_TASK_ID '$1==TaskID {print $2}' $config)
idiom_size=$(awk -v TaskID=$SLURM_ARRAY_TASK_ID '$1==TaskID {print $3}' $config)
transformer_size=$(awk -v TaskID=$SLURM_ARRAY_TASK_ID '$1==TaskID {print $4}' $config)
seed=$(awk -v TaskID=$SLURM_ARRAY_TASK_ID '$1==TaskID {print $5}' $config)
use_context=$(awk -v TaskID=$SLURM_ARRAY_TASK_ID '$1==TaskID {print $6}' $config)
command=$(awk -v TaskID=$SLURM_ARRAY_TASK_ID '$1==TaskID {print $7}' $config)


echo "Array task ID: $SLURM_ARRAY_TASK_ID, running on $(hostname)
arguments: Corpus size: $corpus_size 
num idioms: $idiom_size 
transformer size: $transformer_size 
seed: $seed 
use context? $use_context"

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/mengyan3/miniconda3/lib/
source ~/miniconda3/etc/profile.d/conda.sh
conda activate venv


# move files to the scratch of compute node we're on

if ! [ -d /scratch/mengyan3/data ]
then
    echo "Copying data to scratch"
    mkdir -p /scratch/mengyan3/data
    cp -r /compute/tir-0-15/mengyan3/data /scratch/mengyan3/
fi
# actually running the command through the text variable is weird
# let's just run direclty with args

exp_command=(python3 train_transformer.py -c $corpus_size -i $idiom_size -t $transformer_size --seed $seed)
if [ $use_context = "true" ]
then
    exp_command+=(--use_context_beta)
fi
echo "Running command: ${exp_command[@]}"

"${exp_command[@]}"
