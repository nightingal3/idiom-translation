#!/bin/bash
#SBATCH -n 1
#SBATCH --time=0
#SBATCH --gres=gpu:3090:1
#SBATCH --mem=40G
#SBATCH --exclude=tir-0-36,tir-0-3,tir-0-32,tir-1-28,tir-1-32,tir-1-36,tir-0-15
#SBATCH --mail-user=emmy@cmu.edu
#SBATCH --mail-type=END
#SBATCH --output=idiom_learning_ctx_225.out
#SBATCH --job-name=idiom_learning_225

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/mengyan3/miniconda3/lib/
source ~/miniconda3/etc/profile.d/conda.sh
conda activate venv

if ! [ -d /scratch/mengyan3/data ]
then
    echo "Copying data to scratch"
    mkdir -p /scratch/mengyan3/data
    cp -r /compute/tir-0-15/mengyan3/data /scratch/mengyan3/
fi

python train_transformer.py -c 10M -i 1M -t l --seed 46 --use_context_beta