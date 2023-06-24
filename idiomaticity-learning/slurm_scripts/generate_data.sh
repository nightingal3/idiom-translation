#!/bin/bash
#SBATCH -n 1
#SBATCH --time=0
#SBATCH --mem=50G
#SBATCH --array=1-16
#SBATCH --exclude=tir-0-11,tir-0-36,tir-0-3,tir-0-32,tir-1-28,tir-0-17,tir-1-32,tir-1-11,tir-1-36
#SBATCH --mail-user=emmy@cmu.edu
#SBATCH --mail-type=END
#SBATCH --job-name=generate_data_10M_1M_context

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/mengyan3/miniconda3/lib/
source ~/miniconda3/etc/profile.d/conda.sh
conda activate venv

python3 generate_synchronous_data.py -n 10M -i 1M --add_context_beta