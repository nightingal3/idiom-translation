#!/bin/bash
# inspired by this script: https://github.com/y0ast/slurm-for-ml/blob/master/run_file.sh

NUM_JOBS_PARALLEL=4

if [ ! -f "$1" ]
then
    echo "Error: must pass existing file"
    exit 1
fi

n_lines=$(grep -c '^' "$1")

job_name=$(basename "$1" .txt)

OTHER_ARGS="--mem-per-cpu=50G \
--time=0 \
--exclude=tir-0-11,tir-0-3,tir-0-32,tir-1-28,tir-0-17,tir-1-32,tir-1-11,tir-1-36 \
--mail-user=emmy@cmu.edu \
--mail-type=END \
--gres=gpu:1"

sbatch --array=1-${n_lines}%${NUM_JOBS_PARALLEL} --mem=50G --time=0 --gres=gpu:1 --exclude=tir-1-11,tir-0-3,tir-0-32,tir-1-28,tir-1-32,tir-1-36 --job-name=small43_train_transformer $(dirname "$0")/generic.sh "$1"
#sbatch --array=1-${n_lines}%${NUM_JOBS_PARALLEL} "${OTHER_ARGS}" --job-name train_transformer $(dirname "$0")/generic.sh "$1"