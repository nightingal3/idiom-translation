#!/bin/bash

# This is a generic running script. It can run in two configurations:
# Single job mode: pass the python arguments to this script
# Batch job mode: pass a file with first the job tag and second the commands per line

#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

set -e # fail fully on first line failure

# Customize this line to point to conda installation
path_to_conda="~/miniconda3"

echo "Running on $(hostname)"

if [ -z "$SLURM_ARRAY_TASK_ID" ]
then
    # Not in Slurm Job Array - running in single mode

    JOB_ID=$SLURM_JOB_ID

    # Just read in what was passed over cmdline
    # modifying this to read in a list of variables instead of just a command...
    #JOB_CMD="${@}"
    corpus_size=$(awk -v TaskID=$SLURM_ARRAY_TASK_ID '$1==TaskID {print $2}' $config)
    idiom_size=$(awk -v TaskID=$SLURM_ARRAY_TASK_ID '$1==TaskID {print $3}' $config)
    transformer_size=$(awk -v TaskID=$SLURM_ARRAY_TASK_ID '$1==TaskID {print $4}' $config)
    seed=$(awk -v TaskID=$SLURM_ARRAY_TASK_ID '$1==TaskID {print $5}' $config)
    use_context=$(awk -v TaskID=$SLURM_ARRAY_TASK_ID '$1==TaskID {print $6}' $config)
    command=$(awk -v TaskID=$SLURM_ARRAY_TASK_ID '$1==TaskID {print $7}' $config)

else
    # In array

    JOB_ID="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

    # Get the line corresponding to the task id
    JOB_CMD=$(head -n ${SLURM_ARRAY_TASK_ID} "$1" | tail -1)
fi

# Find what was passed to --output_folder
regexp="--output_folder\s+(\S+)"
if [[ $JOB_CMD =~ $regexp ]]
then
    JOB_OUTPUT=${BASH_REMATCH[1]}
else
    echo "Did not find a --output_folder argument"
    echo "Setting default output folder slurm_outputs..."
    JOB_OUTPUT="/projects/tir5/users/mengyan3/idiomaticity-learning/slurm_scripts/slurm_outputs"
fi

# Check if results exists, if so remove slurm log and skip
if [ -f  "$JOB_OUTPUT/results.json" ]
then
    echo "Results already done - exiting"
    rm "slurm-${JOB_ID}.out"
    exit 0
fi

# Check if the output folder exists at all. We could remove the folder in that case.
if [ -d  "$JOB_OUTPUT" ]
then
    echo "Folder exists, but was unfinished or is ongoing (no results.json)."
    echo "Starting job as usual"
    # It might be worth removing the folder at this point:
    # echo "Removing current output before continuing"
    # rm -r "$JOB_OUTPUT"
    # Since this is a destructive action it is not on by default
fi

# Use this line if you need to create the environment first on a machine
# ./run_locked.sh ${path_to_conda}/bin/conda-env update -f environment.yml

# Activate the environment
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/mengyan3/miniconda3/lib/
source ~/miniconda3/etc/profile.d/conda.sh
conda activate venv

# Train the model
srun python3 $JOB_CMD

# Move the log file to the job folder
mv "slurm-${JOB_ID}.out" "${JOB_OUTPUT}/"
