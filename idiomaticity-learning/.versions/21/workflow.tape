### /projects/tir5/users/mengyan3/idiomaticity-learning/tapes/shell-submitter.tape ###
submitter sbatch :: job_name mem time gpus 
                :: COMMANDS
                :: TASK REALIZATION CONFIGURATION {
    action run {
        wrapper="ducttape_job.sh"
        echo "#!/bin/bash" >> $wrapper
        echo "#SBATCH -n 1" >> $wrapper
        echo "#SBATCH --time=$time" >> $wrapper
        echo "#SBATCH --gres=gpu:$gpus" >> $wrapper
        echo "#SBATCH --mem=$mem" >> $wrapper
        echo "#SBATCH --exclude=tir-0-11,tir-0-36,tir-0-3,tir-0-32,tir-1-28,tir-0-17,tir-1-32,tir-1-36" >> $wrapper
        echo "#SBATCH --mail-user=emmy@cmu.edu" >> $wrapper
        echo "#SBATCH --mail-type=END" >> $wrapper
        echo "#SBATCH --job-name=$job_name" >> $wrapper
        echo "set -e # stop on errors" >> $wrapper
        echo "set -i pipefail # stop on pipeline errors" >> $wrapper
        echo "set -u # stop on undeclared variables" >> $wrapper
        echo "set -x # show each command as it is executed" >> $wrapper
        # The current working directory will also be changed by most schedulers
        echo "cd $PWD" >> $wrapper
        echo "$COMMANDS" >> $wrapper
    }
}

submitter slurm :: COMMANDS 
                :: TASK REALIZATION CONFIGURATION {
    action run > exit_code {
        # Returns true iff the first parameter 
        # is the name of a defined variable
        function var_defined {
            eval '[[ ${!'$1'[@]} ]]'	     
        }

        # define script 
        wrapper="ducttape_job.sh"

        echo "#!/bin/bash" > $wrapper
        echo "set -e # stop on errors" >> $wrapper
        echo "set -o pipefail # stop on pipeline errors" >> $wrapper
        echo "set -u # stop on undeclared variables" >> $wrapper

        # print actual jobs 
        echo "$COMMANDS" >> $wrapper

        SLURM_ARGS="--job-name=$TASK"
        if (var_defined cpus); then
            SLURM_ARGS+=" -c $cpus"
        fi
        if (var_defined gres); then
            SLURM_ARGS+=" --gres=$gres"
        fi
        if (var_defined mem); then
            SLURM_ARGS+=" --mem=$mem"
        fi
        if (var_defined time); then
            SLURM_ARGS+=" --time=$time"
        fi
        if (var_defined nodelist); then
            SLURM_ARGS+=" --nodelist=$nodelist"
        fi
        if (var_defined exclude); then
            SLURM_ARGS+=" --exclude=$exclude"
        fi

        echo $SLURM_ARGS

        srun $SLURM_ARGS bash $wrapper
  }
}
global {
    ducttape_experimental_submitters=enable
}

task hello_slurm :: .submitter=slurm .mem="10G" .time="1:00:00" .gpus="1" .exclude="tir-0-11,tir-0-36,tir-0-3,tir-0-32,tir-1-28,tir-0-17,tir-1-32,tir-1-36" 
{
exec ~/miniconda3/etc/profile.d/conda.sh
conda activate venv
python3 generate_synchronous_data.py -n 10M -i 1M --add_context_beta
echo "done!"
}

