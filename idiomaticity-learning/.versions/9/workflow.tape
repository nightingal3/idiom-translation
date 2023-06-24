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
    }
}

global {
    ducttape_experimental_submitters=enable
}

task hello_sbatch :: .submitter=sbatch .job_name=example_job .mem="10G" .time="1:00:00" .gpus="1" {
    echo hello
}

