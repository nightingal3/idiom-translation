### /projects/tir5/users/mengyan3/idiomaticity-learning/tapes/shell-submitter.tape ###
task hello_shell {
    echo hello
}

submitter shell :: COMMANDS {
    action run > exit_code {
        eval $COMMANDS
    }
}
