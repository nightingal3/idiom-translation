### /projects/tir5/users/mengyan3/idiomaticity-learning/tapes/hello_world.tape ###
#!/usr/bin/env ducttape

task hello_world < a=/etc/passwd b=/etc/hosts {
    echo "I will be reading files $a and $b and doing nothing useful with them"
    cat $a>/dev/null
    cat $b>/dev/null
}

global {
    ducttape_structure=flat
}
