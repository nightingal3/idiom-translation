### /projects/tir5/users/mengyan3/idiomaticity-learning/tapes/hello_world.tape ###
#!/usr/bin/env ducttape

task param_step < in=/etc/passwd > out :: N=5 M=6 {
    echo "$in has $(wc -l < $in) lines"
    echo "The parameter N is $N"
    echo "The parameter M is $M"
}

global {
    ducttape_structure=flat
}
