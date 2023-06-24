### /projects/tir5/users/mengyan3/idiomaticity-learning/tapes/hello_world.tape ###
#!/usr/bin/env ducttape

task hello_world > x y_txt {
    echo writing files $x and $y_txt...
    echo hello > $x
    echo hello > $y_txt
}

task named_output < x=/etc/passwd > named=x.gz {
    cat $x | gzip > $named
}

global {
    ducttape_structure=flat
}
