### /projects/tir5/users/mengyan3/idiomaticity-learning/tapes/hello_world.tape ###
#!/usr/bin/env ducttape

task hello_world {
    echo hi 
    echo >&2 hello
}

global {
    ducttape_structure=flat
}
