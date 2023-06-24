### /projects/tir5/users/mengyan3/idiomaticity-learning/tapes/hello_world.tape ###
#!/usr/bin/env ducttape

global {
    someone=world
    ducttape_experimental_packages=enable
}

task has_branches :: N=(WhichSize: smaller=5 bigger=6) > out {
    echo the number is $N
}

task parent_has_branches < in=$out@has_branches {
    echo received $in from branch
}
