### /projects/tir5/users/mengyan3/idiomaticity-learning/tapes/hello_world.tape ###
#!/usr/bin/env ducttape

global {
    someone=world
    ducttape_experimental_packages=enable
}

task first > x {
    for i in {1..10}; do
        echo $i >> $x
    done
}

task and_then < a=$x@first > x {
    cat < $a > $x
}

global {
    ducttape_structure=flat
}
