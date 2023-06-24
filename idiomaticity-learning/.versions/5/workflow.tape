### /projects/tir5/users/mengyan3/idiomaticity-learning/tapes/hello_world.tape ###
#!/usr/bin/env ducttape

global {
    someone=world
    ducttape_experimental_packages=enable
}

task hello_external_world :: who=$someone {
    echo hello $who
}

global {
    ducttape_structure=flat
}
