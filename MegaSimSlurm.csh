#!/bin/csh

# csh MegaSimSlurm.csh SimSlurm.sh directory

set slurm = $1
set dirname = $2

foreach subdir (`ls $2 | grep F`)
    
    echo $subdir
    sbatch $slurm $2/$subdir
end