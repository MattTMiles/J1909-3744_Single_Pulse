#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --job-name=simulations
#SBATCH --mem=6gb
#SBATCH --tmp=6gb



cd $1

for sample in */ ; do
    
    cd $1/$sample ;
    echo $PWD
    if [ ! -d "sim_999" ]; then
        python /home/mmiles/soft/SP/SP_t2sim.py -par J1909-3744.par -tim *tim -avgpar J1909-3744_avg.par -output . ;
    fi
done

#for f in `find 1284/*dm -type f -printf "%f\n"`; do
#    if [ ! -f ${f%%.*}.f32 ]; then
#        pam -f 32 1284/${f%%.*}.dm -e f32;
#    fi
#done


echo done