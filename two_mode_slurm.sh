#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --job-name=alldata_FDM
#SBATCH --mem=1gb
#SBATCH --tmp=1gb

#cd /fred/oz002/users/mmiles/SinglePulse/two-mode-timing-slurm-uniform-pool
touch ${1}".alldata_FDM"

srun python /home/mmiles/soft/SP/alldata_smoothed_weakStrongTemplatePython_TMtoa_Gauss_FDM_dyn.py $1

rm -f ${1}".alldata_FDM"

echo done