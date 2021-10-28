#!/bin/bash
#SBATCH --job-name=noT
#SBATCH -c 10
#SBATCH --mem=20G
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --array=[0-8]%20
#SBATCH -o %x_%A_%a.out
#SBATCH -e %x_%A_%a.err

FILES=($(ls -1 x*LL))

FQ1=${FILES[$SLURM_ARRAY_TASK_ID]}

module load python/3.6.3
sh $FQ1
