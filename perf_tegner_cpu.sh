#!/bin/bash -l

#SBATCH -A edu21.dd2360
#SBATCH -N 1
#SBATCH -t 0:10:0

# sbatch perf_tegner_cpu.sh

TIMEFORMAT=%R

image_dir="/cfs/klemming/scratch/j/jacobwah/cpu-${SLURM_JOB_ID}"
time_dir="/afs/pdc.kth.se/home/j/jacobwah/Private/dd2360-time/cpu-${SLURM_JOB_ID}"

mkdir -p ${image_dir}
mkdir -p ${time_dir}

module load anaconda/py37

date

for dim in 10 100 1000; do
    time_file="${time_dir}/${dim}"
    for i in $(seq 10); do
        image_file="${image_dir}/${dim}-${i}.raw"
        cmd="srun python raytracing.py ${dim} ${dim} ${image_file}"
        echo ${cmd}
        { time ${cmd} 2>&1; } 2>> ${time_file}
    done
done

date
