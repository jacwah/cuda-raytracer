#!/bin/bash -l

#SBATCH -A edu21.dd2360
#SBATCH -N 1

# sbatch --gres=gpu:K420:1 perf_tegner.sh sm30
# sbatch --gres=gpu:K80:2 perf_tegner.sh sm37

TIMEFORMAT=%R

outdir=${1?}

image_dir="/cfs/klemming/scratch/j/jacobwah/${outdir}-${SLURM_JOB_ID}"
time_dir="/afs/pdc.kth.se/home/j/jacobwah/Private/dd2360-time/${outdir}-${SLURM_JOB_ID}"

mkdir -p ${image_dir}
mkdir -p ${time_dir}

date

for dim in 10 100 1000 10000 100000; do
    for block in 4 8 16 32; do
        for v in 0 1 2; do
            time_file="${time_dir}/${v}-${dim}-${block}"
            for i in $(seq 10); do
                image_file="${image_dir}/${v}-${dim}-${block}-${i}.ppm"
                cmd="${outdir}/ray${v}.out ${dim} ${dim} ${image_file} ${block}"
                echo ${cmd}
                { time ${cmd}; } 2>> ${time_file}
        done
    done
done

date
