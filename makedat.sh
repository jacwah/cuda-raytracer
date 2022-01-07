#!/bin/bash

indir=${1?}
dim=${2?}

for v in 0 1 2; do
    for block in 4 8 16 32; do
        f=${indir}/${v}-${dim}-${block}
        ./stats.pl ${f} | tr "\n" " "
    done
    echo
done
