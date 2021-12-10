#!/bin/bash

module load cuda/10.1

make CFLAGS="-std=c++11 -arch=sm_30" outdir=sm30
make CFLAGS="-std=c++11 -arch=sm_30 -DPINNED" outdir=sm30p
make CFLAGS="-std=c++11 -arch=sm_37" outdir=sm37
make CFLAGS="-std=c++11 -arch=sm_37 -DPINNED" outdir=sm37p
