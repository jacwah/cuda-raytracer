module load cuda/10.1

make CFLAGS="-arch=sm_30" outdir=sm30
make CFLAGS="-arch=sm_30 -DPINNED" outdir=sm30p
make CFLAGS="-arch=sm_37" outdir=sm37
make CFLAGS="-arch=sm_37 -DPINNED" outdir=sm37p
