module load cuda/10.1

nvcc -O3 -o ray30 -arch=sm_30 ray.cu
nvcc -O3 -o ray37 -arch=sm_37 ray.cu
