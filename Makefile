src = $(wildcard ray*.cu)
out = $(patsubst %.cu,%.out,$(src))

all: $(out)

%.out: %.cu
	nvcc -O3 -g -o $@ $^
