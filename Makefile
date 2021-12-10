outdir ?= .
src = $(wildcard ray*.cu)
out = $(patsubst %.cu,$(outdir)/%.out,$(src))

all: $(out)

$(outdir)/%.out: %.cu | $(outdir)
	nvcc $(CFLAGS) -O3 -g -o $@ $<

$(outdir):
	mkdir -p $(outdir)

PHONY: clean

clean:
	$(RM) $(out)
ifneq ($(outdir),.)
	rmdir $(outdir)
endif
