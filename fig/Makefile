all: sm30-1000.pdf sm30-100000.pdf sm37-1000.pdf

%.pdf: results/%.cat.dat plot.gpi
	GNUTERM=pdfcairo gnuplot -c plot.gpi $< > $@

%.cat.dat: %.dat %-p.dat
	cat $^ > $@
