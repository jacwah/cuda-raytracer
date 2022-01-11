all: sm30-1000.pdf sm30-100000.pdf sm37-1000.pdf sm37-100000.pdf sm30-10000.pdf sm37-10000.pdf

%.pdf: results/%.cat.dat results/%.min plot.gpi
	GNUTERM=pdfcairo gnuplot -c plot.gpi $< $$(<results/$*.min) > $@

%.cat.dat: %.dat %-p.dat
	cat $^ > $@

%.min: %.cat.dat
	cut -d " " -f 1,3,5 $< | tr " " "\n" | sort -g | head -1 > $@
