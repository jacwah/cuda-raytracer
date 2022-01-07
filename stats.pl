#!/usr/bin/perl

use Statistics::Basic;

my @a = <>;

my $x = Statistics::Basic::mean(@a);
my $e = Statistics::Basic::stddev(@a);

printf("%e %e\n", $x, $e);
