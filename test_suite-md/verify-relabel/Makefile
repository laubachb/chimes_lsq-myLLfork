## Test to see whether 1 element and 2 element systems are identical
## if the 1 element system is relabeled as a 2 element system.
##
all: 2body 3body 4body

2body:
	../../build/chimes_md run_md.ch.2b.in > run_md.ch.2b.out
	mv forceout.txt forceout.ch.2b.txt
	../../build/chimes_md run_md.c.2b.in > run_md.c.2b.out
	mv forceout.txt forceout.c.2b.txt
	perl ../../contrib/compare/compare.pl forceout.ch.2b.txt forceout.c.2b.txt


3body:
	../../build/chimes_md run_md.ch.3b.in > run_md.ch.3b.out
	mv forceout.txt forceout.ch.3b.txt
	../../build/chimes_md run_md.c.3b.in > run_md.c.3b.out
	mv forceout.txt forceout.c.3b.txt
	perl ../../contrib/compare/compare.pl forceout.ch.3b.txt forceout.c.3b.txt

4body:
	../../build/chimes_md run_md.ch.4b.in > run_md.ch.4b.out
	mv forceout.txt forceout.ch.4b.txt
	../../build/chimes_md run_md.c.4b.in > run_md.c.4b.out
	mv forceout.txt forceout.c.4b.txt
	perl ../../contrib/compare/compare.pl forceout.ch.4b.txt forceout.c.4b.txt

clean:
	rm -f forceout*.txt *.out *.gen *.bak traj_bad.xyz

