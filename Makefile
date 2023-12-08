cc := g++ -Wall -Werror -Wpedantic -fopenmp 

plot: seq
	@./$< > plot.dat
	./plot

heat: seq
	@./$< > heat.dat
	./heatmap

debug: seq
	gdb --tui $<

%.o: %.cpp
	$(cc) -c $<

par: parallel.o utils.o
	$(cc) $^ -o $@

seq: sequential.o utils.o
	$(cc) -O2 $^ -o $@
