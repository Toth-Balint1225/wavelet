cc := g++ -Wall -Werror -Wpedantic -fopenmp 

wavelet: wavelet.cu utils.o
	nvcc -c wavelet.cu
	nvcc wavelet.o utils.o -o wavelet
	./wavelet > heat.dat


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
