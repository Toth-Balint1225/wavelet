cc := g++ -Wall -fopenmp 

seqtest: seqtest.cpp utils.o
	$(cc) -c seqtest.cpp
	$(cc) seqtest.o utils.o -o $@

wavelet: wavelet.cu utils.o
	nvcc -c wavelet.cu
	nvcc wavelet.o utils.o -o wavelet
#./wavelet > heat.dat

tester: main.cu utils.o
	nvcc -c main.cu
	nvcc main.o utils.o -o $@
	./$@ 10 1000 64

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
