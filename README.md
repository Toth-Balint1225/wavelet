# Wavelet transformation project

Requires GCC, CUDA and GNUplot. 
- sequential.cpp contains experimental code and several iterations of the serial implementation
- conv.cu is the basic convolution impementation in CUDA
- wavelet.cu contains iterations of the wavelet transformer

## Compile and run
- Makefile is present for the sequential case. 
- Use `make heat` to compile, run and pipe the result into heat.dat. Then  GNUplot displays the matrix as a heatmap.
  For the heat target, you must only use a single `plot_trafo` function to output the points into a matrix data file
- Use `make plot` to compile, run and pipe the result into plot.dat. Then  GNUplot displays the function.
  For the plot target, you must only use a single `plot_func` function to output the points of a data series.
