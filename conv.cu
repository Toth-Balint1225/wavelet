#include <cuda.h>
#include <iostream>
#include <stdio.h>
#include <cstdlib>

#define COLAB
#ifdef COLAB
#include "timetest.h"
#include "error.h"
#else
#include <timetest.h>
#include <error.h>
#endif

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

void control(float* u, size_t n, float* v, size_t m, float* res) {
	// exactly like MATLAB's implementation
	for (size_t i = 0; i < m + n - 1; i++)
		for (size_t j = 0; j < n; j++)
			if (i - j < m) res[i] += u[j] * v[i - j];
}


/**
 * parallel convolution with CUDA
 * draft codes
 * ---
 * 
 * Convolution Sum
 * where signal is s, |s| = M
 *       window is w, |w| = N
 *       result is c, |c| = M + N -1
 * 
 * for k := 0 to M + N - 1 do
 *   c[k] := 0
 *   for n := 0 to N do
 *     c[k] += w[n] * s[k-n] 
 *   end
 * end
 * 
 * notes:
 * - a thread is an item in the result
 * - a thread needs the entire window, the window size equivalent amount of items 
 * - what happens on block edges ??
 *   - block cluster with unified shared memory ?
 *   - each block's data is sized in a way, that some signal values overlap ?
 *   - can actually stream it ?
*/

/**
 * Basic convolution: 
 * - everything is in global memory
 * - basic reduction (a.k.a. for loop)
*/
__global__ void conv_kernel1(float* s, float* w, float* c, size_t M, size_t N)
{
    size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    float acc = 0.0;
    for (size_t n=0;n<N;n++) {
        if (tid - n < M)
            acc += w[n] * s[tid - n];
    }
    c[tid] = acc;
}

void conv_driver1(size_t M, size_t N)
{
    float* s = new float[M];
    float* w = new float[N];
    float* res = new float[M+N-1];
    float* control_res = new float[M+N-1];

    // fill up with dummy variables
    for (auto i=0;i<M;i++) {
        s[i] = std::rand() % 10;
    }
    for (auto i=0;i<N;i++) {
        w[i] = std::rand() % 10;
    }
    
    // GPU setup
    float *d_s, *d_w, *d_res;
    CHECK_ERR(cudaMalloc((void**)&d_s, M*sizeof(float)));
    CHECK_ERR(cudaMalloc((void**)&d_w, N*sizeof(float)));
    CHECK_ERR(cudaMalloc((void**)&d_res, (M+N-1)*sizeof(float)));

    CHECK_ERR(cudaMemcpy(d_s, s, M*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_ERR(cudaMemcpy(d_w, w, N*sizeof(float), cudaMemcpyHostToDevice));

    // execute
    conv_kernel1<<<(M+N+1022)/1024, 1024>>>(d_s, d_w, d_res, M, N);
    LAST_ERR();

    CHECK_ERR(cudaMemcpy(res, d_res, (M+N-1)*sizeof(float), cudaMemcpyDeviceToHost));
    // safety check
    control(s, M, w, N, control_res);
    for (size_t i=0;i<(M+N-1);i++) {
        std::cout << res[i] - control_res[i] << " ";
    }
    std::cout << std::endl;

    CHECK_ERR(cudaFree(d_s));
    CHECK_ERR(cudaFree(d_w));
    CHECK_ERR(cudaFree(d_res));
    delete res;

    CHECK_ERR(cudaDeviceReset());
}

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        std::cerr << "Usage: ./" << argv[0] << " <signal_length> <window_length>" << std::endl;
        std::exit(1);
    }
    std::srand(20001225);
    conv_driver1(std::atoi(argv[1]), std::atoi(argv[2]));
    return 0;
}