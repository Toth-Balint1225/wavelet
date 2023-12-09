#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include "utils.h"
#include <stdio.h>

#define COLAB
#ifdef COLAB
#include "timetest.h"
#include "error.h"
#else
#include <timetest.h>
#include <error.h>
#endif

__global__ void wav_kernel_advanced(float* s, float* w_mat_re, float* w_mat_im, float* c, size_t M, size_t N, size_t F)
{
    size_t col = blockDim.x * blockIdx.x + threadIdx.x;
    size_t row = blockDim.y * blockIdx.y + threadIdx.y;

	// important, that it only does stuff, if it's actually in the transformation range
	if (! (col < M+M-1 && row < F))
		return;


	// outer loop iterates over rows of the wavelet matrix (frequencies)
	float acc_re = 0.0;
	float acc_im = 0.0;
	// inner loop does the convolution sum
	// now I think that this maybe probably and quite possibly can be converted to a parallel reduction kernel thingy
	for (size_t n=0;n<N;n++) {
		if (col - n < M) {
			acc_re += w_mat_re[row * N + n] * s[col - n];
			acc_im += w_mat_im[row * N + n] * s[col - n];
		}
	}
	// finally, we use the complex norm in the i'th row of the transform
	c[row * (M+N-1) + col] = sqrtf(acc_re*acc_re + acc_im*acc_im);
	//printf("tid: %lu iteration %lu writing %f into c[%lu] \n", tid, i, acc, i * (M+N-1) + tid);
}

__global__ void gen_wavelets(size_t N, size_t F, float fmin, float fdiff, float h, float wavelet_min, float morlet_base, float* wavelet_mat_re, float* wavelet_mat_im)
{
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
	size_t row = blockIdx.y * blockDim.y + threadIdx.y;
	if (! (row < F && col < N))
		return;
	// get what row we're in -> implies freq
	float freq = fmin + fdiff * row;
	float tx = freq * (wavelet_min + col * h);

	// tester real and imag components will be shared memory, so we won't need to pass it into the function
	wavelet_mat_re[row * N + col] = freq * cosf(morlet_base * tx) * expf(-1 * (tx * tx) / 2);
	wavelet_mat_im[row * N + col] = freq * sinf(morlet_base * tx) * expf(-1 * (tx * tx) / 2);
	// printf("k: %lu (%f, %f)\n", k, tester_r[k], tester_i[k]);
}


Trafo flattened_transformer(std::vector<float> signal_vector, size_t M, float fmin, float fmax, size_t freq_num, float sample_freq)
{
	float wavelet_min = -4;
	float wavelet_max = 4;
	size_t F = freq_num;
	float morlet_base = M_PI * 2;
	float h = 1.0 / sample_freq;
	float fdiff = (fmax - fmin) / freq_num;
	size_t N = (wavelet_max - wavelet_min) * sample_freq;

	float* signal;
	CHECK_ERR(cudaMallocHost((void**)&signal, M * sizeof(float)));
	float* transform = new float[(M+N-1) * F];
	float* transform_control  = new float[(M+N-1) * F];

	// GPU handles
	float *d_signal, *d_wavelet_mat_re, *d_wavelet_mat_im, *d_transform;
	CHECK_ERR(cudaMalloc((void**)&d_signal, M * sizeof(float)));
	CHECK_ERR(cudaMalloc((void**)&d_wavelet_mat_re, N * F * sizeof(float)));
	CHECK_ERR(cudaMalloc((void**)&d_wavelet_mat_im, N * F * sizeof(float)));
	CHECK_ERR(cudaMalloc((void**)&d_transform, (M+N-1) * F * sizeof(float)));

	// init
	for (size_t i=0;i<M;i++)
		signal[i] = signal_vector[i];
	CHECK_ERR(cudaMemcpyAsync(d_signal, signal, M * sizeof(float), cudaMemcpyHostToDevice, cudaStreamDefault));

	cudaEvent_t gen_finished;
	CHECK_ERR(cudaEventCreate(&gen_finished));

	// generate the wavelet matrix
	unsigned T_wav = 32;
	dim3 grid_wav = dim3((N + T_wav - 1) / T_wav, (F + T_wav - 1) / T_wav);
	dim3 block_wav = dim3(T_wav, T_wav);
	gen_wavelets<<<grid_wav, block_wav>>>(N, F, fmin, fdiff, h, wavelet_min, morlet_base, d_wavelet_mat_re, d_wavelet_mat_im);
	CHECK_ERR(cudaEventRecord(gen_finished));

	
	// wait until the template generation finishes
	CHECK_ERR(cudaEventSynchronize(gen_finished));
	// initial block copied like before

	size_t T_main = 32;
	dim3 grid_main = dim3((N + M + T_main - 2) / T_main, (F + T_main - 1) / T_main);
	dim3 block_main = dim3(T_main, T_main);
	wav_kernel_advanced<<<grid_main, block_main>>>(d_signal, d_wavelet_mat_re, d_wavelet_mat_im, d_transform, M, N, F);
	LAST_ERR();


	//control_trafo(signal, wavelet_mat_re, wavelet_mat_im, transform_control, M, N, F);
	//compare(transform, transform_control, M, N, F);

	CHECK_ERR(cudaMemcpy(transform, d_transform, (M+N-1) * F * sizeof(float), cudaMemcpyDeviceToHost));
	Trafo res_vec(F);
	for (size_t i=0;i<F;i++) {
		res_vec[F-i-1] = std::vector<float>(N+M-1);
		for (size_t j=0;j<N+M-1;j++) {
			res_vec[F-i-1][j] = transform[i * (N+M-1) + j];
		}
	}

	CHECK_ERR(cudaFree(d_signal));
	CHECK_ERR(cudaFree(d_wavelet_mat_re));
	CHECK_ERR(cudaFree(d_wavelet_mat_im));
	CHECK_ERR(cudaFree(d_transform));
	CHECK_ERR(cudaFreeHost(signal));
	delete[] transform;

	return res_vec;

}

int main(int argc, char** argv)
{
    if (argc < 4) {
        std::cerr << "Usage: ./" << argv[0] << " <signal length (sec)> <sampling frequency (Hz)> <spectrum size (#)> [print heatmap (0/1), default 0]" << std::endl;
        std::exit(1);
    }

    // parse args
    float signal_length = std::atof(argv[1]);
	float sample_freq = std::atof(argv[2]);
    float spectrum = std::atoi(argv[3]);

    bool print_heat = false;
    if (argc == 5) {
        print_heat = std::atoi(argv[4]) == 1 ? true : false;
    }

    // set up test case
	auto x = sample(0, signal_length, sample_freq);
	auto message_sin = function(x, [](float x) -> float {
		return std::sin(2*PI*x) + 2 + std::sin(20*2*PI*x) * std::cos(4*PI*x);
	});
	// noise in the message
	for (size_t i=0;i<x.size();i++)
		message_sin[i] += (std::cos(x[i]) * std::sin(70*PI*x[i])) * 0.2;

	auto signal = freq_modulate(message_sin, 10, sample_freq, 1, 50);

    // noise in the signal
	for (size_t i=0;i<x.size();i++)
		signal[i] += (std::cos(x[i]) * std::sin(140*PI*x[i])) * 0.2;


	auto trafo = flattened_transformer(signal, signal.size(), 1, 100, spectrum, sample_freq);

    if (print_heat)
        plot_trafo(trafo);

    // for the profiler
	CHECK_ERR(cudaDeviceReset());
    return 0;
}