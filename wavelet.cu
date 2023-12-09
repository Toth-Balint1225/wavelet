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

// wavelet transform test codes and drafts



// control solution (copied from the serial CPU implementation)
using Trafo = std::vector<std::vector<float>>;
void control(float* s, float* w_mat, float* c, size_t M, size_t N, size_t F)
{
	for (size_t k=0;k<F;k++)
		for (size_t i = 0; i < M + N - 1; i++) {
			c[k*(M+N-1) + i] = 0.0;
			for (size_t j = 0; j < M; j++)
				if (i - j < N) c[k * (M+N-1) + i] += s[j] * w_mat[k * N + (i - j)];
		}
}

void control_trafo(float* s, float* w_mat_re, float* w_mat_im, float* c, size_t M, size_t N, size_t F)
{
	float re = 0.0;
	float im = 0.0;
	for (size_t k=0;k<F;k++) {
		for (size_t i = 0; i < M + N - 1; i++) {
			re = 0.0;
			im = 0.0;	
			for (size_t j = 0; j < M; j++) {
				if (i - j < N) {
					re += s[j] * w_mat_re[k * N + (i - j)];
					im += s[j] * w_mat_im[k * N + (i - j)];
				}
			}
			c[k*(M+N-1) + i] = std::sqrt(re*re + im*im);
		}
	}
}

// control checker for the tests
void compare(float* trafo, float* control, size_t M, size_t N, size_t F)
{
	for (size_t i=0;i<F;i++) {
		for (size_t j=0;j<(N+M-1);j++) {
			float diff = trafo[i*(M+N-1) + j] - control[i*(M+N-1) + j];
			if (std::abs(diff) > 0.01)
				std::cout << "Diff " << diff << std::endl;
		}
	}
}

// working
// this generates the specified frequency wavelet filter
__device__ void gen_filter(float freq, size_t m, float wavelet_start, float morlet_base, float h, float* tester_r, float* tester_i)
{
    size_t k = blockIdx.x * blockDim.x + threadIdx.x;
	if (k < m) {
		float tx = freq * (wavelet_start + k * h);
		// tester real and imag components will be shared memory, so we won't need to pass it into the function
		tester_r[k] = freq * cos(morlet_base * tx) * exp(-1 * (tx * tx) / 2);
		tester_i[k] = freq * sin(morlet_base * tx) * exp(-1 * (tx * tx) / 2);
		printf("k: %lu (%f, %f)\n", k, tester_r[k], tester_i[k]);
	}
}

// function just to call the wavelet generator
__global__ void test_kernel(float* signal, size_t n, size_t sample_num, float fmin, float fdiff, float* tester_r, float* tester_i, float wavelet_start, float morlet_base, float h, size_t m,  float* res, size_t res_width)
{
	float freq = 0.0;
	for (size_t i=0;i<sample_num;i++) {
		printf("i: %lu freq: %f\n", i, freq);
		freq = fmin + i * fdiff;
		gen_filter(freq, m, wavelet_start, morlet_base, h, tester_r, tester_i);
		__syncthreads();
	}
}

// not working, probably because of the grid cannot be synchronised
__global__ void basic_kernel(float* signal, size_t n, size_t sample_num, float fmin, float fdiff, float* tester_r, float* tester_i, float wavelet_start, float morlet_base, float h, size_t m,  float* res, size_t res_width)
{
    // set up indices
    size_t k = blockIdx.x * blockDim.x + threadIdx.x;
    //size_t tid = threadIdx.x;
    float freq = 0.0;

    // i steps the frequency
	for (size_t i = 0; i < sample_num; i++) {
		// wavelet creation
		freq = fmin + i * fdiff;
        // make the frequency specific wavelet (into global memory for now)
		if (k < m) {
			float tx = freq * (wavelet_start + k * h);
			// tester real and imag components will be shared memory, so we won't need to pass it into the function
			tester_r[k] = freq * cos(morlet_base * tx) * exp(-1 * (tx * tx) / 2);
			tester_i[k] = freq * sin(morlet_base * tx) * exp(-1 * (tx * tx) / 2);
			printf("(%f, %f)\n", tester_r[k], tester_i[k]);
		}
        __syncthreads();

		float real, imag;
        real = 0.0;
        imag = 0.0;

		// convolution

        for (size_t j = 0; j < n; j++) {
            if (k - j < m) {
                real += signal[j] * tester_r[k - j];
                imag += signal[j] * tester_i[k - j];
            }
        }
		res[(sample_num - i - 1)* res_width + k] = sqrt(real * real + imag * imag);
        // sync up the global everything (or is it?)
        __syncthreads();
	}

}

// driver call for the previous kernel
Trafo transform(std::vector<float> const& signal, float fmin, float fmax, size_t sample_num, float sample_freq)
{
	float fdiff = (fmax - fmin) / sample_num;
	float morlet_base = 6.0;
	//float freq = fmin;

	// variables for the function generation
	float wavelet_start = -4;
	float wavelet_end = 4;
	float h = 1.0 / sample_freq;
	size_t wavelet_sample_num = (wavelet_end - wavelet_start) * sample_freq;
	auto tester_r = new float[wavelet_sample_num];
	auto tester_i = new float[wavelet_sample_num];
	
	// variables for the convolution
	size_t n = signal.size();
	size_t m = wavelet_sample_num;
	// result pointer to be passed to the kernel
	size_t res_width = n + m - 1;
	size_t res_len = res_width * sample_num;
	auto res = new float[res_len];
	auto signal_raw = new float[n];
	std::copy(signal.begin(), signal.end(), signal_raw);

	// this will be the kernel call
	// transform_array_impl(signal_raw, n, tester_r, tester_i, m, h, wavelet_start, morlet_base, fmin, fdiff, freq, sample_num, res, res_width);

    // prepare kernel call
    float* d_signal;
    float* d_tester_r;
    float* d_tester_i;
    float* d_res;
    CHECK_ERR(cudaMalloc((void**)&d_signal, n * sizeof(float)));
    CHECK_ERR(cudaMalloc((void**)&d_tester_r, wavelet_sample_num * sizeof(float)));
    CHECK_ERR(cudaMalloc((void**)&d_tester_i, wavelet_sample_num * sizeof(float)));
    CHECK_ERR(cudaMalloc((void**)&d_res, res_len * sizeof(float)));

    CHECK_ERR(cudaMemcpy(d_signal, signal_raw, n * sizeof(float), cudaMemcpyHostToDevice));

    // call
    const size_t T = 1024;
    basic_kernel<<<(res_width+T-1)/T, T>>>(d_signal, n, sample_num, fmin, fdiff, d_tester_r, d_tester_i, wavelet_start, morlet_base, h, m, d_res, res_width);
    LAST_ERR();

    CHECK_ERR(cudaMemcpy(res, d_res, res_len * sizeof(float), cudaMemcpyDeviceToHost));

	Trafo res_vec(sample_num);
	for (size_t i=0;i<sample_num;i++) {
		res_vec[i] = std::vector<float>(res_width);
		for (size_t j=0;j<res_width;j++) {
			res_vec[i][j] = res[i * res_width + j];
		}
	}

	delete signal_raw;
	delete tester_i;
	delete tester_r;
	delete res;

    CHECK_ERR(cudaFree(d_signal));
    CHECK_ERR(cudaFree(d_tester_r));
    CHECK_ERR(cudaFree(d_tester_i));
    CHECK_ERR(cudaFree(d_res));

	return res_vec;
}

/**
 * New approach:
 * - generate the kernel on the host and copy it over as a matrix
 * - compute the transform in a column-wise block thins
 * - this will probably be better, because in the general case M >> N and M, N >> F
 *   so in this case we can reduce memory reads and writes hopefully
*/

__global__ void conv_kernel_nogen(float* s, float* w_mat, float* c, size_t M, size_t N, size_t F)
{
    size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (! (tid < M+N-1))
		return;


	for (size_t i=0;i<F;i++)
	{
		float acc = 0.0;
		for (size_t n=0;n<N;n++) {
			if (tid - n < M)
				acc += w_mat[i * N + n] * s[tid - n];
		}
		c[i * (M+N-1) + tid] = acc;
		//printf("tid: %lu iteration %lu writing %f into c[%lu] \n", tid, i, acc, i * (M+N-1) + tid);
	}
}

// test driver for the previous kernel
void conv_kernel_nogen_driver(size_t M, size_t N, size_t F)
{
	float* signal = new float[M];
	float* wavelet_mat = new float[N * F];
	float* transform = new float[(M+N-1) * F];
	float* transform_control  = new float[(M+N-1) * F];

	// GPU handles
	float *d_signal, *d_wavelet_mat, *d_transform;
	CHECK_ERR(cudaMalloc((void**)&d_signal, M * sizeof(float)));
	CHECK_ERR(cudaMalloc((void**)&d_wavelet_mat, N * F * sizeof(float)));
	CHECK_ERR(cudaMalloc((void**)&d_transform, (M+N-1) * F * sizeof(float)));

	// init
	for (size_t i=0;i<M;i++)
		signal[i] = std::rand() % 10;

	for (size_t i=0;i<N*F;i++)
		wavelet_mat[i] = std::rand() % 10;

	CHECK_ERR(cudaMemcpy(d_signal, signal, M * sizeof(float), cudaMemcpyHostToDevice));
	CHECK_ERR(cudaMemcpy(d_wavelet_mat, wavelet_mat, N * F * sizeof(float), cudaMemcpyHostToDevice));

	size_t T = 1024;
	conv_kernel_nogen<<<(N + M + T - 2) / T, T>>>(d_signal, d_wavelet_mat, d_transform, M, N, F);
	LAST_ERR();

	CHECK_ERR(cudaMemcpy(transform, d_transform, (M+N-1) * F * sizeof(float), cudaMemcpyDeviceToHost));

	//control(signal, wavelet_mat, transform_control, M, N, F);
	//compare(transform, transform_control, M, N, F);

	CHECK_ERR(cudaFree(d_signal));
	CHECK_ERR(cudaFree(d_wavelet_mat));
	CHECK_ERR(cudaFree(d_transform));
	delete[] signal;
	delete[] wavelet_mat;
	delete[] transform;

}

// effectively the wavelet transformation kernel, ready for streamification
__global__ void wav_kernel_nogen(float* s, float* w_mat_re, float* w_mat_im, float* c, size_t M, size_t N, size_t F)
{
    size_t tid = blockDim.x * blockIdx.x + threadIdx.x;

	// important, that it only does stuff, if it's actually in the transformation range
	if (! (tid < M+N-1))
		return;


	// outer loop iterates over rows of the wavelet matrix (frequencies)
	for (size_t i=0;i<F;i++)
	{
		float acc_re = 0.0;
		float acc_im = 0.0;
		// inner loop does the convolution sum
		for (size_t n=0;n<N;n++) {
			if (tid - n < M) {
				acc_re += w_mat_re[i * N + n] * s[tid - n];
				acc_im += w_mat_im[i * N + n] * s[tid - n];
			}
		}
		// finally, we use the complex norm in the i'th row of the transform
		c[i * (M+N-1) + tid] = sqrtf(acc_re*acc_re + acc_im*acc_im);
		//printf("tid: %lu iteration %lu writing %f into c[%lu] \n", tid, i, acc, i * (M+N-1) + tid);
	}
}

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

__global__ void identical(size_t N, size_t F, float* re, float* im)
{
	size_t item = blockIdx.x * blockDim.x + threadIdx.x;
	if (!(item < N*F))
		return;
	printf("item %lu: (%f, %f)\n", item, re[item], im[item]);
}

void gen_wavelet_inplace(size_t sample_freq, float fmin, float fmax, size_t freq_num)
{
	size_t F = freq_num;
	float wavelet_min = -4;
	float wavelet_max = 4;
	float morlet_base = M_PI * 2;
	float h = 1.0 / sample_freq;
	float fdiff = (fmax - fmin) / freq_num;
	size_t N = (wavelet_max - wavelet_min) * sample_freq;

	float* re = new float[N * F];
	float* im = new float[N * F];


	float *d_wav_re, *d_wav_im;
	CHECK_ERR(cudaMalloc((void**)&d_wav_re, N * F * sizeof(float)));
	CHECK_ERR(cudaMalloc((void**)&d_wav_im, N * F * sizeof(float)));

	unsigned T = 32;
	dim3 grid = dim3((N + T - 1) / T, (F + T - 1) / T);
	dim3 block = dim3(T, T);
	cudaEvent_t gen_finished;
	cudaEventCreate(&gen_finished);
	gen_wavelets<<<grid, block>>>(N, F, fmin, fdiff, h, wavelet_min, morlet_base, d_wav_re, d_wav_im);
	cudaEventRecord(gen_finished);
	LAST_ERR();

	cudaEventSynchronize(gen_finished);
	identical<<<(N + 1024 - 1) / 1024, 1024>>>(N, F, d_wav_re, d_wav_im);
	LAST_ERR();

	CHECK_ERR(cudaMemcpy(re, d_wav_re, N * F * sizeof(float), cudaMemcpyDeviceToHost));
	CHECK_ERR(cudaMemcpy(im, d_wav_im, N * F * sizeof(float), cudaMemcpyDeviceToHost));

	/*for (size_t i=0;i<N;i++) {
		std::cout << re[i] << ", ";
	}
	std::cout << std::endl;*/

	CHECK_ERR(cudaFree(d_wav_re));
	CHECK_ERR(cudaFree(d_wav_im));
	delete [] re;
	delete [] im;
}

// test driver, with randomly generated data
void wav_kernel_nogen_driver(size_t M, size_t N, size_t F)
{
	float* signal = new float[M];
	float* wavelet_mat_re = new float[N * F];
	float* wavelet_mat_im = new float[N * F];
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
		signal[i] = std::rand() % 10;

	// this later will be the wavelet generator (maybe on the GPU, I have no idea yet)
	for (size_t i=0;i<N*F;i++) {
		wavelet_mat_re[i] = std::rand() % 10;
		wavelet_mat_im[i] = std::rand() % 10;
	}

	CHECK_ERR(cudaMemcpy(d_signal, signal, M * sizeof(float), cudaMemcpyHostToDevice));
	CHECK_ERR(cudaMemcpy(d_wavelet_mat_re, wavelet_mat_re, N * F * sizeof(float), cudaMemcpyHostToDevice));
	CHECK_ERR(cudaMemcpy(d_wavelet_mat_im, wavelet_mat_im, N * F * sizeof(float), cudaMemcpyHostToDevice));

	size_t T = 1024;
	wav_kernel_nogen<<<(N + M + T - 2) / T, T>>>(d_signal, d_wavelet_mat_re, d_wavelet_mat_im, d_transform, M, N, F);
	LAST_ERR();

	CHECK_ERR(cudaMemcpy(transform, d_transform, (M+N-1) * F * sizeof(float), cudaMemcpyDeviceToHost));

	//control(signal, wavelet_mat, transform_control, M, N, F);
	//compare(transform, transform_control, M, N, F);

/*
	for (size_t i=0;i<(N+M-1) * F; i++) {
		std::cout << transform[i] << " ";
	}
	std::cout << std::endl;
*/

	CHECK_ERR(cudaFree(d_signal));
	CHECK_ERR(cudaFree(d_wavelet_mat_re));
	CHECK_ERR(cudaFree(d_wavelet_mat_im));
	CHECK_ERR(cudaFree(d_transform));
	delete[] signal;
	delete[] wavelet_mat_re;
	delete[] wavelet_mat_im;
	delete[] transform;

}

Trafo simple_transformer(std::vector<float> signal_vector, size_t M, float fmin, float fmax, size_t freq_num, float sample_freq)
{
	float wavelet_min = -4;
	float wavelet_max = 4;
	size_t F = freq_num;
	float morlet_base = M_PI * 2;
	float h = 1.0 / sample_freq;
	float fdiff = (fmax - fmin) / freq_num;
	size_t N = (wavelet_max - wavelet_min) * sample_freq;
	float* signal = new float[M];
	float* wavelet_mat_re = new float[N * F];
	float* wavelet_mat_im = new float[N * F];
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

	// this later will be the wavelet generator (maybe on the GPU, I have no idea yet)
	float tx = 0.0;
	float freq = 0;
	for (size_t i=0;i<F;i++) {
		freq = fmin + i * fdiff;
		for (size_t j=0;j<N;j++) {
			tx = freq * (wavelet_min + j * h);
			wavelet_mat_re[i*N+j] = freq * cos(morlet_base * tx) * exp(-1 * (tx * tx) / 2);
			wavelet_mat_im[i*N+j] = freq * sin(morlet_base * tx) * exp(-1 * (tx * tx) / 2);
		}
	}

	CHECK_ERR(cudaMemcpy(d_signal, signal, M * sizeof(float), cudaMemcpyHostToDevice));
	CHECK_ERR(cudaMemcpy(d_wavelet_mat_re, wavelet_mat_re, N * F * sizeof(float), cudaMemcpyHostToDevice));
	CHECK_ERR(cudaMemcpy(d_wavelet_mat_im, wavelet_mat_im, N * F * sizeof(float), cudaMemcpyHostToDevice));

	size_t T = 1024;
	wav_kernel_nogen<<<(N + M + T - 2) / T, T>>>(d_signal, d_wavelet_mat_re, d_wavelet_mat_im, d_transform, M, N, F);
	LAST_ERR();

	CHECK_ERR(cudaMemcpy(transform, d_transform, (M+N-1) * F * sizeof(float), cudaMemcpyDeviceToHost));

	//control_trafo(signal, wavelet_mat_re, wavelet_mat_im, transform_control, M, N, F);
	//compare(transform, transform_control, M, N, F);

/*
	for (size_t i=0;i<(N+M-1) * F; i++) {
		std::cout << transform[i] << " ";
	}
	std::cout << std::endl;
*/
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
	delete[] signal;
	delete[] wavelet_mat_re;
	delete[] wavelet_mat_im;
	delete[] transform;

	return res_vec;

}

Trafo stream_transformer(std::vector<float> signal_vector, size_t M, float fmin, float fmax, size_t freq_num, float sample_freq)
{
	float wavelet_min = -4;
	float wavelet_max = 4;
	size_t F = freq_num;
	float morlet_base = M_PI * 2;
	float h = 1.0 / sample_freq;
	float fdiff = (fmax - fmin) / freq_num;
	size_t N = (wavelet_max - wavelet_min) * sample_freq;

	float* signal = new float[M];
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
	CHECK_ERR(cudaMemcpy(d_signal, signal, M * sizeof(float), cudaMemcpyHostToDevice));

	cudaEvent_t gen_finished;
	CHECK_ERR(cudaEventCreate(&gen_finished));

	unsigned T_wav = 32;
	dim3 grid_wav = dim3((N + T_wav - 1) / T_wav, (F + T_wav - 1) / T_wav);
	dim3 block_wav = dim3(T_wav, T_wav);
	gen_wavelets<<<grid_wav, block_wav>>>(N, F, fmin, fdiff, h, wavelet_min, morlet_base, d_wavelet_mat_re, d_wavelet_mat_im);
	CHECK_ERR(cudaEventRecord(gen_finished));

	size_t T = 1024;

	// how many streams we need? - ceil(M / N)
	/*
	const size_t STREAM_NUM = std::ceil((float)M / (float)N);
	// stride will be N
	cudaStream_t stream[STREAM_NUM];
	for (size_t i=0;i<STREAM_NUM;i++) {
		cudaStreamCreate(&stream[i]);
	}
	*/
	
	// wait until the template generation finishes
	CHECK_ERR(cudaEventSynchronize(gen_finished));
	// initial block copied like before
	// CHECK_ERR(cudaMemcpy(d_signal,signal,N*sizeof(float), cudaMemcpyHostToDevice));

	// start the stream copy 
	/*
	for (size_t i=0;i<STREAM_NUM;i++) {
		// every stream will copy the block next to it
		CHECK_ERR(cudaMemcpyAsync(d_signal + (1+i)*N, signal + (1+i)*N, N*sizeof(float), cudaMemcpyHostToDevice, stream[i]));
		wav_kernel_nogen<<<(N + M + T - 2) / T, T, 0, stream[i]>>>(d_signal + (i*N), d_wavelet_mat_re, d_wavelet_mat_im, d_transform, M, N, F);
		CHECK_ERR(cudaMemcpyAsync(transform, d_transform, (M+N-1) * F * sizeof(float), cudaMemcpyDeviceToHost, stream[i]));
	}
	*/

	wav_kernel_nogen<<<(N + M + T - 2) / T, T>>>(d_signal, d_wavelet_mat_re, d_wavelet_mat_im, d_transform, M, N, F);
	LAST_ERR();


	//control_trafo(signal, wavelet_mat_re, wavelet_mat_im, transform_control, M, N, F);
	//compare(transform, transform_control, M, N, F);

/*
	for (size_t i=0;i<(N+M-1) * F; i++) {
		std::cout << transform[i] << " ";
	}
	std::cout << std::endl;
*/

	CHECK_ERR(cudaMemcpy(transform, d_transform, (M+N-1) * F * sizeof(float), cudaMemcpyDeviceToHost));
	Trafo res_vec(F);
	for (size_t i=0;i<F;i++) {
		res_vec[F-i-1] = std::vector<float>(N+M-1);
		for (size_t j=0;j<N+M-1;j++) {
			res_vec[F-i-1][j] = transform[i * (N+M-1) + j];
		}
	}

	/*
	for (size_t i=0;i<STREAM_NUM;i++) {
		cudaStreamDestroy(stream[i]);
	}
	*/
	CHECK_ERR(cudaFree(d_signal));
	CHECK_ERR(cudaFree(d_wavelet_mat_re));
	CHECK_ERR(cudaFree(d_wavelet_mat_im));
	CHECK_ERR(cudaFree(d_transform));
	delete[] signal;
	delete[] transform;

	return res_vec;

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

// test cases
int main()
{
	float sample_freq = 1000;
	auto x = sample(0, 10, sample_freq);
	auto message_step = function(x, [](float x) -> float {
		if (x < 1)
			return 0;
		else if (x < 2)
			return 1;
		else if (x < 3)
			return 0;
		else if (x < 4)	
			return 1;
		else 
			return 0;
	});
	auto message_sin = function(x, [](float x) -> float {
		return std::sin(2*PI*x) + 2 + std::sin(20*2*PI*x) * std::cos(4*PI*x);
	});
	auto message_exp = function(x, [](float x) -> float {
		return std::exp(x / 6);
	});
	auto message_idk = function(x, [](float x) -> float {
		return std::sin(2*PI*x) * std::sin(3*PI*x);
	});
	auto message_sin2 = function(x, [](float x) -> float {
		return std::sin(x*x);
	});
	// just a bit of noise
	for (size_t i=0;i<x.size();i++)
		message_step[i] += (std::cos(x[i]) * std::sin(70*PI*x[i])) * 0.2;

	auto signal = freq_modulate(message_sin, 10, sample_freq, 1, 50);

	for (size_t i=0;i<x.size();i++)
		signal[i] += (std::cos(x[i]) * std::sin(140*PI*x[i])) * 0.2;

	/*
	plot_func(x, signal);
    auto trafo = transform(signal, 1, 10, 10, sample_freq);
    plot_trafo(trafo);
	*/

	// tests for benchmarking
	// wav_kernel_nogen_driver(10000, 8000, 64);
	// auto trafo = simple_transformer(signal, signal.size(), 1, 100, 64, sample_freq);
	// auto trafo = stream_transformer(signal, signal.size(), 1, 100, 64, sample_freq);
	auto trafo = flattened_transformer(signal, signal.size(), 1, 100, 64, sample_freq);

	plot_trafo(trafo);

	// gen_wavelet_inplace(1000, 1, 100, 64);

	CHECK_ERR(cudaDeviceReset());
    return 0;
}