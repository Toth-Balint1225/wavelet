#include <iostream>
#include <cstdlib>

#include "utils.h"

#include "timetest.h"

auto transform_array_impl(float const* signal, size_t n, float* tester_r, float* tester_i, size_t m, float h, float wavelet_start, float morlet_base, float fmin, float fdiff, float freq, size_t sample_num, float* res, size_t res_width) -> void {
	for (size_t i = 0; i < sample_num; i++) {
		freq = fmin + i * fdiff;
		for (size_t j = 0; j < m; j++) {
			float tx = freq * (wavelet_start + j * h);
			// tester real and imag components will be shared memory, so we won't need to pass it into the function
			tester_r[j] = freq * std::cos(morlet_base * tx) * std::exp(-1 * (tx * tx) / 2);
			tester_i[j] = freq * std::sin(morlet_base * tx) * std::exp(-1 * (tx * tx) / 2);
		}

		float real, imag;
		for (size_t k = 0; k < m + n - 1; k++) {
			real = 0.0;
			imag = 0.0;
			// this is the problematic part, because parts of the vector can be in a different SMP
			for (size_t j = 0; j < n; j++) {
				if (k - j < m) {
					real += signal[j] * tester_r[k - j];
					imag += signal[j] * tester_i[k - j];
				}
			}
			res[(sample_num - i - 1)* res_width + k] = std::sqrt(real * real + imag * imag);
		}
	}
}

// probably CPU driver code for the GPU code
// effectively sets up the GPU call
auto transform_wavelet_array(std::vector<float> const& signal, float fmin, float fmax, size_t sample_num, float sample_freq) -> Trafo {
	float fdiff = (fmax - fmin) / sample_num;
	float morlet_base = 6.0;
	float freq = fmin;

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
	TIME_START();
	transform_array_impl(signal_raw, n, tester_r, tester_i, m, h, wavelet_start, morlet_base, fmin, fdiff, freq, sample_num, res, res_width);
	TIME_END();

	Trafo res_vec(sample_num);
	for (size_t i=0;i<sample_num;i++) {
		res_vec[i] = std::vector<float>(res_width);
		for (size_t j=0;j<res_width;j++) {
			res_vec[i][j] = res[i * res_width + j];
		}
	}

	delete [] signal_raw;
	delete [] tester_i;
	delete [] tester_r;
	delete [] res;
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

	auto trafo = transform_wavelet_array(signal, 1, 100, spectrum, sample_freq);

    if (print_heat) {
		plot_trafo(trafo);
	}

	(void)cuda_measure_start;
	(void)cuda_measure_end;
	(void)cuda_measure_elapsed;
	return 0;
}
