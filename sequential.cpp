#include <iostream>
#include <ostream>
#include <vector>
#include <functional>
#include <cmath>

#include "utils.h"

/*
 * Try different ideas: 
 * - 1. convolution of the signal and the Morlet wavelet
 * - 2. Apply the Wavelet transform directly
 * -> for both: check numerical precision
 * -> try and find a reference implementation for checking
 */

// (obsolete)
// trapezoid rule for numerical integration
auto trapezoid(std::vector<float> const& values, float from, float to) -> float {
	size_t n = values.size();
	float h = (to - from) / n;
	float acc = 0.0f;
	for (size_t i=1;i<n-1;i++)
		acc += values[i];
	return (values[0] + 2 * acc + values[n-1]) * (h / 2);
}

// (obsolete)
// Simpson's 1/3 rule for numerical integration
auto simpson1_3(std::vector<float> const& values, float from, float to) -> float {
	size_t n = values.size();
	float h = (to - from) / n;
	float acc = 0.0f;
	for (size_t i=1;i<(n/2);i++)
		acc += 4 * values[2*i-1] + 2 * values[2*i];
	return (values[0] + acc + values[n-1]) * (h / 3);
}

// the basic convolution implementation (stolen from MATLAB)
auto conv(std::vector<float> const& u, std::vector<float> const& v) -> std::vector<float> {
	// exactly like MATLAB's implementation
	size_t n = u.size();
	size_t m = v.size();
	std::vector<float> w(m + n - 1);
	for (size_t i = 0; i < m + n - 1; i++)
		for (size_t j = 0; j < n; j++)
			if (i - j < m) w[i] += u[j] * v[i - j];
			//std::cout << "w[" << i << "] = u[" << j << "] * v[" << i-j << "]" << std::endl;
	return w;
}

// basic implementation with the convolution function, to verify the process is working
auto transform_basic(std::vector<float> signal, float fmin, float fmax, size_t sample_num, float sample_freq) -> std::vector<std::vector<float>> {

	float fdiff = (fmax - fmin) / sample_num;
	std::vector<std::vector<float>> res(sample_num);
	float morlet_base = 2 * PI;
	float freq = fmin;
	// convolution can be implemented inside this function -> only one complete iteration
	// this is just for clarity
	auto wavelet_x = sample(-4, 4, sample_freq);
	for (size_t i = 0; i < sample_num; i++) {
		freq = fmin + i * fdiff;
		auto tester_r = function(wavelet_x, [freq, morlet_base](float x) -> float {
			float tx = freq * x;
			return  freq * std::cos(morlet_base * tx) * std::exp(-1 * (tx * tx) / 2);
		});
		auto tester_i = function(wavelet_x, [freq, morlet_base](float x) -> float {
			float tx = freq * x;
			return  freq * std::sin(morlet_base * tx) * std::exp(-1 * (tx * tx) / 2);
		});
		auto real = conv(signal, tester_r);
		auto imag = conv(signal, tester_i);
		// std::vector<float> spectrum(real.size());
		size_t idx = sample_num - i - 1;
		res[idx] = std::vector<float>(real.size());
		for (size_t j=0;j<real.size();j++)
			res[idx][j] = std::sqrt(real[j] * real[j] + imag[j] * imag[j]);
	}
	return res;
}

// moved every operation in the loop (re-implemented convolution in the inner loop)
auto transform_oneloop(std::vector<float> signal, float fmin, float fmax, size_t sample_num, float sample_freq) -> std::vector<std::vector<float>> {
	float fdiff = (fmax - fmin) / sample_num;
	std::vector<std::vector<float>> res(sample_num);
	float morlet_base = 6.0;
	float freq = fmin;
	// convolution can be implemented inside this function -> only one complete iteration
	// this is just for clarity
	//#pragma omp parallel for
	for (size_t i = 0; i < sample_num; i++) {
		freq = fmin + i * fdiff;
		auto tester_r = function(sample(-3, 3, sample_freq), [morlet_base, freq](float x) -> float {
			float tx = freq * x;
			return freq * std::cos(morlet_base * tx) * std::exp(-1 * (tx * tx) / 2);
		});
		auto tester_i = function(sample(-3,3,sample_freq), [morlet_base, freq](float x) -> float {
			float tx = freq * x;
			return  freq * std::sin(morlet_base * tx) * std::exp(-1 * (tx * tx) / 2);
		});
		size_t n = signal.size();
		size_t m = tester_r.size();

		size_t idx = sample_num - i - 1;
		res[idx] = std::vector<float>(m + n - 1);
		float real, imag;
		for (size_t k = 0; k < m + n - 1; k++) {
			real = 0.0;
			imag = 0.0;
			for (size_t j = 0; j < n; j++) {
				if (k - j < m) {
					real += signal[j] * tester_r[k - j];
					imag += signal[j] * tester_i[k - j];
				}
			}
			res[idx][k] = std::sqrt(real * real + imag * imag);
		}
	}
	return res;
}

// fastest CPU implementation with the re-implementation of the function generation
auto transform_wavelet_inplace(std::vector<float> signal, float fmin, float fmax, size_t sample_num, float sample_freq) -> std::vector<std::vector<float>> {
	float fdiff = (fmax - fmin) / sample_num;
	std::vector<std::vector<float>> res(sample_num);
	float morlet_base = 6.0;
	float freq = fmin;

	// variables for the function generation
	float wavelet_start = -4;
	float wavelet_end = 4;
	float h = 1.0 / sample_freq;
	size_t wavelet_sample_num = (wavelet_end - wavelet_start) * sample_freq;
	std::vector<float> tester_r(wavelet_sample_num);
	std::vector<float> tester_i(wavelet_sample_num);
	
	// variables for the convolution
	size_t n = signal.size();
	size_t m = wavelet_sample_num;
	// FIXME: vector operations seem to be messed up with the paralle for
	//#pragma omp parallel for schedule(dynamic)
	for (size_t i = 0; i < sample_num; i++) {
		freq = fmin + i * fdiff;
		for (size_t j = 0; j < wavelet_sample_num; j++) {
			float tx = freq * (wavelet_start + j * h);
			tester_r[j] = freq * std::cos(morlet_base * tx) * std::exp(-1 * (tx * tx) / 2);
			tester_i[j] = freq * std::sin(morlet_base * tx) * std::exp(-1 * (tx * tx) / 2);
		}


		size_t idx = sample_num - i - 1;
		res[idx] = std::vector<float>(m + n - 1);
		float real, imag;
		for (size_t k = 0; k < m + n - 1; k++) {
			real = 0.0;
			imag = 0.0;
			for (size_t j = 0; j < n; j++) {
				if (k - j < m) {
					real += signal[j] * tester_r[k - j];
					imag += signal[j] * tester_i[k - j];
				}
			}
			res[idx][k] = std::sqrt(real * real + imag * imag);
		}
	}
	return res;
}

// the oneloop version with pointer based arrays instead of C++ template data structures
// the repetitive number crunching code, that will run on the GPU
// probably device / global kernel
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

using Trafo = std::vector<std::vector<float>>;

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
	transform_array_impl(signal_raw, n, tester_r, tester_i, m, h, wavelet_start, morlet_base, fmin, fdiff, freq, sample_num, res, res_width);

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


// currently used testing code to visually verify the implementation
auto main_advanced() -> int {
	// sampling frequency
	float sample_freq = 1000;
	// domain for input signal
	auto x = sample(0, 10, sample_freq);
	// different messages for testing (message = what we modulate into a sin wave)
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
	// just a bit of noise for the base message
	for (size_t i=0;i<x.size();i++)
		message_step[i] += (std::cos(x[i]) * std::sin(70*PI*x[i])) * 0.2;

	// creating the FM signal from the message (can change the message variable)
	auto signal = freq_modulate(message_sin, 10, sample_freq, 1, 50);

	// noise for the FM signal as well
	for (size_t i=0;i<x.size();i++)
		signal[i] += (std::cos(x[i]) * std::sin(140*PI*x[i])) * 0.2;
	
	// uncomment this and run make heat to show the wavelet transform in a plot window
	auto trafo = transform_wavelet_inplace(message_sin, 1, 100, 64, sample_freq);
	plot_trafo(trafo);
	////////////////////////////////////////////////////////////////////////////

	// uncomment this and run with make plot to show the FM signal
	//auto signal_x = sample(0, signal.size() / sample_freq, sample_freq);
	//plot_func(signal_x, signal);
	////////////////////////////////////////////////////////////////////////////
	
	// uncomment this and run with make plot to show the original message 
	//auto message_x = sample(0, 5, sample_freq);
	//plot_func(message_x, message_sin);
	////////////////////////////////////////////////////////////////////////////

	return 0;
}

// test for the basic control transformer
auto main_trafo() -> int {
	float sample_freq = 128;
	// 10s long signal
	auto x = sample(0, 10, sample_freq);
	auto signal = function(x, [](float x) -> float {
		// create a 2 Hz signal with some 3Hz noise
		return std::sin(2*2*PI*x) + std::sin(4*2*PI*x);
	});

	// look for 1 - 10 Hz range with the sampe sample frequency
	auto trafo = transform_basic(signal, 1, 10, 64, sample_freq);
	//plot_trafo(trafo);
	/*
	std::cout << "trafo = [";
	for (size_t row = 0; row < trafo.size()-1; row++) {
		for (size_t col = 0; col < trafo[row].size()-1; col++)  {
			std::cout << trafo[row][col] << ",";
		}
		std::cout << trafo[row][trafo[row].size()-1] << ";";
	}
	for (size_t col = 0; col < trafo[trafo.size()-1].size()-1; col++)  {
		std::cout << trafo[trafo.size()-1][col] << ",";
	}
	std::cout << trafo[trafo.size()-1][trafo[trafo.size()-1].size()-1] << "];" << std::endl;
	std::cout << "figure" << std::endl << "imagesc(trafo)" << std::endl;
	*/

	return 0;
}

// test for validating the convolution implementation
auto main_conv() -> int {
	float sample_freq = 100;
	float signal_max = 10;
	auto x = sample(0, signal_max, sample_freq);
	// this is actually a functor application by the way
	// if we had easy functors, this would be simpler
	auto signal = function(x, [](float x) -> float {
		return std::sin(2*PI*x) + std::sin(2*2*PI*x);
	});

	float morlet_base = 2*PI;
	float target_frequency = 2;
	float wavelet_min = -4;
	float wavelet_max = 4;
	auto wavelet_x = sample(wavelet_min,wavelet_max,sample_freq);
	auto tester_r = function(wavelet_x, [target_frequency, morlet_base](float x) -> float {
		float tx = target_frequency * x;
		return std::sqrt(target_frequency) * std::cos(morlet_base * tx) * std::exp(-1 * (tx * tx) / 2);
	});
	auto tester_i = function(wavelet_x, [target_frequency, morlet_base](float x) -> float {
		float tx = target_frequency * x;
		return std::sqrt(target_frequency) * std::sin(morlet_base * tx) * std::exp(-1 * (tx * tx) / 2);
	});
	auto real = conv(signal, tester_r);
	auto imag = conv(signal, tester_i);
	std::vector<float> spectrum(real.size());
	for (size_t i=0;i<real.size();i++)
		spectrum[i] = std::sqrt(real[i] * real[i] + imag[i] * imag[i]);

	auto trafo_x = sample(0, spectrum.size() / sample_freq, sample_freq);
	plot_func(trafo_x, spectrum);
	// std::cout << trafo_x.size() << " =?= " << signal.size() + tester_r.size() - 1 << std::endl;
	/*
	std::cout << "x = " << x << ";" << std::endl 
			  << "signal = " << signal << ";" <<  std::endl 
			  << "wavelet_x = " << wavelet_x << ";" << std::endl
			  << "wavelet_r = " << tester_r << ";" << std::endl
			  << "wavelet_i = " << tester_i << ";" << std::endl
			  << "trafo_x = " << trafo_x << ";" << std::endl
			  << "real = " << real << ";" << std::endl
			  << "imag = " << imag << ";" << std::endl
			  << "spectrum = " << spectrum << ";" << std::endl;
	std::cout << "figure" << std::endl;
	std::cout << "plot(x,signal)" << std::endl;
	std::cout << "figure" << std::endl;
	std::cout << "plot(wavelet_x, wavelet_r, '-r', wavelet_x, wavelet_i, '--b')" << std::endl;
	std::cout << "figure" << std::endl;
	std::cout << "plot(trafo_x, real, '-r', trafo_x, imag, '--b')" << std::endl;
	std::cout << "figure" << std::endl;
	std::cout << "plot(trafo_x, spectrum)" << std::endl;
	*/

	//std::cout << trafo_x.size() << " =?= " << spectrum.size() << std::endl;
	return 0;
}

// test for the validation of wavelet generator
auto main_wavelets() -> int {
	float freq = 1;
	auto x = sample(-4, 4, 1000);
	auto wavelet = function(x, [freq](float x) -> float {
		float tx = freq * x;
		return std::cos(2* PI * tx) * std::exp(-1 * tx * tx / 2);
	});
	plot_func(x, wavelet);
	return 0;
}

// test for sampling
auto main_basic() -> int {
	auto x = sample(0, 10, 2);
	std::cout << x.size() << std::endl;
	return 0;
}

auto main() -> int {
	main_wavelets();
	// main_trafo();
	//main_advanced();
	// main_basic();
	// main_conv();
	return 0;
}
