#include "utils.h"

auto  operator <<(std::ostream& os, std::vector<float> const& vec) -> std::ostream& {
	os << "[";
	for (size_t i=0;i<vec.size()-1;i++) {
		os << vec[i] << ", ";
	}
	os << vec[vec.size()-1];
	os << "]";
	return os;
}

auto plot_func(std::vector<float> const& x, std::vector<float> const& y) -> void {
	for (size_t i=0;i<x.size()-1;i++) {
		std::cout << x[i] << " " << y[i] << std::endl;
	}
	std::cout << std::endl;
}

auto plot_trafo(std::vector<std::vector<float>> const& data) -> void {
	for (size_t row = data.size()-1; row > 0; row--) {
		for (size_t col = 0; col < data[row].size()-1; col++)  {
			std::cout << data[row][col] << ", ";
		}
		std::cout << std::endl;
	}
}

auto sample(float start, float end, float frequency) -> std::vector<float> {
	// get number of samples from the frequency
	// frequency is sample num in a second (in this case a unit)
	size_t sample_num = (end - start) * frequency;
	float h = 1.0 / frequency;
	std::vector<float> res(sample_num);
	for (size_t i=0; i<sample_num; i++)
		res[i] = start + h * i;
	return res;
}

auto function(std::vector<float> const& dom, std::function<float(float)> fn) -> std::vector<float> {
	std::vector<float> res(dom.size());
	for (size_t i=0;i<dom.size();i++)
		res[i] = fn(dom[i]);
	return res;
}

auto freq_modulate(std::vector<float> const& message, float base_freq, float sample_freq, float amplitude = 1, float sensitivity = 1) -> std::vector<float> {
	std::vector<float> signal(message.size());
	float period = 1.0 / sample_freq;
	for (size_t n=0;n<message.size();n++) {
		float theta_m = 0;
		for (size_t k=0;k<n;k++) {
			theta_m += message[k];
		}
		signal[n] = amplitude * std::cos(2*PI*base_freq*n*period + sensitivity*theta_m*period);
	}
	return signal;
}
