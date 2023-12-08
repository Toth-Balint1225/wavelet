#pragma once

#include <iostream>
#include <functional>
#include <vector>
#include <cmath>

constexpr float PI = 3.1415926535;

auto  operator <<(std::ostream& os, std::vector<float> const& vec) -> std::ostream&;
auto plot_func(std::vector<float> const& x, std::vector<float> const& y) -> void;
auto plot_trafo(std::vector<std::vector<float>> const& data) -> void;
auto sample(float start, float end, float frequency) -> std::vector<float>;
auto function(std::vector<float> const& dom, std::function<float(float)> fn) -> std::vector<float>;
auto freq_modulate(std::vector<float> const& message, float base_freq, float sample_freq, float amplitude, float sensitivity) -> std::vector<float>;