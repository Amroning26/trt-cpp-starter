#include <stdexcept>
#include <cmath>

#include "postprocess/PostProcessor.h"

Prediction PostProcessor::predict(const std::vector<float>& output) {
    if (output.empty()) {
        throw std::invalid_argument("Inference output is empty");
    }

    int best_index = 0;

    for (std::size_t i = 1; i < output.size(); ++i) {
        if (output[i] > output[best_index]) {
            best_index = static_cast<int>(i);
        }
    }

    float max_value = output[best_index];
    float sum = 0.0f;

    for (float value : output) {
        sum += std::exp(value - max_value);
    }

    float probability = std::exp(output[best_index] - max_value) / sum;

    return Prediction{best_index, probability};
}