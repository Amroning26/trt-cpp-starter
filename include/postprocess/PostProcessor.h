#pragma once

#include <vector>

struct Prediction {
    int label;
    float probability;
};

class PostProcessor {
public:
    static Prediction predict(const std::vector<float>& output);
};