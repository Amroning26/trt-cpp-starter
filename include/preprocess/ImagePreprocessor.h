#pragma once

#include <string>
#include <vector>

#include <opencv2/core.hpp>

class ImagePreprocessor {
public:
    explicit ImagePreprocessor(cv::Size input_size = cv::Size(28, 28), float scale = 1.0f / 255.0f);

    cv::Mat process(const std::string& image_path) const;

    static std::vector<float> hwc2chw(const cv::Mat& image);

private:
    cv::Size input_size_;
    float scale_;
};