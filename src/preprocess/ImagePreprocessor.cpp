#include <stdexcept>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "preprocess/ImagePreprocessor.h"

ImagePreprocessor::ImagePreprocessor(cv::Size input_size, float scale) 
    : input_size_(input_size), scale_(scale) {}

cv::Mat ImagePreprocessor::process(const std::string& image_path) const {
    cv::Mat image = cv::imread(image_path, cv::IMREAD_UNCHANGED);

    if (image.empty()) {
        throw std::runtime_error("Failed to load image: " + image_path);
    }

    cv::Mat gray;

    if (image.channels() == 1) {
        gray = image;
    } 
    else if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } 
    else if (image.channels() == 4) {
        cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
    } 
    else {
        throw std::runtime_error("Unsupported image channel count");
    }

    cv::Mat resized;
    cv::resize(gray, resized, input_size_);

    cv::Mat result;
    resized.convertTo(result, CV_32FC1, scale_);

    return result;
}

std::vector<float> ImagePreprocessor::hwc2chw(const cv::Mat& image) {
    if (image.empty()) {
        throw std::invalid_argument("Cannot convert an empty image");
    }

    cv::Mat float_image;

    if (image.depth() == CV_32F) {
        float_image = image;
    } 
    else {
        image.convertTo(float_image, CV_MAKETYPE(CV_32F, image.channels()));
    }

    std::vector<cv::Mat> channels;
    cv::split(float_image, channels);

    std::vector<float> result;
    result.reserve(static_cast<std::size_t>(image.rows * image.cols * image.channels()));

    for (const cv::Mat& channel : channels) {
        if (!channel.isContinuous()) {
            cv::Mat continuous_channel = channel.clone();

            result.insert(
                result.end(),
                continuous_channel.ptr<float>(),
                continuous_channel.ptr<float>() + continuous_channel.total());
        } 
        else {
            result.insert(
                result.end(),
                channel.ptr<float>(),
                channel.ptr<float>() + channel.total());
        }
    }

    return result;
}