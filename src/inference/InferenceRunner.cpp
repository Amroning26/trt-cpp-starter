#include <limits>
#include <stdexcept>
#include <string>

#include "inference/InferenceRunner.h"

namespace {

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

} // namespace

CudaStream::CudaStream() {
    checkCuda(cudaStreamCreate(&stream_), "cudaStreamCreate");
}

CudaStream::~CudaStream() {
    if (stream_ != nullptr) {
        cudaStreamDestroy(stream_);
    }
}

CudaBuffer::CudaBuffer(std::size_t bytes) {
    allocate(bytes);
}

void CudaBuffer::allocate(std::size_t bytes) {
    if (data_ != nullptr) {
        throw std::logic_error("CUDA buffer has already been allocated");
    }

    if (bytes == 0) {
        throw std::invalid_argument("CUDA buffer size must be greater than zero");
    }

    checkCuda(cudaMalloc(&data_, bytes), "cudaMalloc");
}

CudaBuffer::~CudaBuffer() {
    if (data_ != nullptr) {
        cudaFree(data_);
    }
}

int InferenceRunner::tensorSize(const nvinfer1::Dims& dims) {
    long long size = 1;

    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] <= 0) {
            throw std::runtime_error("Dynamic or invalid tensor dimension");
        }

        size *= dims.d[i];

        if (size > std::numeric_limits<int>::max()) {
            throw std::runtime_error("Tensor is too large");
        }
    }

    return static_cast<int>(size);
}

InferenceRunner::InferenceRunner(nvinfer1::ICudaEngine& engine) : engine_(engine) {
    const nvinfer1::Dims input_dims = engine_.getTensorShape(input_name_);

    const nvinfer1::Dims output_dims = engine_.getTensorShape(output_name_);

    input_size_ = tensorSize(input_dims);
    output_size_ = tensorSize(output_dims);

    context_.reset(engine_.createExecutionContext());

    if (!context_) {
        throw std::runtime_error("Failed to create TensorRT execution context");
    }

    device_input_.allocate(static_cast<std::size_t>(input_size_) * sizeof(float));

    device_output_.allocate(static_cast<std::size_t>(output_size_) * sizeof(float));

    if (!context_->setTensorAddress(input_name_, device_input_.data())) {
        throw std::runtime_error("Failed to set input tensor address");
    }

    if (!context_->setTensorAddress(output_name_, device_output_.data())) {
        throw std::runtime_error("Failed to set output tensor address");
    }
}

std::vector<float> InferenceRunner::infer(const cv::Mat& input) {
    if (input.empty()) {
        throw std::invalid_argument("Input image is empty");
    }

    if (input.depth() != CV_32F || input.channels() != 1) {
        throw std::invalid_argument("Inference input must be a single-channel CV_32F matrix");
    }

    if (!input.isContinuous()) {
        throw std::invalid_argument("Inference input must be continuous");
    }

    const std::size_t input_elements = input.total() * static_cast<std::size_t>(input.channels());

    if (input_elements != static_cast<std::size_t>(input_size_)) {
        throw std::invalid_argument("Input size does not match engine input tensor");
    }

    checkCuda(
        cudaMemcpyAsync(device_input_.data(), input.ptr<float>(), input_size_ * sizeof(float), cudaMemcpyHostToDevice, stream_.get()),
        "cudaMemcpyAsync host to device");

    if (!context_->enqueueV3(stream_.get())) {
        throw std::runtime_error("TensorRT enqueueV3 failed");
    }

    std::vector<float> output(static_cast<std::size_t>(output_size_));

    checkCuda(
        cudaMemcpyAsync(output.data(), device_output_.data(), output_size_ * sizeof(float), cudaMemcpyDeviceToHost, stream_.get()),
        "cudaMemcpyAsync device to host");

    checkCuda(cudaStreamSynchronize(stream_.get()), "cudaStreamSynchronize");

    return output;
}