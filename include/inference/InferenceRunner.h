#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/core.hpp>

struct TrtContextDeleter {
    void operator()(nvinfer1::IExecutionContext* context) const {
        delete context;
    }
};

class CudaStream {
public:
    CudaStream();
    ~CudaStream();

    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;

    cudaStream_t get() const noexcept {
        return stream_;
    }

private:
    cudaStream_t stream_{nullptr};
};

class CudaBuffer {
public:
    CudaBuffer() = default;
    explicit CudaBuffer(std::size_t bytes);

    ~CudaBuffer();

    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;

    void allocate(std::size_t bytes);

    void* data() const noexcept {
        return data_;
    }

private:
    void* data_{nullptr};
};

class InferenceRunner {
public:
    explicit InferenceRunner(nvinfer1::ICudaEngine& engine);

    InferenceRunner(const InferenceRunner&) = delete;
    InferenceRunner& operator=(const InferenceRunner&) = delete;

    std::vector<float> infer(const cv::Mat& input);

private:
    static int tensorSize(const nvinfer1::Dims& dims);

private:
    nvinfer1::ICudaEngine& engine_;

    const char* input_name_{"Input3"};
    const char* output_name_{"Plus214_Output_0"};

    int input_size_{0};
    int output_size_{0};

    std::unique_ptr<nvinfer1::IExecutionContext, TrtContextDeleter> context_;

    CudaStream stream_;
    CudaBuffer device_input_;
    CudaBuffer device_output_;
};