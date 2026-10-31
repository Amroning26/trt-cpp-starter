#pragma once

#include <memory>
#include <string>

#include <NvInfer.h>

template <typename T>
struct TrtDeleter {
    void operator()(T* object) const {
        delete object;
    }
};

class EngineManager {
public:
    EngineManager(std::string onnx_path, std::string engine_path);

    EngineManager(const EngineManager&) = delete;
    EngineManager& operator=(const EngineManager&) = delete;

    bool loadOrBuild();

    nvinfer1::ICudaEngine* engine() const noexcept {
        return engine_.get();
    }

private:
    bool loadSerializedEngine();
    bool buildEngineFromOnnx();
    bool serializeEngine();

private:
    std::string onnx_path_;
    std::string engine_path_;

    std::unique_ptr<nvinfer1::IRuntime, TrtDeleter<nvinfer1::IRuntime>> runtime_;

    std::unique_ptr<nvinfer1::ICudaEngine, TrtDeleter<nvinfer1::ICudaEngine>> engine_;
};