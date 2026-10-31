#include <filesystem>
#include <fstream>
#include <utility>
#include <vector>

#include <logger.h>
#include <NvOnnxParser.h>

#include "model/EngineManager.h"

using BuilderPtr = std::unique_ptr<nvinfer1::IBuilder, TrtDeleter<nvinfer1::IBuilder>>;

using NetworkPtr = std::unique_ptr<nvinfer1::INetworkDefinition, TrtDeleter<nvinfer1::INetworkDefinition>>;

using ConfigPtr = std::unique_ptr<nvinfer1::IBuilderConfig, TrtDeleter<nvinfer1::IBuilderConfig>>;

using ParserPtr = std::unique_ptr<nvonnxparser::IParser, TrtDeleter<nvonnxparser::IParser>>;

using HostMemoryPtr = std::unique_ptr<nvinfer1::IHostMemory, TrtDeleter<nvinfer1::IHostMemory>>;

EngineManager::EngineManager(std::string onnx_path, std::string engine_path)
    : onnx_path_(std::move(onnx_path)), engine_path_(std::move(engine_path)) {}

bool EngineManager::loadOrBuild() {
    runtime_.reset(nvinfer1::createInferRuntime(sample::gLogger));

    if (!runtime_) {
        return false;
    }

    if (std::filesystem::exists(engine_path_)) {
        return loadSerializedEngine();
    }

    return buildEngineFromOnnx();
}

bool EngineManager::loadSerializedEngine() {
    std::ifstream file(engine_path_, std::ios::binary | std::ios::ate);

    if (!file) {
        return false;
    }

    const std::streamsize size = file.tellg();

    if (size <= 0) {
        return false;
    }

    file.seekg(0, std::ios::beg);

    std::vector<char> data(static_cast<std::size_t>(size));

    if (!file.read(data.data(), size)) {
        return false;
    }

    engine_.reset(runtime_->deserializeCudaEngine(data.data(), data.size()));

    return engine_ != nullptr;
}

bool EngineManager::buildEngineFromOnnx() {
    BuilderPtr builder{nvinfer1::createInferBuilder(sample::gLogger)};

    if (!builder) {
        return false;
    }

    const auto explicit_batch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

    NetworkPtr network{builder->createNetworkV2(explicit_batch)};

    if (!network) {
        return false;
    }

    ParserPtr parser{nvonnxparser::createParser(*network, sample::gLogger)};

    if (!parser) {
        return false;
    }

    if (!parser->parseFromFile(onnx_path_.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kINFO))) {
        return false;
    }

    ConfigPtr config{builder->createBuilderConfig()};

    if (!config) {
        return false;
    }

    if (builder->platformHasFastFp16()) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }

    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,256ULL << 20);

    engine_.reset(builder->buildEngineWithConfig(*network, *config));

    if (!engine_) {
        return false;
    }

    return serializeEngine();
}

bool EngineManager::serializeEngine() {
    HostMemoryPtr serialized_engine{engine_->serialize()};

    if (!serialized_engine) {
        return false;
    }

    std::ofstream file(engine_path_, std::ios::binary);

    if (!file) {
        return false;
    }

    file.write(static_cast<const char*>(serialized_engine->data()), serialized_engine->size());

    return file.good();
}
