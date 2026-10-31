cmake_minimum_required(VERSION 3.18)

project(trt_cpp_starter LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
find_package(CUDAToolkit REQUIRED)

set(TENSORRT_ROOT "" CACHE PATH "TensorRT installation prefix")
set(TENSORRT_SAMPLES_ROOT "" CACHE PATH "TensorRT samples/common directory")

find_path(TENSORRT_INCLUDE_DIR
    NAMES NvInfer.h
    HINTS
        ${TENSORRT_ROOT}/include
    PATH_SUFFIXES
        x86_64-linux-gnu
)

find_path(TENSORRT_SAMPLES_INCLUDE_DIR
    NAMES logger.h
    HINTS
        ${TENSORRT_SAMPLES_ROOT}
        ${TENSORRT_ROOT}/samples/common
        /usr/src/tensorrt/samples/common
)

find_library(TENSORRT_LIBRARY
    NAMES nvinfer
    HINTS
        ${TENSORRT_ROOT}/lib
        ${TENSORRT_ROOT}/lib64
    PATH_SUFFIXES
        x86_64-linux-gnu
)

find_library(TENSORRT_ONNX_PARSER_LIBRARY
    NAMES nvonnxparser
    HINTS
        ${TENSORRT_ROOT}/lib
        ${TENSORRT_ROOT}/lib64
    PATH_SUFFIXES
        x86_64-linux-gnu
)

if(NOT TENSORRT_INCLUDE_DIR OR
   NOT TENSORRT_SAMPLES_INCLUDE_DIR OR
   NOT TENSORRT_LIBRARY OR
   NOT TENSORRT_ONNX_PARSER_LIBRARY)
    message(FATAL_ERROR "TensorRT or CUDA development files were not found")
endif()

find_file(TENSORRT_LOGGER_SOURCE
    NAMES logger.cpp
    HINTS ${TENSORRT_SAMPLES_INCLUDE_DIR}
)

if(NOT TENSORRT_LOGGER_SOURCE)
    message(FATAL_ERROR "TensorRT sample logger.cpp was not found")
endif()

add_executable(trt_cpp_starter
    main.cpp
    src/model/EngineManager.cpp
    src/preprocess/ImagePreprocessor.cpp
    src/inference/InferenceRunner.cpp
    src/postprocess/PostProcess.cpp
    ${TENSORRT_LOGGER_SOURCE}
)

target_include_directories(trt_cpp_starter
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
        ${TENSORRT_INCLUDE_DIR}
        ${TENSORRT_SAMPLES_INCLUDE_DIR}
)

target_link_libraries(trt_cpp_starter
    PRIVATE
        ${OpenCV_LIBS}
        ${TENSORRT_LIBRARY}
        ${TENSORRT_ONNX_PARSER_LIBRARY}
        CUDA::cudart
)

