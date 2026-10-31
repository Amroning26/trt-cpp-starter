# TensorRT C++ Starter

用于学习 TensorRT、CUDA、OpenCV 和 CMake 的 C++ 推理示例。
A C++ inference example for learning TensorRT, CUDA, OpenCV, and CMake.

## 环境/Requirements

- C++17
- CMake
- CUDA
- TensorRT
- OpenCV

## 编译/Build

```bash
mkdir -p build
cd build
cmake ..
cmake --build . -j
```

## 运行/Run

./trt_cpp_starter

注意：当前代码中的路径是相对于 `build` 目录写的，所以需要从 `build` 目录运行程序。
Note: The paths in the current code are relative to the `build` directory, so the program must be run from that directory.