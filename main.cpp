#include <exception>
#include <iostream>
#include <iomanip>

#include "model/EngineManager.h"
#include "inference/InferenceRunner.h"
#include "postprocess/PostProcessor.h"
#include "preprocess/ImagePreprocessor.h"

int main() {
    try {
        EngineManager manager("../mnist.onnx", "../mnist.engine");

        if (!manager.loadOrBuild()) {
            std::cerr << "Failed to load or build engine\n";
            return 1;
        }

        ImagePreprocessor preprocessor;
        cv::Mat input = preprocessor.process("../num.jpg");

        InferenceRunner runner(*manager.engine());
        std::vector<float> output = runner.infer(input);

        Prediction prediction = PostProcessor::predict(output);

        std::cout << std::fixed << std::setprecision(4);
        std::cout << "Predicted digit: " << prediction.label << '\n';
        std::cout << "Probability: " << prediction.probability * 100.0f << "%\n";

        return 0;
    } 
    catch (const std::exception& error) {
        std::cerr << "Inference failed: " << error.what() << '\n';
        return 1;
    }
}