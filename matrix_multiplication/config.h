#pragma once


#include <string>

enum class DataType {
    Float32,
    Float16
};

enum class ComputeRoof {
    FP32CudaCores,
    FP16TensorCores
};

struct Config {
    DataType data_type = DataType::Float32;

    // This describes which hardware/instruction roof to compare against.
    // Most of your kernels use FP32CudaCores, even when InputT = __half,
    // because they convert to float and use fmaf.
    ComputeRoof compute_roof = ComputeRoof::FP32CudaCores;
};