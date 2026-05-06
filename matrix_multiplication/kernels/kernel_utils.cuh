#pragma once

#include <cuda_fp16.h>

template <typename InputT>
__device__ __forceinline__
float input_to_float_device(InputT x)
{
    return static_cast<float>(x);
}

template <>
__device__ __forceinline__
float input_to_float_device<__half>(__half x)
{
    return __half2float(x);
}