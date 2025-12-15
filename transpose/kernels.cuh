# pragma once


__global__ void Transpose(const float* __ restrict__ input,
                          float* __restrict__ output,
                          int width,
                          int height);

