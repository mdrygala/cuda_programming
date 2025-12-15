#pragma once


__global__ void SumReductionBaseline(const float* __restrict__ input,
                                  float* __restrict__ output,
                                  int N);


__global__ void SumReductionWarpShuffle(const float* input,
                                  float* blockResults,
                                  int N);