#pragma once
#include <cuda_runtime.h>
#include "config.h"
#include "param_init.cuh"
#include "load_helpers.cuh"
#include "compute_helpers.cuh"
#include "store_helpers.cuh"


__global__
void GEMMSubTiling(int M, int N, int K,
                          float alpha,
                          const float* __restrict__ A,
                          const float* __restrict__ B,
                          float beta,
                          float* __restrict__ C);
