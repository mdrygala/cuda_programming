#include <cuda_runtime.h>
#include <cstdio>
#include "config.h"
#include "kernel_utils.cuh"

#ifndef BASELINE_TILE
#define BASELINE_TILE 32
#endif

// ================ BEST BASELINE CONFIGS ================

// DATATYPE:   float
// CONFIG:     datatype=float BASELINE_TILE=32 THREADS=1024
// TFLOPS:     3.00
// Efficiency: 15.41%

// DATATYPE:   half
// CONFIG:     datatype=half BASELINE_TILE=32 THREADS=1024
// TFLOPS:     3.09
// Efficiency: 15.84%
// =======================================================

template <typename InputT>
__global__ void GEMMBaseline(int M, int N, int K,
                             float alpha,
                             const InputT* __restrict__ A,
                             const InputT* __restrict__ B,
                             float beta,
                             float* __restrict__ C)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; k++){
        float a = input_to_float_device<InputT>(A[row * K + k]);
        float b = input_to_float_device<InputT>(B[k * N + col]);
        sum = fmaf(a, b, sum);
    }

    int idx = row * N + col;
    float cold = (beta != 0.0f) ? C[idx] : 0.0f;
    C[idx] = alpha * sum + beta * cold;   // FIX: don’t read C twice
}