#include <cuda_runtime.h>
#include <cstdio>
#include "config.h"
#include "kernels.cuh"

// -------------------- Baseline --------------------
__global__ void GEMMBaseline(int M, int N, int K,
                             float alpha,
                             const float* __restrict__ A,
                             const float* __restrict__ B,
                             float beta,
                             float* __restrict__ C)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; k++){
        sum = fmaf(A[row * K + k], B[k * N + col], sum);
    }

    int idx = row * N + col;
    float cold = (beta != 0.0f) ? C[idx] : 0.0f;
    C[idx] = alpha * sum + beta * cold;   // FIX: don’t read C twice
}