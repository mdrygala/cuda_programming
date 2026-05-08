#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include "config.h"
#include "kernel_utils.cuh"

#ifndef TILE
#define TILE 32
#endif

template <typename InputT>
__global__ void GEMMTiling(int M, int N, int K,
                           float alpha,
                           const InputT* __restrict__ A,
                           const InputT* __restrict__ B,
                           float beta,
                           float* __restrict__ C)
{
    __shared__ InputT ATile[TILE][TILE];
    __shared__ InputT BTile[TILE][TILE];

    int startRow = blockIdx.y * TILE;
    int startCol = blockIdx.x * TILE;

    int threadRowGlobal = startRow + threadIdx.y;
    int threadColGlobal = startCol + threadIdx.x;

    float sum = 0.0f;

    for (int chunk = 0; chunk < K; chunk += TILE){
        int rowA = threadRowGlobal;
        int colA = chunk + threadIdx.x;

        int rowB = chunk + threadIdx.y;
        int colB = threadColGlobal;

        ATile[threadIdx.y][threadIdx.x] =
            (rowA < M && colA < K) ? A[rowA * K + colA] : InputT(0);

        BTile[threadIdx.y][threadIdx.x] =
            (rowB < K && colB < N) ? B[rowB * N + colB] : InputT(0);

        __syncthreads();

        int kmax = min(TILE, K - chunk);
        #pragma unroll
        for (int k = 0; k < kmax; k++){
            float a = input_to_float_device<InputT>(ATile[threadIdx.y][k]);
            float b = input_to_float_device<InputT>(BTile[k][threadIdx.x]);
            sum = fmaf(a, b, sum);
        }
        __syncthreads();
    }

    if (threadRowGlobal < M && threadColGlobal < N){
        int idx = threadRowGlobal * N + threadColGlobal;
        float cold = (beta != 0.0f) ? C[idx] : 0.0f;
        C[idx] = alpha * sum + beta * cold;
    }
}