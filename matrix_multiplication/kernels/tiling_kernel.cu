#include <cuda_runtime.h>
#include <cstdio>
#include "config.h"
#include "kernels.cuh"



// -------------------- Tiling --------------------
__global__ void GEMMTiling(int M, int N, int K,
                           float alpha,
                           const float* __restrict__ A,
                           const float* __restrict__ B,
                           float beta,
                           float* __restrict__ C)
{
    __shared__ float ATile[TILE][TILE];
    __shared__ float BTile[TILE][TILE];

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
            (rowA < M && colA < K) ? A[rowA * K + colA] : 0.0f;

        BTile[threadIdx.y][threadIdx.x] =
            (rowB < K && colB < N) ? B[rowB * N + colB] : 0.0f;

        __syncthreads();

        int kmax = min(TILE, K - chunk);
        #pragma unroll
        for (int k = 0; k < kmax; k++){
            sum = fmaf(ATile[threadIdx.y][k], BTile[k][threadIdx.x], sum);
        }
        __syncthreads();
    }

    if (threadRowGlobal < M && threadColGlobal < N){
        int idx = threadRowGlobal * N + threadColGlobal;
        float cold = (beta != 0.0f) ? C[idx] : 0.0f;
        C[idx] = alpha * sum + beta * cold;
    }
}