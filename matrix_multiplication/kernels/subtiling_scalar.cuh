#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include "config.h"
#include "kernel_utils.cuh"
#include "helpers/compute_helpers.cuh"
#include "helpers/store_helpers.cuh"

#ifndef TILE_REGISTER_SCALAR
#define TILE_REGISTER_SCALAR 64
#endif

#ifndef THREAD_DIM_REGISTER_SCALAR
#define THREAD_DIM_REGISTER_SCALAR 4
#endif

#ifndef PADDING_REGISTER_SCALAR
#define PADDING_REGISTER_SCALAR 0
#endif

template <typename InputT>
__device__ __forceinline__
void load_subtile_naive(const InputT* __restrict__ A,
                        InputT ATile[TILE_REGISTER_SCALAR][TILE_REGISTER_SCALAR+PADDING_REGISTER_SCALAR],
                        const InputT* __restrict__ B,
                        InputT BTile[TILE_REGISTER_SCALAR][TILE_REGISTER_SCALAR+PADDING_REGISTER_SCALAR],
                        int M, int K, int N,
                        int startRow, int startCol, int chunk,
                        int threadRowTile, int threadColTile)
{
    int threadRowGlobalOriginA = startRow;
    int threadColGlobalOriginA = chunk;

    int threadRowGlobalOriginB = chunk;
    int threadColGlobalOriginB = startCol;

    #pragma unroll
    for (int i = 0; i < THREAD_DIM_REGISTER_SCALAR; i++){
        int rowTile = threadRowTile + i;
        int rowA = threadRowGlobalOriginA + rowTile;
        int rowB = threadRowGlobalOriginB + rowTile;

        #pragma unroll
        for (int j = 0; j < THREAD_DIM_REGISTER_SCALAR; j++){
            int colTile = threadColTile + j;
            int colA = threadColGlobalOriginA + colTile;
            int colB = threadColGlobalOriginB + colTile;
        

            //Loads into shared memory,
            ATile[rowTile][colTile] = (rowA < M && colA < K) ? A[rowA * K + colA] : InputT(0);
            BTile[rowTile][colTile] = (rowB < K && colB < N) ? B[rowB * N + colB] : InputT(0);
        }
    }
}


template <typename InputT>
__global__
void GEMMSubTilingScalar(int M, int N, int K,
                          float alpha,
                          const InputT* __restrict__ A,
                          const InputT* __restrict__ B,
                          float beta,
                          float* __restrict__ C)
{
    __shared__ InputT ATile[TILE_REGISTER_SCALAR][TILE_REGISTER_SCALAR + PADDING_REGISTER_SCALAR];
    __shared__ InputT BTile[TILE_REGISTER_SCALAR][TILE_REGISTER_SCALAR + PADDING_REGISTER_SCALAR];

    int startRow = blockIdx.y * TILE_REGISTER_SCALAR;
    int startCol = blockIdx.x * TILE_REGISTER_SCALAR;

    int threadRowTile = threadIdx.y * THREAD_DIM_REGISTER_SCALAR;
    int threadColTile = threadIdx.x * THREAD_DIM_REGISTER_SCALAR;

    float sum[THREAD_DIM_REGISTER_SCALAR][THREAD_DIM_REGISTER_SCALAR];
    #pragma unroll
    for (int i = 0; i < THREAD_DIM_REGISTER_SCALAR; i++) {
        #pragma unroll
        for (int j = 0; j < THREAD_DIM_REGISTER_SCALAR; j++) {
            sum[i][j] = 0.0f;
        }
    }

    for (int chunk = 0; chunk < K; chunk += TILE_REGISTER_SCALAR) {
        load_subtile_naive<InputT>(
            A, ATile,
            B, BTile,
            M, K, N,
            startRow, startCol,
            chunk, threadRowTile, threadColTile
        );
        __syncthreads();

        int kmax = min(TILE_REGISTER_SCALAR, K - chunk);
        compute_subtile<InputT, TILE_REGISTER_SCALAR, TILE_REGISTER_SCALAR, TILE_REGISTER_SCALAR, THREAD_DIM_REGISTER_SCALAR, PADDING_REGISTER_SCALAR>(
            ATile, BTile,
            kmax,
            sum,
            threadRowTile, threadColTile
        );
        __syncthreads();
    }

    store_subtile_scalar<THREAD_DIM_REGISTER_SCALAR>(
        sum, C, M, N,
        startRow, startCol,
        threadRowTile, threadColTile,
        alpha, beta
    );
}
