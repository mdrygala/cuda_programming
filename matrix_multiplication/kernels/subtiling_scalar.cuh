#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include "config.h"
#include "kernel_utils.cuh"
// #include "compute_helpers.cuh"
#include "store_helpers.cuh"

template <typename InputT>
__device__ __forceinline__
void load_subtile_naive(const InputT* __restrict__ A,
                        InputT ATile[SUBTILE][SUBTILE+PADDING],
                        const InputT* __restrict__ B,
                        InputT BTile[SUBTILE][SUBTILE+PADDING],
                        int M, int K, int N,
                        int startRow, int startCol, int chunk,
                        int threadRowTile, int threadColTile)
{
    int threadRowGlobalOriginA = startRow;
    int threadColGlobalOriginA = chunk;

    int threadRowGlobalOriginB = chunk;
    int threadColGlobalOriginB = startCol;

    #pragma unroll
    for (int i = 0; i < SUB; i++){
        int rowTile = threadRowTile + i;
        int rowA = threadRowGlobalOriginA + rowTile;
        int rowB = threadRowGlobalOriginB + rowTile;

        #pragma unroll
        for (int j = 0; j < SUB; j++){
            int colTile = threadColTile + j;
            int colA = threadColGlobalOriginA + colTile;
            int colB = threadColGlobalOriginB + colTile;
        

            //Loads into shared memory, in 
            ATile[rowTile][colTile] = (rowA < M && colA < K) ? A[rowA * K + colA] : InputT(0);
            BTile[rowTile][colTile] = (rowB < K && colB < N) ? B[rowB * N + colB] : InputT(0);
        }
    }
}


template <typename InputT>
__device__ __forceinline__
void compute_subtile_temp(const InputT ATile[SUBTILE][SUBTILE+PADDING],
                     const InputT BTile[SUBTILE][SUBTILE+PADDING],
                     int K, int kmax,
                     float sum[SUB][SUB], int threadRowTile, int threadColTile)
{
              
    #pragma unroll
    for (int k = 0; k < kmax; k++){
        float AReg[SUB];
        float BReg[SUB];
        #pragma unroll
        for (int i=0; i < SUB; i++){
            AReg[i] = input_to_float_device<InputT>(ATile[threadRowTile + i][k]);
        }
        #pragma unroll
        for (int j=0; j < SUB; j++){
            BReg[j] = input_to_float_device<InputT>(BTile[k][threadColTile + j]);
        }

        #pragma unroll
        for (int i = 0; i < SUB; i++){
            #pragma unroll
            for (int j = 0; j < SUB; j++){
                sum[i][j] = fmaf(AReg[i], BReg[j], sum[i][j]);
            }

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
    __shared__ InputT ATile[SUBTILE][SUBTILE + PADDING];
    __shared__ InputT BTile[SUBTILE][SUBTILE + PADDING];

    int startRow = blockIdx.y * SUBTILE;
    int startCol = blockIdx.x * SUBTILE;

    int threadRowTile = threadIdx.y * SUB;
    int threadColTile = threadIdx.x * SUB;

    float sum[SUB][SUB];
    #pragma unroll
    for (int i = 0; i < SUB; i++) {
        #pragma unroll
        for (int j = 0; j < SUB; j++) {
            sum[i][j] = 0.0f;
        }
    }

    for (int chunk = 0; chunk < K; chunk += SUBTILE) {
        load_subtile_naive<InputT>(
            A, ATile,
            B, BTile,
            M, K, N,
            startRow, startCol,
            chunk, threadRowTile, threadColTile
        );
        __syncthreads();

        int kmax = min(SUBTILE, K - chunk);
        compute_subtile_temp<InputT>(
            ATile, BTile,
            K, kmax,
            sum,
            threadRowTile, threadColTile
        );
        __syncthreads();
    }

    store_subtile_scalar(
        sum, C, M, N,
        startRow, startCol,
        threadRowTile, threadColTile,
        alpha, beta
    );
}
