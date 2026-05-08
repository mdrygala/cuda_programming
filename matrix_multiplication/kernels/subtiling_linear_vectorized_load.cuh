#pragma once
#include <cuda_runtime.h>
#include <cstdio>

#include "config.h"
#include "kernel_utils.cuh"
#include "param_init.cuh"
#include "helpers/load_helpers.cuh"
#include "helpers/compute_helpers.cuh"
#include "helpers/store_helpers.cuh"


// ================ BEST LINEAR VECTORIZED LOAD CONFIGS ================

// DATATYPE:   float
// CONFIG:     datatype=float TILE_LINEAR_LOAD_M=64 TILE_LINEAR_LOAD_N=64 TILE_LINEAR_LOAD_K=64 THREAD_DIM_LINEAR_LOAD=4 PADDING_LINEAR_LOAD=0 THREADS=256
// TFLOPS:     15.89
// Efficiency: 81.50%

// DATATYPE:   half
// CONFIG:     datatype=half TILE_LINEAR_LOAD_M=32 TILE_LINEAR_LOAD_N=128 TILE_LINEAR_LOAD_K=16 THREAD_DIM_LINEAR_LOAD=8 PADDING_LINEAR_LOAD=0 THREADS=64
// TFLOPS:     16.15
// Efficiency: 82.85%
// =====================================================================


#ifndef TILE_LINEAR_LOAD_M
#define TILE_LINEAR_LOAD_M 64
#endif

#ifndef TILE_LINEAR_LOAD_N
#define TILE_LINEAR_LOAD_N 64
#endif


#ifndef TILE_LINEAR_LOAD_K
#define TILE_LINEAR_LOAD_K 32
#endif

#ifndef PADDING_LINEAR_LOAD
#define PADDING_LINEAR_LOAD 0
#endif


#ifndef THREAD_DIM_LINEAR_LOAD
#define THREAD_DIM_LINEAR_LOAD 4
#endif

#define NUM_THREADS_LINEAR_LOAD_M (TILE_LINEAR_LOAD_M / THREAD_DIM_LINEAR_LOAD)
#define NUM_THREADS_LINEAR_LOAD_N (TILE_LINEAR_LOAD_N / THREAD_DIM_LINEAR_LOAD)
#define NUM_THREADS_PER_BLOCK_LINEAR_LOAD (NUM_THREADS_LINEAR_LOAD_M * NUM_THREADS_LINEAR_LOAD_N)
#define NUM_WARPS_PER_BLOCK_LINEAR_LOAD (NUM_THREADS_PER_BLOCK_LINEAR_LOAD / 32)

template <typename InputT>
__global__
void GEMMSubTilingLinearVectorized(int M, int N, int K,
                          float alpha,
                          const InputT* __restrict__ A,
                          const InputT* __restrict__ B,
                          float beta,
                          float* __restrict__ C)
{
    __shared__ InputT ATile[TILE_LINEAR_LOAD_M][TILE_LINEAR_LOAD_K + PADDING_LINEAR_LOAD];
    __shared__ InputT BTile[TILE_LINEAR_LOAD_K][TILE_LINEAR_LOAD_N + PADDING_LINEAR_LOAD];

    int startRow = blockIdx.y * TILE_LINEAR_LOAD_M;
    int startCol = blockIdx.x * TILE_LINEAR_LOAD_N;

    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    int threadRowTile = threadIdx.y * THREAD_DIM_LINEAR_LOAD;
    int threadColTile = threadIdx.x * THREAD_DIM_LINEAR_LOAD;


    float sum[THREAD_DIM_LINEAR_LOAD][THREAD_DIM_LINEAR_LOAD];
    #pragma unroll
    for (int i = 0; i < THREAD_DIM_LINEAR_LOAD; i++) {
        #pragma unroll
        for (int j = 0; j < THREAD_DIM_LINEAR_LOAD; j++) {
            sum[i][j] = 0.0f;
        }
    }

    for (int chunk = 0; chunk < K; chunk += TILE_LINEAR_LOAD_K) {
        load_subtile_linear<InputT, NUM_THREADS_PER_BLOCK_LINEAR_LOAD, PADDING_LINEAR_LOAD, TILE_LINEAR_LOAD_M, TILE_LINEAR_LOAD_N, TILE_LINEAR_LOAD_K>(
            A, ATile,
            B, BTile,
            M, K, N,
            startRow, startCol,
            chunk, tid
        );
        __syncthreads();

        int kmax = min(TILE_LINEAR_LOAD_K, K - chunk);
        compute_subtile<InputT, TILE_LINEAR_LOAD_M, TILE_LINEAR_LOAD_N, TILE_LINEAR_LOAD_K, THREAD_DIM_LINEAR_LOAD, PADDING_LINEAR_LOAD>(
            ATile, BTile,
            kmax,
            sum,
            threadRowTile, threadColTile
        );
        __syncthreads();
    }

    store_subtile_vec4<THREAD_DIM_LINEAR_LOAD>(
        sum, C, M, N,
        startRow, startCol,
        threadRowTile, threadColTile,
        alpha, beta
    );
}


