#pragma once
#include <cuda_runtime.h>
#include <cstdio>

#include "config.h"
#include "kernel_utils.cuh"
#include "param_init.cuh"
#include "helpers/load_helpers.cuh"
#include "helpers/compute_helpers.cuh"
#include "helpers/store_helpers.cuh"


// DATATYPE:   half
// CONFIG:     datatype=half TILE_WARP_LOAD_M=64 TILE_WARP_LOAD_N=128 TILE_WARP_LOAD_K=16 THREAD_DIM_WARP_LOAD=8 PADDING_WARP_LOAD=0 THREADS=128
// TFLOPS:     15.94
// Efficiency: 81.78%

#ifndef TILE_WARP_LOAD_M
#define TILE_WARP_LOAD_M 64
#endif

#ifndef TILE_WARP_LOAD_N
#define TILE_WARP_LOAD_N 64
#endif


#ifndef TILE_WARP_LOAD_K
#define TILE_WARP_LOAD_K 32
#endif

#ifndef PADDING_WARP_LOAD
#define PADDING_WARP_LOAD 0
#endif


#ifndef THREAD_DIM_WARP_LOAD
#define THREAD_DIM_WARP_LOAD 4
#endif

#define NUM_THREADS_WARP_LOAD_M (TILE_WARP_LOAD_M / THREAD_DIM_WARP_LOAD)
#define NUM_THREADS_WARP_LOAD_N (TILE_WARP_LOAD_N / THREAD_DIM_WARP_LOAD)
#define NUM_THREADS_PER_BLOCK_WARP_LOAD (NUM_THREADS_WARP_LOAD_M * NUM_THREADS_WARP_LOAD_N)
#define NUM_WARPS_PER_BLOCK_WARP_LOAD (NUM_THREADS_PER_BLOCK_WARP_LOAD / 32)

template <typename InputT>
__global__
void GEMMSubTilingLoadSlabGenDims(int M, int N, int K,
                          float alpha,
                          const InputT* __restrict__ A,
                          const InputT* __restrict__ B,
                          float beta,
                          float* __restrict__ C)
{
    __shared__ InputT ATile[TILE_WARP_LOAD_M][TILE_WARP_LOAD_K + PADDING_WARP_LOAD];
    __shared__ InputT BTile[TILE_WARP_LOAD_K][TILE_WARP_LOAD_N + PADDING_WARP_LOAD];

    int startRow = blockIdx.y * TILE_WARP_LOAD_M;
    int startCol = blockIdx.x * TILE_WARP_LOAD_N;

    int threadRowTile = threadIdx.y * THREAD_DIM_WARP_LOAD;
    int threadColTile = threadIdx.x * THREAD_DIM_WARP_LOAD;

    SlabParamsGenDim params = make_linear_slab_params_gendim<InputT,  TILE_WARP_LOAD_M, TILE_WARP_LOAD_N, TILE_WARP_LOAD_K>();

    float sum[THREAD_DIM_WARP_LOAD][THREAD_DIM_WARP_LOAD];
    #pragma unroll
    for (int i = 0; i < THREAD_DIM_WARP_LOAD; i++) {
        #pragma unroll
        for (int j = 0; j < THREAD_DIM_WARP_LOAD; j++) {
            sum[i][j] = 0.0f;
        }
    }

    for (int chunk = 0; chunk < K; chunk += TILE_WARP_LOAD_K) {
        load_subtile_warp<InputT, NUM_WARPS_PER_BLOCK_WARP_LOAD, PADDING_WARP_LOAD, TILE_WARP_LOAD_M, TILE_WARP_LOAD_N, TILE_WARP_LOAD_K>(
            A, ATile,
            B, BTile,
            M, K, N,
            startRow, startCol,
            chunk, params
        );
        __syncthreads();

        int kmax = min(TILE_WARP_LOAD_K, K - chunk);
        compute_subtile<InputT, TILE_WARP_LOAD_M, TILE_WARP_LOAD_N, TILE_WARP_LOAD_K, THREAD_DIM_WARP_LOAD, PADDING_WARP_LOAD>(
            ATile, BTile,
            kmax,
            sum,
            threadRowTile, threadColTile
        );
        __syncthreads();
    }

    store_subtile_vec4<THREAD_DIM_WARP_LOAD>(
        sum, C, M, N,
        startRow, startCol,
        threadRowTile, threadColTile,
        alpha, beta
    );
}


