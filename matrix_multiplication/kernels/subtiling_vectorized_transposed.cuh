#include <cuda_runtime.h>
#include <cstdio>
#include "config.h"
#include "kernels.cuh"
#include "helpers/load_helpers.cuh"
#include "helpers/compute_helpers.cuh"
#include "helpers/store_helpers.cuh"

// ================ BEST REGISTER VEC4 TRANSPOSED CONFIGS ================

// DATATYPE:   float
// CONFIG:     datatype=float TILE_REGISTER_VEC_TRANSPOSED_M=64 TILE_REGISTER_VEC_TRANSPOSED_N=128 TILE_REGISTER_VEC_TRANSPOSED_K=16 THREAD_DIM_REGISTER_VEC_TRANSPOSED=8 PADDING_REGISTER_VEC_TRANSPOSED=0 THREADS=128
// TFLOPS:     16.29
// Efficiency: 83.60%

// DATATYPE:   half
// CONFIG:     datatype=half TILE_REGISTER_VEC_TRANSPOSED_M=64 TILE_REGISTER_VEC_TRANSPOSED_N=128 TILE_REGISTER_VEC_TRANSPOSED_K=16 THREAD_DIM_REGISTER_VEC_TRANSPOSED=8 PADDING_REGISTER_VEC_TRANSPOSED=8 THREADS=128
// TFLOPS:     17.43
// Efficiency: 89.43%
// =======================================================================

#ifndef TILE_REGISTER_VEC_TRANSPOSED_M
#define TILE_REGISTER_VEC_TRANSPOSED_M 64
#endif

#ifndef TILE_REGISTER_VEC_TRANSPOSED_N
#define TILE_REGISTER_VEC_TRANSPOSED_N 128
#endif


#ifndef TILE_REGISTER_VEC_TRANSPOSED_K
#define TILE_REGISTER_VEC_TRANSPOSED_K 16
#endif

#ifndef PADDING_REGISTER_VEC_TRANSPOSED
#define PADDING_REGISTER_VEC_TRANSPOSED 0
#endif

#ifndef THREAD_DIM_REGISTER_VEC_TRANSPOSED
#define THREAD_DIM_REGISTER_VEC_TRANSPOSED 8
#endif


#define NUM_THREADS_REGISTER_VEC_TRANSPOSED_M (TILE_REGISTER_VEC_TRANSPOSED_M / THREAD_DIM_REGISTER_VEC_TRANSPOSED)
#define NUM_THREADS_REGISTER_VEC_TRANSPOSED_N (TILE_REGISTER_VEC_TRANSPOSED_N / THREAD_DIM_REGISTER_VEC_TRANSPOSED)
#define NUM_THREADS_PER_BLOCK_REGISTER_VEC_TRANSPOSED (NUM_THREADS_REGISTER_VEC_TRANSPOSED_M * NUM_THREADS_REGISTER_VEC_TRANSPOSED_N)
#define NUM_WARPS_PER_BLOCK_REGISTER_VEC_TRANSPOSED (NUM_THREADS_PER_BLOCK_REGISTER_VEC_TRANSPOSED / 32)



template<typename InputT>
__global__
void GEMMSubTilingVec4Transposed(int M, int N, int K,
                          float alpha,
                          const InputT* __restrict__ A,
                          const InputT* __restrict__ B,
                          float beta,
                          float* __restrict__ C)
{
    __shared__ InputT ATileT[TILE_REGISTER_VEC_TRANSPOSED_K]
                           [TILE_REGISTER_VEC_TRANSPOSED_M + PADDING_REGISTER_VEC_TRANSPOSED];
    __shared__ InputT BTile[TILE_REGISTER_VEC_TRANSPOSED_K]
                          [TILE_REGISTER_VEC_TRANSPOSED_N + PADDING_REGISTER_VEC_TRANSPOSED];

    int startRow = blockIdx.y * TILE_REGISTER_VEC_TRANSPOSED_M;
    int startCol = blockIdx.x *TILE_REGISTER_VEC_TRANSPOSED_N;

    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    int threadRowTile = threadIdx.y * THREAD_DIM_REGISTER_VEC_TRANSPOSED;
    int threadColTile = threadIdx.x * THREAD_DIM_REGISTER_VEC_TRANSPOSED;

    float sum[THREAD_DIM_REGISTER_VEC_TRANSPOSED][THREAD_DIM_REGISTER_VEC_TRANSPOSED];
    #pragma unroll
    for (int i = 0; i < THREAD_DIM_REGISTER_VEC_TRANSPOSED; i++) {
        #pragma unroll
        for (int j = 0; j < THREAD_DIM_REGISTER_VEC_TRANSPOSED; j++) {
            sum[i][j] = 0.0f;
        }
    }

    for (int chunk = 0; chunk < K; chunk += TILE_REGISTER_VEC_TRANSPOSED_K) {
        load_subtile_linear_transposed<InputT, NUM_THREADS_PER_BLOCK_REGISTER_VEC_TRANSPOSED, PADDING_REGISTER_VEC_TRANSPOSED, TILE_REGISTER_VEC_TRANSPOSED_M, TILE_REGISTER_VEC_TRANSPOSED_N, TILE_REGISTER_VEC_TRANSPOSED_K>(
            A, ATileT,
            B, BTile,
            M, K, N,
            startRow, startCol,
            chunk, tid
        );
        __syncthreads();

        int kmax = min(TILE_REGISTER_VEC_TRANSPOSED_K, K - chunk);
        compute_subtile_transposed<
            InputT,
            TILE_REGISTER_VEC_TRANSPOSED_M,
            TILE_REGISTER_VEC_TRANSPOSED_N,
            TILE_REGISTER_VEC_TRANSPOSED_K,
            THREAD_DIM_REGISTER_VEC_TRANSPOSED,
            PADDING_REGISTER_VEC_TRANSPOSED
        >(
            ATileT, BTile,
            kmax,
            sum,
            threadRowTile, threadColTile
        );
        __syncthreads();
    }

    store_subtile_vec4<THREAD_DIM_REGISTER_VEC_TRANSPOSED>(
        sum, C, M, N,
        startRow, startCol,
        threadRowTile, threadColTile,
        alpha, beta
    );
}
