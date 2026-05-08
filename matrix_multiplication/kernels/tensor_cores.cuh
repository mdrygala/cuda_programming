#pragma once
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>

#include <cstdio>
#include "config.h"

#include "param_init.cuh"
#include "helpers/load_helpers.cuh"
#include "helpers/compute_helpers.cuh"
#include "helpers/store_helpers.cuh"
#include "helpers/tensor_core_types.cuh"


#ifndef SUBTILE_TENSOR_CORE_M
#define SUBTILE_TENSOR_CORE_M 64
#endif

#ifndef SUBTILE_TENSOR_CORE_N
#define SUBTILE_TENSOR_CORE_N 64
#endif


#ifndef SUBTILE_TENSOR_CORE_K
#define SUBTILE_TENSOR_CORE_K 16
#endif

#ifndef PADDING_TENSOR_CORE
#define PADDING_TENSOR_CORE 8
#endif

#ifndef FRAGMENT_M
#define FRAGMENT_M 16
#endif

#ifndef FRAGMENT_N
#define FRAGMENT_N 16
#endif

#ifndef FRAGMENT_K
#define FRAGMENT_K 16
#endif

#ifndef WARP_M  
#define WARP_M 2
#endif

#ifndef WARP_N  
#define WARP_N 2
#endif

#define WARP_TILE_M (WARP_M * FRAGMENT_M)
#define WARP_TILE_N (WARP_N * FRAGMENT_N)
#define NUM_WARPS_M (SUBTILE_TENSOR_CORE_M / WARP_TILE_M)
#define NUM_WARPS_N (SUBTILE_TENSOR_CORE_N / WARP_TILE_N)
#define NUM_WARPS_PER_BLOCK_TENSOR_CORE (NUM_WARPS_M *  NUM_WARPS_N)
#define NUM_THREADS_PER_BLOCK_TENSOR_CORE (NUM_WARPS_PER_BLOCK_TENSOR_CORE  * 32)





__device__ __forceinline__
void store_step_tensor_cores(int N, float alpha, float beta, AccFrag& c_frag, float* __restrict__ C,
                             AccFrag (&sum)[WARP_M][WARP_N], int warpRow, int warpCol){
    #pragma unroll
            for (int i = 0; i < WARP_M; i++){
                int rowTile = warpRow * WARP_TILE_M + FRAGMENT_M * i;
                int globalRow = blockIdx.y * SUBTILE_TENSOR_CORE_M + rowTile;
                #pragma unroll
                for (int j = 0; j < WARP_N; j++){
                    int colTile = warpCol * WARP_TILE_N + FRAGMENT_N * j;
                    int globalCol = blockIdx.x * SUBTILE_TENSOR_CORE_N + colTile;
                    int globalIdx = globalRow * N + globalCol;
                    
                    if (beta != 0.0f) {
                        wmma::load_matrix_sync(
                            c_frag,
                            &C[globalIdx],
                            N,
                            wmma::mem_row_major
                        );

                        #pragma unroll
                        for (int t = 0; t < c_frag.num_elements; t++) {
                            c_frag.x[t] = alpha * sum[i][j].x[t]
                                        + beta  * c_frag.x[t];
                        }
                    } else {
                        #pragma unroll
                        for (int t = 0; t < c_frag.num_elements; t++) {
                            c_frag.x[t] = alpha * sum[i][j].x[t];
                        }
                    }
                    wmma::store_matrix_sync(
                        &C[globalIdx],
                        c_frag,
                        N,
                        wmma::mem_row_major
                    );
                }
            }
}




// template <typename InputT>
__global__
void GEMMTensorCores(int M, int N, int K,
                          float alpha,
                          const __half* __restrict__ A,
                          const __half* __restrict__ B,
                          float beta,
                          float* __restrict__ C)
{   
    static_assert(SUBTILE_TENSOR_CORE_K % FRAGMENT_K == 0);
    static_assert(WARP_TILE_M % FRAGMENT_M == 0);
    static_assert(WARP_TILE_N % FRAGMENT_N == 0);
    static_assert(SUBTILE_TENSOR_CORE_M % WARP_TILE_M == 0);
    static_assert(SUBTILE_TENSOR_CORE_N % WARP_TILE_N == 0);
    static_assert(NUM_WARPS_PER_BLOCK_TENSOR_CORE == NUM_WARPS_M * NUM_WARPS_N);

    __shared__ __half ATile[SUBTILE_TENSOR_CORE_M][SUBTILE_TENSOR_CORE_K + PADDING_TENSOR_CORE];
    __shared__ __half BTile[SUBTILE_TENSOR_CORE_K][SUBTILE_TENSOR_CORE_N + PADDING_TENSOR_CORE];

    AFrag<FRAGMENT_M, FRAGMENT_N, FRAGMENT_K> a_frag[WARP_M];
    BFrag<FRAGMENT_M, FRAGMENT_N, FRAGMENT_K> b_frag[WARP_N];
    AccFrag<FRAGMENT_M, FRAGMENT_N, FRAGMENT_K> c_frag;
    AccFrag<FRAGMENT_M, FRAGMENT_N, FRAGMENT_K> sum[WARP_M][WARP_N];

    SlabParamsGenDim params = make_linear_slab_params_gendim<__half, SUBTILE_TENSOR_CORE_M, SUBTILE_TENSOR_CORE_N, SUBTILE_TENSOR_CORE_K>();
    int startRow = blockIdx.y * SUBTILE_TENSOR_CORE_M;
    int startCol = blockIdx.x * SUBTILE_TENSOR_CORE_N;

    #pragma unroll
    for (int i = 0; i < WARP_M; i++) {
        #pragma unroll
        for (int j = 0; j < WARP_N; j++) {
            wmma::fill_fragment(sum[i][j], 0.0f);
        }
    }

    const int warpRow = params.warpId / NUM_WARPS_N;
    const int warpCol = params.warpId % NUM_WARPS_N;

    for (int chunk = 0; chunk < K; chunk += SUBTILE_TENSOR_CORE_K) {
        load_subtile_warp<__half, NUM_WARPS_PER_BLOCK_TENSOR_CORE, PADDING_TENSOR_CORE, SUBTILE_TENSOR_CORE_M, SUBTILE_TENSOR_CORE_N, SUBTILE_TENSOR_CORE_K>(
            A, ATile,
            B, BTile,
            M, K, N,
            startRow, startCol,
            chunk, params
        );
        __syncthreads();

        compute_step_tensor_cores(ATile, BTile, a_frag, b_frag, sum, warpRow, warpCol);
        
        __syncthreads();
    }

    store_step_tensor_cores(N, alpha, beta, c_frag, C, sum, warpRow, warpCol);
    
}



__global__
void GEMMTensorCoresDB(int M, int N, int K,
                          float alpha,
                          const __half* __restrict__ A,
                          const __half* __restrict__ B,
                          float beta,
                          float* __restrict__ C)
{   
    static_assert(SUBTILE_TENSOR_CORE_K % FRAGMENT_K == 0);
    static_assert(WARP_TILE_M % FRAGMENT_M == 0);
    static_assert(WARP_TILE_N % FRAGMENT_N == 0);
    static_assert(SUBTILE_TENSOR_CORE_M % WARP_TILE_M == 0);
    static_assert(SUBTILE_TENSOR_CORE_N % WARP_TILE_N == 0);
    static_assert(NUM_WARPS_PER_BLOCK_TENSOR_CORE == NUM_WARPS_M * NUM_WARPS_N);

    __shared__ __half ATile[2][SUBTILE_TENSOR_CORE_M][SUBTILE_TENSOR_CORE_K + PADDING_TENSOR_CORE];
    __shared__ __half BTile[2][SUBTILE_TENSOR_CORE_K][SUBTILE_TENSOR_CORE_N + PADDING_TENSOR_CORE];

    AFrag a_frag[WARP_M];
    BFrag b_frag[WARP_N];
    AccFrag c_frag;
    AccFrag sum[WARP_M][WARP_N];

    SlabParamsGenDim params = make_linear_slab_params_gendim<__half, SUBTILE_TENSOR_CORE_M, SUBTILE_TENSOR_CORE_N, SUBTILE_TENSOR_CORE_K>();
    int startRow = blockIdx.y * SUBTILE_TENSOR_CORE_M;
    int startCol = blockIdx.x * SUBTILE_TENSOR_CORE_N;

    #pragma unroll
    for (int i = 0; i < WARP_M; i++) {
        #pragma unroll
        for (int j = 0; j < WARP_N; j++) {
            wmma::fill_fragment(sum[i][j], 0.0f);
        }
    }

    const int warpRow = params.warpId / NUM_WARPS_N;
    const int warpCol = params.warpId % NUM_WARPS_N;

    int read_buffer_idx = 0;
    load_subtile_warp<__half, NUM_WARPS_PER_BLOCK_TENSOR_CORE, PADDING_TENSOR_CORE, SUBTILE_TENSOR_CORE_M, SUBTILE_TENSOR_CORE_N, SUBTILE_TENSOR_CORE_K>(
            A, ATile[read_buffer_idx],
            B, BTile[read_buffer_idx],
            M, K, N,
            startRow, startCol,
            0, params
        );
     __syncthreads();

    for (int chunk = 0; chunk < K; chunk += SUBTILE_TENSOR_CORE_K) {
        int write_buffer_idx = read_buffer_idx ^ 1;
        int nextChunk = chunk + SUBTILE_TENSOR_CORE_K;
        if (nextChunk < K){
            load_subtile_warp<__half, NUM_WARPS_PER_BLOCK_TENSOR_CORE, PADDING_TENSOR_CORE, SUBTILE_TENSOR_CORE_M, SUBTILE_TENSOR_CORE_N, SUBTILE_TENSOR_CORE_K>(
                A, ATile[write_buffer_idx],
                B, BTile[write_buffer_idx],
                M, K, N,
                startRow, startCol,
                nextChunk, params
            );
        }

        compute_step_tensor_cores(ATile[read_buffer_idx], BTile[read_buffer_idx], a_frag, b_frag, sum, warpRow, warpCol);
        
        __syncthreads();
        if (nextChunk < K){
            read_buffer_idx = write_buffer_idx;
        }
    }

    store_step_tensor_cores(N, alpha, beta, c_frag, C, sum, warpRow, warpCol);
    
}