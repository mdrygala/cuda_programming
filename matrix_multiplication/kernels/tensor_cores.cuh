#pragma once
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>

#include <cstdio>
#include "config.h"
// #include "kernels.cuh"
// #include "kernel_utils.cuh"
#include "param_init.cuh"
#include "load_helpers.cuh"
// #include "compute_helpers.cuh"
// #include "store_helpers.cuh"

using namespace nvcuda;




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
    static_assert(SUBTILE_TENSOR_CORE_MN % WARP_TILE_M == 0);
    static_assert(SUBTILE_TENSOR_CORE_MN % WARP_TILE_N == 0);
    static_assert(NUM_WARPS_PER_BLOCK_TENSOR_CORE == NUM_WARPS_M * NUM_WARPS_N);

    __shared__ __half ATile[SUBTILE_TENSOR_CORE_MN][SUBTILE_TENSOR_CORE_K + PADDING_TENSOR_CORE];
    __shared__ __half BTile[SUBTILE_TENSOR_CORE_K][SUBTILE_TENSOR_CORE_MN + PADDING_TENSOR_CORE];

    wmma::fragment<wmma::matrix_a, FRAGMENT_M, FRAGMENT_N, FRAGMENT_K, __half, wmma::row_major> a_frag[WARP_M];
    wmma::fragment<wmma::matrix_b, FRAGMENT_M, FRAGMENT_N, FRAGMENT_K, __half, wmma::row_major> b_frag[WARP_N];
    wmma::fragment<wmma::accumulator, FRAGMENT_M, FRAGMENT_N, FRAGMENT_K, float> c_frag;
    wmma::fragment<wmma::accumulator, FRAGMENT_M, FRAGMENT_N, FRAGMENT_K, float> sum[WARP_M][WARP_N];

    SlabParamsGenDim params = make_linear_slab_params_gendim<__half, SUBTILE_TENSOR_CORE_MN, SUBTILE_TENSOR_CORE_K>();
    int startRow = blockIdx.y * SUBTILE_TENSOR_CORE_MN;
    int startCol = blockIdx.x * SUBTILE_TENSOR_CORE_MN;

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
        load_subtile_linear_slab_gendim<__half, NUM_WARPS_PER_BLOCK_TENSOR_CORE, PADDING_TENSOR_CORE, SUBTILE_TENSOR_CORE_MN, SUBTILE_TENSOR_CORE_K>(
            A, ATile,
            B, BTile,
            M, K, N,
            startRow, startCol,
            chunk, params
        );
        __syncthreads();

        //compute here
        for (int wmma_chunk = 0; wmma_chunk < SUBTILE_TENSOR_CORE_K; wmma_chunk += FRAGMENT_K){
            #pragma unroll
            for (int i = 0; i < WARP_M; i++){
                int rowTile = warpRow * WARP_TILE_M + FRAGMENT_M * i;
                wmma::load_matrix_sync(a_frag[i], &ATile[rowTile][wmma_chunk] ,
                                        SUBTILE_TENSOR_CORE_K + PADDING_TENSOR_CORE);
            }
            #pragma unroll
            for (int j = 0; j < WARP_N; j++){
                int colTile = warpCol * WARP_TILE_N + FRAGMENT_N * j;
                wmma::load_matrix_sync(b_frag[j], &BTile[wmma_chunk][colTile],
                                        SUBTILE_TENSOR_CORE_MN + PADDING_TENSOR_CORE);
            }
            

            #pragma unroll
            for (int i = 0; i < WARP_M; i++){
                for (int j = 0; j < WARP_N; j++){
                    wmma::mma_sync(sum[i][j], a_frag[i], b_frag[j], sum[i][j]);
                }
            }
        }
        __syncthreads();
    }

    //storing
    #pragma unroll
            for (int i = 0; i < WARP_M; i++){
                int rowTile = warpRow * WARP_TILE_M + FRAGMENT_M * i;
                int globalRow = blockIdx.y * SUBTILE_TENSOR_CORE_MN + rowTile;
                #pragma unroll
                for (int j = 0; j < WARP_N; j++){
                    int colTile = warpCol * WARP_TILE_N + FRAGMENT_N * j;
                    int globalCol = blockIdx.x * SUBTILE_TENSOR_CORE_MN + colTile;
                    int globalIdx = globalRow * N + globalCol;
                    if (beta != 0.0f){
                        wmma::load_matrix_sync(c_frag, &C[globalIdx], N, wmma::mem_row_major);
                    }
                    #pragma unroll
                    for (int t = 0; t < c_frag.num_elements; t++) {
                        c_frag.x[t] = alpha * sum[i][j].x[t] + beta * c_frag.x[t];
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