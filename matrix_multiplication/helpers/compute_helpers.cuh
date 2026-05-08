#pragma once
#include <cuda_runtime.h>
#include "config.h"
#include "param_init.cuh"
#include "kernels/kernel_utils.cuh"
#include "helpers/tensor_core_types.cuh"
//Thread coordinate inputs

//used for subtiling scalar and vectorized
template <typename InputT, int TILE_M, int TILE_N, int TILE_K, int THREAD_DIM, int PAD>
__device__ __forceinline__
void compute_subtile(const InputT ATile[TILE_M][TILE_K+PAD],
                     const InputT BTile[TILE_K][TILE_N+PAD],
                     int kmax,
                     float sum[THREAD_DIM][THREAD_DIM], int threadRowTile, int threadColTile)
{
              
    #pragma unroll
    for (int k = 0; k < kmax; k++){
        float AReg[THREAD_DIM];
        float BReg[THREAD_DIM];
        #pragma unroll
        for (int i=0; i < THREAD_DIM; i++){
            AReg[i] = input_to_float_device<InputT>(ATile[threadRowTile + i][k]);
        }
        #pragma unroll
        for (int j=0; j < THREAD_DIM; j++){
            BReg[j] = input_to_float_device<InputT>(BTile[k][threadColTile + j]);
        }

        #pragma unroll
        for (int i = 0; i < THREAD_DIM; i++){
            #pragma unroll
            for (int j = 0; j < THREAD_DIM; j++){
                sum[i][j] = fmaf(AReg[i], BReg[j], sum[i][j]);
            }

        }

    }
}


//used for subtiling scalar and vectorized (both transposed versions)
template <typename InputT, int TILE_M, int TILE_N, int TILE_K, int THREAD_DIM, int PAD>
__device__ __forceinline__
void compute_subtile_transposed(const InputT ATileT[TILE_K][TILE_M + PAD],
                     const InputT BTile[TILE_K][TILE_N + PAD],
                     int kmax,
                     float sum[THREAD_DIM][THREAD_DIM], int threadRowTile, int threadColTile)
{
              
    #pragma unroll
    for (int k = 0; k < kmax; k++){
        float AReg[THREAD_DIM];
        float BReg[THREAD_DIM];
        #pragma unroll
        for (int i=0; i < THREAD_DIM; i++){
            AReg[i] = ATileT[k][threadRowTile + i];
        }
        #pragma unroll
        for (int j=0; j < THREAD_DIM; j++){
            BReg[j] = BTile[k][threadColTile + j];
        }

        #pragma unroll
        for (int i = 0; i < THREAD_DIM; i++){
            #pragma unroll
            for (int j = 0; j < THREAD_DIM; j++){
                sum[i][j] = fmaf(AReg[i], BReg[j], sum[i][j]);
            }

        }

    }
}


template <int TILE_M, int TILE_N, int TILE_K,
          int WARP_TILE_M, int WARP_TILE_N, int WARP_TILE_K, int WARP_FRAGMENTS_M,
          int WARP_FRAGMENTS_N, int FRAG_DIM_M, int FRAG_DIM_N, int FRAG_DIM_K, int PAD>
__device__ __forceinline__
void compute_step_tensor_cores(const __half ATile[TILE_M][TILE_K + PAD],
                               const __half BTile[TILE_K][TILE_N + PAD],
                               AFrag<FRAG_DIM_M, FRAG_DIM_N, FRAG_DIM_K> (&a_frag)[WARP_FRAGMENTS_M],
                               BFrag<FRAG_DIM_M, FRAG_DIM_N, FRAG_DIM_K> (&b_frag)[WARP_FRAGMENTS_N],
                               AccFrag<FRAG_DIM_M, FRAG_DIM_N, FRAG_DIM_K> (&sum)[WARP_FRAGMENTS_M][WARP_FRAGMENTS_N],
                               int warpRow, int warpCol){
    for (int wmma_chunk = 0; wmma_chunk < TILE_K; wmma_chunk += FRAG_DIM_K){
            #pragma unroll
            for (int i = 0; i < WARP_FRAGMENTS_M; i++){
                int rowTile = warpRow * WARP_TILE_M + FRAG_DIM_M * i;
                wmma::load_matrix_sync(a_frag[i], &ATile[rowTile][wmma_chunk] ,
                                        TILE_K + PAD);
            }
            #pragma unroll
            for (int j = 0; j < WARP_FRAGMENTS_N; j++){
                int colTile = warpCol * WARP_TILE_N + FRAG_DIM_N * j;
                wmma::load_matrix_sync(b_frag[j], &BTile[wmma_chunk][colTile],
                                        TILE_N + PAD);
            }
            

            #pragma unroll
            for (int i = 0; i < WARP_FRAGMENTS_M; i++){
                #pragma unroll
                for (int j = 0; j < WARP_FRAGMENTS_N; j++){
                    wmma::mma_sync(sum[i][j], a_frag[i], b_frag[j], sum[i][j]);
                }
            }
        }
}