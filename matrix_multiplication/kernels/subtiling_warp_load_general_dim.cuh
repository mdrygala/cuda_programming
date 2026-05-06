#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include "config.h"
// #include "kernels.cuh"
#include "kernel_utils.cuh"
#include "param_init.cuh"
#include "load_helpers.cuh"
// #include "compute_helpers.cuh"
#include "store_helpers.cuh"




//used for warp loader general dims
template <typename Params, typename InputT>
__device__ __forceinline__
void compute_subtile_regtangluar(const InputT ATile[SUBTILE_MN][SUBTILE_K+PADDING_GEN_DIM],
                     const InputT BTile[SUBTILE_K][SUBTILE_MN+PADDING_GEN_DIM],
                     int kmax,
                     float sum[SUB][SUB],
                     const Params& params)
{
              
    #pragma unroll
    for (int k = 0; k < kmax; k++){
        float AReg[SUB];
        float BReg[SUB];
        #pragma unroll
        for (int i=0; i < SUB; i++){
            AReg[i] = input_to_float_device<InputT>(ATile[params.threadRowTile + i][k]);
        }
        #pragma unroll
        for (int j=0; j < SUB; j++){
            BReg[j] = input_to_float_device<InputT>(BTile[k][params.threadColTile + j]);
        }

        #pragma unroll
        for (int i = 0; i < SUB; i++){
            // float a = ATile[params.threadRowTile + i][k];
            #pragma unroll
            for (int j = 0; j < SUB; j++){
                sum[i][j] = fmaf(AReg[i], BReg[j], sum[i][j]);
                // sum[i][j] = fmaf(a, BReg[j], sum[i][j]);
            }

        }

    }
}

template <typename InputT>
__global__
void GEMMSubTilingLoadSlabGenDims(int M, int N, int K,
                          float alpha,
                          const InputT* __restrict__ A,
                          const InputT* __restrict__ B,
                          float beta,
                          float* __restrict__ C)
{
    __shared__ InputT ATile[SUBTILE_MN][SUBTILE_K + PADDING_GEN_DIM];
    __shared__ InputT BTile[SUBTILE_K][SUBTILE_MN + PADDING_GEN_DIM];

    int startRow = blockIdx.y * SUBTILE_MN;
    int startCol = blockIdx.x * SUBTILE_MN;

    SlabParamsGenDim params = make_linear_slab_params_gendim<InputT,  SUBTILE_MN, SUBTILE_K>();

    float sum[SUB][SUB];
    #pragma unroll
    for (int i = 0; i < SUB; i++) {
        #pragma unroll
        for (int j = 0; j < SUB; j++) {
            sum[i][j] = 0.0f;
        }
    }

    for (int chunk = 0; chunk < K; chunk += SUBTILE_K) {
        load_subtile_linear_slab_gendim<InputT, NUM_WARPS_PER_BLOCK, PADDING_GEN_DIM, SUBTILE_MN, SUBTILE_K>(
            A, ATile,
            B, BTile,
            M, K, N,
            startRow, startCol,
            chunk, params
        );
        __syncthreads();

        int kmax = min(SUBTILE_K, K - chunk);
        compute_subtile_regtangluar<SlabParamsGenDim, InputT>(
            ATile, BTile,
            kmax,
            sum,
            params
        );
        __syncthreads();
    }

    store_subtile_vec4(
        sum, C, M, N,
        startRow, startCol,
        params.threadRowTile, params.threadColTile,
        alpha, beta
    );
}


