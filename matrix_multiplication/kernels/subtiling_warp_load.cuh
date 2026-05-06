#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include "config.h"
#include "kernel_utils.cuh"
#include "param_init.cuh"
#include "load_helpers.cuh"
#include "compute_helpers.cuh"
#include "store_helpers.cuh"



//used for warp loader
template <typename Params, typename InputT>
__device__ __forceinline__
void compute_subtile(const InputT ATile[SUBTILE][SUBTILE+PADDING_WARP],
                     const InputT BTile[SUBTILE][SUBTILE+PADDING_WARP],
                     int kmax,
                     float sum[SUB][SUB], const Params& params)
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
            #pragma unroll
            for (int j = 0; j < SUB; j++){
                sum[i][j] = fmaf(AReg[i], BReg[j], sum[i][j]);
            }

        }

    }
}

template <typename InputT>
__global__
void GEMMSubTilingLoadSlabLinear(int M, int N, int K,
                          float alpha,
                          const InputT* __restrict__ A,
                          const InputT* __restrict__ B,
                          float beta,
                          float* __restrict__ C)
{
    __shared__ InputT ATile[SUBTILE][SUBTILE + PADDING_WARP];
    __shared__ InputT BTile[SUBTILE][SUBTILE + PADDING_WARP];

    int startRow = blockIdx.y * SUBTILE;
    int startCol = blockIdx.x * SUBTILE;

    SlabParams params = make_linear_slab_params<InputT>();

    float sum[SUB][SUB];
    #pragma unroll
    for (int i = 0; i < SUB; i++) {
        #pragma unroll
        for (int j = 0; j < SUB; j++) {
            sum[i][j] = 0.0f;
        }
    }

    for (int chunk = 0; chunk < K; chunk += SUBTILE) {
        load_subtile_linear_slab<InputT>(
            A, ATile,
            B, BTile,
            M, K, N,
            startRow, startCol,
            chunk, params
        );
        __syncthreads();

        int kmax = min(SUBTILE, K - chunk);
        compute_subtile<SlabParams, InputT>(
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


