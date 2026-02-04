
#pragma once
#include <cuda_runtime.h>
#include "gemm_config.h"

__device__ __forceinline__
void compute_subtile_naive(const float ATile[SUBTILE][SUBTILE+1],
                     const float BTile[SUBTILE][SUBTILE+1],
                     int K, int kmax,
                     float sum[SUB][SUB], int threadRowTile, int threadColTile)
{
              
    #pragma unroll
    for (int k = 0; k < kmax; k++){
        float AReg[SUB];
        float BReg[SUB];
        #pragma unroll
        for (int i=0; i < SUB; i++){
            AReg[i] = ATile[threadRowTile + i][k];
        }
        #pragma unroll
        for (int j=0; j < SUB; j++){
            BReg[j] = BTile[k][threadColTile + j];
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


__device__ __forceinline__
void compute_subtile_swizzle(const float ATile[SUBTILE][SUBTILE+1],
                            const float BTile[SUBTILE][SUBTILE+1],
                            int /*K*/, int kmax,
                            float sum[SUB][SUB],
                            int threadRowTile, int threadColTile)
{
    #pragma unroll
    for (int k = 0; k < kmax; k++){
        float AReg[SUB];
        float BReg[SUB];

        #pragma unroll
        for (int i = 0; i < SUB; i++){
            AReg[i] = ATile[threadRowTile + i][k];
        }

        const int colSwz = (k & SWZ_MASK) << SWZ_SHIFT;

        #pragma unroll
        for (int j = 0; j < SUB; j++){
            BReg[j] = BTile[k][(threadColTile + j) ^ colSwz];
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









struct ComputeNaive {
    __device__ __forceinline__
    static void run(
        const float ATile[SUBTILE][SUBTILE+1],
        const float BTile[SUBTILE][SUBTILE+1],
        int K, int kmax,
        float sum[SUB][SUB],
        int threadRowTile, int threadColTile
    ) {
        compute_subtile_naive(
            ATile, BTile,
            K,
            kmax,
            sum,
            threadRowTile, threadColTile
        );
    }
};


struct ComputeSwizzle {
    __device__ __forceinline__
    static void run(
        const float ATile[SUBTILE][SUBTILE+1],
        const float BTile[SUBTILE][SUBTILE+1],
        int K, int kmax,
        float sum[SUB][SUB],
        int threadRowTile, int threadColTile
    ) {
        compute_subtile_swizzle(
            ATile, BTile,
            K, kmax,
            sum,
            threadRowTile, threadColTile
        );
    }
};
