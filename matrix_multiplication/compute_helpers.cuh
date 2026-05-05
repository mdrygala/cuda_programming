
#pragma once
#include <cuda_runtime.h>
#include "config.h"
#include "param_init.cuh"

__device__ __forceinline__
void compute_subtile(const float ATile[SUBTILE][SUBTILE+1],
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
void compute_subtile_transposed(const float ATileT[SUBTILE_K][SUBTILE_MN + 1],
                     const float BTile[SUBTILE_K][SUBTILE_MN + 1],
                     int kmax,
                     float sum[SUB][SUB], int threadRowTile, int threadColTile)
{
              
    #pragma unroll
    for (int k = 0; k < kmax; k++){
        float AReg[SUB];
        float BReg[SUB];
        #pragma unroll
        for (int i=0; i < SUB; i++){
            AReg[i] = ATileT[k][threadRowTile + i];
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


template <typename Params>
__device__ __forceinline__
void compute_subtile(const float ATile[SUBTILE][SUBTILE+PADDING],
                     const float BTile[SUBTILE][SUBTILE+PADDING],
                     int kmax,
                     float sum[SUB][SUB], const Params& params)
{
              
    #pragma unroll
    for (int k = 0; k < kmax; k++){
        float AReg[SUB];
        float BReg[SUB];
        #pragma unroll
        for (int i=0; i < SUB; i++){
            AReg[i] = ATile[params.threadRowTile + i][k];
        }
        #pragma unroll
        for (int j=0; j < SUB; j++){
            BReg[j] = BTile[k][params.threadColTile + j];
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


template <typename Params>
__device__ __forceinline__
void compute_subtile_regtangluar(const float ATile[SUBTILE_MN][SUBTILE_K+PADDING_GEN_DIM],
                     const float BTile[SUBTILE_K][SUBTILE_MN+PADDING_GEN_DIM],
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
            AReg[i] = ATile[params.threadRowTile + i][k];
        }
        #pragma unroll
        for (int j=0; j < SUB; j++){
            BReg[j] = BTile[k][params.threadColTile + j];
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




template <typename Params>
__device__ __forceinline__
void compute_subtile_transposed(const float ATileT[SUBTILE_K][SUBTILE_MN + 1],
                     const float BTile[SUBTILE_K][SUBTILE_MN + 1],
                     int kmax,
                     float sum[SUB][SUB], const Params& params)
{
              
    #pragma unroll
    for (int k = 0; k < kmax; k++){
        float AReg[SUB];
        float BReg[SUB];
        #pragma unroll
        for (int i=0; i < SUB; i++){
            AReg[i] = ATileT[k][params.threadRowTile + i];
        }
        #pragma unroll
        for (int j=0; j < SUB; j++){
            BReg[j] = BTile[k][params.threadColTile + j];
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