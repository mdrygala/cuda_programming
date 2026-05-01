
#pragma once
#include <cuda_runtime.h>
#include "config.h"
#include "param_init.cuh"

template <typename Params>
__device__ __forceinline__
void compute_subtile(const float ATile[SUBTILE][SUBTILE+1],
                     const float BTile[SUBTILE][SUBTILE+1],
                     int K, int kmax,
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
            BReg[j] = BTile[k][params.computeColTile + j];
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






