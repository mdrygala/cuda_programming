#include <cuda_runtime.h>
#include <cstdio>
#include "config.h"
#include "kernels.cuh"
#include "helpers/load_helpers.cuh"
#include "helpers/compute_helpers.cuh"
#include "helpers/store_helpers.cuh"

#ifndef TILE_REGISTER_VEC
#define TILE_REGISTER_VEC 64
#endif

#ifndef THREAD_DIM_REGISTER_VEC
#define THREAD_DIM_REGISTER_VEC 4
#endif

#ifndef PADDING_REGISTER_VEC
#define PADDING_REGISTER_VEC 0
#endif


__device__ __forceinline__
void load_subtile_vec4(const float* __restrict__ A,
                       float ATile[TILE_REGISTER_VEC][TILE_REGISTER_VEC+PADDING_REGISTER_VEC],
                       const float* __restrict__ B,
                       float BTile[TILE_REGISTER_VEC][TILE_REGISTER_VEC+PADDING_REGISTER_VEC],
                       int M, int K, int N,
                       int startRow, int startCol, int chunk,
                       int threadRowTile, int threadColTile)
{   
    int threadRowGlobalOriginA = startRow;
    int threadColGlobalOriginA = chunk;

    int threadRowGlobalOriginB = chunk;
    int threadColGlobalOriginB = startCol;

    #pragma unroll
    for (int i = 0; i < THREAD_DIM_REGISTER_VEC; i++){
        int rowTile = threadRowTile + i;
        int rowA = threadRowGlobalOriginA + rowTile;
        int rowB = threadRowGlobalOriginB + rowTile;

        
        #pragma unroll
        for (int j = 0; j < THREAD_DIM_REGISTER_VEC; j+=4){
            int colTile = threadColTile + j;
            int colA = threadColGlobalOriginA + colTile;
            int colB = threadColGlobalOriginB + colTile;

        
            load_to_shared<float>(A, rowA, colA, K,
                                   M, K, &ATile[rowTile][0], colTile);
            load_to_shared<float>(B, rowB, colB, N,
                                   K, N, &BTile[rowTile][0], colTile);

    
        }
    }
}



__global__
void GEMMSubTilingVec4(int M, int N, int K,
                          float alpha,
                          const float* __restrict__ A,
                          const float* __restrict__ B,
                          float beta,
                          float* __restrict__ C)
{
    __shared__ float ATile[TILE_REGISTER_VEC][TILE_REGISTER_VEC + PADDING_REGISTER_VEC];
    __shared__ float BTile[TILE_REGISTER_VEC][TILE_REGISTER_VEC + PADDING_REGISTER_VEC];

    int startRow = blockIdx.y * TILE_REGISTER_VEC;
    int startCol = blockIdx.x * TILE_REGISTER_VEC;

    int threadRowTile = threadIdx.y * THREAD_DIM_REGISTER_VEC;
    int threadColTile = threadIdx.x * THREAD_DIM_REGISTER_VEC;

    float sum[THREAD_DIM_REGISTER_VEC][THREAD_DIM_REGISTER_VEC];
    #pragma unroll
    for (int i = 0; i < THREAD_DIM_REGISTER_VEC; i++) {
        #pragma unroll
        for (int j = 0; j < THREAD_DIM_REGISTER_VEC; j++) {
            sum[i][j] = 0.0f;
        }
    }

    for (int chunk = 0; chunk < K; chunk += TILE_REGISTER_VEC) {
        load_subtile_vec4(
            A, ATile,
            B, BTile,
            M, K, N,
            startRow, startCol,
            chunk, threadRowTile, threadColTile
        );
        __syncthreads();

        int kmax = min(TILE_REGISTER_VEC, K - chunk);
        compute_subtile<float, TILE_REGISTER_VEC, TILE_REGISTER_VEC, TILE_REGISTER_VEC, THREAD_DIM_REGISTER_VEC, PADDING_REGISTER_VEC>(
            ATile, BTile,
            kmax,
            sum,
            threadRowTile, threadColTile
        );
        __syncthreads();
    }

    store_subtile_vec4<THREAD_DIM_REGISTER_VEC>(
        sum, C, M, N,
        startRow, startCol,
        threadRowTile, threadColTile,
        alpha, beta
    );
}


