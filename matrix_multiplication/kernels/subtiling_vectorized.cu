#include <cuda_runtime.h>
#include <cstdio>
#include "config.h"
#include "kernels.cuh"
#include "load_helpers.cuh"
#include "compute_helpers.cuh"
#include "store_helpers.cuh"


__device__ __forceinline__
void load_subtile_vec4(const float* __restrict__ A,
                       float ATile[SUBTILE][SUBTILE+1],
                       const float* __restrict__ B,
                       float BTile[SUBTILE][SUBTILE+1],
                       int M, int K, int N,
                       int startRow, int startCol, int chunk,
                       int threadRowTile, int threadColTile)
{   
    int threadRowGlobalOriginA = startRow;
    int threadColGlobalOriginA = chunk;

    int threadRowGlobalOriginB = chunk;
    int threadColGlobalOriginB = startCol;

    #pragma unroll
    for (int i = 0; i < SUB; i++){
        int rowTile = threadRowTile + i;
        int rowA = threadRowGlobalOriginA + rowTile;
        int rowB = threadRowGlobalOriginB + rowTile;

        
        #pragma unroll
        for (int j = 0; j < SUB; j+=4){
            int colTile = threadColTile + j;
            int colA = threadColGlobalOriginA + colTile;
            int colB = threadColGlobalOriginB + colTile;

        
            load_vec4_or_scalar_to_shared(A, rowA, colA, K,
                                   M, K, &ATile[rowTile][0], colTile);
            load_vec4_or_scalar_to_shared(B, rowB, colB, N,
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
    __shared__ float ATile[SUBTILE][SUBTILE + 1];
    __shared__ float BTile[SUBTILE][SUBTILE + 1];

    int startRow = blockIdx.y * SUBTILE;
    int startCol = blockIdx.x * SUBTILE;

    int threadRowTile = threadIdx.y * SUB;
    int threadColTile = threadIdx.x * SUB;

    float sum[SUB][SUB];
    #pragma unroll
    for (int i = 0; i < SUB; i++) {
        #pragma unroll
        for (int j = 0; j < SUB; j++) {
            sum[i][j] = 0.0f;
        }
    }

    for (int chunk = 0; chunk < K; chunk += SUBTILE) {
        load_subtile_vec4(
            A, ATile,
            B, BTile,
            M, K, N,
            startRow, startCol,
            chunk, threadRowTile, threadColTile
        );
        __syncthreads();

        int kmax = min(SUBTILE, K - chunk);
        compute_subtile(
            ATile, BTile,
            K, kmax,
            sum,
            threadRowTile, threadColTile
        );
        __syncthreads();
    }

    store_subtile_vec4(
        sum, C, M, N,
        startRow, startCol,
        threadRowTile, threadColTile,
        alpha, beta
    );
}


