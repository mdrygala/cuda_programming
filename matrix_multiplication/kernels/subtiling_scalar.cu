#include <cuda_runtime.h>
#include <cstdio>
#include "config.h"
#include "kernels.cuh"
#include "compute_helpers.cuh"

__device__ __forceinline__
void load_subtile_naive(const float* __restrict__ A,
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
        for (int j = 0; j < SUB; j++){
            int colTile = threadColTile + j;
            int colA = threadColGlobalOriginA + colTile;
            int colB = threadColGlobalOriginB + colTile;
        

            //Loads into shared memory, in 
            ATile[rowTile][colTile] = (rowA < M && colA < K) ? A[rowA * K + colA] : 0.0f;
            BTile[rowTile][colTile] = (rowB < K && colB < N) ? B[rowB * N + colB] : 0.0f;
        }
    }
}



__device__ __forceinline__
void store_subtile_scalar(float sum[SUB][SUB],
                         float*  __restrict__ C, int M, int N, 
                         int startRow, int startCol,
                         int threadRowTile, int threadColTile,
                        float alpha, float beta)
{
    int threadRowGlobalOrigin = startRow + threadRowTile;
    int threadColGlobalOrigin = startCol + threadColTile;
    #pragma unroll
    for (int i = 0; i < SUB; i++){
        int r = threadRowGlobalOrigin + i;
        if (r >= M) break;
        #pragma unroll
        for (int j = 0; j < SUB; j++){
            int c = threadColGlobalOrigin + j;
            if (c >= N) break;
            int idx = r * N + c;
            float cold = (beta != 0.0f) ? C[idx] : 0.0f;
            C[idx] = alpha * sum[i][j] + beta * cold;
        }
    }
}


__global__
void GEMMSubTilingScalar(int M, int N, int K,
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
        load_subtile_naive(
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

    store_subtile_scalar(
        sum, C, M, N,
        startRow, startCol,
        threadRowTile, threadColTile,
        alpha, beta
    );
}



