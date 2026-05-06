#include <cuda_runtime.h>
#include <cstdio>
#include "config.h"
#include "kernels.cuh"
#include "compute_helpers.cuh"
#include "store_helpers.cuh"

__device__ __forceinline__
void load_subtile_scalar_transposed(const float* __restrict__ A,
                        float ATileT[SUBTILE_K][SUBTILE_MN+1],
                        const float* __restrict__ B,
                        float BTile[SUBTILE_K][SUBTILE_MN+1],
                        int M, int K, int N,
                        int startRow, int startCol, int chunk,
                        int tid, int numThreads)
{
    constexpr int NUM_LOADS = SUBTILE_MN * SUBTILE_K;

    int threadRowGlobalOriginA = startRow;
    int threadColGlobalOriginA = chunk;

    int threadRowGlobalOriginB = chunk;
    int threadColGlobalOriginB = startCol;

    for (int idx = tid; idx < NUM_LOADS; idx += NUM_THREADS_PER_BLOCK) {
        // A tile: global A[M][K] -> shared ATileT[K][M]
        int rowTileA = idx / SUBTILE_K;
        int colTileA = idx % SUBTILE_K;

        int rowA = threadRowGlobalOriginA + rowTileA;
        int colA = threadColGlobalOriginA + colTileA;

        ATileT[colTileA][rowTileA] =
            (rowA < M && colA < K)
            ? A[rowA * K + colA]
            : 0.0f;

        // B tile: global B[K][N] -> shared BTile[K][N]
        int rowTileB = idx / SUBTILE_MN;
        int colTileB = idx % SUBTILE_MN;

        int rowB = threadRowGlobalOriginB + rowTileB;
        int colB = threadColGlobalOriginB + colTileB;

        BTile[rowTileB][colTileB] =
            (rowB < K && colB < N)
            ? B[rowB * N + colB]
            : 0.0f;
    }
}




__global__
void GEMMSubTilingScalarTransposed(int M, int N, int K,
                          float alpha,
                          const float* __restrict__ A,
                          const float* __restrict__ B,
                          float beta,
                          float* __restrict__ C)
{
    __shared__ float ATileT[SUBTILE_K][SUBTILE_MN + 1];
    __shared__ float BTile[SUBTILE_K][SUBTILE_MN + 1];

    int startRow = blockIdx.y * SUBTILE_MN;
    int startCol = blockIdx.x * SUBTILE_MN;

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int numThreads = blockDim.x * blockDim.y;

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

    for (int chunk = 0; chunk < K; chunk += SUBTILE_K) {
        load_subtile_scalar_transposed(
            A, ATileT,
            B, BTile,
            M, K, N,
            startRow, startCol,
            chunk, tid, numThreads
        );
        __syncthreads();

        int kmax = min(SUBTILE_K, K - chunk);
        compute_subtile_transposed(
            ATileT, BTile,
            kmax,
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



