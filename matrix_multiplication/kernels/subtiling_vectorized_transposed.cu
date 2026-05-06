#include <cuda_runtime.h>
#include <cstdio>
#include "config.h"
#include "kernels.cuh"
#include "load_helpers.cuh"
#include "store_helpers.cuh"


__device__ __forceinline__
void load_subtile_vec4_transposed(
    const float* __restrict__ A,
    float ATileT[SUBTILE_K][SUBTILE_MN + 1],
    const float* __restrict__ B,
    float BTile[SUBTILE_K][SUBTILE_MN + 1],
    int M, int K, int N,
    int startRow, int startCol, int chunk,
    int tid, int numThreads)
{
    constexpr int VEC = 4;

    static_assert(SUBTILE_K  % VEC == 0, "SUBTILE_K must be divisible by 4");
    static_assert(SUBTILE_MN % VEC == 0, "SUBTILE_MN must be divisible by 4");

    constexpr int A_VEC_COLS = SUBTILE_K  / VEC;
    constexpr int B_VEC_COLS = SUBTILE_MN / VEC;

    constexpr int NUM_LOADS = SUBTILE_MN * SUBTILE_K / VEC;

    int threadRowGlobalOriginA = startRow;
    int threadColGlobalOriginA = chunk;

    int threadRowGlobalOriginB = chunk;
    int threadColGlobalOriginB = startCol;

    for (int idx = tid; idx < NUM_LOADS; idx += numThreads) {
        int rowTileA = idx / A_VEC_COLS;
        int vecColA  = idx % A_VEC_COLS;
        int colTileA = vecColA * VEC;

        int rowA = threadRowGlobalOriginA + rowTileA;
        int colA = threadColGlobalOriginA + colTileA;

        load_vec4_or_scalar_to_shared_transposed(
            A,
            rowA, colA, K,
            M, K,
            &ATileT[0][0],
            SUBTILE_MN + 1,
            rowTileA,
            colTileA
        );
        int rowTileB = idx / B_VEC_COLS;
        int vecColB  = idx % B_VEC_COLS;
        int colTileB = vecColB * VEC;

        int rowB = threadRowGlobalOriginB + rowTileB;
        int colB = threadColGlobalOriginB + colTileB;

        load_to_shared<float>(
            B,
            rowB, colB, N,
            K, N,
            &BTile[rowTileB][0],
            colTileB
        );
    }
}



__global__
void GEMMSubTilingVec4Transposed(int M, int N, int K,
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
        load_subtile_vec4_transposed(
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

    store_subtile_vec4(
        sum, C, M, N,
        startRow, startCol,
        threadRowTile, threadColTile,
        alpha, beta
    );
}
