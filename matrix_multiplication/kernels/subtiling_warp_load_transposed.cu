#include <cuda_runtime.h>
#include <cstdio>
#include "config.h"
#include "kernels.cuh"
#include "param_init.cuh"
#include "load_helpers.cuh"
#include "compute_helpers.cuh"
#include "store_helpers.cuh"




__device__ __forceinline__
void load_subtile_linear_slab_transposed(
    const float* __restrict__ A,
    float ATileT[SUBTILE_K][SUBTILE_MN + 1],
    const float* __restrict__ B,
    float BTile[SUBTILE_K][SUBTILE_MN + 1],
    int M, int K, int N,
    int startRow, int startCol,
    int chunk,
    const SlabParamsLinearTransposed& params)
{
    constexpr int VEC = 4;
    constexpr int WARP_SIZE = 32;

    static_assert(SUBTILE_K  % VEC == 0, "SUBTILE_K must be divisible by 4");
    static_assert(SUBTILE_MN % VEC == 0, "SUBTILE_MN must be divisible by 4");

    constexpr int A_VEC_COLS = SUBTILE_K  / VEC;
    constexpr int B_VEC_COLS = SUBTILE_MN / VEC;

    static_assert(WARP_SIZE % A_VEC_COLS == 0,
                  "SUBTILE_K / 4 must divide 32");
    static_assert(WARP_SIZE % B_VEC_COLS == 0,
                  "SUBTILE_MN / 4 must divide 32");

    constexpr int A_SLAB_ROWS = WARP_SIZE / A_VEC_COLS;
    constexpr int B_SLAB_ROWS = WARP_SIZE / B_VEC_COLS;

    static_assert((SUBTILE_MN * SUBTILE_K) % (WARP_SIZE * VEC) == 0,
                  "SUBTILE_MN * SUBTILE_K must be divisible by 128");

    constexpr int TOTAL_SLABS =
        (SUBTILE_MN * SUBTILE_K) / (WARP_SIZE * VEC);

    int threadRowGlobalOriginA = startRow;
    int threadColGlobalOriginA = chunk;

    int threadRowGlobalOriginB = chunk;
    int threadColGlobalOriginB = startCol;

    for (int slabNum = params.warpId;
         slabNum < TOTAL_SLABS;
         slabNum += params.numWarps)
    {
        int slabRowIdxA = params.laneId / A_VEC_COLS;
        int slabColIdxA = params.laneId % A_VEC_COLS;

        int rowTileA = A_SLAB_ROWS * slabNum + slabRowIdxA;
        int colTileA = VEC * slabColIdxA;

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

        int slabRowIdxB = params.laneId / B_VEC_COLS;
        int slabColIdxB = params.laneId % B_VEC_COLS;

        int rowTileB = B_SLAB_ROWS * slabNum + slabRowIdxB;
        int colTileB = VEC * slabColIdxB;

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
void GEMMSubTilingLoadSlabLinearTransposed(int M, int N, int K,
                                           float alpha,
                                           const float* __restrict__ A,
                                           const float* __restrict__ B,
                                           float beta,
                                           float* __restrict__ C)
{
    __shared__ float ATileT[SUBTILE_K][SUBTILE_MN + 1];
    __shared__ float BTile [SUBTILE_K][SUBTILE_MN + 1];

    int startRow = blockIdx.y * SUBTILE_MN;
    int startCol = blockIdx.x * SUBTILE_MN;

    SlabParamsLinearTransposed params =
        make_slab_params_linear_transposed();

    float sum[SUB][SUB];

#pragma unroll
    for (int i = 0; i < SUB; i++) {
#pragma unroll
        for (int j = 0; j < SUB; j++) {
            sum[i][j] = 0.0f;
        }
    }

    for (int chunk = 0; chunk < K; chunk += SUBTILE_K) {
        load_subtile_linear_slab_transposed(
            A, ATileT,
            B, BTile,
            M, K, N,
            startRow, startCol,
            chunk,
            params
        );

        __syncthreads();

        int kmax = min(SUBTILE_K, K - chunk);

        compute_subtile_transposed(
            ATileT, BTile,
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




// struct SlabParamsLinearTransposed {
//     int threadRowTile;
//     int threadColTile;

//     int warpId;
//     int laneId;
//     int numWarps;

//     int slabRowIdxA;
//     int slabColIdxA;

//     int slabRowIdxB;
//     int slabColIdxB;
// };


// __device__ __forceinline__
// SlabParamsLinearTransposed make_slab_params_linear_transposed()
// {
//     constexpr int VEC = 4;
//     constexpr int WARP_SIZE = 32;

//     static_assert(SUBTILE_K  % VEC == 0, "SUBTILE_K must be divisible by 4");
//     static_assert(SUBTILE_MN % VEC == 0, "SUBTILE_MN must be divisible by 4");

//     constexpr int A_VEC_COLS = SUBTILE_K  / VEC;
//     constexpr int B_VEC_COLS = SUBTILE_MN / VEC;

//     static_assert(WARP_SIZE % A_VEC_COLS == 0,
//                   "SUBTILE_K / 4 must divide 32");
//     static_assert(WARP_SIZE % B_VEC_COLS == 0,
//                   "SUBTILE_MN / 4 must divide 32");

//     SlabParamsLinearTransposed params;

//     params.threadRowTile = threadIdx.y * SUB;
//     params.threadColTile = threadIdx.x * SUB;

//     int threadBlockIdx = threadIdx.y * blockDim.x + threadIdx.x;

//     params.warpId   = threadBlockIdx >> 5;
//     params.laneId   = threadBlockIdx & 31;
//     params.numWarps = (blockDim.x * blockDim.y) >> 5;

//     params.slabRowIdxA = params.laneId / A_VEC_COLS;
//     params.slabColIdxA = params.laneId % A_VEC_COLS;

//     params.slabRowIdxB = params.laneId / B_VEC_COLS;
//     params.slabColIdxB = params.laneId % B_VEC_COLS;

//     return params;
// }

// __device__ __forceinline__
// void load_subtile_linear_slab_transposed(
//     const float* __restrict__ A,
//     float ATileT[SUBTILE_K][SUBTILE_MN + 1],
//     const float* __restrict__ B,
//     float BTile[SUBTILE_K][SUBTILE_MN + 1],
//     int M, int K, int N,
//     int startRow, int startCol,
//     int chunk,
//     const SlabParamsLinearTransposed& params)
// {
//     constexpr int VEC = 4;
//     constexpr int WARP_SIZE = 32;

//     constexpr int A_VEC_COLS = SUBTILE_K  / VEC;
//     constexpr int B_VEC_COLS = SUBTILE_MN / VEC;

//     constexpr int A_SLAB_ROWS = WARP_SIZE / A_VEC_COLS;
//     constexpr int B_SLAB_ROWS = WARP_SIZE / B_VEC_COLS;

//     static_assert((SUBTILE_MN * SUBTILE_K) % (WARP_SIZE * VEC) == 0,
//                   "SUBTILE_MN * SUBTILE_K must be divisible by 128");

//     constexpr int TOTAL_SLABS =
//         (SUBTILE_MN * SUBTILE_K) / (WARP_SIZE * VEC);

//     int threadRowGlobalOriginA = startRow;
//     int threadColGlobalOriginA = chunk;

//     int threadRowGlobalOriginB = chunk;
//     int threadColGlobalOriginB = startCol;

//     for (int slabNum = params.warpId;
//          slabNum < TOTAL_SLABS;
//          slabNum += params.numWarps)
//     {
//         int rowTileA = A_SLAB_ROWS * slabNum + params.slabRowIdxA;
//         int colTileA = VEC * params.slabColIdxA;

//         int rowA = threadRowGlobalOriginA + rowTileA;
//         int colA = threadColGlobalOriginA + colTileA;

//         load_vec4_or_scalar_to_shared_transposed(
//             A,
//             rowA, colA, K,
//             M, K,
//             &ATileT[0][0],
//             SUBTILE_MN + 1,
//             rowTileA,
//             colTileA
//         );

//         int rowTileB = B_SLAB_ROWS * slabNum + params.slabRowIdxB;
//         int colTileB = VEC * params.slabColIdxB;

//         int rowB = threadRowGlobalOriginB + rowTileB;
//         int colB = threadColGlobalOriginB + colTileB;

//         load_vec4_or_scalar_to_shared(
//             B,
//             rowB, colB, N,
//             K, N,
//             &BTile[rowTileB][0],
//             colTileB
//         );
//     }
// }