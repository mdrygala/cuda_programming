// #include <cuda_runtime.h>
// #include <cstdio>
// #include "config.h"
// #include "kernels.cuh"
// #include "param_init.cuh"
// #include "load_helpers.cuh"
// #include "compute_helpers.cuh"
// #include "store_helpers.cuh"

// struct FakeSwizzleParams {
//     int threadRowTile;
//     int threadColTile;
//     int computeColTile;

//     int slabDimRows;
//     int slabDimCols;
//     int numWarps;

//     int warpRowGroup;
//     int warpsPerColGroup;
//     int slabRowIdx;

//     int colTile;
//     int newColTile;
// };

// __device__ __forceinline__
// FakeSwizzleParams make_fake_swizzle_params()
// {
//     FakeSwizzleParams params;

//     params.threadRowTile = threadIdx.y * SUB;
//     params.threadColTile = threadIdx.x * SUB;
    

//     params.slabDimRows = (SUBTILE + 3) >> 2;
//     params.slabDimCols = (SUBTILE + 31) >> 5;
//     params.numWarps    = (blockDim.x * blockDim.y) >> 5;

//     int threadBlockIdx = threadIdx.y * blockDim.x + threadIdx.x;
//     int warpId = threadBlockIdx >> 5;
//     int laneId = threadBlockIdx & 31;

//     int warpColGroup = warpId % params.slabDimCols;
//     params.warpRowGroup = warpId / params.slabDimCols;
//     params.warpsPerColGroup = params.numWarps / params.slabDimCols;

//     params.slabRowIdx = laneId >> 3;
//     int slabColIdx = laneId & 7;

//     params.colTile = 32 * warpColGroup + 4 * slabColIdx;
    

//     int shared_segment = params.colTile >> 5;
//     int shared_bank_idx = params.colTile & 31;
//     int new_shared_bank_idx = (shared_segment + shared_bank_idx) & 31;
//     params.newColTile = (shared_segment << 5) + new_shared_bank_idx;

//     params.newColTile = params.colTile;

//     // params.computeColTile = params.newColTile;
//     params.computeColTile = params.threadColTile;

//     return params;
// }

// __device__ __forceinline__
// void load_subtile_slab_fake_swizzle(const float* __restrict__ A,
//                        float ATile[SUBTILE][SUBTILE+1],
//                        const float* __restrict__ B,
//                        float BTile[SUBTILE][SUBTILE+1],
//                        int M, int K, int N,
//                        int threadRowGlobalOriginA, int threadColGlobalOriginA,
//                        int threadRowGlobalOriginB, int threadColGlobalOriginB,
//                        const FakeSwizzleParams& params){

 

// for (int slabRowStart = params.warpRowGroup; slabRowStart < params.slabDimRows; slabRowStart += params.warpsPerColGroup){ // loop over all slabs
//     int rowTile = 4 * slabRowStart + params.slabRowIdx; // tells us which row of the tile the thread is working on

//     //Load in A
//     int rowA = threadRowGlobalOriginA + rowTile;
//     int colA = threadColGlobalOriginA + params.colTile;
//     load_vec4_or_scalar_to_shared(A, rowA, colA, K,
//                                    M, K, &ATile[rowTile][0], params.colTile);


//     //Load in B
//     int rowB = threadRowGlobalOriginB + rowTile;
//     int colB = threadColGlobalOriginB + params.colTile;
//     load_vec4_or_scalar_to_shared(B, rowB, colB, N,
//                                    K, N, &BTile[rowTile][0], params.newColTile);
// }


// }




// __device__ __forceinline__
// void load_with_params(const FakeSwizzleParams& params,
//                       const float* __restrict__ A,
//                       float ATile[SUBTILE][SUBTILE+1],
//                       const float* __restrict__ B,
//                       float BTile[SUBTILE][SUBTILE+1],
//                       int M, int K, int N,
//                       int startRow, int startCol,
//                       int chunk)
// {
//     int threadRowGlobalOriginA = startRow;
//     int threadColGlobalOriginA = chunk;

//     int threadRowGlobalOriginB = chunk;
//     int threadColGlobalOriginB = startCol;

//     load_subtile_slab_fake_swizzle(
//         A, ATile, B, BTile,
//         M, K, N,
//         threadRowGlobalOriginA, threadColGlobalOriginA,
//         threadRowGlobalOriginB, threadColGlobalOriginB,
//         params
//     );
// }

// template <typename Params>
// __device__ __forceinline__
// Params make_params();


// template <>
// __device__ __forceinline__
// FakeSwizzleParams make_params<FakeSwizzleParams>()
// {
//     return make_fake_swizzle_params();
// }

// __global__
// void GEMMSubTilingLoadSlab2D(int M, int N, int K,
//                           float alpha,
//                           const float* __restrict__ A,
//                           const float* __restrict__ B,
//                           float beta,
//                           float* __restrict__ C)
// {
//     __shared__ float ATile[SUBTILE][SUBTILE + 1];
//     __shared__ float BTile[SUBTILE][SUBTILE + 1];

//     int startRow = blockIdx.y * SUBTILE;
//     int startCol = blockIdx.x * SUBTILE;

//     FakeSwizzleParams params = make_params<FakeSwizzleParams>();

//     // int threadRowTile = threadIdx.y * SUB;
//     // int threadColTile = threadIdx.x * SUB;

//     float sum[SUB][SUB];
//     #pragma unroll
//     for (int i = 0; i < SUB; i++) {
//         #pragma unroll
//         for (int j = 0; j < SUB; j++) {
//             sum[i][j] = 0.0f;
//         }
//     }

//     for (int chunk = 0; chunk < K; chunk += SUBTILE) {
//         load_with_params(
//             params,
//             A, ATile,
//             B, BTile,
//             M, K, N,
//             startRow, startCol,
//             chunk
//         );
//         __syncthreads();

//         int kmax = min(SUBTILE, K - chunk);
//         compute_subtile(
//             ATile, BTile,
//             K, kmax,
//             sum,
//             params
//         );
//         __syncthreads();
//     }

//     store_subtile_vec4(
//         sum, C, M, N,
//         startRow, startCol,
//         params.threadRowTile, params.computeColTile,
//         alpha, beta
//     );
// }


