#include <cuda_runtime.h>
#include <cstdio>
#include "config.h"
#include "load_helpers.cuh"
#include "compute_helpers.cuh"
#include "store_helpers.cuh"
#include "kernels.cuh"          // include declarations

// -------------------- Baseline --------------------
__global__ void GEMMBaseline(int M, int N, int K,
                             float alpha,
                             const float* __restrict__ A,
                             const float* __restrict__ B,
                             float beta,
                             float* __restrict__ C)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; k++){
        sum = fmaf(A[row * K + k], B[k * N + col], sum);
    }

    int idx = row * N + col;
    float cold = (beta != 0.0f) ? C[idx] : 0.0f;
    C[idx] = alpha * sum + beta * cold;   // FIX: don’t read C twice
}

// -------------------- Tiling --------------------
__global__ void GEMMTiling(int M, int N, int K,
                           float alpha,
                           const float* __restrict__ A,
                           const float* __restrict__ B,
                           float beta,
                           float* __restrict__ C)
{
    __shared__ float ATile[TILE][TILE];
    __shared__ float BTile[TILE][TILE];

    int startRow = blockIdx.y * TILE;
    int startCol = blockIdx.x * TILE;

    int threadRowGlobal = startRow + threadIdx.y;
    int threadColGlobal = startCol + threadIdx.x;

    float sum = 0.0f;

    for (int chunk = 0; chunk < K; chunk += TILE){
        int rowA = threadRowGlobal;
        int colA = chunk + threadIdx.x;

        int rowB = chunk + threadIdx.y;
        int colB = threadColGlobal;

        ATile[threadIdx.y][threadIdx.x] =
            (rowA < M && colA < K) ? A[rowA * K + colA] : 0.0f;

        BTile[threadIdx.y][threadIdx.x] =
            (rowB < K && colB < N) ? B[rowB * N + colB] : 0.0f;

        __syncthreads();

        int kmax = min(TILE, K - chunk);
        #pragma unroll
        for (int k = 0; k < kmax; k++){
            sum = fmaf(ATile[threadIdx.y][k], BTile[k][threadIdx.x], sum);
        }
        __syncthreads();
    }

    if (threadRowGlobal < M && threadColGlobal < N){
        int idx = threadRowGlobal * N + threadColGlobal;
        float cold = (beta != 0.0f) ? C[idx] : 0.0f;
        C[idx] = alpha * sum + beta * cold;
    }
}

// -------------------- Subtiling template (definition stays in .cu) --------------------
template <typename LoaderFunc, typename ComputeFunc, typename StoreFunc>
__device__ __forceinline__ void GEMMSubTiling(int M, int N, int K,
                              float alpha,
                              const float* __restrict__ A,
                              const float* __restrict__ B,
                              float beta,
                              float* __restrict__ C)
{
    __shared__ float ATile[SUBTILE][SUBTILE+1];
    __shared__ float BTile[SUBTILE][SUBTILE+1];

    int startRow = blockIdx.y * SUBTILE;
    int startCol = blockIdx.x * SUBTILE;

    int threadRowTile = threadIdx.y * SUB;
    int threadColTile = threadIdx.x * SUB;

    float sum[SUB][SUB];
    #pragma unroll
    for (int i = 0; i < SUB; i++)
        #pragma unroll
        for (int j = 0; j < SUB; j++)
            sum[i][j] = 0.0f;

    for (int chunk = 0; chunk < K; chunk += SUBTILE){
        int threadRowGlobalOriginA = startRow;
        int threadColGlobalOriginA = chunk;

        int threadRowGlobalOriginB = chunk;
        int threadColGlobalOriginB =  startCol;

        LoaderFunc::run(A, ATile, B, BTile, M, K, N,
                        startRow, startCol,
                        threadRowTile, threadColTile,
                        threadRowGlobalOriginA, threadColGlobalOriginA,
                        threadRowGlobalOriginB, threadColGlobalOriginB);
        __syncthreads();


        int kmax = min(SUBTILE, K - chunk);
        ComputeFunc::run(ATile, BTile, K, kmax,
                            sum, threadRowTile, threadColTile);
        __syncthreads();
    }
    

    StoreFunc::run(sum, C, M, N, startRow, startCol,
                threadRowTile, threadColTile, alpha, beta);
        
}



// -------------------- Subtiling slabs --------------------
template <typename LoaderFunc, typename ComputeFunc, typename StoreFunc>
__device__ __forceinline__ void GEMMSubTilingSlab(int M, int N, int K,
                              float alpha,
                              const float* __restrict__ A,
                              const float* __restrict__ B,
                              float beta,
                              float* __restrict__ C)
{
    __shared__ float ATile[SUBTILE][SUBTILE+1];
    __shared__ float BTile[SUBTILE][SUBTILE+1];

    int startRow = blockIdx.y * SUBTILE;
    int startCol = blockIdx.x * SUBTILE;
    //for compute
    int threadRowTile = threadIdx.y * SUB;
    int threadColTile = threadIdx.x * SUB;


    //for loading
    int slabDimRows = (SUBTILE + 3) >> 2; // num of slabs that fit vertically within tile
    int slabDimCols = (SUBTILE + 31) >> 5;  // num of slabs that fit horizontally within tile
    int totalSlabs =  slabDimRows * slabDimCols;

    int numWarps = blockDim.x * blockDim.y >> 5; // num of warps within a block

    int threadBlockIdx = threadIdx.y * blockDim.x + threadIdx.x; // thread index within a block
    int warpId = threadBlockIdx >> 5; // Id of the warp within a block
    int laneId = threadBlockIdx & 31; // Within a warp the id of a thread

    int slabRowIdx = laneId >> 3; // Among the 4 rows in a slab which one the thread is assigned to
    int slabColIdx = laneId & 7; // among the 8 columns in a slab which one is the thread assigned to

    float sum[SUB][SUB];
    #pragma unroll
    for (int i = 0; i < SUB; i++)
        #pragma unroll
        for (int j = 0; j < SUB; j++)
            sum[i][j] = 0.0f;

    for (int chunk = 0; chunk < K; chunk += SUBTILE){
        int threadRowGlobalOriginA = startRow;
        int threadColGlobalOriginA = chunk;

        int threadRowGlobalOriginB = chunk;
        int threadColGlobalOriginB =  startCol;

        LoaderFunc::run(A, ATile, B, BTile, M, K, N,
                        startRow, startCol,
                        threadRowTile, threadColTile,
                        threadRowGlobalOriginA, threadColGlobalOriginA,
                        threadRowGlobalOriginB, threadColGlobalOriginB,
                        warpId, totalSlabs, numWarps, slabRowIdx, slabColIdx, slabDimCols
                    );
        __syncthreads();


        int kmax = min(SUBTILE, K - chunk);
        ComputeFunc::run(ATile, BTile, K, kmax,
                            sum, threadRowTile, threadColTile);
        __syncthreads();
    }
    

    StoreFunc::run(sum, C, M, N, startRow, startCol,
                threadRowTile, threadColTile, alpha, beta);
        
}





template <typename LoaderFunc, typename ComputeFunc, typename StoreFunc>
__device__ __forceinline__ void GEMMSubTilingSwzl(int M, int N, int K,
                              float alpha,
                              const float* __restrict__ A,
                              const float* __restrict__ B,
                              float beta,
                              float* __restrict__ C)
{
    __shared__ float ATile[SUBTILE][SUBTILE+1];
    __shared__ float BTile[SUBTILE][SUBTILE+1];

    int startRow = blockIdx.y * SUBTILE;
    int startCol = blockIdx.x * SUBTILE;

    // for compute
    int threadRowTile = threadIdx.y * SUB;
    int threadColTile = threadIdx.x * SUB;

    float sum[SUB][SUB];
    #pragma unroll
    for (int i = 0; i < SUB; i++)
        #pragma unroll
        for (int j = 0; j < SUB; j++)
            sum[i][j] = 0.0f;

    
    //for loading
    int slabDimRows = (SUBTILE + 3) >> 2; // num of slabs that fit vertically within tile
    int slabDimCols = (SUBTILE + 31) >> 5;  // num of slabs that fit horizontally within tile
    int numWarps = blockDim.x * blockDim.y >> 5; // num of warps within a block
    int threadBlockIdx = threadIdx.y * blockDim.x + threadIdx.x; // thread index within a block
    int warpId = threadBlockIdx >> 5; // Id of the warp within a block
    int laneId = threadBlockIdx & 31; // Within a warp the id of a thread

    int warpColGroup = warpId % slabDimCols;
    int warpRowGroup = warpId / slabDimCols;
    int warpsPerColGroup = numWarps / slabDimCols;

    int slabRowIdx = laneId >> 3; // Among the 4 rows in a slab which one the thread is assigned to
    int slabColIdx = laneId & 7; // among the 8 columns in a slab which one is the thread assigned to

    int colTile = 32 * warpColGroup + 4 * slabColIdx;
    int shared_segment = colTile >> 5;
    int shared_bank_idx = colTile & 31;
    int new_shared_bank_idx = (shared_segment + shared_bank_idx) & 31;
    int newColTile = (shared_segment << 5) + new_shared_bank_idx;        

    for (int chunk = 0; chunk < K; chunk += SUBTILE){
        int threadRowGlobalOriginA = startRow;
        int threadColGlobalOriginA = chunk;

        int threadRowGlobalOriginB = chunk;
        int threadColGlobalOriginB =  startCol;

        LoaderFunc::run(A, ATile, B, BTile, M, K, N,
                        startRow, startCol,
                        threadRowTile, threadColTile,
                        threadRowGlobalOriginA, threadColGlobalOriginA,
                        threadRowGlobalOriginB, threadColGlobalOriginB,
                        slabDimRows, slabDimCols,
                        warpRowGroup, warpsPerColGroup,
                        slabRowIdx,
                        colTile, newColTile);
        __syncthreads();


        int kmax = min(SUBTILE, K - chunk);
        ComputeFunc::run(ATile, BTile, K, kmax,
                            sum, threadRowTile, threadColTile, newColTile);
        __syncthreads();
    }
    

    StoreFunc::run(sum, C, M, N, startRow, startCol,
                threadRowTile, threadColTile, alpha, beta);
        
}


__global__ void GEMMSubtileRegNaive(int M,int N,int K,
                                 float alpha,
                                 const float* __restrict__ A,
                                 const float* __restrict__ B,
                                 float beta,
                                 float* __restrict__ C)
{
    GEMMSubTiling<LoaderNaive, ComputeNaive, StoreVec4>(M, N, K, alpha, A, B, beta, C);
}

__global__ void GEMMSubtileRegVec4(int M,int N,int K,
                                 float alpha,
                                 const float* __restrict__ A,
                                 const float* __restrict__ B,
                                 float beta,
                                 float* __restrict__ C)
{
    GEMMSubTiling<LoaderVec4, ComputeNaive, StoreVec4>(M, N, K, alpha, A, B, beta, C);
}

__global__ void GEMMSubtileWarpSlab(int M,int N,int K,
                                 float alpha,
                                 const float* __restrict__ A,
                                 const float* __restrict__ B,
                                 float beta,
                                 float* __restrict__ C)
{
    GEMMSubTilingSlab<LoaderSlab, ComputeNaive, StoreVec4>(M, N, K, alpha, A, B, beta, C);
}


__global__ void GEMMSubtileSwzl(int M,int N,int K,
                                 float alpha,
                                 const float* __restrict__ A,
                                 const float* __restrict__ B,
                                 float beta,
                                 float* __restrict__ C)
{
    GEMMSubTilingSwzl<LoaderSwzl, ComputeSwzl, StoreVec4>(M, N, K, alpha, A, B, beta, C);
}

