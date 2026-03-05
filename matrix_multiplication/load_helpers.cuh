#pragma once
#include <cuda_runtime.h>
#include "config.h"

__device__ __forceinline__
void load_subtile_naive(const float* __restrict__ A,
                        float ATile[SUBTILE][SUBTILE+1],
                        const float* __restrict__ B,
                        float BTile[SUBTILE][SUBTILE+1],
                        int M, int K, int N,
                        int startRow, int startCol,
                        int threadRowTile, int threadColTile,
                        int threadRowGlobalOriginA, int threadColGlobalOriginA,
                        int threadRowGlobalOriginB, int threadColGlobalOriginB)
{

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


// Vectorized version (float4 loads)

__device__ __forceinline__
void load_subtile_vec4(const float* __restrict__ A,
                       float ATile[SUBTILE][SUBTILE+1],
                       const float* __restrict__ B,
                       float BTile[SUBTILE][SUBTILE+1],
                       int M, int K, int N,
                       int startRow, int startCol,
                       int threadRowTile, int threadColTile,
                       int threadRowGlobalOriginA, int threadColGlobalOriginA,
                       int threadRowGlobalOriginB, int threadColGlobalOriginB)
{

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

            

            //Prefer the floa4 load when possible
            int idxA = rowA * K + colA;
            if (rowA < M && colA + 3 < K && ((idxA & 3) == 0)){
                float4 tmpA = reinterpret_cast<const float4*>(&A[rowA * K + colA])[0];
                ATile[rowTile][colTile + 0] = tmpA.x;
                ATile[rowTile][colTile + 1] = tmpA.y;
                ATile[rowTile][colTile + 2] = tmpA.z;
                ATile[rowTile][colTile + 3] = tmpA.w;
            } 
            //Otherwise fall back to scalar loads
            else {
                #pragma unroll
                for (int k =0; k < 4; k++){
                    int colAGlobal = colA + k;
                    ATile[rowTile][colTile + k] =
                        (rowA < M && colAGlobal < K) ? A[rowA * K + colAGlobal] : 0.0f;
                }
            }
            int idxB = rowB * N + colB;
            if (rowB < K && colB + 3 < N && ((idxB & 3) == 0)){
                float4 tmpB = reinterpret_cast<const float4*>(&B[rowB * N +  colB])[0];
                BTile[rowTile][colTile + 0] = tmpB.x;
                BTile[rowTile][colTile + 1] = tmpB.y;
                BTile[rowTile][colTile + 2] = tmpB.z;
                BTile[rowTile][colTile + 3] = tmpB.w;
            } else {
                #pragma unroll
                for (int k =0; k < 4; k++){
                    int colBGlobal = colB + k;
                    BTile[rowTile][colTile + k] =
                        (rowB < K && colBGlobal < N) ? B[rowB * N + colBGlobal] : 0.0f;
                }
            }
    
        }
    }
}


__device__ __forceinline__
void load_subtile_slab(const float* __restrict__ A,
                       float ATile[SUBTILE][SUBTILE+1],
                       const float* __restrict__ B,
                       float BTile[SUBTILE][SUBTILE+1],
                       int M, int K, int N,
                       int startRow, int startCol,
                       int threadRowTile, int threadColTile,
                       int threadRowGlobalOriginA, int threadColGlobalOriginA,
                       int threadRowGlobalOriginB, int threadColGlobalOriginB)
{
int slabDimRows = (SUBTILE + 3) >> 2; // num of slabs that fit vertically within tile
int slabDimCols = (SUBTILE + 31) >> 5;  // num of slabs that fit horizontally within tile
int totalSlabs =  slabDimRows * slabDimCols;

int numWarps = blockDim.x * blockDim.y >> 5; // num of warps within a block

int threadBlockIdx = threadIdx.y * blockDim.x + threadIdx.x; // thread index within a block
int warpId = threadBlockIdx >> 5; // Id of the warp within a block
int laneId = threadBlockIdx & 31; // Within a warp the id of a thread

int slabRowIdx = laneId >> 3; // Among the 4 rows in a slab which one the thread is assigned to
int slabColIdx = laneId & 7; // among the 8 columns in a slab which one is the thread assigned to

 

for (int slabNum = warpId; slabNum < totalSlabs; slabNum += numWarps){ // loop over all slabs
    int slabRowStart = slabNum / slabDimCols; // map back to the starting row of tile for that slab
    int slabColStart = slabNum % slabDimCols; // map back to starting col of tile for that slab
    int rowTile = 4 * slabRowStart + slabRowIdx; // tells us which row of the tile the thread is working on
    int colTile = 32 * slabColStart + 4 * slabColIdx; // tells us which col of the tile the thread is starting on

    //Load in A
    int rowA = threadRowGlobalOriginA + rowTile;
    int colA = threadColGlobalOriginA + colTile;
    int idxA = rowA * K + colA;
    if (rowA < M && colA + 3 < K && ((idxA & 3) == 0)){
                float4 tmpA = reinterpret_cast<const float4*>(&A[idxA])[0];
                ATile[rowTile][colTile + 0] = tmpA.x;
                ATile[rowTile][colTile + 1] = tmpA.y;
                ATile[rowTile][colTile + 2] = tmpA.z;
                ATile[rowTile][colTile + 3] = tmpA.w;
            } 
    //Otherwise fall back to scalar loads
    else {
        #pragma unroll
        for (int k =0; k < 4; k++){
            int colAGlobal = colA + k;
            ATile[rowTile][colTile + k] =
                (rowA < M && colAGlobal < K) ? A[rowA * K + colAGlobal] : 0.0f;
        }
    }


    //Load in B
    int rowB = threadRowGlobalOriginB + rowTile;
    int colB = threadColGlobalOriginB + colTile;
    int idxB = rowB * N +  colB;
    if (rowB < K && colB + 3 < N && ((idxB & 3) == 0)){
                float4 tmpB = reinterpret_cast<const float4*>(&B[idxB])[0];
                BTile[rowTile][colTile + 0] = tmpB.x;
                BTile[rowTile][colTile + 1] = tmpB.y;
                BTile[rowTile][colTile + 2] = tmpB.z;
                BTile[rowTile][colTile + 3] = tmpB.w;
            } else {
                #pragma unroll
                for (int k =0; k < 4; k++){
                    int colBGlobal = colB + k;
                    BTile[rowTile][colTile + k] =
                        (rowB < K && colBGlobal < N) ? B[rowB * N + colBGlobal] : 0.0f;
                }
            }
}


}



__device__ __forceinline__
void load_subtile_slab_swizzle(const float* __restrict__ A,
                       float ATile[SUBTILE][SUBTILE+1],
                       const float* __restrict__ B,
                       float BTile[SUBTILE][SUBTILE+1],
                       int M, int K, int N,
                       int startRow, int startCol,
                       int threadRowTile, int threadColTile,
                       int threadRowGlobalOriginA, int threadColGlobalOriginA,
                       int threadRowGlobalOriginB, int threadColGlobalOriginB){
int slabDimRows = (SUBTILE + 3) >> 2; // num of slabs that fit vertically within tile
int slabDimCols = (SUBTILE + 31) >> 5;  // num of slabs that fit horizontally within tile
int totalSlabs =  slabDimRows * slabDimCols;

int numWarps = blockDim.x * blockDim.y >> 5; // num of warps within a block

int threadBlockIdx = threadIdx.y * blockDim.x + threadIdx.x; // thread index within a block
int warpId = threadBlockIdx >> 5; // Id of the warp within a block
int laneId = threadBlockIdx & 31; // Within a warp the id of a thread

int slabRowIdx = laneId >> 3; // Among the 4 rows in a slab which one the thread is assigned to
int slabColIdx = laneId & 7; // among the 8 columns in a slab which one is the thread assigned to

 

for (int slabNum = warpId; slabNum < totalSlabs; slabNum += numWarps){ // loop over all slabs
    int slabRowStart = slabNum / slabDimCols; // map back to the starting row of tile for that slab
    int slabColStart = slabNum % slabDimCols; // map back to starting col of tile for that slab
    int rowTile = 4 * slabRowStart + slabRowIdx; // tells us which row of the tile the thread is working on
    int colTile = 32 * slabColStart + 4 * slabColIdx; // tells us which col of the tile the thread is starting on

    //Load in A
    int rowA = threadRowGlobalOriginA + rowTile;
    int colA = threadColGlobalOriginA + colTile;
    int idxA = rowA * K + colA;
    if (rowA < M && colA + 3 < K && ((idxA & 3) == 0)){
                float4 tmpA = reinterpret_cast<const float4*>(&A[idxA])[0];
                ATile[rowTile][colTile + 0] = tmpA.x;
                ATile[rowTile][colTile + 1] = tmpA.y;
                ATile[rowTile][colTile + 2] = tmpA.z;
                ATile[rowTile][colTile + 3] = tmpA.w;
            } 
    //Otherwise fall back to scalar loads
    else {
        #pragma unroll
        for (int k =0; k < 4; k++){
            int colAGlobal = colA + k;
            ATile[rowTile][colTile + k] =
                (rowA < M && colAGlobal < K) ? A[rowA * K + colAGlobal] : 0.0f;
        }
    }


    //Load in B
    int rowB = threadRowGlobalOriginB + rowTile;
    int colB = threadColGlobalOriginB + colTile;
    int idxB = rowB * N +  colB;

    int shared_segment = colTile >> 5;
    int shared_bank_idx = colTile & 31;
    int new_shared_bank_idx = (shared_segment + shared_bank_idx) & 31;
    int newColTile = (shared_segment << 5) + new_shared_bank_idx;


    if (rowB < K && colB + 3 < N && ((idxB & 3) == 0)){
                float4 tmpB = reinterpret_cast<const float4*>(&B[idxB])[0];
                BTile[rowTile][newColTile + 0] = tmpB.x;
                BTile[rowTile][newColTile + 1] = tmpB.y;
                BTile[rowTile][newColTile + 2] = tmpB.z;
                BTile[rowTile][newColTile + 3] = tmpB.w;
            } else {
                #pragma unroll
                for (int k =0; k < 4; k++){
                    int colBGlobal = colB + k;
                    BTile[rowTile][newColTile+ k] =
                        (rowB < K && colBGlobal < N) ? B[rowB * N + colBGlobal] : 0.0f;
                }
            }
}


}




struct LoaderNaive {
    __device__ __forceinline__
    static void run(
        const float* __restrict__ A,
        float ATile[SUBTILE][SUBTILE+1],
        const float* __restrict__ B,
        float BTile[SUBTILE][SUBTILE+1],
        int M, int K, int N,
        int startRow, int startCol,
        int threadRowTile, int threadColTile,
        int threadRowGlobalOriginA, int threadColGlobalOriginA,
        int threadRowGlobalOriginB, int threadColGlobalOriginB
    ) {
        load_subtile_naive(
            A, ATile, B, BTile,
            M, K, N,
            startRow, startCol,
            threadRowTile, threadColTile,
            threadRowGlobalOriginA, threadColGlobalOriginA,
            threadRowGlobalOriginB, threadColGlobalOriginB
        );
    }
};


struct LoaderVec4 {
    __device__ __forceinline__
    static void run(
        const float* __restrict__ A,
        float ATile[SUBTILE][SUBTILE+1],
        const float* __restrict__ B,
        float BTile[SUBTILE][SUBTILE+1],
        int M, int K, int N,
        int startRow, int startCol,
        int threadRowTile, int threadColTile,
        int threadRowGlobalOriginA, int threadColGlobalOriginA,
        int threadRowGlobalOriginB, int threadColGlobalOriginB
    ) {
        load_subtile_vec4(
            A, ATile, B, BTile,
            M, K, N,
            startRow, startCol,
            threadRowTile, threadColTile,
            threadRowGlobalOriginA, threadColGlobalOriginA,
            threadRowGlobalOriginB, threadColGlobalOriginB
        );
    }
};


struct LoaderSlab {
    __device__ __forceinline__
    static void run(
        const float* __restrict__ A,
        float ATile[SUBTILE][SUBTILE+1],
        const float* __restrict__ B,
        float BTile[SUBTILE][SUBTILE+1],
        int M, int K, int N,
        int startRow, int startCol,
        int threadRowTile, int threadColTile,
        int threadRowGlobalOriginA, int threadColGlobalOriginA,
        int threadRowGlobalOriginB, int threadColGlobalOriginB
    ) {
        load_subtile_slab(
            A, ATile, B, BTile,
            M, K, N,
            startRow, startCol,
            threadRowTile, threadColTile,
            threadRowGlobalOriginA, threadColGlobalOriginA,
            threadRowGlobalOriginB, threadColGlobalOriginB
        );
    }
};





struct LoaderSwzl {
    __device__ __forceinline__
    static void run(
        const float* __restrict__ A,
        float ATile[SUBTILE][SUBTILE+1],
        const float* __restrict__ B,
        float BTile[SUBTILE][SUBTILE+1],
        int M, int K, int N,
        int startRow, int startCol,
        int threadRowTile, int threadColTile,
        int threadRowGlobalOriginA, int threadColGlobalOriginA,
        int threadRowGlobalOriginB, int threadColGlobalOriginB
    ) {
        load_subtile_slab_swizzle(
            A, ATile, B, BTile,
            M, K, N,
            startRow, startCol,
            threadRowTile, threadColTile,
            threadRowGlobalOriginA, threadColGlobalOriginA,
            threadRowGlobalOriginB, threadColGlobalOriginB
        );
    }
};


