#pragma once
#include <cuda_runtime.h>
#include "config.h"
#include "param_init.cuh"

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


__device__ __forceinline__
void load_vec4_or_scalar_to_shared(const float* __restrict__ src,
                                   int row, int col, int ld,
                                   int rowBound, int colBound,
                                   float* dst, int dstCol)
{
    int idx = row * ld + col;

    if (row < rowBound && col + 3 < colBound && ((idx & 3) == 0)) {
        float4 tmp = reinterpret_cast<const float4*>(&src[idx])[0];
        dst[dstCol + 0] = tmp.x;
        dst[dstCol + 1] = tmp.y;
        dst[dstCol + 2] = tmp.z;
        dst[dstCol + 3] = tmp.w;
    } else {
        dst[dstCol + 0] = (row < rowBound && col + 0 < colBound) ? src[idx + 0] : 0.0f;
        dst[dstCol + 1] = (row < rowBound && col + 1 < colBound) ? src[idx + 1] : 0.0f;
        dst[dstCol + 2] = (row < rowBound && col + 2 < colBound) ? src[idx + 2] : 0.0f;
        dst[dstCol + 3] = (row < rowBound && col + 3 < colBound) ? src[idx + 3] : 0.0f;
    }
}


__device__ __forceinline__
void load_vec4_or_scalar_to_shared_B(const float* __restrict__ src,
                                   int row, int col, int ld,
                                   int rowBound, int colBound,
                                   float* dst, int dstCol)
{
    int idx = row * ld + col;

    if (row < rowBound && col + 3 < colBound && ((idx & 3) == 0)) {
        float4 tmp = reinterpret_cast<const float4*>(&src[idx])[0];
        dst[dstCol + 0] = tmp.x;
        dst[dstCol + 1] = tmp.y;
        dst[dstCol + 2] = tmp.z;
        dst[dstCol + 3] = tmp.w;
    } else {
        dst[dstCol + 0] = (row < rowBound && col + 0 < colBound) ? src[idx + 0] : 0.0f;
        dst[dstCol + 1] = (row < rowBound && col + 1 < colBound) ? src[idx + 1] : 0.0f;
        dst[dstCol + 2] = (row < rowBound && col + 2 < colBound) ? src[idx + 2] : 0.0f;
        dst[dstCol + 3] = (row < rowBound && col + 3 < colBound) ? src[idx + 3] : 0.0f;
    }
}

__device__ __forceinline__
void load_subtile_vec4(const float* __restrict__ A,
                       float ATile[SUBTILE][SUBTILE+1],
                       const float* __restrict__ B,
                       float BTile[SUBTILE][SUBTILE+1],
                       int M, int K, int N,
                       int threadRowGlobalOriginA, int threadColGlobalOriginA,
                       int threadRowGlobalOriginB, int threadColGlobalOriginB,
                       const NaiveParams& params)
{

    #pragma unroll
    for (int i = 0; i < SUB; i++){
        int rowTile = params.threadRowTile + i;
        int rowA = threadRowGlobalOriginA + rowTile;
        int rowB = threadRowGlobalOriginB + rowTile;

        
        #pragma unroll
        for (int j = 0; j < SUB; j+=4){
            int colTile = params.threadColTile + j;
            int colA = threadColGlobalOriginA + colTile;
            int colB = threadColGlobalOriginB + colTile;

        
            load_vec4_or_scalar_to_shared(A, rowA, colA, K,
                                   M, K, &ATile[rowTile][0], colTile);
            load_vec4_or_scalar_to_shared(B, rowB, colB, N,
                                   K, N, &BTile[rowTile][0], colTile);

    
        }
    }
}


__device__ __forceinline__
void load_subtile_slab(const float* __restrict__ A,
                       float ATile[SUBTILE][SUBTILE+1],
                       const float* __restrict__ B,
                       float BTile[SUBTILE][SUBTILE+1],
                       int M, int K, int N,
                       int threadRowGlobalOriginA, int threadColGlobalOriginA,
                       int threadRowGlobalOriginB, int threadColGlobalOriginB,
                       const SlabParams& params)
{


for (int slabNum = params.warpId; slabNum < params.totalSlabs; slabNum += params.numWarps){ // loop over all slabs
    int slabRowStart = slabNum / params.slabDimCols; // map back to the starting row of tile for that slab
    int slabColStart = slabNum % params.slabDimCols; // map back to starting col of tile for that slab
    int rowTile = 4 * slabRowStart +params. slabRowIdx; // tells us which row of the tile the thread is working on
    int colTile = 32 * slabColStart + 4 * params.slabColIdx; // tells us which col of the tile the thread is starting on

    //Load in A
    int rowA = threadRowGlobalOriginA + rowTile;
    int colA = threadColGlobalOriginA + colTile;

    load_vec4_or_scalar_to_shared(A, rowA, colA, K,
                                   M, K, &ATile[rowTile][0], colTile);



    //Load in B
    int rowB = threadRowGlobalOriginB + rowTile;
    int colB = threadColGlobalOriginB + colTile;
    load_vec4_or_scalar_to_shared(B, rowB, colB, N,
                                   K, N, &BTile[rowTile][0], colTile);
    
}


}




__device__ __forceinline__
void load_subtile_slab_swizzle(const float* __restrict__ A,
                       float ATile[SUBTILE][SUBTILE+1],
                       const float* __restrict__ B,
                       float BTile[SUBTILE][SUBTILE+1],
                       int M, int K, int N,
                       int threadRowGlobalOriginA, int threadColGlobalOriginA,
                       int threadRowGlobalOriginB, int threadColGlobalOriginB,
                       const SwizzleParams& params){

 

for (int slabRowStart = params.warpRowGroup; slabRowStart < params.slabDimRows; slabRowStart += params.warpsPerColGroup){ // loop over all slabs
    int rowTile = 4 * slabRowStart + params.slabRowIdx; // tells us which row of the tile the thread is working on

    //Load in A
    int rowA = threadRowGlobalOriginA + rowTile;
    int colA = threadColGlobalOriginA + params.colTile;
    load_vec4_or_scalar_to_shared(A, rowA, colA, K,
                                   M, K, &ATile[rowTile][0], params.colTile);


    //Load in B
    int rowB = threadRowGlobalOriginB + rowTile;
    int colB = threadColGlobalOriginB + params.colTile;
    load_vec4_or_scalar_to_shared_B(B, rowB, colB, N,
                                   K, N, &BTile[rowTile][0], params.newColTile);
}


}



__device__ __forceinline__
void load_subtile_slab_fake_swizzle(const float* __restrict__ A,
                       float ATile[SUBTILE][SUBTILE+1],
                       const float* __restrict__ B,
                       float BTile[SUBTILE][SUBTILE+1],
                       int M, int K, int N,
                       int threadRowGlobalOriginA, int threadColGlobalOriginA,
                       int threadRowGlobalOriginB, int threadColGlobalOriginB,
                       const FakeSwizzleParams& params){

 

for (int slabRowStart = params.warpRowGroup; slabRowStart < params.slabDimRows; slabRowStart += params.warpsPerColGroup){ // loop over all slabs
    int rowTile = 4 * slabRowStart + params.slabRowIdx; // tells us which row of the tile the thread is working on

    //Load in A
    int rowA = threadRowGlobalOriginA + rowTile;
    int colA = threadColGlobalOriginA + params.colTile;
    load_vec4_or_scalar_to_shared(A, rowA, colA, K,
                                   M, K, &ATile[rowTile][0], params.colTile);


    //Load in B
    int rowB = threadRowGlobalOriginB + rowTile;
    int colB = threadColGlobalOriginB + params.colTile;
    load_vec4_or_scalar_to_shared_B(B, rowB, colB, N,
                                   K, N, &BTile[rowTile][0], params.newColTile);
}


}




__device__ __forceinline__
void load_with_params(const NaiveParams& params,
                      const float* __restrict__ A,
                      float ATile[SUBTILE][SUBTILE+1],
                      const float* __restrict__ B,
                      float BTile[SUBTILE][SUBTILE+1],
                      int M, int K, int N,
                      int startRow, int startCol,
                      int chunk)
{
    int threadRowGlobalOriginA = startRow;
    int threadColGlobalOriginA = chunk;

    int threadRowGlobalOriginB = chunk;
    int threadColGlobalOriginB = startCol;

    load_subtile_vec4(
        A, ATile, B, BTile,
        M, K, N,
        threadRowGlobalOriginA, threadColGlobalOriginA,
        threadRowGlobalOriginB, threadColGlobalOriginB,
        params
    );
}


__device__ __forceinline__
void load_with_params(const SlabParams& params,
                      const float* __restrict__ A,
                      float ATile[SUBTILE][SUBTILE+1],
                      const float* __restrict__ B,
                      float BTile[SUBTILE][SUBTILE+1],
                      int M, int K, int N,
                      int startRow, int startCol,
                      int chunk)
{
    int threadRowGlobalOriginA = startRow;
    int threadColGlobalOriginA = chunk;

    int threadRowGlobalOriginB = chunk;
    int threadColGlobalOriginB = startCol;

    load_subtile_slab(
        A, ATile, B, BTile,
        M, K, N,
        threadRowGlobalOriginA, threadColGlobalOriginA,
        threadRowGlobalOriginB, threadColGlobalOriginB,
        params
    );
}


__device__ __forceinline__
void load_with_params(const SwizzleParams& params,
                      const float* __restrict__ A,
                      float ATile[SUBTILE][SUBTILE+1],
                      const float* __restrict__ B,
                      float BTile[SUBTILE][SUBTILE+1],
                      int M, int K, int N,
                      int startRow, int startCol,
                      int chunk)
{
    int threadRowGlobalOriginA = startRow;
    int threadColGlobalOriginA = chunk;

    int threadRowGlobalOriginB = chunk;
    int threadColGlobalOriginB = startCol;

    load_subtile_slab_swizzle(
        A, ATile, B, BTile,
        M, K, N,
        threadRowGlobalOriginA, threadColGlobalOriginA,
        threadRowGlobalOriginB, threadColGlobalOriginB,
        params
    );
}



__device__ __forceinline__
void load_with_params(const FakeSwizzleParams& params,
                      const float* __restrict__ A,
                      float ATile[SUBTILE][SUBTILE+1],
                      const float* __restrict__ B,
                      float BTile[SUBTILE][SUBTILE+1],
                      int M, int K, int N,
                      int startRow, int startCol,
                      int chunk)
{
    int threadRowGlobalOriginA = startRow;
    int threadColGlobalOriginA = chunk;

    int threadRowGlobalOriginB = chunk;
    int threadColGlobalOriginB = startCol;

    load_subtile_slab_fake_swizzle(
        A, ATile, B, BTile,
        M, K, N,
        threadRowGlobalOriginA, threadColGlobalOriginA,
        threadRowGlobalOriginB, threadColGlobalOriginB,
        params
    );
}