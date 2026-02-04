#pragma once
#include <cuda_runtime.h>
#include "config.h"

__device__ __forceinline__
void load_subtile_naive(const float* __restrict__ A,
                        float ATile[SUBTILE][SUBTILE+1],
                        const float* __restrict__ B,
                        float BTile[SUBTILE][SUBTILE+1],
                        int M, int K, int N,
                        int startRow, int startCol, int chunk,
                        int threadRowTile, int threadColTile,
                        int threadRowGlobalOriginA, int threadColGlobalOriginA,
                        int threadRowGlobalOriginB, int threadColGlobalOriginB)
{

    #pragma unroll
    for (int i = 0; i < SUB; i++){
        int rowA = threadRowGlobalOriginA + i;
        int rowB = threadRowGlobalOriginB + i;

        int rowTile = threadRowTile + i;

        #pragma unroll
        for (int j = 0; j < SUB; j++){

            int colA = threadColGlobalOriginA + j;
            int colB = threadColGlobalOriginB + j;

            int colTile = threadColTile + j;
        

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
                       int startRow, int startCol, int chunk,
                       int threadRowTile, int threadColTile,
                       int threadRowGlobalOriginA, int threadColGlobalOriginA,
                       int threadRowGlobalOriginB, int threadColGlobalOriginB)
{

    #pragma unroll
    for (int i = 0; i < SUB; i++){
        int rowA = threadRowGlobalOriginA + i;
        int rowB = threadRowGlobalOriginB + i;

        int rowTile = threadRowTile + i;
        #pragma unroll
        for (int j = 0; j < SUB; j+=4){

            int colA = threadColGlobalOriginA + j;
            int colB = threadColGlobalOriginB + j;

            int colTile = threadColTile + j;

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


// Vectorized + swizzle to avoid bank conflicts in B
__device__ __forceinline__
void load_subtile_vec4_swizzle(const float* __restrict__ A,
                               float ATile[SUBTILE][SUBTILE+1],
                               const float* __restrict__ B,
                               float BTile[SUBTILE][SUBTILE+1],
                               int M, int K, int N,
                               int startRow, int startCol, int chunk,
                               int threadRowTile, int threadColTile,
                               int threadRowGlobalOriginA, int threadColGlobalOriginA,
                               int threadRowGlobalOriginB, int threadColGlobalOriginB)
{
    static_assert(SUB % 4 == 0, "SUB must be multiple of 4 for vec4 loads");

    #pragma unroll
    for (int i = 0; i < SUB; i++){
        int rowA = threadRowGlobalOriginA + i;
        int rowB = threadRowGlobalOriginB + i;

        int rowTile = threadRowTile + i;

        // Swizzle based on shared row index (rowTile)
        const int colSwz = (rowTile & SWZ_MASK) << SWZ_SHIFT;

        #pragma unroll
        for (int j = 0; j < SUB; j += 4){
            int colA = threadColGlobalOriginA + j;
            int colB = threadColGlobalOriginB + j;

            int colTileA = threadColTile + j;
            int colTileB = threadColTile + j;

            int colTileBSwz = colTileB ^ colSwz;

            // ---- A load
            int idxA = rowA * K + colA;
            if (rowA < M && colA + 3 < K && ((idxA & 3) == 0)){
                float4 tmpA = *reinterpret_cast<const float4*>(&A[idxA]);
                ATile[rowTile][colTileA + 0] = tmpA.x;
                ATile[rowTile][colTileA + 1] = tmpA.y;
                ATile[rowTile][colTileA + 2] = tmpA.z;
                ATile[rowTile][colTileA + 3] = tmpA.w;
            } else {
                #pragma unroll
                for (int t = 0; t < 4; t++){
                    int c = colA + t;
                    ATile[rowTile][colTileA + t] =
                        (rowA < M && c < K) ? A[rowA * K + c] : 0.0f;
                }
            }

            // ---- B load (store swizzled in shared)
            int idxB = rowB * N + colB;
            if (rowB < K && colB + 3 < N && ((idxB & 3) == 0)){
                float4 tmpB = *reinterpret_cast<const float4*>(&B[idxB]);
                BTile[rowTile][colTileBSwz + 0] = tmpB.x;
                BTile[rowTile][colTileBSwz + 1] = tmpB.y;
                BTile[rowTile][colTileBSwz + 2] = tmpB.z;
                BTile[rowTile][colTileBSwz + 3] = tmpB.w;
            } else {
                #pragma unroll
                for (int t = 0; t < 4; t++){
                    int c = colB + t;
                    BTile[rowTile][colTileBSwz + t] =
                        (rowB < K && c < N) ? B[rowB * N + c] : 0.0f;
                }
            }
        }
    }
}


//Warp Tiling
__device__ __forceinline__
void warp_load_slab_vec4_swizzleB(
    const float* __restrict__ A,
    float ATile[SUBTILE][SUBTILE+1],
    const float* __restrict__ B,
    float BTile[SUBTILE][SUBTILE+1],
    int M, int K, int N,
    int startRow, int startCol, int chunk,
    int rowBase, int colBase,
    int laneId
){
    int laneRow  = laneId >> 3;     // 0..3
    int laneCol4 = laneId & 7;      // 0..7
    int rTile = rowBase + laneRow;
    int cTile = colBase + laneCol4 * 4;

    // Guard against accidental OOB in shared
    if (rTile >= SUBTILE || cTile + 3 >= SUBTILE) return;

    // ---- A
    int aRow = startRow + rTile;
    int aCol = chunk + cTile;
    int idxA = aRow * K + aCol;

    float4 a4 = make_float4(0,0,0,0);
    if (aRow < M && aCol + 3 < K && ((idxA & 3) == 0)) {
        a4 = *reinterpret_cast<const float4*>(&A[idxA]);
    } else if (aRow < M) {
        a4.x = (aCol+0 < K) ? A[idxA+0] : 0.f;
        a4.y = (aCol+1 < K) ? A[idxA+1] : 0.f;
        a4.z = (aCol+2 < K) ? A[idxA+2] : 0.f;
        a4.w = (aCol+3 < K) ? A[idxA+3] : 0.f;
    }

    ATile[rTile][cTile+0] = a4.x;
    ATile[rTile][cTile+1] = a4.y;
    ATile[rTile][cTile+2] = a4.z;
    ATile[rTile][cTile+3] = a4.w;

    // ---- B
    int bRow = chunk + rTile;
    int bCol = startCol + cTile;
    int idxB = bRow * N + bCol;

    const int swz  = (rTile & SWZ_MASK) << SWZ_SHIFT;
    const int cSwz = cTile ^ swz;

    if (cSwz < 0 || cSwz + 3 >= SUBTILE) return;

    float4 b4 = make_float4(0,0,0,0);
    if (bRow < K && bCol + 3 < N && ((idxB & 3) == 0)) {
        b4 = *reinterpret_cast<const float4*>(&B[idxB]);
    } else if (bRow < K) {
        b4.x = (bCol+0 < N) ? B[idxB+0] : 0.f;
        b4.y = (bCol+1 < N) ? B[idxB+1] : 0.f;
        b4.z = (bCol+2 < N) ? B[idxB+2] : 0.f;
        b4.w = (bCol+3 < N) ? B[idxB+3] : 0.f;
    }

    BTile[rTile][cSwz+0] = b4.x;
    BTile[rTile][cSwz+1] = b4.y;
    BTile[rTile][cSwz+2] = b4.z;
    BTile[rTile][cSwz+3] = b4.w;
}


struct LoaderNaive {
    __device__ __forceinline__
    static void run(
        const float* __restrict__ A,
        float ATile[SUBTILE][SUBTILE+1],
        const float* __restrict__ B,
        float BTile[SUBTILE][SUBTILE+1],
        int M, int K, int N,
        int startRow, int startCol, int chunk,
        int threadRowTile, int threadColTile,
        int threadRowGlobalOriginA, int threadColGlobalOriginA,
        int threadRowGlobalOriginB, int threadColGlobalOriginB
    ) {
        load_subtile_naive(
            A, ATile, B, BTile,
            M, K, N,
            startRow, startCol, chunk,
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
        int startRow, int startCol, int chunk,
        int threadRowTile, int threadColTile,
        int threadRowGlobalOriginA, int threadColGlobalOriginA,
        int threadRowGlobalOriginB, int threadColGlobalOriginB
    ) {
        load_subtile_vec4(
            A, ATile, B, BTile,
            M, K, N,
            startRow, startCol, chunk,
            threadRowTile, threadColTile,
            threadRowGlobalOriginA, threadColGlobalOriginA,
            threadRowGlobalOriginB, threadColGlobalOriginB
        );
    }
};


struct LoaderVec4Swizzle {

    __device__ __forceinline__
    static void run(
        const float* __restrict__ A,
        float ATile[SUBTILE][SUBTILE+1],
        const float* __restrict__ B,
        float BTile[SUBTILE][SUBTILE+1],
        int M, int K, int N,
        int startRow, int startCol, int chunk,
        int threadRowTile, int threadColTile,
        int threadRowGlobalOriginA, int threadColGlobalOriginA,
        int threadRowGlobalOriginB, int threadColGlobalOriginB
    ) {
        load_subtile_vec4_swizzle(
            A, ATile, B, BTile,
            M, K, N,
            startRow, startCol, chunk,
            threadRowTile, threadColTile,
            threadRowGlobalOriginA, threadColGlobalOriginA,
            threadRowGlobalOriginB, threadColGlobalOriginB
        );
    }
};


struct LoaderWarpSlabVec4SwizzleB {
    __device__ __forceinline__
    static void run(
        const float* __restrict__ A,
        float ATile[SUBTILE][SUBTILE+1],
        const float* __restrict__ B,
        float BTile[SUBTILE][SUBTILE+1],
        int M, int K, int N,
        int startRow, int startCol, int chunk,
        int /*threadRowTile*/, int /*threadColTile*/,
        int /*threadRowGlobalOriginA*/, int /*threadColGlobalOriginA*/,
        int /*threadRowGlobalOriginB*/, int /*threadColGlobalOriginB*/
    ) {
        // NOTE: warp-slab loader doesn't use the per-thread microtile coordinates.
        // You must call it from the kernel with rowBase/colBase loops.
        // So this wrapper isn't very meaningful unless you redesign the interface.
    }
};
