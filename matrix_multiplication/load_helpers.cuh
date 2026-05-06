#pragma once
#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include "config.h"
#include "param_init.cuh"



__device__ __forceinline__
void load_vec4_or_scalar_to_shared_transposed(
    const float* __restrict__ src,
    int row, int col, int ld,
    int rowBound, int colBound,
    float* dst, int dst_ld,   // full shared mem pointer + leading dim
    int rowTile, int colTile)
{
    int idx = row * ld + col;

    if (row < rowBound && col + 3 < colBound && ((idx & 3) == 0)) {
        float4 tmp = reinterpret_cast<const float4*>(&src[idx])[0];

        dst[(colTile + 0) * dst_ld + rowTile] = tmp.x;
        dst[(colTile + 1) * dst_ld + rowTile] = tmp.y;
        dst[(colTile + 2) * dst_ld + rowTile] = tmp.z;
        dst[(colTile + 3) * dst_ld + rowTile] = tmp.w;
    } else {
        dst[(colTile + 0) * dst_ld + rowTile] =
            (row < rowBound && col + 0 < colBound) ? src[idx + 0] : 0.0f;

        dst[(colTile + 1) * dst_ld + rowTile] =
            (row < rowBound && col + 1 < colBound) ? src[idx + 1] : 0.0f;

        dst[(colTile + 2) * dst_ld + rowTile] =
            (row < rowBound && col + 2 < colBound) ? src[idx + 2] : 0.0f;

        dst[(colTile + 3) * dst_ld + rowTile] =
            (row < rowBound && col + 3 < colBound) ? src[idx + 3] : 0.0f;
    }
}




template <typename InputT>
__device__ __forceinline__
void load_to_shared(const InputT* __restrict__ src,
                    int row, int col, int ld,
                    int rowBound, int colBound,
                    InputT* dst,
                    int dstCol)
{
    constexpr int VEC_BYTES = 16;
    constexpr int VEC_ELEMS = VEC_BYTES / sizeof(InputT);

    int idx = row * ld + col;

    if (row < rowBound &&
        col + VEC_ELEMS - 1 < colBound &&
        ((reinterpret_cast<uintptr_t>(&src[idx]) & 0xF) == 0))
    {
        int4 raw = reinterpret_cast<const int4*>(&src[idx])[0];
        InputT* vals = reinterpret_cast<InputT*>(&raw);

#pragma unroll
        for (int v = 0; v < VEC_ELEMS; v++) {
            dst[dstCol + v] = vals[v];
        }
    }
    else {
#pragma unroll
        for (int v = 0; v < VEC_ELEMS; v++) {
            dst[dstCol + v] =
                (row < rowBound && col + v < colBound)
                    ? src[idx + v]
                    : InputT{};
        }
    }
}


template <typename InputT>
__device__ __forceinline__
void load_to_shared_pad_free(const InputT* __restrict__ src,
                             int row, int col, int ld,
                             int rowBound, int colBound,
                             InputT* dst,
                             int dstCol)
{
    constexpr int VEC_BYTES = 16;
    constexpr int VEC_ELEMS = VEC_BYTES / sizeof(InputT);

    int idx = row * ld + col;

    if (row < rowBound &&
        col + VEC_ELEMS - 1 < colBound &&
        ((reinterpret_cast<uintptr_t>(&src[idx]) & 0xF) == 0) &&
        ((reinterpret_cast<uintptr_t>(&dst[dstCol]) & 0xF) == 0))
    {
        *reinterpret_cast<int4*>(&dst[dstCol]) =
            *reinterpret_cast<const int4*>(&src[idx]);
    }
    else {
#pragma unroll
        for (int v = 0; v < VEC_ELEMS; v++) {
            dst[dstCol + v] =
                (row < rowBound && col + v < colBound)
                    ? src[idx + v]
                    : InputT{};
        }
    }
}



template <typename InputT>
__device__ __forceinline__
void load_subtile_linear_slab(const InputT* __restrict__ A,
                       InputT ATile[SUBTILE][SUBTILE+PADDING_WARP],
                       const InputT* __restrict__ B,
                       InputT BTile[SUBTILE][SUBTILE+PADDING_WARP],
                       int M, int K, int N,
                       int startRow, int startCol,
                       int chunk,
                       const SlabParams& params)
{   

    constexpr int VEC_BYTES = 16;
    constexpr int VEC_ELEMS = VEC_BYTES / sizeof(InputT);
    constexpr int VEC_COLS_PER_SLAB = 32 / VEC_ELEMS;
    constexpr int SLAB_ROWS = 32 / VEC_COLS_PER_SLAB;

    constexpr int SLAB_DIM_ROWS =(SUBTILE + SLAB_ROWS - 1) / SLAB_ROWS;
    constexpr int SLAB_DIM_COLS = (SUBTILE + 31) >> 5;
    constexpr int TOTAL_SLABS   = SLAB_DIM_ROWS * SLAB_DIM_COLS;

    int threadRowGlobalOriginA = startRow;
    int threadColGlobalOriginA = chunk;

    int threadRowGlobalOriginB = chunk;
    int threadColGlobalOriginB = startCol;

    for (int slabNum = params.warpId; slabNum < TOTAL_SLABS; slabNum += NUM_WARPS_PER_BLOCK){ // loop over all slabs
        int slabRowStart = slabNum / SLAB_DIM_COLS; // map back to the starting row of tile for that slab
        int slabColStart = slabNum % SLAB_DIM_COLS; // map back to starting col of tile for that slab
        int rowTile = SLAB_ROWS * slabRowStart + params.slabRowIdx; // tells us which row of the tile the thread is working on
        int colTile = 32 * slabColStart + VEC_ELEMS * params.slabColIdx; // tells us which col of the tile the thread is starting on

        //Load in A
        int rowA = threadRowGlobalOriginA + rowTile;
        int colA = threadColGlobalOriginA + colTile;

        if constexpr ((SUBTILE + PADDING_WARP) % VEC_ELEMS == 0){
            load_to_shared_pad_free<InputT>(A, rowA, colA, K,
                                    M, K, &ATile[rowTile][0], colTile);
        } else{
            load_to_shared<InputT>(A, rowA, colA, K,
                                    M, K, &ATile[rowTile][0], colTile);
        }
        



        //Load in B
        int rowB = threadRowGlobalOriginB + rowTile;
        int colB = threadColGlobalOriginB + colTile;
        if constexpr ((SUBTILE + PADDING_WARP)% VEC_ELEMS == 0){
            load_to_shared_pad_free<InputT>(B, rowB, colB, N,
                                    K, N, &BTile[rowTile][0], colTile);
        }
        else{
            load_to_shared<InputT>(B, rowB, colB, N,
                                        K, N, &BTile[rowTile][0], colTile);
        }
        
    }


}

template <typename InputT, int NumWarps, int Padding>
__device__ __forceinline__
void load_subtile_linear_slab_gendim(
    const InputT* __restrict__ A,
    InputT ATile[SUBTILE_MN][SUBTILE_K + Padding],
    const InputT* __restrict__ B,
    InputT BTile[SUBTILE_K][SUBTILE_MN + Padding],
    int M, int K, int N,
    int startRow, int startCol,
    int chunk,
    const SlabParamsGenDim& params)
{
    constexpr int VEC_BYTES = 16;
    constexpr int VEC_ELEMS = VEC_BYTES / sizeof(InputT);
    constexpr int WARP_SIZE = 32;

    constexpr int A_VEC_COLS = SUBTILE_K  / VEC_ELEMS;
    constexpr int B_VEC_COLS = SUBTILE_MN / VEC_ELEMS;

    constexpr int A_SLAB_ROWS = WARP_SIZE / A_VEC_COLS;
    constexpr int B_SLAB_ROWS = WARP_SIZE / B_VEC_COLS;

  

    constexpr int TOTAL_SLABS =
        (SUBTILE_MN * SUBTILE_K) / (WARP_SIZE * VEC_ELEMS);

    int threadRowGlobalOriginA = startRow;
    int threadColGlobalOriginA = chunk;

    int threadRowGlobalOriginB = chunk;
    int threadColGlobalOriginB = startCol;

    for (int slabNum = params.warpId;
         slabNum < TOTAL_SLABS;
         slabNum += NumWarps)
    {
        int rowTileA = A_SLAB_ROWS * slabNum + params.slabRowIdxA;
        int colTileA = VEC_ELEMS * params.slabColIdxA;

        int rowA = threadRowGlobalOriginA + rowTileA;
        int colA = threadColGlobalOriginA + colTileA;

        if constexpr ((SUBTILE_K + PADDING_GEN_DIM) % VEC_ELEMS == 0){
            load_to_shared_pad_free<InputT>(
                A,
                rowA, colA, K,
                M, K,
                &ATile[rowTileA][0],
                colTileA
            );
            
        } else {
            load_to_shared<InputT>(
            A,
            rowA, colA, K,
            M, K,
            &ATile[rowTileA][0],
            colTileA
        );
        }

        int rowTileB = B_SLAB_ROWS * slabNum + params.slabRowIdxB;
        int colTileB = VEC_ELEMS * params.slabColIdxB;

        int rowB = threadRowGlobalOriginB + rowTileB;
        int colB = threadColGlobalOriginB + colTileB;
        if constexpr ((SUBTILE_MN + PADDING_GEN_DIM) % VEC_ELEMS== 0){
            load_to_shared_pad_free<InputT>(
            B,
            rowB, colB, N,
            K, N,
            &BTile[rowTileB][0],
            colTileB
        );
        } else{
            load_to_shared<InputT>(
                B,
                rowB, colB, N,
                K, N,
                &BTile[rowTileB][0],
                colTileB
            );
        }
    }
}