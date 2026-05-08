#pragma once
#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include "config.h"
#include "param_init.cuh"
#include "load_utils.cuh"


// WARP-TILE LOADING
template <typename InputT, int NumWarps, int PAD, int SUBDIM_M, int SUBDIM_N, int SUBDIM_K>
__device__ __forceinline__
void load_subtile_warp(
    const InputT* __restrict__ A,
    InputT ATile[SUBDIM_M][SUBDIM_K + PAD],
    const InputT* __restrict__ B,
    InputT BTile[SUBDIM_K][SUBDIM_N + PAD],
    int M, int K, int N,
    int startRow, int startCol,
    int chunk,
    const SlabParamsGenDim& params)
{
    constexpr int VEC_BYTES = 16;
    constexpr int VEC_ELEMS = VEC_BYTES / sizeof(InputT);
    constexpr int WARP_SIZE = 32;

    constexpr int A_VEC_COLS = SUBDIM_K  / VEC_ELEMS;
    constexpr int B_VEC_COLS = SUBDIM_N / VEC_ELEMS;

    constexpr int A_SLAB_ROWS = WARP_SIZE / A_VEC_COLS;
    constexpr int B_SLAB_ROWS = WARP_SIZE / B_VEC_COLS;

  

    constexpr int TOTAL_SLABS_A =
        (SUBDIM_M * SUBDIM_K) / (WARP_SIZE * VEC_ELEMS);
    
    constexpr int TOTAL_SLABS_B =
        (SUBDIM_N * SUBDIM_K) / (WARP_SIZE * VEC_ELEMS);

    int threadRowGlobalOriginA = startRow;
    int threadColGlobalOriginA = chunk;

    int threadRowGlobalOriginB = chunk;
    int threadColGlobalOriginB = startCol;

    for (int slabNum = params.warpId;
         slabNum < TOTAL_SLABS_A;
         slabNum += NumWarps)
    {
        int rowTileA = A_SLAB_ROWS * slabNum + params.slabRowIdxA;
        int colTileA = VEC_ELEMS * params.slabColIdxA;

        int rowA = threadRowGlobalOriginA + rowTileA;
        int colA = threadColGlobalOriginA + colTileA;

        if constexpr ((SUBDIM_K + PAD) % VEC_ELEMS == 0){
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
    }
    for (int slabNum = params.warpId;
         slabNum < TOTAL_SLABS_B;
         slabNum += NumWarps){

        int rowTileB = B_SLAB_ROWS * slabNum + params.slabRowIdxB;
        int colTileB = VEC_ELEMS * params.slabColIdxB;

        int rowB = threadRowGlobalOriginB + rowTileB;
        int colB = threadColGlobalOriginB + colTileB;
        if constexpr ((SUBDIM_N + PAD) % VEC_ELEMS== 0){
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

//LINEAR LOADING
template <typename InputT, int NumThreads, int PAD,
          int SUBDIM_M, int SUBDIM_N, int SUBDIM_K>
__device__ __forceinline__
void load_subtile_linear(
    const InputT* __restrict__ A,
    InputT ATile[SUBDIM_M][SUBDIM_K + PAD],
    const InputT* __restrict__ B,
    InputT BTile[SUBDIM_K][SUBDIM_N + PAD],
    int M, int K, int N,
    int startRow, int startCol, int chunk,
    int tid)
{
    constexpr int VEC_BYTES = 16;
    constexpr int VEC_ELEMS = VEC_BYTES / sizeof(InputT);

    static_assert(SUBDIM_K % VEC_ELEMS == 0, "SUBDIM_K must be divisible by VEC_ELEMS");
    static_assert(SUBDIM_N % VEC_ELEMS == 0, "SUBDIM_N must be divisible by VEC_ELEMS");

    constexpr int A_VEC_COLS = SUBDIM_K / VEC_ELEMS;
    constexpr int B_VEC_COLS = SUBDIM_N / VEC_ELEMS;

    constexpr int NUM_LOADS_A = SUBDIM_M * A_VEC_COLS;
    constexpr int NUM_LOADS_B = SUBDIM_K * B_VEC_COLS;

    for (int idx = tid; idx < NUM_LOADS_A; idx += NumThreads) {
        int rowTileA = idx / A_VEC_COLS;
        int vecColA  = idx % A_VEC_COLS;
        int colTileA = vecColA * VEC_ELEMS;

        int rowA = startRow + rowTileA;
        int colA = chunk + colTileA;

        load_to_shared<InputT>(
            A,
            rowA, colA, K,
            M, K,
            &ATile[rowTileA][0],
            colTileA
        );
    }

    for (int idx = tid; idx < NUM_LOADS_B; idx += NumThreads) {
        int rowTileB = idx / B_VEC_COLS;
        int vecColB  = idx % B_VEC_COLS;
        int colTileB = vecColB * VEC_ELEMS;

        int rowB = chunk + rowTileB;
        int colB = startCol + colTileB;

        load_to_shared<InputT>(
            B,
            rowB, colB, N,
            K, N,
            &BTile[rowTileB][0],
            colTileB
        );
    }
}


//LINEAR LOADING TRANSPOSE

template <typename InputT, int NumThreads, int PAD, int SUBDIM_M, int SUBDIM_N, int SUBDIM_K>
__device__ __forceinline__
void load_subtile_linear_transposed(
    const InputT* __restrict__ A,
    InputT ATileT[SUBDIM_K][SUBDIM_M + PAD],
    const InputT* __restrict__ B,
    InputT BTile[SUBDIM_K][SUBDIM_N + PAD],
    int M, int K, int N,
    int startRow, int startCol, int chunk,
    int tid)
{
    constexpr int VEC_BYTES = 16;
    constexpr int VEC_ELEMS = VEC_BYTES / sizeof(InputT);
    constexpr int WARP_SIZE = 32;

    static_assert(SUBDIM_K  % VEC_ELEMS == 0, "SUBDIM_K must be divisible by 4");
    static_assert(SUBDIM_M % VEC_ELEMS == 0, "SUBDIM_M must be divisible by 4");
    static_assert(SUBDIM_N % VEC_ELEMS == 0, "SUBDIM_N must be divisible by 4");

    constexpr int A_VEC_COLS = SUBDIM_K  / VEC_ELEMS;
    constexpr int B_VEC_COLS = SUBDIM_N / VEC_ELEMS;

    constexpr int NUM_LOADS_A = SUBDIM_M * A_VEC_COLS;
    constexpr int NUM_LOADS_B = SUBDIM_K * B_VEC_COLS;

    int threadRowGlobalOriginA = startRow;
    int threadColGlobalOriginA = chunk;

    int threadRowGlobalOriginB = chunk;
    int threadColGlobalOriginB = startCol;

    for (int idx = tid; idx < NUM_LOADS_A; idx += NumThreads) {
        int rowTileA = idx / A_VEC_COLS;
        int vecColA  = idx % A_VEC_COLS;
        int colTileA = vecColA * VEC_ELEMS;

        int rowA = threadRowGlobalOriginA + rowTileA;
        int colA = threadColGlobalOriginA + colTileA;

        load_to_shared_transposed<InputT>(
            A,
            rowA, colA, K,
            M, K,
            &ATileT[0][0],
            SUBDIM_M + PAD,
            rowTileA,
            colTileA
        );
    }
    for (int idx = tid; idx < NUM_LOADS_B; idx += NumThreads) {
        int rowTileB = idx / B_VEC_COLS;
        int vecColB  = idx % B_VEC_COLS;
        int colTileB = vecColB * VEC_ELEMS;

        int rowB = threadRowGlobalOriginB + rowTileB;
        int colB = threadColGlobalOriginB + colTileB;

        load_to_shared<InputT>(
            B,
            rowB, colB, N,
            K, N,
            &BTile[rowTileB][0],
            colTileB
        );
    }
}