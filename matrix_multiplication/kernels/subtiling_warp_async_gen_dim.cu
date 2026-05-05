#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include <cstdio>
#include "config.h"
#include "kernels.cuh"
#include "param_init.cuh"
#include "load_helpers.cuh"
#include "compute_helpers.cuh"
#include "store_helpers.cuh"

__device__ __forceinline__
void load_subtile_linear_slab_gendim_async(
    const float* __restrict__ A,
    float ATile[SUBTILE_MN][SUBTILE_K + PADDING_GEN_DIM],
    const float* __restrict__ B,
    float BTile[SUBTILE_K][SUBTILE_MN + PADDING_GEN_DIM],
    int M, int K, int N,
    int startRow, int startCol,
    int chunk,
    const SlabParamsGenDim& params)
{
    constexpr int VEC = 4;
    constexpr int WARP_SIZE = 32;

    constexpr int A_VEC_COLS = SUBTILE_K  / VEC;
    constexpr int B_VEC_COLS = SUBTILE_MN / VEC;

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
        int rowTileA = A_SLAB_ROWS * slabNum + params.slabRowIdxA;
        int colTileA = VEC * params.slabColIdxA;

        int rowA = threadRowGlobalOriginA + rowTileA;
        int colA = threadColGlobalOriginA + colTileA;

        load_vec4_or_scalar_to_shared_async(
            A,
            rowA, colA, K,
            M, K,
            &ATile[rowTileA][0],
            colTileA
        );

        int rowTileB = B_SLAB_ROWS * slabNum + params.slabRowIdxB;
        int colTileB = VEC * params.slabColIdxB;

        int rowB = threadRowGlobalOriginB + rowTileB;
        int colB = threadColGlobalOriginB + colTileB;

        load_vec4_or_scalar_to_shared_async(
            B,
            rowB, colB, N,
            K, N,
            &BTile[rowTileB][0],
            colTileB
        );
    }
}


__global__
void GEMMSubTilingLoadSlabGenDimsDoubleBuffered(int M, int N, int K,
                                                float alpha,
                                                const float* __restrict__ A,
                                                const float* __restrict__ B,
                                                float beta,
                                                float* __restrict__ C)
{
    __shared__ float ATile[2][SUBTILE_MN][SUBTILE_K + PADDING_GEN_DIM];
    __shared__ float BTile[2][SUBTILE_K][SUBTILE_MN + PADDING_GEN_DIM];

    int startRow = blockIdx.y * SUBTILE_MN;
    int startCol = blockIdx.x * SUBTILE_MN;

    SlabParamsGenDim params = make_linear_slab_params_gendim();

    float sum[SUB][SUB];

#pragma unroll
    for (int i = 0; i < SUB; i++) {
#pragma unroll
        for (int j = 0; j < SUB; j++) {
            sum[i][j] = 0.0f;
        }
    }

    int stage = 0;

    load_subtile_linear_slab_gendim_async(
        A, ATile[stage],
        B, BTile[stage],
        M, K, N,
        startRow, startCol,
        0,
        params
    );

    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();

    for (int chunk = 0; chunk < K; chunk += SUBTILE_K) {

        int nextChunk = chunk + SUBTILE_K;
        int nextStage = stage ^ 1;

        if (nextChunk < K) {
            load_subtile_linear_slab_gendim_async(
                A, ATile[nextStage],
                B, BTile[nextStage],
                M, K, N,
                startRow, startCol,
                nextChunk,
                params
            );

            __pipeline_commit();
        }

        __syncthreads();

        int kmax = min(SUBTILE_K, K - chunk);

        compute_subtile_regtangluar(
            ATile[stage],
            BTile[stage],
            kmax,
            sum,
            params
        );

        if (nextChunk < K) {
            __pipeline_wait_prior(0);
            __syncthreads();
        }

        stage = nextStage;
    }

    store_subtile_vec4(
        sum, C, M, N,
        startRow, startCol,
        params.threadRowTile,
        params.threadColTile,
        alpha, beta
    );
}