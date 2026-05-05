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
void load_subtile_linear_slab_async(
    const float* __restrict__ A,
    float ATile[SUBTILE][SUBTILE + PADDING],
    const float* __restrict__ B,
    float BTile[SUBTILE][SUBTILE + PADDING],
    int M, int K, int N,
    int startRow, int startCol,
    int chunk,
    const SlabParams& params)
{
    constexpr int SLAB_DIM_ROWS = (SUBTILE + 3) >> 2;
    constexpr int SLAB_DIM_COLS = (SUBTILE + 31) >> 5;
    constexpr int TOTAL_SLABS   = SLAB_DIM_ROWS * SLAB_DIM_COLS;

    int threadRowGlobalOriginA = startRow;
    int threadColGlobalOriginA = chunk;

    int threadRowGlobalOriginB = chunk;
    int threadColGlobalOriginB = startCol;

    for (int slabNum = params.warpId;
         slabNum < TOTAL_SLABS;
         slabNum += params.numWarps)
    {
        int slabRowStart = slabNum / SLAB_DIM_COLS;
        int slabColStart = slabNum % SLAB_DIM_COLS;

        int rowTile = 4 * slabRowStart + params.slabRowIdx;
        int colTile = 32 * slabColStart + 4 * params.slabColIdx;

        int rowA = threadRowGlobalOriginA + rowTile;
        int colA = threadColGlobalOriginA + colTile;

        load_vec4_or_scalar_to_shared_async(
            A,
            rowA, colA, K,
            M, K,
            &ATile[rowTile][0],
            colTile
        );

        int rowB = threadRowGlobalOriginB + rowTile;
        int colB = threadColGlobalOriginB + colTile;

        load_vec4_or_scalar_to_shared_async(
            B,
            rowB, colB, N,
            K, N,
            &BTile[rowTile][0],
            colTile
        );
    }
}


__global__
void GEMMSubTilingLoadSlabLinearAsync(int M, int N, int K,
                                      float alpha,
                                      const float* __restrict__ A,
                                      const float* __restrict__ B,
                                      float beta,
                                      float* __restrict__ C)
{
    __shared__ float ATile[2][SUBTILE][SUBTILE + PADDING];
    __shared__ float BTile[2][SUBTILE][SUBTILE + PADDING];

    int startRow = blockIdx.y * SUBTILE;
    int startCol = blockIdx.x * SUBTILE;

    SlabParams params = make_linear_slab_params();

    float sum[SUB][SUB];

#pragma unroll
    for (int i = 0; i < SUB; i++) {
#pragma unroll
        for (int j = 0; j < SUB; j++) {
            sum[i][j] = 0.0f;
        }
    }

    int stage = 0;

    // Preload first tile.
    load_subtile_linear_slab_async(
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

    for (int chunk = 0; chunk < K; chunk += SUBTILE) {
        int nextChunk = chunk + SUBTILE;
        int nextStage = stage ^ 1;

        if (nextChunk < K) {
            load_subtile_linear_slab_async(
                A, ATile[nextStage],
                B, BTile[nextStage],
                M, K, N,
                startRow, startCol,
                nextChunk,
                params
            );

            __pipeline_commit();
        }

        int kmax = min(SUBTILE, K - chunk);

        compute_subtile(
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